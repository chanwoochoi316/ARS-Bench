from __future__ import annotations
from typing import List, Sequence, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
DEFAULT_PIGUARD_MODEL_PATH = "models/piguard"

class PIGuardDefender:
    """Filter retrieved passages with a local pretrained PIGuard checkpoint."""

    def __init__(
        self,
        model_path: str = DEFAULT_PIGUARD_MODEL_PATH,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ):
        self.model_path = model_path
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))

        # local_files_only=True prevents accidental network access and guarantees
        # that the explicitly requested local snapshot is the one being used.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Official PIGuard config: 0=benign, 1=injection. Resolve by label name
        # so the adapter remains robust if the numeric mapping is represented
        # with string keys in config.json.
        label2id = {
            str(k).lower(): int(v)
            for k, v in (getattr(self.model.config, "label2id", {}) or {}).items()
        }
        self.injection_label_id = label2id.get("injection", 1)
        self.benign_label_id = label2id.get("benign", 0)

        print(
            "[PIGuard] Loaded local model "
            f"from {self.model_path} on {self.device} "
            f"(max_length={self.max_length}, batch_size={self.batch_size})"
        )

    def _predict_batch(self, passages: Sequence[str]) -> List[dict]:
        encoded = self.tokenizer(
            list(passages),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)

        id2label = getattr(self.model.config, "id2label", {}) or {}
        predictions = []
        for row_idx, pred_id_tensor in enumerate(pred_ids):
            pred_id = int(pred_id_tensor.item())
            label = id2label.get(pred_id, id2label.get(str(pred_id), str(pred_id)))
            score = float(probs[row_idx, pred_id].item())
            predictions.append(
                {
                    "label_id": pred_id,
                    "label": str(label),
                    "score": score,
                }
            )
        return predictions

    def defend(
        self,
        query: str,
        passages: Sequence[str],
        return_indices: bool = False,
    ) -> Union[List[str], Tuple[List[str], List[int]]]:
        """Return passages after removing those PIGuard predicts as injection.

        Args:
            query: Current retrieval query. Kept only for compatibility with
                E5WikiEnv/RAGDefender's interface; PIGuard does not use it.
            passages: Retrieved documents, normally "title\\ncontents" strings.
            return_indices: If True, also return original passage indices that
                were removed. This is the mode E5WikiEnv currently uses.
        """
        del query  # Explicitly document that query is not used by PIGuard.

        passages = list(passages)
        if not passages:
            return ([], []) if return_indices else []

        predictions: List[dict] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            predictions.extend(self._predict_batch(batch))

        removed_indices: List[int] = []
        filtered_passages: List[str] = []

        for idx, (passage, pred) in enumerate(zip(passages, predictions)):
            is_injection = pred["label_id"] == self.injection_label_id
            decision = "REMOVE" if is_injection else "KEEP"
            print(
                f"[PIGuard] doc={idx} label={pred['label']} "
                f"score={pred['score']:.6f} decision={decision}"
            )

            if is_injection:
                removed_indices.append(idx)
            else:
                filtered_passages.append(passage)

        if return_indices:
            return filtered_passages, removed_indices
        return filtered_passages
