from typing import List, Optional, Sequence, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
DEFAULT_PIGUARD_MODEL_PATH = None

class PIGuardDefender:
    """Filter passages classified as prompt injection by the pretrained PIGuard model."""

    def __init__(
        self,
        model_path: str = DEFAULT_PIGUARD_MODEL_PATH,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
        threshold: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        pipeline_device = self._pipeline_device(device)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
        )

    @staticmethod
    def _pipeline_device(device: str) -> Union[int, str]:
        """Convert user-friendly device names to a transformers pipeline device."""
        text = str(device).strip().lower()
        if text in {"cpu", "-1"}:
            return -1
        if text == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PIGuard requested CUDA, but torch.cuda.is_available() is False.")
            return 0
        if text.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("PIGuard requested CUDA, but torch.cuda.is_available() is False.")
            try:
                return int(text.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid PIGuard device: {device!r}") from exc
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported PIGuard device {device!r}; use 'cpu', 'cuda', or 'cuda:N'."
            ) from exc

    @staticmethod
    def _is_injection_label(label: object) -> bool:
        """Support both PIGuard's named label and generic HF LABEL_1 fallback."""
        normalized = str(label).strip().lower()
        return normalized in {"injection", "label_1", "1"}

    def defend(
        self,
        query: str,
        passages: Sequence[str],
        return_indices: bool = False,
    ):
        """Return benign passages and, optionally, indices removed as injections.

        ``query`` is accepted for interface compatibility with RAGDefender. PIGuard
        classifies each retrieved passage independently and does not use the query.
        """
        del query

        passages = list(passages)
        if not passages:
            return (passages, []) if return_indices else passages

        predictions = self.classifier(
            passages,
            truncation=True,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )

        removed_indices: List[int] = []
        kept_passages: List[str] = []

        for index, (passage, prediction) in enumerate(zip(passages, predictions)):
            label = prediction.get("label", "")
            score = float(prediction.get("score", 0.0))
            is_injection = self._is_injection_label(label)

            if self.threshold is not None:
                is_injection = is_injection and score >= float(self.threshold)

            if is_injection:
                removed_indices.append(index)
            else:
                kept_passages.append(passage)

        if return_indices:
            return kept_passages, removed_indices
        return kept_passages
