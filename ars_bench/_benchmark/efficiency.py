import re
from typing import Any

def estimate_token_count(text: Any) -> int:
    """Estimate subword tokens using the existing benchmark heuristic."""
    text = str(text or "")
    if not text:
        return 0

    word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
    punctuation_count = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))
    non_ascii_count = sum(
        1 for character in text if not character.isascii() and not character.isspace()
    )
    return max(
        1,
        round(
            max(
                len(text) / 4,
                word_count / 0.75,
                punctuation_count,
                non_ascii_count / 1.5,
            )
        ),
    )

def relative_increase(value: float, baseline: float, metric_name: str) -> float:
    if baseline == 0:
        raise ValueError(f"Clean {metric_name} is zero; relative increase is undefined")
    return (value - baseline) / baseline
