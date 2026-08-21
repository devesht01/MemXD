import re
from collections import defaultdict

_USER_PREFIX = re.compile(r"^(the\s+)?user\s+", re.IGNORECASE)


def strip_user_prefix(text: str) -> str:
    return _USER_PREFIX.sub("", text, count=1).strip()


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)


def rrf_fuse(
    rankings: list[list[str]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must be the same length")

    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += weight * (1.0 / (k + rank + 1))

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
