from collections import Counter


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.lower().strip().split())


# ==========================================================
# GENERATION METRICS
# ==========================================================

def exact_match(prediction: str, expected: str) -> bool:
    """Returns True if prediction exactly matches expected."""
    return normalize(prediction) == normalize(expected)


def token_overlap(prediction: str, expected: str) -> float:
    """Computes token overlap ratio."""

    pred = set(normalize(prediction).split())
    exp = set(normalize(expected).split())

    if not exp:
        return 0.0

    return len(pred & exp) / len(exp)


def precision(prediction: str, expected: str) -> float:
    """Computes token-level precision."""

    pred = normalize(prediction).split()
    exp = normalize(expected).split()

    if not pred:
        return 0.0

    pred_counter = Counter(pred)
    exp_counter = Counter(exp)

    common = pred_counter & exp_counter
    tp = sum(common.values())

    return tp / len(pred)


def recall(prediction: str, expected: str) -> float:
    """Computes token-level recall."""

    pred = normalize(prediction).split()
    exp = normalize(expected).split()

    if not exp:
        return 0.0

    pred_counter = Counter(pred)
    exp_counter = Counter(exp)

    common = pred_counter & exp_counter
    tp = sum(common.values())

    return tp / len(exp)


def f1_score(prediction: str, expected: str) -> float:
    """Computes F1 score."""

    p = precision(prediction, expected)
    r = recall(prediction, expected)

    if (p + r) == 0:
        return 0.0

    return (2 * p * r) / (p + r)


# ==========================================================
# RETRIEVAL METRICS
# ==========================================================

def top_k_hit(expected_chunks: list, retrieved_chunks: list, k: int) -> bool:
    """
    Returns True if any expected chunk appears
    within the first k retrieved chunks.
    """

    retrieved_ids = [
        item["chunk_id"]
        for item in retrieved_chunks[:k]
    ]

    return any(
        chunk in retrieved_ids
        for chunk in expected_chunks
    )


def retrieval_accuracy(correct: int, total: int) -> float:
    """Computes retrieval accuracy."""

    if total == 0:
        return 0.0

    return correct / total


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    expected = (
        "A database management system stores and manages data."
    )

    prediction = (
        "Database management system stores data."
    )

    print("=" * 60)
    print("GENERATION EVALUATION METRICS")
    print("=" * 60)

    print("Exact Match   :", exact_match(prediction, expected))
    print("Token Overlap :", round(token_overlap(prediction, expected), 4))
    print("Precision     :", round(precision(prediction, expected), 4))
    print("Recall        :", round(recall(prediction, expected), 4))
    print("F1 Score      :", round(f1_score(prediction, expected), 4))

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION METRICS")
    print("=" * 60)

    expected_chunks = [
        "AI_chunk_1"
    ]

    retrieved_chunks = [
        {"chunk_id": "AI_chunk_1"},
        {"chunk_id": "AI_chunk_4"},
        {"chunk_id": "AI_chunk_8"},
    ]

    print("Top-1 Hit :", top_k_hit(expected_chunks, retrieved_chunks, 1))
    print("Top-3 Hit :", top_k_hit(expected_chunks, retrieved_chunks, 3))
    print("Top-5 Hit :", top_k_hit(expected_chunks, retrieved_chunks, 5))

    print("Retrieval Accuracy :", round(retrieval_accuracy(8, 10), 4))

    print("=" * 60)