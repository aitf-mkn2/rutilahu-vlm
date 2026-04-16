from typing import Any

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    
def compute_konflik_metrics(y_true: list[bool], y_pred: list[bool]) -> dict[str, Any]:
    """
    Metrik untuk prediksi konflik dinding (binary classification).
    Recall diprioritaskan agar kasus konflik tidak terlewat.
    """
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_samples": 0,
        }

    yt = [int(v) for v in y_true]
    yp = [int(v) for v in y_pred]

    if HAS_SKLEARN:
        precision = float(precision_score(yt, yp, zero_division=0))
        recall = float(recall_score(yt, yp, zero_division=0))
        f1 = float(f1_score(yt, yp, zero_division=0))
        accuracy = float(accuracy_score(yt, yp))
    else:
        tp = sum(t == 1 and p == 1 for t, p in zip(yt, yp))
        fp = sum(t == 0 and p == 1 for t, p in zip(yt, yp))
        fn = sum(t == 1 and p == 0 for t, p in zip(yt, yp))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        accuracy = sum(t == p for t, p in zip(yt, yp)) / len(yt)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_samples": len(y_true),
    }
