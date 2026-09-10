import csv
from pathlib import Path

REPORT_PATH = Path("reports/retrieval_per_query.csv")


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    rows = []

    with REPORT_PATH.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["method"] != "reranked":
                continue

            score_raw = row["top1_score"]

            if not score_raw:
                continue

            rows.append(
                {
                    "score": float(score_raw),
                    "answerable": (row["answerable"].strip().lower() == "true"),
                }
            )

    if not rows:
        raise RuntimeError("No reranked rows found.")

    scores = sorted({row["score"] for row in rows})

    candidates = [
        scores[0] - 1e-6,
        *scores,
        scores[-1] + 1e-6,
    ]

    best = None

    for threshold in candidates:
        tp = 0
        fp = 0
        fn = 0

        for row in rows:
            predicted = row["score"] >= threshold

            actual = row["answerable"]

            if predicted and actual:
                tp += 1

            elif predicted and not actual:
                fp += 1

            elif not predicted and actual:
                fn += 1

        precision = safe_divide(
            tp,
            tp + fp,
        )

        recall = safe_divide(
            tp,
            tp + fn,
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if best is None or result["f1"] > best["f1"]:
            best = result

    print("BEST THRESHOLD")
    print()

    print(f"threshold: {best['threshold']:.6f}")

    print(f"precision: {best['precision']:.3f}")

    print(f"recall: {best['recall']:.3f}")

    print(f"F1: {best['f1']:.3f}")

    print()
    print("Important: this is calibrated on a very small demo dataset.")


if __name__ == "__main__":
    main()
