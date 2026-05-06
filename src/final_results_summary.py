import csv
from pathlib import Path


# ============================================================
# NeuroMemory Robot - Final Results Summary
#
# Collects the main simulation, dataset validation, embedding,
# and query-gallery retrieval results into one final CSV table.
# ============================================================

OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

FINAL_SUMMARY_PATH = OUTPUT_TABLES / "final_results_summary.csv"


def read_csv_first_row(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        return None

    return rows[0]


def read_csv_rows(path):
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        return list(reader)


def percent(value):
    return round(float(value) * 100.0, 2)


def add_row(rows, level, module, metric, value, unit, interpretation):
    rows.append({
        "level": level,
        "module": module,
        "metric": metric,
        "value": value,
        "unit": unit,
        "interpretation": interpretation,
    })


def main():
    final_rows = []

    # --------------------------------------------------------
    # Level 2: External dataset validation
    # --------------------------------------------------------
    dataset_summary = read_csv_rows(OUTPUT_TABLES / "dataset_validation_summary.csv")

    same_row = None
    diff_row = None
    sep_row = None

    for row in dataset_summary:
        if row.get("group") == "same_identity_pairs":
            same_row = row
        elif row.get("group") == "different_identity_pairs":
            diff_row = row
        elif row.get("group") == "separation_margin":
            sep_row = row

    if same_row:
        add_row(
            final_rows,
            "Level 2",
            "External ReID feature validation",
            "Same-identity mean similarity",
            round(float(same_row["mean_similarity"]) * 100.0, 2),
            "%",
            "Average similarity between images of the same identity."
        )

    if diff_row:
        add_row(
            final_rows,
            "Level 2",
            "External ReID feature validation",
            "Different-identity mean similarity",
            round(float(diff_row["mean_similarity"]) * 100.0, 2),
            "%",
            "Average similarity between images of different identities."
        )

    if sep_row:
        add_row(
            final_rows,
            "Level 2",
            "External ReID feature validation",
            "Separation margin",
            round(float(sep_row["mean_similarity"]) * 100.0, 2),
            "%",
            "Difference between same-identity and different-identity mean similarity."
        )

    threshold_row = read_csv_first_row(OUTPUT_TABLES / "threshold_calibration_summary.csv")

    if threshold_row:
        add_row(
            final_rows,
            "Level 2",
            "Threshold calibration",
            "Selected similarity threshold",
            threshold_row["selected_threshold"],
            "score",
            "Threshold calibrated from same/different identity distributions."
        )

        add_row(
            final_rows,
            "Level 2",
            "Threshold calibration",
            "Accuracy",
            round(float(threshold_row["accuracy"]) * 100.0, 2),
            "%",
            "Binary same/different decision accuracy at the selected threshold."
        )

        add_row(
            final_rows,
            "Level 2",
            "Threshold calibration",
            "Precision",
            round(float(threshold_row["precision"]) * 100.0, 2),
            "%",
            "How often predicted same-identity matches were correct."
        )

        add_row(
            final_rows,
            "Level 2",
            "Threshold calibration",
            "Recall",
            round(float(threshold_row["recall"]) * 100.0, 2),
            "%",
            "How many true same-identity pairs were recovered."
        )

        add_row(
            final_rows,
            "Level 2",
            "Threshold calibration",
            "F1 score",
            round(float(threshold_row["f1_score"]) * 100.0, 2),
            "%",
            "Balance between precision and recall."
        )

    # --------------------------------------------------------
    # Level 3: Deep embedding validation
    # --------------------------------------------------------
    resnet_row = read_csv_first_row(OUTPUT_TABLES / "pretrained_embedding_summary.csv")

    if resnet_row:
        add_row(
            final_rows,
            "Level 3",
            "Generic ResNet18 embedding",
            "Same-identity mean similarity",
            round(float(resnet_row["same_identity_mean_similarity"]) * 100.0, 2),
            "%",
            "Generic ImageNet-pretrained embedding similarity for same identities."
        )

        add_row(
            final_rows,
            "Level 3",
            "Generic ResNet18 embedding",
            "Different-identity mean similarity",
            round(float(resnet_row["different_identity_mean_similarity"]) * 100.0, 2),
            "%",
            "Generic ImageNet-pretrained embedding similarity for different identities."
        )

        add_row(
            final_rows,
            "Level 3",
            "Generic ResNet18 embedding",
            "Separation margin",
            round(float(resnet_row["separation_margin"]) * 100.0, 2),
            "%",
            "Shows that generic embeddings are less suitable for ReID without task-specific training."
        )

    osnet_row = read_csv_first_row(OUTPUT_TABLES / "reid_specific_embedding_summary.csv")

    if osnet_row:
        add_row(
            final_rows,
            "Level 3",
            "ReID-specific OSNet embedding",
            "Same-identity mean similarity",
            round(float(osnet_row["same_identity_mean_similarity"]) * 100.0, 2),
            "%",
            "OSNet embedding similarity for same identities."
        )

        add_row(
            final_rows,
            "Level 3",
            "ReID-specific OSNet embedding",
            "Different-identity mean similarity",
            round(float(osnet_row["different_identity_mean_similarity"]) * 100.0, 2),
            "%",
            "OSNet embedding similarity for different identities."
        )

        add_row(
            final_rows,
            "Level 3",
            "ReID-specific OSNet embedding",
            "Separation margin",
            round(float(osnet_row["separation_margin"]) * 100.0, 2),
            "%",
            "ReID-specific embedding provides better separation than generic ResNet18."
        )

    retrieval_rows = read_csv_rows(OUTPUT_TABLES / "query_gallery_retrieval_summary.csv")

    for row in retrieval_rows:
        method = row["method"]

        add_row(
            final_rows,
            "Level 3",
            f"Query-gallery retrieval / {method}",
            "Top-1 accuracy",
            round(float(row["top1_accuracy"]) * 100.0, 2),
            "%",
            "Correct identity retrieved at rank 1."
        )

        add_row(
            final_rows,
            "Level 3",
            f"Query-gallery retrieval / {method}",
            "Top-3 accuracy",
            round(float(row["top3_accuracy"]) * 100.0, 2),
            "%",
            "Correct identity retrieved within the first three ranked candidates."
        )

        add_row(
            final_rows,
            "Level 3",
            f"Query-gallery retrieval / {method}",
            "Top-5 accuracy",
            round(float(row["top5_accuracy"]) * 100.0, 2),
            "%",
            "Correct identity retrieved within the first five ranked candidates."
        )

        add_row(
            final_rows,
            "Level 3",
            f"Query-gallery retrieval / {method}",
            "Mean reciprocal rank",
            round(float(row["mean_reciprocal_rank"]) * 100.0, 2),
            "%",
            "Ranking quality metric; higher means correct identity appears earlier."
        )

        if method == "Hybrid NeuroMemory":
            add_row(
                final_rows,
                "Level 3",
                "Hybrid NeuroMemory",
                "Selected OSNet weight",
                row.get("selected_osnet_weight", ""),
                "weight",
                "Selected contribution of the ReID-specific embedding signal."
            )

            add_row(
                final_rows,
                "Level 3",
                "Hybrid NeuroMemory",
                "Selected explainable feature weight",
                row.get("selected_explainable_weight", ""),
                "weight",
                "Selected contribution of explainable body-region visual memory features."
            )

    # --------------------------------------------------------
    # Write final summary
    # --------------------------------------------------------
    with open(FINAL_SUMMARY_PATH, "w", newline="", encoding="utf-8") as fp:
        fieldnames = [
            "level",
            "module",
            "metric",
            "value",
            "unit",
            "interpretation",
        ]

        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved final summary: {FINAL_SUMMARY_PATH}")
    print(f"Total rows: {len(final_rows)}")

    print("\nKey final results:")
    for row in final_rows:
        if row["metric"] in [
            "Separation margin",
            "Top-1 accuracy",
            "Top-3 accuracy",
            "Selected similarity threshold",
            "F1 score",
        ]:
            print(
                f"- {row['module']} | {row['metric']}: "
                f"{row['value']} {row['unit']}"
            )


if __name__ == "__main__":
    main()