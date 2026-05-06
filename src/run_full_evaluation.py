import subprocess
import sys
from pathlib import Path


# ============================================================
# NeuroMemory Robot - Full Evaluation Runner
#
# Runs all analysis scripts and regenerates final tables/figures.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "src/dataset_feature_validation.py",
    "src/pretrained_embedding_validation.py",
    "src/reid_specific_embedding_validation.py",
    "src/reid_query_gallery_validation.py",
]


def run_script(script_path):
    print("\n" + "=" * 70)
    print(f"Running: {script_path}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_path}")

    print(f"Finished: {script_path}")


def main():
    print("NeuroMemory Robot - Full Evaluation Runner")
    print(f"Project root: {PROJECT_ROOT}")

    for script in SCRIPTS:
        run_script(script)

    print("\n" + "=" * 70)
    print("All evaluations completed successfully.")
    print("=" * 70)

    print("\nGenerated key outputs:")
    print("- outputs/tables/dataset_validation_summary.csv")
    print("- outputs/tables/threshold_calibration_summary.csv")
    print("- outputs/tables/pretrained_embedding_summary.csv")
    print("- outputs/tables/reid_specific_embedding_summary.csv")
    print("- outputs/tables/query_gallery_retrieval_summary.csv")
    print("- outputs/figures/same_vs_different_similarity_graph.png")
    print("- outputs/figures/threshold_calibration_curve.png")
    print("- outputs/figures/reid_method_comparison_graph.png")
    print("- outputs/figures/query_gallery_method_comparison.png")
    print("- outputs/figures/query_gallery_retrieval_example.png")


if __name__ == "__main__":
    main()