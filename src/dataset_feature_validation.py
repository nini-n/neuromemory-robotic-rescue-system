import os
import csv
import math
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# NeuroMemory Robot - Enhanced Public ReID Dataset Validation
#
# Purpose:
# Validate visual-memory similarity on a public person ReID subset
# using explainable body-region features.
#
# Features:
# - Full-body HSV histogram
# - Upper-body HSV histogram
# - Lower-body HSV histogram
# - Brightness, contrast, texture
# - Aspect ratio
# - Threshold calibration
# - Same/different identity evaluation
#
# Important:
# This is not face recognition.
# It is feature-level visual-memory validation.
# Final identity decisions remain human-supervised.
# ============================================================

DATA_DIR = Path("data/sample_reid")
OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_FIGURES = Path("outputs/figures")

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Professional plotting style
# ------------------------------------------------------------

COLORS = {
    "navy": "#14213D",
    "blue": "#2F6690",
    "teal": "#2A9D8F",
    "green": "#52B788",
    "amber": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "red": "#C44536",
    "slate": "#5C677D",
    "light_slate": "#94A3B8",
    "bg": "#F7F9FC",
    "grid": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
    "white": "#FFFFFF",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": COLORS["white"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["light_slate"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "text.color": COLORS["text"],
})


def style_axes(ax, both_axes=False):
    ax.set_facecolor(COLORS["bg"])

    if both_axes:
        ax.grid(True, linestyle="--", linewidth=0.8, color=COLORS["grid"], alpha=0.65)
    else:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, color=COLORS["grid"], alpha=0.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["light_slate"])
    ax.spines["bottom"].set_color(COLORS["light_slate"])


def add_footnote(fig, text):
    fig.text(
        0.5,
        0.014,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"],
    )


# ============================================================
# Utility functions
# ============================================================

def parse_identity(filename: str) -> str:
    """
    Expected selected filename:
    id_0001_sample_1_0001_c1s1_001051_03.jpg
    """
    parts = filename.split("_")
    if len(parts) >= 2 and parts[0] == "id":
        return parts[1]

    # Fallback for raw Market-1501 filenames:
    return filename.split("_")[0]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_image(path: Path, size=(128, 256)):
    """
    Person ReID crops are usually tall.
    We resize to width=128, height=256 to preserve person-like shape.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return np.asarray(img).astype(np.float32) / 255.0


def rgb_to_hsv_np(rgb):
    """
    Lightweight RGB to HSV conversion using matplotlib.
    Input and output ranges are [0, 1].
    """
    return mpl.colors.rgb_to_hsv(rgb)


def normalized_hist(values, bins, value_range=(0.0, 1.0)):
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist


def hsv_histogram(region, h_bins=16, s_bins=8, v_bins=8):
    """
    HSV histogram for a body region.
    H is more important for clothing color.
    """
    hsv = rgb_to_hsv_np(region)

    h_hist = normalized_hist(hsv[:, :, 0].ravel(), bins=h_bins)
    s_hist = normalized_hist(hsv[:, :, 1].ravel(), bins=s_bins)
    v_hist = normalized_hist(hsv[:, :, 2].ravel(), bins=v_bins)

    return np.concatenate([h_hist, s_hist, v_hist])


def brightness_contrast(img):
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return float(gray.mean()), float(gray.std())


def texture_density(img):
    """
    Simple texture/silhouette proxy from finite differences.
    """
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()

    return float(dx + dy)


def histogram_intersection(h1, h2):
    return float(np.minimum(h1, h2).sum() / (np.maximum(h1, h2).sum() + 1e-8))


def scalar_similarity(a, b, scale):
    return clamp(1.0 - abs(a - b) / scale, 0.0, 1.0)


# ============================================================
# Feature extraction
# ============================================================

def split_body_regions(img):
    """
    Splits person crop into upper and lower body regions.

    Market-1501 images are already person bounding boxes.
    Approximate split:
    - top 10% ignored slightly because head/background may vary
    - upper body: 15% to 55%
    - lower body: 55% to 95%
    """
    h, w = img.shape[:2]

    y_upper_start = int(0.15 * h)
    y_upper_end = int(0.55 * h)
    y_lower_start = int(0.55 * h)
    y_lower_end = int(0.95 * h)

    upper = img[y_upper_start:y_upper_end, :, :]
    lower = img[y_lower_start:y_lower_end, :, :]

    return upper, lower


def extract_features(path: Path):
    img = load_image(path)

    upper, lower = split_body_regions(img)

    full_hsv = hsv_histogram(img)
    upper_hsv = hsv_histogram(upper)
    lower_hsv = hsv_histogram(lower)

    brightness, contrast = brightness_contrast(img)
    texture = texture_density(img)

    h, w = img.shape[:2]
    aspect_ratio = float(h / w)

    return {
        "path": str(path),
        "filename": path.name,
        "identity": parse_identity(path.name),
        "full_hsv": full_hsv,
        "upper_hsv": upper_hsv,
        "lower_hsv": lower_hsv,
        "brightness": brightness,
        "contrast": contrast,
        "texture": texture,
        "aspect_ratio": aspect_ratio,
    }


def pair_similarity(f1, f2):
    full_color_sim = histogram_intersection(f1["full_hsv"], f2["full_hsv"])
    upper_color_sim = histogram_intersection(f1["upper_hsv"], f2["upper_hsv"])
    lower_color_sim = histogram_intersection(f1["lower_hsv"], f2["lower_hsv"])

    brightness_sim = scalar_similarity(f1["brightness"], f2["brightness"], scale=0.45)
    contrast_sim = scalar_similarity(f1["contrast"], f2["contrast"], scale=0.35)
    texture_sim = scalar_similarity(f1["texture"], f2["texture"], scale=0.25)
    aspect_sim = scalar_similarity(f1["aspect_ratio"], f2["aspect_ratio"], scale=1.0)

    # Body-region visual memory score.
    # Upper and lower clothing regions are weighted strongly.
    weighted = (
        0.18 * full_color_sim
        + 0.26 * upper_color_sim
        + 0.26 * lower_color_sim
        + 0.08 * brightness_sim
        + 0.08 * contrast_sim
        + 0.09 * texture_sim
        + 0.05 * aspect_sim
    )

    return {
        "full_color_similarity": full_color_sim,
        "upper_body_similarity": upper_color_sim,
        "lower_body_similarity": lower_color_sim,
        "brightness_similarity": brightness_sim,
        "contrast_similarity": contrast_sim,
        "texture_similarity": texture_sim,
        "aspect_similarity": aspect_sim,
        "weighted_similarity": weighted,
    }


def decision_from_similarity(score, threshold):
    if score >= threshold:
        return "probable same identity"
    elif score >= threshold - 0.12:
        return "human verification advised"
    elif score >= threshold - 0.25:
        return "uncertain / re-observe"
    else:
        return "different or insufficient evidence"


# ============================================================
# Threshold calibration
# ============================================================

def compute_binary_metrics(pair_rows, threshold):
    """
    Treat same identity as positive class.
    Similarity >= threshold predicts same identity.
    """
    tp = fp = tn = fn = 0

    for row in pair_rows:
        actual_same = bool(row["same_identity"])
        predicted_same = row["weighted_similarity"] >= threshold

        if actual_same and predicted_same:
            tp += 1
        elif not actual_same and predicted_same:
            fp += 1
        elif not actual_same and not predicted_same:
            tn += 1
        elif actual_same and not predicted_same:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_positive_count": tp,
        "true_negative_count": tn,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def calibrate_threshold(pair_rows):
    thresholds = np.arange(0.45, 0.96, 0.01)
    rows = []

    for threshold in thresholds:
        metrics = compute_binary_metrics(pair_rows, float(threshold))
        rows.append(metrics)

    # Conservative choice:
    # maximize F1 first, then accuracy, then prefer higher threshold.
    best = sorted(
        rows,
        key=lambda r: (r["f1_score"], r["accuracy"], r["threshold"]),
        reverse=True
    )[0]

    return best, rows


# ============================================================
# Plotting
# ============================================================

def plot_same_vs_different(same_scores, diff_scores):
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    style_axes(ax)

    data = [
        [s * 100 for s in same_scores],
        [s * 100 for s in diff_scores],
    ]

    box = ax.boxplot(
        data,
        labels=["Same identity", "Different identity"],
        patch_artist=True,
        widths=0.48,
        showmeans=True,
        meanline=True,
    )

    box_colors = [COLORS["teal"], COLORS["coral"]]
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(COLORS["navy"])

    for median in box["medians"]:
        median.set_color(COLORS["navy"])
        median.set_linewidth(2)

    for mean in box["means"]:
        mean.set_color(COLORS["white"])
        mean.set_linewidth(2)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(data, start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=26,
            color=COLORS["navy"],
            alpha=0.55,
            zorder=3,
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Body-region visual-memory similarity (%)")
    ax.set_title(
        "Public ReID Subset: Body-Region Feature Similarity",
        pad=14,
        fontweight="bold",
    )

    add_footnote(
        fig,
        "External validation uses explainable body-region features; no face recognition model is used."
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = OUTPUT_FIGURES / "same_vs_different_similarity_graph.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {path}")


def plot_similarity_matrix(features):
    n = len(features)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                sim = pair_similarity(features[i], features[j])
                matrix[i, j] = sim["weighted_similarity"]

    labels = [
        f"{f['identity']}-{idx + 1}"
        for idx, f in enumerate(features)
    ]

    fig, ax = plt.subplots(figsize=(12.0, 10.0))
    im = ax.imshow(matrix * 100, cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    ax.set_title(
        "Pairwise Body-Region Visual-Memory Similarity Matrix",
        pad=14,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity (%)")

    # Annotate only if not too many images.
    if n <= 25:
        for i in range(n):
            for j in range(n):
                value = matrix[i, j] * 100
                color = "white" if value < 55 else "black"
                ax.text(
                    j, i, f"{value:.0f}",
                    ha="center", va="center",
                    fontsize=6,
                    color=color,
                )

    fig.tight_layout()
    path = OUTPUT_FIGURES / "dataset_similarity_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {path}")


def plot_threshold_calibration(calibration_rows, best_threshold):
    thresholds = [r["threshold"] for r in calibration_rows]
    accuracies = [r["accuracy"] * 100 for r in calibration_rows]
    precisions = [r["precision"] * 100 for r in calibration_rows]
    recalls = [r["recall"] * 100 for r in calibration_rows]
    f1s = [r["f1_score"] * 100 for r in calibration_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    style_axes(ax, both_axes=True)

    ax.plot(thresholds, accuracies, linewidth=2.3, color=COLORS["teal"], label="Accuracy")
    ax.plot(thresholds, precisions, linewidth=2.3, color=COLORS["blue"], label="Precision")
    ax.plot(thresholds, recalls, linewidth=2.3, color=COLORS["amber"], label="Recall")
    ax.plot(thresholds, f1s, linewidth=2.6, color=COLORS["coral"], label="F1 score")

    ax.axvline(
        best_threshold,
        linestyle="--",
        linewidth=1.8,
        color=COLORS["navy"],
        alpha=0.85,
    )

    ax.text(
        best_threshold + 0.005,
        8,
        f"selected threshold = {best_threshold:.2f}",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=9,
        color=COLORS["navy"],
    )

    ax.set_xlim(min(thresholds), max(thresholds))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Similarity decision threshold")
    ax.set_ylabel("Metric (%)")
    ax.set_title(
        "Threshold Calibration on Public ReID Subset",
        pad=14,
        fontweight="bold",
    )

    ax.legend(
        frameon=True,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grid"],
        loc="lower left",
    )

    add_footnote(
        fig,
        "Threshold is calibrated from same-identity and different-identity similarity distributions."
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = OUTPUT_FIGURES / "threshold_calibration_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {path}")


# ============================================================
# Main
# ============================================================

def main():
    image_files = sorted([
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(image_files) < 4:
        raise RuntimeError(
            f"Not enough images in {DATA_DIR}. Put at least 4 ReID images first."
        )

    print(f"Found {len(image_files)} images in {DATA_DIR}")

    features = [extract_features(p) for p in image_files]

    # --------------------------------------------------------
    # Save image feature table
    # --------------------------------------------------------
    feature_table = []

    for f in features:
        feature_table.append({
            "filename": f["filename"],
            "identity": f["identity"],
            "brightness": round(f["brightness"], 4),
            "contrast": round(f["contrast"], 4),
            "texture": round(f["texture"], 4),
            "aspect_ratio": round(f["aspect_ratio"], 4),
        })

    feature_csv = OUTPUT_TABLES / "dataset_image_features.csv"
    with open(feature_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=feature_table[0].keys())
        writer.writeheader()
        writer.writerows(feature_table)

    print(f"Saved table: {feature_csv}")

    # --------------------------------------------------------
    # Pairwise validation
    # --------------------------------------------------------
    pair_rows = []

    for f1, f2 in combinations(features, 2):
        sim = pair_similarity(f1, f2)
        same_identity = f1["identity"] == f2["identity"]

        pair_rows.append({
            "image_1": f1["filename"],
            "image_2": f2["filename"],
            "id_1": f1["identity"],
            "id_2": f2["identity"],
            "same_identity": same_identity,
            "full_color_similarity": round(sim["full_color_similarity"], 4),
            "upper_body_similarity": round(sim["upper_body_similarity"], 4),
            "lower_body_similarity": round(sim["lower_body_similarity"], 4),
            "brightness_similarity": round(sim["brightness_similarity"], 4),
            "contrast_similarity": round(sim["contrast_similarity"], 4),
            "texture_similarity": round(sim["texture_similarity"], 4),
            "aspect_similarity": round(sim["aspect_similarity"], 4),
            "weighted_similarity": round(sim["weighted_similarity"], 4),
        })

    # Calibrate threshold before assigning final decision text.
    best_threshold_row, calibration_rows = calibrate_threshold(pair_rows)
    best_threshold = best_threshold_row["threshold"]

    for row in pair_rows:
        row["decision_threshold"] = round(best_threshold, 3)
        row["decision"] = decision_from_similarity(
            row["weighted_similarity"],
            best_threshold
        )

    pair_csv = OUTPUT_TABLES / "dataset_feature_validation.csv"
    with open(pair_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"Saved table: {pair_csv}")

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------
    same_scores = [r["weighted_similarity"] for r in pair_rows if r["same_identity"]]
    diff_scores = [r["weighted_similarity"] for r in pair_rows if not r["same_identity"]]

    same_mean = float(np.mean(same_scores)) if same_scores else 0.0
    diff_mean = float(np.mean(diff_scores)) if diff_scores else 0.0
    separation = same_mean - diff_mean

    summary_rows = [
        {
            "group": "same_identity_pairs",
            "pair_count": len(same_scores),
            "mean_similarity": round(same_mean, 4),
            "min_similarity": round(float(np.min(same_scores)), 4) if same_scores else None,
            "max_similarity": round(float(np.max(same_scores)), 4) if same_scores else None,
        },
        {
            "group": "different_identity_pairs",
            "pair_count": len(diff_scores),
            "mean_similarity": round(diff_mean, 4),
            "min_similarity": round(float(np.min(diff_scores)), 4) if diff_scores else None,
            "max_similarity": round(float(np.max(diff_scores)), 4) if diff_scores else None,
        },
        {
            "group": "separation_margin",
            "pair_count": "",
            "mean_similarity": round(separation, 4),
            "min_similarity": "",
            "max_similarity": "",
        },
    ]

    summary_csv = OUTPUT_TABLES / "dataset_validation_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved table: {summary_csv}")

    # --------------------------------------------------------
    # Threshold calibration tables
    # --------------------------------------------------------
    calibration_table = []

    for r in calibration_rows:
        calibration_table.append({
            "threshold": round(r["threshold"], 3),
            "accuracy": round(r["accuracy"], 4),
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1_score": round(r["f1_score"], 4),
            "false_positive_count": r["false_positive_count"],
            "false_negative_count": r["false_negative_count"],
            "true_positive_count": r["true_positive_count"],
            "true_negative_count": r["true_negative_count"],
            "false_positive_rate": round(r["false_positive_rate"], 4),
            "false_negative_rate": round(r["false_negative_rate"], 4),
        })

    calibration_csv = OUTPUT_TABLES / "threshold_calibration_curve.csv"
    with open(calibration_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=calibration_table[0].keys())
        writer.writeheader()
        writer.writerows(calibration_table)

    print(f"Saved table: {calibration_csv}")

    best_summary = [{
        "selected_threshold": round(best_threshold, 3),
        "accuracy": round(best_threshold_row["accuracy"], 4),
        "precision": round(best_threshold_row["precision"], 4),
        "recall": round(best_threshold_row["recall"], 4),
        "f1_score": round(best_threshold_row["f1_score"], 4),
        "false_positive_count": best_threshold_row["false_positive_count"],
        "false_negative_count": best_threshold_row["false_negative_count"],
        "true_positive_count": best_threshold_row["true_positive_count"],
        "true_negative_count": best_threshold_row["true_negative_count"],
    }]

    best_csv = OUTPUT_TABLES / "threshold_calibration_summary.csv"
    with open(best_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=best_summary[0].keys())
        writer.writeheader()
        writer.writerows(best_summary)

    print(f"Saved table: {best_csv}")

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_same_vs_different(same_scores, diff_scores)
    plot_similarity_matrix(features)
    plot_threshold_calibration(calibration_rows, best_threshold)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\nDone.")
    print(f"Image count: {len(image_files)}")
    print(f"Same-identity pair count: {len(same_scores)}")
    print(f"Different-identity pair count: {len(diff_scores)}")
    print(f"Same-identity mean similarity: {same_mean * 100:.1f}%")
    print(f"Different-identity mean similarity: {diff_mean * 100:.1f}%")
    print(f"Separation margin: {separation * 100:.1f}%")
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Threshold accuracy: {best_threshold_row['accuracy'] * 100:.1f}%")
    print(f"Threshold precision: {best_threshold_row['precision'] * 100:.1f}%")
    print(f"Threshold recall: {best_threshold_row['recall'] * 100:.1f}%")
    print(f"Threshold F1: {best_threshold_row['f1_score'] * 100:.1f}%")

    print("\nGenerated:")
    print("- dataset_image_features.csv")
    print("- dataset_feature_validation.csv")
    print("- dataset_validation_summary.csv")
    print("- threshold_calibration_curve.csv")
    print("- threshold_calibration_summary.csv")
    print("- same_vs_different_similarity_graph.png")
    print("- dataset_similarity_matrix.png")
    print("- threshold_calibration_curve.png")


if __name__ == "__main__":
    main()