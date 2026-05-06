import csv
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import torchreid


# ============================================================
# NeuroMemory Robot - ReID-Specific Embedding Validation
#
# Uses OSNet from Torchreid as a person re-identification feature extractor.
#
# Important:
# This is not face recognition.
# The model is used only for offline feature-level validation.
# Final identity-related decisions remain human-supervised.
# ============================================================

DATA_DIR = Path("data/sample_reid")
OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_FIGURES = Path("outputs/figures")

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
COLORS = {
    "navy": "#14213D",
    "blue": "#2F6690",
    "teal": "#2A9D8F",
    "green": "#52B788",
    "amber": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
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


def style_axes(ax):
    ax.set_facecolor(COLORS["bg"])
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


def parse_identity(filename: str) -> str:
    parts = filename.split("_")
    if len(parts) >= 2 and parts[0] == "id":
        return parts[1]
    return filename.split("_")[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def decision_from_similarity(score):
    if score >= 0.82:
        return "probable same identity"
    elif score >= 0.70:
        return "human verification advised"
    elif score >= 0.55:
        return "uncertain / re-observe"
    else:
        return "different or insufficient evidence"


# ------------------------------------------------------------
# ReID-specific OSNet extractor
# ------------------------------------------------------------

class OSNetFeatureExtractor:
    def __init__(self, device):
        self.device = device

        # Person ReID models typically use 256x128 person crops.
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # num_classes is only needed to construct the model.
        # In evaluation mode we use the feature embedding output.
        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=751,
            loss="softmax",
            pretrained=True,
            use_gpu=(device.type == "cuda"),
        )

        self.model.to(device)
        self.model.eval()

    def extract(self, image_path: Path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

            # Some torchreid models may return tuple/list depending on mode.
            if isinstance(features, (tuple, list)):
                features = features[0]

            features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy()[0].astype(np.float32)


def load_low_level_summary():
    summary_path = OUTPUT_TABLES / "dataset_validation_summary.csv"
    if not summary_path.exists():
        return None

    low_same = None
    low_diff = None

    with open(summary_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row.get("group") == "same_identity_pairs":
                low_same = float(row["mean_similarity"])
            elif row.get("group") == "different_identity_pairs":
                low_diff = float(row["mean_similarity"])

    if low_same is None or low_diff is None:
        return None

    return {
        "method": "Explainable features",
        "same": low_same,
        "different": low_diff,
        "separation": low_same - low_diff,
    }


def load_resnet_summary():
    summary_path = OUTPUT_TABLES / "pretrained_embedding_summary.csv"
    if not summary_path.exists():
        return None

    with open(summary_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    if not rows:
        return None

    row = rows[0]
    same = float(row["same_identity_mean_similarity"])
    different = float(row["different_identity_mean_similarity"])

    return {
        "method": "Generic ResNet18",
        "same": same,
        "different": different,
        "separation": same - different,
    }


def main():
    image_files = sorted([
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(image_files) < 4:
        raise RuntimeError(f"Not enough images in {DATA_DIR}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading OSNet ReID-specific feature extractor...")
    extractor = OSNetFeatureExtractor(device)

    records = []

    print("Extracting OSNet embeddings...")
    for path in image_files:
        identity = parse_identity(path.name)
        embedding = extractor.extract(path)

        records.append({
            "filename": path.name,
            "identity": identity,
            "embedding": embedding,
        })

    # Save embedding feature table
    feature_rows = []
    for record in records:
        feature_rows.append({
            "filename": record["filename"],
            "identity": record["identity"],
            "embedding_dim": len(record["embedding"]),
            "embedding_norm": round(float(np.linalg.norm(record["embedding"])), 4),
        })

    feature_csv = OUTPUT_TABLES / "reid_specific_embedding_features.csv"
    with open(feature_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=feature_rows[0].keys())
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"Saved table: {feature_csv}")

    # Pairwise similarities
    pair_rows = []

    for r1, r2 in combinations(records, 2):
        sim = cosine_similarity(r1["embedding"], r2["embedding"])
        same_identity = r1["identity"] == r2["identity"]

        pair_rows.append({
            "image_1": r1["filename"],
            "image_2": r2["filename"],
            "id_1": r1["identity"],
            "id_2": r2["identity"],
            "same_identity": same_identity,
            "reid_embedding_similarity": round(sim, 4),
            "decision": decision_from_similarity(sim),
        })

    pair_csv = OUTPUT_TABLES / "reid_specific_embedding_validation.csv"
    with open(pair_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"Saved table: {pair_csv}")

    same_scores = [r["reid_embedding_similarity"] for r in pair_rows if r["same_identity"]]
    diff_scores = [r["reid_embedding_similarity"] for r in pair_rows if not r["same_identity"]]

    same_mean = float(np.mean(same_scores)) if same_scores else 0.0
    diff_mean = float(np.mean(diff_scores)) if diff_scores else 0.0
    separation = same_mean - diff_mean

    summary_rows = [{
        "method": "reid_specific_osnet_embedding",
        "same_identity_pair_count": len(same_scores),
        "different_identity_pair_count": len(diff_scores),
        "same_identity_mean_similarity": round(same_mean, 4),
        "different_identity_mean_similarity": round(diff_mean, 4),
        "separation_margin": round(separation, 4),
        "same_identity_min": round(float(np.min(same_scores)), 4) if same_scores else None,
        "same_identity_max": round(float(np.max(same_scores)), 4) if same_scores else None,
        "different_identity_min": round(float(np.min(diff_scores)), 4) if diff_scores else None,
        "different_identity_max": round(float(np.max(diff_scores)), 4) if diff_scores else None,
    }]

    summary_csv = OUTPUT_TABLES / "reid_specific_embedding_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved table: {summary_csv}")

    # --------------------------------------------------------
    # Figure 1: same vs different OSNet ReID graph
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.7, 5.8))
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
            s=35,
            color=COLORS["navy"],
            alpha=0.65,
            zorder=3,
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("OSNet ReID embedding cosine similarity (%)")
    ax.set_title("Public ReID Subset: ReID-Specific OSNet Similarity", pad=14, fontweight="bold")

    add_footnote(
        fig,
        "OSNet is used as an offline ReID-specific feature extractor; final identity decisions remain human-supervised."
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    graph_path = OUTPUT_FIGURES / "reid_specific_same_vs_different_graph.png"
    fig.savefig(graph_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {graph_path}")

    # --------------------------------------------------------
    # Figure 2: pairwise similarity matrix
    # --------------------------------------------------------
    n = len(records)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = cosine_similarity(records[i]["embedding"], records[j]["embedding"])

    labels = [f"{record['identity']}-{idx + 1}" for idx, record in enumerate(records)]

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    im = ax.imshow(matrix * 100, cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title("OSNet ReID Pairwise Similarity Matrix", pad=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine similarity (%)")

    for i in range(n):
        for j in range(n):
            value = matrix[i, j] * 100
            color = "white" if value < 55 else "black"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    matrix_path = OUTPUT_FIGURES / "reid_specific_similarity_matrix.png"
    fig.savefig(matrix_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {matrix_path}")

    # --------------------------------------------------------
    # Figure 3: method comparison
    # --------------------------------------------------------
    methods = []

    low = load_low_level_summary()
    if low is not None:
        methods.append(low)

    resnet = load_resnet_summary()
    if resnet is not None:
        methods.append(resnet)

    methods.append({
        "method": "ReID-specific OSNet",
        "same": same_mean,
        "different": diff_mean,
        "separation": separation,
    })

    method_names = [m["method"] for m in methods]
    same_means = [m["same"] * 100 for m in methods]
    diff_means = [m["different"] * 100 for m in methods]
    separations = [m["separation"] * 100 for m in methods]

    x = np.arange(len(methods))
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    style_axes(ax)

    ax.bar(
        x - bar_w,
        same_means,
        width=bar_w,
        color=COLORS["teal"],
        edgecolor=COLORS["navy"],
        linewidth=0.6,
        label="Same identity mean",
    )

    ax.bar(
        x,
        diff_means,
        width=bar_w,
        color=COLORS["coral"],
        edgecolor=COLORS["navy"],
        linewidth=0.6,
        label="Different identity mean",
    )

    ax.bar(
        x + bar_w,
        separations,
        width=bar_w,
        color=COLORS["blue"],
        edgecolor=COLORS["navy"],
        linewidth=0.6,
        label="Separation margin",
    )

    for i in range(len(methods)):
        ax.text(x[i] - bar_w, same_means[i] + 2, f"{same_means[i]:.1f}", ha="center", fontsize=9, color=COLORS["teal"])
        ax.text(x[i], diff_means[i] + 2, f"{diff_means[i]:.1f}", ha="center", fontsize=9, color=COLORS["coral"])
        ax.text(x[i] + bar_w, separations[i] + 2, f"{separations[i]:.1f}", ha="center", fontsize=9, color=COLORS["blue"])

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=8, ha="center")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Similarity / separation (%)")
    ax.set_title("External ReID Subset Validation: Feature Method Comparison", pad=14, fontweight="bold")

    ax.legend(
        frameon=True,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grid"],
    )

    add_footnote(
        fig,
        "External validation compares explainable features, generic ImageNet embedding, and ReID-specific OSNet embedding."
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    comparison_path = OUTPUT_FIGURES / "reid_method_comparison_graph.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {comparison_path}")

    print("\nDone.")
    print(f"Same-identity mean OSNet similarity: {same_mean * 100:.1f}%")
    print(f"Different-identity mean OSNet similarity: {diff_mean * 100:.1f}%")
    print(f"Separation margin: {separation * 100:.1f}%")
    print("\nGenerated:")
    print("- reid_specific_embedding_features.csv")
    print("- reid_specific_embedding_validation.csv")
    print("- reid_specific_embedding_summary.csv")
    print("- reid_specific_same_vs_different_graph.png")
    print("- reid_specific_similarity_matrix.png")
    print("- reid_method_comparison_graph.png")


if __name__ == "__main__":
    main()