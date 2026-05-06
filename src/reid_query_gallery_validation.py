import csv
import random
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchreid


# ============================================================
# NeuroMemory Robot - Query/Gallery ReID Validation
#
# Adds:
# 1. Market-1501 query-gallery validation
# 2. OSNet ReID-specific embedding retrieval
# 3. Explainable body-region feature retrieval
# 4. Hybrid NeuroMemory score
# 5. Top-1 / Top-3 / Top-5 accuracy
# 6. Retrieval example figure with cross-camera preference
#
# This is not face recognition.
# This is public ReID dataset feature-level validation.
# ============================================================

EXTRACTED_ROOT = Path("data/market1501_extracted")
OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_FIGURES = Path("outputs/figures")

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

MAX_IDENTITIES = 10
QUERIES_PER_ID = 2
GALLERY_PER_ID = 4
DISTRACTOR_IDENTITIES = 10
DISTRACTOR_IMAGES_PER_ID = 2

HYBRID_WEIGHTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# ------------------------------------------------------------
# Plot style
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


# ============================================================
# Dataset utilities
# ============================================================

def find_folder(root: Path, folder_name: str) -> Path:
    matches = [p for p in root.rglob(folder_name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find folder '{folder_name}' inside {root}")
    return matches[0]


def parse_market_filename(path):
    """
    Market-1501 filename format:
    0001_c1s1_001051_03.jpg

    identity = 0001
    camera = c1

    Accepts either a Path object or filename string.
    """
    filename = Path(path).name
    parts = filename.split("_")

    person_id = parts[0]

    camera = "unknown"
    if len(parts) >= 2 and parts[1].startswith("c"):
        camera = parts[1][:2]

    return person_id, camera


def collect_by_identity(folder: Path):
    grouped = {}

    for path in folder.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        pid, _ = parse_market_filename(path)

        if pid in ["-1", "0000"]:
            continue

        grouped.setdefault(pid, []).append(path)

    for pid in grouped:
        grouped[pid] = sorted(grouped[pid])

    return grouped


def select_query_gallery(query_grouped, gallery_grouped):
    random.seed(RANDOM_SEED)

    common_ids = sorted(set(query_grouped.keys()) & set(gallery_grouped.keys()))

    valid_ids = [
        pid for pid in common_ids
        if len(query_grouped[pid]) >= 1 and len(gallery_grouped[pid]) >= 2
    ]

    if len(valid_ids) < 3:
        raise RuntimeError("Not enough valid identities for query-gallery validation.")

    selected_ids = valid_ids[:MAX_IDENTITIES]

    queries = []
    gallery = []

    for pid in selected_ids:
        q_imgs = query_grouped[pid][:QUERIES_PER_ID]
        g_imgs = gallery_grouped[pid][:GALLERY_PER_ID]

        queries.extend(q_imgs)
        gallery.extend(g_imgs)

    distractor_candidates = [
        pid for pid in sorted(gallery_grouped.keys())
        if pid not in selected_ids and pid not in ["-1", "0000"]
    ]

    for pid in distractor_candidates[:DISTRACTOR_IDENTITIES]:
        gallery.extend(gallery_grouped[pid][:DISTRACTOR_IMAGES_PER_ID])

    gallery = sorted(list(dict.fromkeys(gallery)))

    return selected_ids, queries, gallery


# ============================================================
# Explainable feature extractor
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def load_image_np(path: Path, size=(128, 256)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return np.asarray(img).astype(np.float32) / 255.0


def rgb_to_hsv_np(rgb):
    return mpl.colors.rgb_to_hsv(rgb)


def normalized_hist(values, bins, value_range=(0.0, 1.0)):
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist


def hsv_histogram(region, h_bins=16, s_bins=8, v_bins=8):
    hsv = rgb_to_hsv_np(region)

    h_hist = normalized_hist(hsv[:, :, 0].ravel(), bins=h_bins)
    s_hist = normalized_hist(hsv[:, :, 1].ravel(), bins=s_bins)
    v_hist = normalized_hist(hsv[:, :, 2].ravel(), bins=v_bins)

    return np.concatenate([h_hist, s_hist, v_hist])


def brightness_contrast(img):
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return float(gray.mean()), float(gray.std())


def texture_density(img):
    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    return float(dx + dy)


def histogram_intersection(h1, h2):
    return float(np.minimum(h1, h2).sum() / (np.maximum(h1, h2).sum() + 1e-8))


def scalar_similarity(a, b, scale):
    return clamp(1.0 - abs(a - b) / scale, 0.0, 1.0)


def split_body_regions(img):
    h, _ = img.shape[:2]

    upper = img[int(0.15 * h):int(0.55 * h), :, :]
    lower = img[int(0.55 * h):int(0.95 * h), :, :]

    return upper, lower


def extract_explainable_features(path: Path):
    img = load_image_np(path)
    upper, lower = split_body_regions(img)

    full_hsv = hsv_histogram(img)
    upper_hsv = hsv_histogram(upper)
    lower_hsv = hsv_histogram(lower)

    brightness, contrast = brightness_contrast(img)
    texture = texture_density(img)

    h, w = img.shape[:2]
    aspect_ratio = float(h / w)

    return {
        "full_hsv": full_hsv,
        "upper_hsv": upper_hsv,
        "lower_hsv": lower_hsv,
        "brightness": brightness,
        "contrast": contrast,
        "texture": texture,
        "aspect_ratio": aspect_ratio,
    }


def explainable_similarity(f1, f2):
    full_color = histogram_intersection(f1["full_hsv"], f2["full_hsv"])
    upper_color = histogram_intersection(f1["upper_hsv"], f2["upper_hsv"])
    lower_color = histogram_intersection(f1["lower_hsv"], f2["lower_hsv"])

    brightness_sim = scalar_similarity(f1["brightness"], f2["brightness"], scale=0.45)
    contrast_sim = scalar_similarity(f1["contrast"], f2["contrast"], scale=0.35)
    texture_sim = scalar_similarity(f1["texture"], f2["texture"], scale=0.25)
    aspect_sim = scalar_similarity(f1["aspect_ratio"], f2["aspect_ratio"], scale=1.0)

    return (
        0.18 * full_color
        + 0.26 * upper_color
        + 0.26 * lower_color
        + 0.08 * brightness_sim
        + 0.08 * contrast_sim
        + 0.09 * texture_sim
        + 0.05 * aspect_sim
    )


# ============================================================
# OSNet ReID extractor
# ============================================================

class OSNetFeatureExtractor:
    def __init__(self, device):
        self.device = device

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=751,
            loss="softmax",
            pretrained=True,
            use_gpu=(device.type == "cuda"),
        )

        self.model.to(device)
        self.model.eval()

    def extract(self, path: Path):
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)

            if isinstance(feat, (tuple, list)):
                feat = feat[0]

            feat = F.normalize(feat, p=2, dim=1)

        return feat.cpu().numpy()[0].astype(np.float32)


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# Retrieval evaluation
# ============================================================

def topk_correct(ranked_gallery, query_id, k):
    top_items = ranked_gallery[:k]
    return any(item["gallery_id"] == query_id for item in top_items)


def evaluate_method(method_name, queries, gallery, score_function):
    rows = []

    for q in queries:
        q_id, q_cam = parse_market_filename(q)

        scored = []

        for g in gallery:
            g_id, g_cam = parse_market_filename(g)

            same_identity = q_id == g_id
            same_camera = q_cam == g_cam

            score = score_function(q, g)

            scored.append({
                "query": q.name,
                "query_id": q_id,
                "query_cam": q_cam,
                "gallery": g.name,
                "gallery_id": g_id,
                "gallery_cam": g_cam,
                "same_identity": same_identity,
                "same_camera": same_camera,
                "score": score,
            })

        ranked = sorted(scored, key=lambda x: x["score"], reverse=True)

        for rank, item in enumerate(ranked, start=1):
            rows.append({
                "method": method_name,
                "query": item["query"],
                "query_id": item["query_id"],
                "query_cam": item["query_cam"],
                "gallery": item["gallery"],
                "gallery_id": item["gallery_id"],
                "gallery_cam": item["gallery_cam"],
                "rank": rank,
                "score": round(item["score"], 4),
                "same_identity": item["same_identity"],
                "same_camera": item["same_camera"],
                "correct_match": item["same_identity"],
            })

    return rows


def summarize_retrieval(rows, method_name):
    query_names = sorted(set(r["query"] for r in rows))

    top1 = top3 = top5 = 0
    mrr_values = []

    for query_name in query_names:
        ranked = sorted(
            [r for r in rows if r["query"] == query_name],
            key=lambda x: x["rank"],
        )

        query_id = ranked[0]["query_id"]

        if topk_correct(ranked, query_id, 1):
            top1 += 1
        if topk_correct(ranked, query_id, 3):
            top3 += 1
        if topk_correct(ranked, query_id, 5):
            top5 += 1

        first_correct_rank = None

        for item in ranked:
            if item["gallery_id"] == query_id:
                first_correct_rank = item["rank"]
                break

        if first_correct_rank is not None:
            mrr_values.append(1.0 / first_correct_rank)
        else:
            mrr_values.append(0.0)

    n = len(query_names)

    return {
        "method": method_name,
        "query_count": n,
        "top1_accuracy": top1 / n if n else 0.0,
        "top3_accuracy": top3 / n if n else 0.0,
        "top5_accuracy": top5 / n if n else 0.0,
        "mean_reciprocal_rank": float(np.mean(mrr_values)) if mrr_values else 0.0,
    }


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No rows to write for {path}")

    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved table: {path}")


# ============================================================
# Figures
# ============================================================

def plot_method_summary(summary_rows):
    methods = [r["method"] for r in summary_rows]
    top1 = [float(r["top1_accuracy"]) * 100 for r in summary_rows]
    top3 = [float(r["top3_accuracy"]) * 100 for r in summary_rows]
    top5 = [float(r["top5_accuracy"]) * 100 for r in summary_rows]
    mrr = [float(r["mean_reciprocal_rank"]) * 100 for r in summary_rows]

    x = np.arange(len(methods))
    bar_w = 0.2

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    style_axes(ax)

    ax.bar(
        x - 1.5 * bar_w,
        top1,
        width=bar_w,
        color=COLORS["coral"],
        edgecolor=COLORS["navy"],
        label="Top-1",
    )

    ax.bar(
        x - 0.5 * bar_w,
        top3,
        width=bar_w,
        color=COLORS["amber"],
        edgecolor=COLORS["navy"],
        label="Top-3",
    )

    ax.bar(
        x + 0.5 * bar_w,
        top5,
        width=bar_w,
        color=COLORS["teal"],
        edgecolor=COLORS["navy"],
        label="Top-5",
    )

    ax.bar(
        x + 1.5 * bar_w,
        mrr,
        width=bar_w,
        color=COLORS["blue"],
        edgecolor=COLORS["navy"],
        label="MRR",
    )

    for i in range(len(methods)):
        ax.text(x[i] - 1.5 * bar_w, top1[i] + 2, f"{top1[i]:.0f}", ha="center", fontsize=8, color=COLORS["coral"])
        ax.text(x[i] - 0.5 * bar_w, top3[i] + 2, f"{top3[i]:.0f}", ha="center", fontsize=8, color=COLORS["amber"])
        ax.text(x[i] + 0.5 * bar_w, top5[i] + 2, f"{top5[i]:.0f}", ha="center", fontsize=8, color=COLORS["teal"])
        ax.text(x[i] + 1.5 * bar_w, mrr[i] + 2, f"{mrr[i]:.0f}", ha="center", fontsize=8, color=COLORS["blue"])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Retrieval metric (%)")
    ax.set_title(
        "Query-Gallery Retrieval Validation on Public ReID Subset",
        pad=14,
        fontweight="bold",
    )

    ax.legend(frameon=True, facecolor=COLORS["white"], edgecolor=COLORS["grid"])

    add_footnote(
        fig,
        "Evaluation uses a small public ReID subset; final identity-related decisions remain human-supervised.",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    path = OUTPUT_FIGURES / "query_gallery_method_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {path}")


def plot_retrieval_example(rows, query_path_map, gallery_path_map, method_name="Hybrid NeuroMemory"):
    method_rows = [r for r in rows if r["method"] == method_name]

    if not method_rows:
        print(f"Retrieval example skipped: no rows found for method '{method_name}'")
        return

    query_names = sorted(set(r["query"] for r in method_rows))

    chosen_query = None
    chosen_ranked = None

    # Prefer a stronger ReID example:
    # correct same-identity retrieval in top-3, but from a different camera.
    for q in query_names:
        ranked = sorted(
            [r for r in method_rows if r["query"] == q],
            key=lambda x: x["rank"],
        )

        has_cross_camera_correct = any(
            bool(r["correct_match"]) and not bool(r["same_camera"])
            for r in ranked[:3]
        )

        if has_cross_camera_correct:
            chosen_query = q
            chosen_ranked = ranked
            break

    # Fallback: any correct match in top-3.
    if chosen_query is None:
        for q in query_names:
            ranked = sorted(
                [r for r in method_rows if r["query"] == q],
                key=lambda x: x["rank"],
            )

            if any(bool(r["correct_match"]) for r in ranked[:3]):
                chosen_query = q
                chosen_ranked = ranked
                break

    # Final fallback: first query.
    if chosen_query is None:
        chosen_query = query_names[0]
        chosen_ranked = sorted(
            [r for r in method_rows if r["query"] == chosen_query],
            key=lambda x: x["rank"],
        )

    top_gallery = chosen_ranked[:5]

    query_img = Image.open(query_path_map[chosen_query]).convert("RGB")

    fig, axes = plt.subplots(1, 6, figsize=(13.5, 4.4))

    fig.suptitle(
        "Query-Gallery Retrieval Example using Hybrid NeuroMemory Score",
        fontsize=15,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.98,
    )

    axes[0].imshow(query_img)
    axes[0].set_title(
        f"Query\nID {chosen_ranked[0]['query_id']}\ncam {chosen_ranked[0]['query_cam']}",
        fontsize=10,
        color=COLORS["navy"],
    )
    axes[0].axis("off")

    for idx, item in enumerate(top_gallery, start=1):
        img = Image.open(gallery_path_map[item["gallery"]]).convert("RGB")
        axes[idx].imshow(img)

        correct = bool(item["correct_match"])
        cross_camera = correct and not bool(item["same_camera"])

        if cross_camera:
            title_color = COLORS["teal"]
            match_note = "cross-cam"
        elif correct:
            title_color = COLORS["green"]
            match_note = "same ID"
        else:
            title_color = COLORS["coral"]
            match_note = "different ID"

        title = (
            f"Rank {idx}\n"
            f"ID {item['gallery_id']} | cam {item['gallery_cam']}\n"
            f"score {float(item['score']):.2f}\n"
            f"{match_note}"
        )

        axes[idx].set_title(title, fontsize=9, color=title_color)
        axes[idx].axis("off")

        for spine in axes[idx].spines.values():
            spine.set_edgecolor(title_color)
            spine.set_linewidth(2.0)

    fig.text(
        0.5,
        0.02,
        "Green/teal labels indicate correct same-identity retrieval; teal highlights cross-camera re-identification.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    path = OUTPUT_FIGURES / "query_gallery_retrieval_example.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {path}")


# ============================================================
# Main
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    query_dir = find_folder(EXTRACTED_ROOT, "query")
    gallery_dir = find_folder(EXTRACTED_ROOT, "bounding_box_test")

    print(f"Query folder: {query_dir}")
    print(f"Gallery folder: {gallery_dir}")

    query_grouped = collect_by_identity(query_dir)
    gallery_grouped = collect_by_identity(gallery_dir)

    selected_ids, queries, gallery = select_query_gallery(query_grouped, gallery_grouped)

    print("\nSelected IDs:")
    print(", ".join(selected_ids))

    print(f"Query count: {len(queries)}")
    print(f"Gallery count: {len(gallery)}")

    query_path_map = {p.name: p for p in queries}
    gallery_path_map = {p.name: p for p in gallery}

    all_paths = sorted(list(dict.fromkeys(queries + gallery)))

    # --------------------------------------------------------
    # Explainable body-region features
    # --------------------------------------------------------
    print("\nExtracting explainable body-region features...")

    explainable_cache = {
        p.name: extract_explainable_features(p)
        for p in all_paths
    }

    def explainable_score(q, g):
        return explainable_similarity(explainable_cache[q.name], explainable_cache[g.name])

    # --------------------------------------------------------
    # OSNet ReID embeddings
    # --------------------------------------------------------
    print("Extracting OSNet embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    osnet = OSNetFeatureExtractor(device)

    osnet_cache = {
        p.name: osnet.extract(p)
        for p in all_paths
    }

    def osnet_score(q, g):
        return cosine_similarity(osnet_cache[q.name], osnet_cache[g.name])

    # --------------------------------------------------------
    # Hybrid scoring
    # --------------------------------------------------------
    def hybrid_score_factory(osnet_weight):
        explainable_weight = 1.0 - osnet_weight

        def score(q, g):
            return (
                explainable_weight * explainable_score(q, g)
                + osnet_weight * osnet_score(q, g)
            )

        return score

    all_result_rows = []
    summary_rows = []

    # --------------------------------------------------------
    # Method 1: Explainable features
    # --------------------------------------------------------
    explainable_rows = evaluate_method(
        "Explainable features",
        queries,
        gallery,
        explainable_score,
    )

    all_result_rows.extend(explainable_rows)

    summary_rows.append(
        summarize_retrieval(explainable_rows, "Explainable features")
    )

    # --------------------------------------------------------
    # Method 2: OSNet
    # --------------------------------------------------------
    osnet_rows = evaluate_method(
        "OSNet ReID embedding",
        queries,
        gallery,
        osnet_score,
    )

    all_result_rows.extend(osnet_rows)

    summary_rows.append(
        summarize_retrieval(osnet_rows, "OSNet ReID embedding")
    )

    # --------------------------------------------------------
    # Method 3: Hybrid weight search
    # --------------------------------------------------------
    hybrid_summaries = []
    hybrid_rows_by_name = {}

    for w in HYBRID_WEIGHTS:
        method_name = f"Hybrid NeuroMemory w={w:.1f}"

        rows = evaluate_method(
            method_name,
            queries,
            gallery,
            hybrid_score_factory(w),
        )

        summary = summarize_retrieval(rows, method_name)
        summary["osnet_weight"] = w
        summary["explainable_weight"] = 1.0 - w

        hybrid_summaries.append(summary)
        hybrid_rows_by_name[method_name] = rows

    best_hybrid = sorted(
        hybrid_summaries,
        key=lambda r: (
            r["top1_accuracy"],
            r["top3_accuracy"],
            r["mean_reciprocal_rank"],
        ),
        reverse=True,
    )[0]

    best_hybrid_name = best_hybrid["method"]
    best_hybrid_rows = hybrid_rows_by_name[best_hybrid_name]

    # Rename selected hybrid rows for final reporting and plotting.
    for row in best_hybrid_rows:
        row["method"] = "Hybrid NeuroMemory"

    all_result_rows.extend(best_hybrid_rows)

    summary_rows.append({
        "method": "Hybrid NeuroMemory",
        "query_count": best_hybrid["query_count"],
        "top1_accuracy": best_hybrid["top1_accuracy"],
        "top3_accuracy": best_hybrid["top3_accuracy"],
        "top5_accuracy": best_hybrid["top5_accuracy"],
        "mean_reciprocal_rank": best_hybrid["mean_reciprocal_rank"],
        "selected_osnet_weight": best_hybrid["osnet_weight"],
        "selected_explainable_weight": best_hybrid["explainable_weight"],
    })

    # --------------------------------------------------------
    # Write detailed retrieval rows
    # --------------------------------------------------------
    retrieval_csv = OUTPUT_TABLES / "query_gallery_retrieval_results.csv"
    write_csv(retrieval_csv, all_result_rows)

    # --------------------------------------------------------
    # Write summary rows with consistent fields
    # --------------------------------------------------------
    summary_out = []

    for r in summary_rows:
        summary_out.append({
            "method": r["method"],
            "query_count": r["query_count"],
            "top1_accuracy": round(float(r["top1_accuracy"]), 4),
            "top3_accuracy": round(float(r["top3_accuracy"]), 4),
            "top5_accuracy": round(float(r["top5_accuracy"]), 4),
            "mean_reciprocal_rank": round(float(r["mean_reciprocal_rank"]), 4),
            "selected_osnet_weight": r.get("selected_osnet_weight", ""),
            "selected_explainable_weight": r.get("selected_explainable_weight", ""),
        })

    summary_csv = OUTPUT_TABLES / "query_gallery_retrieval_summary.csv"
    write_csv(summary_csv, summary_out)

    # --------------------------------------------------------
    # Write hybrid weight search
    # --------------------------------------------------------
    hybrid_csv_rows = []

    for r in hybrid_summaries:
        hybrid_csv_rows.append({
            "method": r["method"],
            "osnet_weight": r["osnet_weight"],
            "explainable_weight": r["explainable_weight"],
            "query_count": r["query_count"],
            "top1_accuracy": round(float(r["top1_accuracy"]), 4),
            "top3_accuracy": round(float(r["top3_accuracy"]), 4),
            "top5_accuracy": round(float(r["top5_accuracy"]), 4),
            "mean_reciprocal_rank": round(float(r["mean_reciprocal_rank"]), 4),
        })

    hybrid_csv = OUTPUT_TABLES / "hybrid_weight_search.csv"
    write_csv(hybrid_csv, hybrid_csv_rows)

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_method_summary(summary_out)
    plot_retrieval_example(
        all_result_rows,
        query_path_map,
        gallery_path_map,
        method_name="Hybrid NeuroMemory",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\nDone.")
    print("Query-gallery validation summary:")

    for row in summary_out:
        print(
            f"  {row['method']}: "
            f"Top-1={row['top1_accuracy'] * 100:.1f}%, "
            f"Top-3={row['top3_accuracy'] * 100:.1f}%, "
            f"Top-5={row['top5_accuracy'] * 100:.1f}%, "
            f"MRR={row['mean_reciprocal_rank'] * 100:.1f}%"
        )

    print(f"\nSelected hybrid: {best_hybrid_name}")
    print(f"OSNet weight: {best_hybrid['osnet_weight']:.1f}")
    print(f"Explainable weight: {1.0 - best_hybrid['osnet_weight']:.1f}")

    print("\nGenerated:")
    print("- query_gallery_retrieval_results.csv")
    print("- query_gallery_retrieval_summary.csv")
    print("- hybrid_weight_search.csv")
    print("- query_gallery_method_comparison.png")
    print("- query_gallery_retrieval_example.png")


if __name__ == "__main__":
    main()