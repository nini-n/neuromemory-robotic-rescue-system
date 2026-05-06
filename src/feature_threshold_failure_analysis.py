import os
import csv
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# NeuroMemory Robot - Feature, Threshold, and Failure Analysis
# Adds:
# 1. Image-derived / simulation-derived visual memory feature analysis
# 2. Threshold sensitivity analysis
# 3. Failure-case / limitation analysis
# ============================================================

OUTPUT_TABLES = "outputs/tables"
OUTPUT_FIGURES = "outputs/figures"

os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)


# ------------------------------------------------------------
# Professional color palette
# ------------------------------------------------------------
COLORS = {
    "navy": "#14213D",
    "blue": "#2F6690",
    "deep_blue": "#1F4E79",
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
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": COLORS["white"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["light_slate"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "text.color": COLORS["text"],
})


# ============================================================
# Helpers
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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
        color=COLORS["muted"]
    )


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def decision_from_threshold(confidence, threshold):
    if confidence >= threshold:
        return "Probable match"
    elif confidence >= threshold - 0.12:
        return "Human verification advised"
    elif confidence >= threshold - 0.25:
        return "Re-observation required"
    else:
        return "Insufficient evidence"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved table: {path}")


# ============================================================
# Part 1 - Simulation-derived visual memory feature analysis
# ============================================================
# We do not use face recognition. The feature vector represents
# explainable visual cues available in the simulation:
#
# [body color consistency,
#  shape consistency,
#  visible area quality,
#  occlusion consistency,
#  viewpoint consistency]
#
# This allows the system to explain why a candidate is considered
# similar or uncertain.


reference_features = {
    "body_color_consistency": 1.00,
    "shape_consistency": 1.00,
    "visible_area_quality": 1.00,
    "occlusion_consistency": 1.00,
    "viewpoint_consistency": 1.00,
}


feature_scenarios = [
    {
        "scenario_id": "SC-01",
        "condition": "Clear visibility",
        "body_color_consistency": 0.96,
        "shape_consistency": 0.94,
        "visible_area_quality": 0.95,
        "occlusion_consistency": 0.98,
        "viewpoint_consistency": 0.93,
    },
    {
        "scenario_id": "SC-02",
        "condition": "Smoke",
        "body_color_consistency": 0.82,
        "shape_consistency": 0.86,
        "visible_area_quality": 0.64,
        "occlusion_consistency": 0.88,
        "viewpoint_consistency": 0.81,
    },
    {
        "scenario_id": "SC-03",
        "condition": "Low light",
        "body_color_consistency": 0.76,
        "shape_consistency": 0.84,
        "visible_area_quality": 0.58,
        "occlusion_consistency": 0.90,
        "viewpoint_consistency": 0.83,
    },
    {
        "scenario_id": "SC-04",
        "condition": "Partial occlusion",
        "body_color_consistency": 0.78,
        "shape_consistency": 0.75,
        "visible_area_quality": 0.62,
        "occlusion_consistency": 0.57,
        "viewpoint_consistency": 0.76,
    },
    {
        "scenario_id": "SC-05",
        "condition": "Smoke + occlusion + viewpoint shift",
        "body_color_consistency": 0.68,
        "shape_consistency": 0.70,
        "visible_area_quality": 0.48,
        "occlusion_consistency": 0.45,
        "viewpoint_consistency": 0.56,
    },
]


FEATURE_WEIGHTS = {
    "body_color_consistency": 0.28,
    "shape_consistency": 0.22,
    "visible_area_quality": 0.20,
    "occlusion_consistency": 0.15,
    "viewpoint_consistency": 0.15,
}


def weighted_feature_similarity(row):
    score = 0.0
    for key, weight in FEATURE_WEIGHTS.items():
        score += weight * row[key]

    return clamp(score, 0.0, 1.0)


def uncertainty_score(row, similarity):
    """
    Uncertainty increases when similarity is low, visible area is low,
    occlusion is high, or viewpoint consistency is poor.
    """
    uncertainty = (
        0.45 * (1.0 - similarity)
        + 0.25 * (1.0 - row["visible_area_quality"])
        + 0.18 * (1.0 - row["occlusion_consistency"])
        + 0.12 * (1.0 - row["viewpoint_consistency"])
    )

    return clamp(uncertainty, 0.0, 1.0)


feature_rows = []

for row in feature_scenarios:
    similarity = weighted_feature_similarity(row)
    uncertainty = uncertainty_score(row, similarity)

    feature_rows.append({
        "scenario_id": row["scenario_id"],
        "condition": row["condition"],
        "body_color_consistency": round(row["body_color_consistency"], 3),
        "shape_consistency": round(row["shape_consistency"], 3),
        "visible_area_quality": round(row["visible_area_quality"], 3),
        "occlusion_consistency": round(row["occlusion_consistency"], 3),
        "viewpoint_consistency": round(row["viewpoint_consistency"], 3),
        "weighted_similarity": round(similarity, 3),
        "uncertainty_score": round(uncertainty, 3),
        "decision_at_0_80_threshold": decision_from_threshold(similarity, 0.80),
    })


visual_memory_csv = os.path.join(OUTPUT_TABLES, "visual_memory_feature_analysis.csv")
write_csv(visual_memory_csv, feature_rows)


# ------------------------------------------------------------
# Figure 1: Visual memory feature graph
# ------------------------------------------------------------

scenario_ids = [r["scenario_id"] for r in feature_rows]
similarities = [r["weighted_similarity"] * 100 for r in feature_rows]
uncertainties = [r["uncertainty_score"] * 100 for r in feature_rows]

x = list(range(len(scenario_ids)))
bar_w = 0.34

fig, ax = plt.subplots(figsize=(10.5, 5.8))
style_axes(ax)

ax.bar(
    [i - bar_w / 2 for i in x],
    similarities,
    width=bar_w,
    color=COLORS["teal"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Weighted visual similarity"
)

ax.bar(
    [i + bar_w / 2 for i in x],
    uncertainties,
    width=bar_w,
    color=COLORS["coral"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Uncertainty score"
)

ax.axhline(80, color=COLORS["slate"], linestyle="--", linewidth=1.4, alpha=0.8)
ax.text(
    len(x) - 0.45,
    82,
    "0.80 decision threshold",
    ha="right",
    va="bottom",
    fontsize=9,
    color=COLORS["slate"]
)

for i, sim in enumerate(similarities):
    ax.text(
        i - bar_w / 2,
        sim + 2,
        f"{sim:.1f}",
        ha="center",
        fontsize=9,
        color=COLORS["teal"]
    )

for i, unc in enumerate(uncertainties):
    ax.text(
        i + bar_w / 2,
        unc + 2,
        f"{unc:.1f}",
        ha="center",
        fontsize=9,
        color=COLORS["coral"]
    )

ax.set_xticks(x)
ax.set_xticklabels(scenario_ids)
ax.set_ylim(0, 105)
ax.set_xlabel("Simulation scenario")
ax.set_ylabel("Score (%)")
ax.set_title("Simulation-Derived Visual Memory Feature Analysis", pad=14, fontweight="bold")
ax.legend(
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Visual-memory features are simulation-derived and explainable; no face recognition or personal identity model is used."
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

feature_graph_path = os.path.join(OUTPUT_FIGURES, "visual_memory_feature_graph.png")
fig.savefig(feature_graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {feature_graph_path}")


# ============================================================
# Part 2 - Threshold sensitivity analysis
# ============================================================

thresholds = [0.60, 0.70, 0.80, 0.90]
threshold_rows = []

for threshold in thresholds:
    probable = 0
    human_verification = 0
    reobserve = 0
    insufficient = 0

    for row in feature_rows:
        similarity = row["weighted_similarity"]
        decision = decision_from_threshold(similarity, threshold)

        if decision == "Probable match":
            probable += 1
        elif decision == "Human verification advised":
            human_verification += 1
        elif decision == "Re-observation required":
            reobserve += 1
        else:
            insufficient += 1

    threshold_rows.append({
        "threshold": threshold,
        "probable_match_count": probable,
        "human_verification_count": human_verification,
        "re_observation_required_count": reobserve,
        "insufficient_evidence_count": insufficient,
    })


threshold_csv = os.path.join(OUTPUT_TABLES, "threshold_sensitivity.csv")
write_csv(threshold_csv, threshold_rows)


# ------------------------------------------------------------
# Figure 2: Threshold sensitivity graph
# ------------------------------------------------------------

threshold_labels = [f"{r['threshold']:.2f}" for r in threshold_rows]

probable_counts = [r["probable_match_count"] for r in threshold_rows]
human_counts = [r["human_verification_count"] for r in threshold_rows]
reobserve_counts = [r["re_observation_required_count"] for r in threshold_rows]
insufficient_counts = [r["insufficient_evidence_count"] for r in threshold_rows]

x = list(range(len(threshold_rows)))

fig, ax = plt.subplots(figsize=(10.5, 5.8))
style_axes(ax)

bottom = [0] * len(x)

ax.bar(
    x,
    probable_counts,
    bottom=bottom,
    color=COLORS["teal"],
    edgecolor=COLORS["navy"],
    linewidth=0.5,
    label="Probable match"
)
bottom = [b + v for b, v in zip(bottom, probable_counts)]

ax.bar(
    x,
    human_counts,
    bottom=bottom,
    color=COLORS["blue"],
    edgecolor=COLORS["navy"],
    linewidth=0.5,
    label="Human verification"
)
bottom = [b + v for b, v in zip(bottom, human_counts)]

ax.bar(
    x,
    reobserve_counts,
    bottom=bottom,
    color=COLORS["amber"],
    edgecolor=COLORS["navy"],
    linewidth=0.5,
    label="Re-observation"
)
bottom = [b + v for b, v in zip(bottom, reobserve_counts)]

ax.bar(
    x,
    insufficient_counts,
    bottom=bottom,
    color=COLORS["coral"],
    edgecolor=COLORS["navy"],
    linewidth=0.5,
    label="Insufficient evidence"
)

ax.set_xticks(x)
ax.set_xticklabels(threshold_labels)
ax.set_ylim(0, len(feature_rows) + 0.8)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Number of scenarios")
ax.set_title("Threshold Sensitivity of Re-Identification Decisions", pad=14, fontweight="bold")

for i in x:
    total = probable_counts[i] + human_counts[i] + reobserve_counts[i] + insufficient_counts[i]
    ax.text(
        i,
        total + 0.15,
        f"n={total}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["muted"]
    )

ax.legend(
    loc="upper right",
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Higher thresholds make the system more conservative and increase human verification or re-observation decisions."
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

threshold_graph_path = os.path.join(OUTPUT_FIGURES, "threshold_sensitivity_graph.png")
fig.savefig(threshold_graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {threshold_graph_path}")


# ============================================================
# Part 3 - Failure-case / limitation analysis
# ============================================================

failure_scenarios = [
    {
        "case_id": "FC-01",
        "condition": "Dense smoke",
        "body_color_consistency": 0.52,
        "shape_consistency": 0.64,
        "visible_area_quality": 0.32,
        "occlusion_consistency": 0.62,
        "viewpoint_consistency": 0.60,
    },
    {
        "case_id": "FC-02",
        "condition": "Severe occlusion",
        "body_color_consistency": 0.60,
        "shape_consistency": 0.55,
        "visible_area_quality": 0.28,
        "occlusion_consistency": 0.24,
        "viewpoint_consistency": 0.68,
    },
    {
        "case_id": "FC-03",
        "condition": "Strong viewpoint shift",
        "body_color_consistency": 0.66,
        "shape_consistency": 0.58,
        "visible_area_quality": 0.56,
        "occlusion_consistency": 0.70,
        "viewpoint_consistency": 0.30,
    },
    {
        "case_id": "FC-04",
        "condition": "Low light + occlusion",
        "body_color_consistency": 0.48,
        "shape_consistency": 0.52,
        "visible_area_quality": 0.24,
        "occlusion_consistency": 0.38,
        "viewpoint_consistency": 0.55,
    },
    {
        "case_id": "FC-05",
        "condition": "Extreme degradation",
        "body_color_consistency": 0.36,
        "shape_consistency": 0.42,
        "visible_area_quality": 0.18,
        "occlusion_consistency": 0.22,
        "viewpoint_consistency": 0.34,
    },
]


failure_rows = []

for row in failure_scenarios:
    similarity = weighted_feature_similarity(row)
    uncertainty = uncertainty_score(row, similarity)

    if similarity >= 0.80:
        safe_decision = "Probable match"
    elif similarity >= 0.65:
        safe_decision = "Human verification advised"
    elif similarity >= 0.50:
        safe_decision = "Re-observation required"
    else:
        safe_decision = "Insufficient evidence"

    failure_rows.append({
        "case_id": row["case_id"],
        "condition": row["condition"],
        "weighted_similarity": round(similarity, 3),
        "uncertainty_score": round(uncertainty, 3),
        "safe_decision": safe_decision,
    })


failure_csv = os.path.join(OUTPUT_TABLES, "failure_case_analysis.csv")
write_csv(failure_csv, failure_rows)


# ------------------------------------------------------------
# Figure 3: Failure-case analysis graph
# ------------------------------------------------------------

case_ids = [r["case_id"] for r in failure_rows]
failure_similarity = [r["weighted_similarity"] * 100 for r in failure_rows]
failure_uncertainty = [r["uncertainty_score"] * 100 for r in failure_rows]

x = list(range(len(case_ids)))

fig, ax = plt.subplots(figsize=(10.5, 5.8))
style_axes(ax, both_axes=True)

ax.plot(
    x,
    failure_similarity,
    marker="o",
    markersize=8,
    linewidth=2.6,
    color=COLORS["teal"],
    markeredgecolor=COLORS["navy"],
    label="Weighted similarity"
)

ax.plot(
    x,
    failure_uncertainty,
    marker="o",
    markersize=8,
    linewidth=2.6,
    color=COLORS["coral"],
    markeredgecolor=COLORS["navy"],
    label="Uncertainty score"
)

ax.axhline(80, color=COLORS["slate"], linestyle="--", linewidth=1.3, alpha=0.75)
ax.text(
    len(x) - 0.4,
    82,
    "confident-match threshold",
    ha="right",
    va="bottom",
    fontsize=9,
    color=COLORS["slate"]
)

for i, row in enumerate(failure_rows):
    ax.text(
        i,
        failure_similarity[i] - 6,
        row["safe_decision"],
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["text"],
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=COLORS["white"],
            edgecolor=COLORS["grid"],
            alpha=0.90
        )
    )

ax.set_xticks(x)
ax.set_xticklabels(case_ids)
ax.set_ylim(0, 100)
ax.set_xlabel("Failure / limitation case")
ax.set_ylabel("Score (%)")
ax.set_title("Failure-Case Analysis Under Extreme Degradation", pad=14, fontweight="bold")
ax.legend(
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Failure cases demonstrate conservative behavior: the system avoids final identity decisions under insufficient evidence."
)

fig.tight_layout(rect=[0, 0.05, 1, 1])

failure_graph_path = os.path.join(OUTPUT_FIGURES, "failure_case_analysis_graph.png")
fig.savefig(failure_graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {failure_graph_path}")


print("\nDone. Generated:")
print("- visual_memory_feature_analysis.csv")
print("- threshold_sensitivity.csv")
print("- failure_case_analysis.csv")
print("- visual_memory_feature_graph.png")
print("- threshold_sensitivity_graph.png")
print("- failure_case_analysis_graph.png")