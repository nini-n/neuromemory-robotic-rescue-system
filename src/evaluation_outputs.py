import os
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# NeuroMemory Robot - Evaluation and Poster Assets v4
# Final polished engineering-style outputs
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
    "deep_blue": "#1F4E79",
    "blue": "#2F6690",
    "teal": "#2A9D8F",
    "green": "#52B788",
    "amber": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "slate": "#5C677D",
    "light_slate": "#94A3B8",
    "bg": "#F7F9FC",
    "panel": "#EEF3F8",
    "grid": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
    "white": "#FFFFFF",

    "layer_perception": "#EAF3FF",
    "layer_memory": "#EEF8F2",
    "layer_human": "#FFF5E8",

    "box_perception": "#FDFEFF",
    "box_memory": "#FCFFFD",
    "box_human": "#FFFCF7",
}


# ------------------------------------------------------------
# Global style
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": COLORS["white"],
    "axes.facecolor": COLORS["bg"],
    "axes.edgecolor": COLORS["light_slate"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "text.color": COLORS["text"],
})


# ------------------------------------------------------------
# Scenario evaluation data
# ------------------------------------------------------------
scenarios = [
    {
        "scenario_id": "SC-01",
        "condition": "Clear visibility",
        "visibility": "High",
        "occlusion": "None",
        "baseline_confidence_percent": 78.4,
        "before_nbv_percent": 71.8,
        "after_nbv_percent": 87.6,
        "improvement_percent": 15.8,
        "decision_output": "Likely match"
    },
    {
        "scenario_id": "SC-02",
        "condition": "Smoke",
        "visibility": "Medium",
        "occlusion": "Low",
        "baseline_confidence_percent": 58.2,
        "before_nbv_percent": 63.7,
        "after_nbv_percent": 86.1,
        "improvement_percent": 22.4,
        "decision_output": "Likely match after re-observation"
    },
    {
        "scenario_id": "SC-03",
        "condition": "Low light",
        "visibility": "Low",
        "occlusion": "Low",
        "baseline_confidence_percent": 52.3,
        "before_nbv_percent": 60.8,
        "after_nbv_percent": 82.4,
        "improvement_percent": 21.6,
        "decision_output": "Likely match after re-observation"
    },
    {
        "scenario_id": "SC-04",
        "condition": "Partial occlusion",
        "visibility": "Medium",
        "occlusion": "Medium",
        "baseline_confidence_percent": 48.1,
        "before_nbv_percent": 57.2,
        "after_nbv_percent": 78.3,
        "improvement_percent": 21.1,
        "decision_output": "Human verification advised"
    },
    {
        "scenario_id": "SC-05",
        "condition": "Smoke + occlusion + viewpoint shift",
        "visibility": "Low",
        "occlusion": "High",
        "baseline_confidence_percent": 41.3,
        "before_nbv_percent": 54.1,
        "after_nbv_percent": 73.2,
        "improvement_percent": 19.1,
        "decision_output": "Uncertain but improved"
    },
]


# ------------------------------------------------------------
# Save scenario table
# ------------------------------------------------------------
csv_path = os.path.join(OUTPUT_TABLES, "scenario_evaluation.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=scenarios[0].keys())
    writer.writeheader()
    writer.writerows(scenarios)

print(f"Saved table: {csv_path}")


# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------
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


def label_box(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=COLORS["white"],
            edgecolor="none",
            alpha=0.92
        )
    )


scenario_ids = [s["scenario_id"] for s in scenarios]
baseline_values = [s["baseline_confidence_percent"] for s in scenarios]
before_values = [s["before_nbv_percent"] for s in scenarios]
after_values = [s["after_nbv_percent"] for s in scenarios]
improvement_values = [s["improvement_percent"] for s in scenarios]


# ------------------------------------------------------------
# Figure 1: Confidence improvement slope plot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 5.8))
style_axes(ax, both_axes=True)

before_x = 0
after_x = 1

ax.axvspan(-0.2, 0.2, color=COLORS["blue"], alpha=0.06)
ax.axvspan(0.8, 1.2, color=COLORS["teal"], alpha=0.08)

for scenario in scenarios:
    before = scenario["before_nbv_percent"]
    after = scenario["after_nbv_percent"]
    improvement = scenario["improvement_percent"]
    sid = scenario["scenario_id"]

    ax.plot(
        [before_x, after_x],
        [before, after],
        color=COLORS["slate"],
        linewidth=1.8,
        alpha=0.75
    )

    ax.scatter(
        before_x,
        before,
        s=80,
        color=COLORS["blue"],
        edgecolor=COLORS["navy"],
        linewidth=0.8,
        zorder=3
    )
    ax.scatter(
        after_x,
        after,
        s=90,
        color=COLORS["teal"],
        edgecolor=COLORS["navy"],
        linewidth=0.8,
        zorder=3
    )

    midpoint_y = (before + after) / 2
    ax.text(
        0.50,
        midpoint_y,
        f"{sid}  +{improvement:.1f}%",
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["text"],
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=COLORS["white"],
            edgecolor=COLORS["grid"],
            alpha=0.92
        )
    )

    ax.text(
        before_x - 0.08,
        before,
        f"{before:.1f}",
        ha="right",
        va="center",
        fontsize=9,
        color=COLORS["deep_blue"]
    )
    ax.text(
        after_x + 0.08,
        after,
        f"{after:.1f}",
        ha="left",
        va="center",
        fontsize=9,
        color=COLORS["teal"]
    )

ax.set_xlim(-0.35, 1.35)
ax.set_ylim(45, 92)
ax.set_xticks([before_x, after_x])
ax.set_xticklabels(["Before NBV", "After NBV"])
ax.set_ylabel("Re-identification confidence (%)")
ax.set_title("Confidence Improvement from Active Re-Observation", pad=14, fontweight="bold")

ax.scatter([], [], s=80, color=COLORS["blue"], edgecolor=COLORS["navy"], label="Before next-best-view")
ax.scatter([], [], s=80, color=COLORS["teal"], edgecolor=COLORS["navy"], label="After next-best-view")
ax.legend(
    loc="lower right",
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Simulation-based proof-of-concept outputs; each line shows confidence change for one representative scenario."
)

fig.tight_layout(rect=[0, 0.045, 1, 1])
slope_graph_path = os.path.join(OUTPUT_FIGURES, "confidence_improvement_slope.png")
fig.savefig(slope_graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {slope_graph_path}")


# ------------------------------------------------------------
# Figure 2: Baseline comparison, polished and spacing fixed
# ------------------------------------------------------------
x = list(range(len(scenario_ids)))

fig, ax = plt.subplots(figsize=(10.5, 5.8))
style_axes(ax, both_axes=True)

ax.fill_between(
    x,
    baseline_values,
    after_values,
    color=COLORS["amber"],
    alpha=0.22,
    label="Confidence margin"
)

ax.plot(
    x,
    baseline_values,
    marker="o",
    markersize=7.5,
    linewidth=2.6,
    color=COLORS["coral"],
    markeredgecolor=COLORS["navy"],
    markeredgewidth=0.6,
    label="Basic detection baseline"
)

ax.plot(
    x,
    after_values,
    marker="o",
    markersize=7.5,
    linewidth=2.8,
    color=COLORS["teal"],
    markeredgecolor=COLORS["navy"],
    markeredgewidth=0.6,
    label="NeuroMemory after NBV"
)

ax.set_xticks(x)
ax.set_xticklabels(scenario_ids)
ax.set_ylim(0, 100)
ax.set_xlabel("Simulation scenario", labelpad=2)
ax.set_ylabel("Confidence (%)")
ax.set_title("Baseline Detection vs NeuroMemory Decision Support", pad=14, fontweight="bold")

for i, (base, neuro) in enumerate(zip(baseline_values, after_values)):
    label_box(ax, i, neuro + 5.0, f"{neuro:.1f}", COLORS["teal"])
    label_box(ax, i, base - 5.5, f"{base:.1f}", COLORS["coral"])

for i, scenario in enumerate(scenarios):
    condition = scenario["condition"]
    short_condition = condition.replace("Smoke + occlusion + viewpoint shift", "Combined degradation")
    ax.text(
        i,
        -0.10,
        short_condition,
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLORS["muted"],
        transform=ax.get_xaxis_transform()
    )

ax.legend(
    loc="upper right",
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Baseline uses detection-only confidence; NeuroMemory adds visual memory, uncertainty check, and active re-observation."
)

fig.tight_layout(rect=[0, 0.11, 1, 1])
baseline_graph_path = os.path.join(OUTPUT_FIGURES, "baseline_comparison_graph.png")
fig.savefig(baseline_graph_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {baseline_graph_path}")


# ------------------------------------------------------------
# Figure 3: Layered architecture diagram, polished
# ------------------------------------------------------------
def draw_layer(ax, x, y, w, h, label, facecolor, edgecolor):
    layer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_patch(layer)
    ax.text(
        x + 0.15,
        y + h - 0.22,
        label,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"]
    )


def draw_arch_box(ax, x, y, w, h, text, facecolor, edgecolor):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.3,
        color=COLORS["text"]
    )


def draw_arch_arrow(ax, start, end, color=COLORS["slate"], lw=1.35, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(12.5, 7.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor(COLORS["white"])

ax.text(
    6,
    7.45,
    "NeuroMemory Robot System Architecture",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=COLORS["navy"]
)

# UZATILMIŞ VE DAHA DENGELİ TITLE ALTI ÇİZGİ
ax.plot([3.2, 8.8], [7.10, 7.10], color=COLORS["teal"], linewidth=2.5)

draw_layer(ax, 0.6, 5.1, 10.7, 1.55, "Perception Layer", COLORS["layer_perception"], COLORS["blue"])
draw_layer(ax, 0.6, 3.2, 10.7, 1.55, "Memory and Planning Layer", COLORS["layer_memory"], COLORS["green"])
draw_layer(ax, 0.6, 1.3, 10.7, 1.55, "Human Decision-Support Layer", COLORS["layer_human"], COLORS["orange"])

draw_arch_box(ax, 1.0, 5.55, 1.7, 0.65, "Robot / Drone\nCamera Input", COLORS["box_perception"], COLORS["blue"])
draw_arch_box(ax, 3.0, 5.55, 1.7, 0.65, "Feature\nExtraction", COLORS["box_perception"], COLORS["blue"])
draw_arch_box(ax, 5.0, 5.55, 1.7, 0.65, "Visual Memory\nEmbedding", COLORS["box_perception"], COLORS["blue"])
draw_arch_box(ax, 7.0, 5.55, 1.7, 0.65, "Similarity\nScoring", COLORS["box_perception"], COLORS["blue"])
draw_arch_box(ax, 9.0, 5.55, 1.7, 0.65, "Confidence\nEstimate", COLORS["box_perception"], COLORS["blue"])

draw_arch_box(ax, 1.1, 3.65, 1.8, 0.65, "Uncertainty\nCheck", COLORS["box_memory"], COLORS["green"])
draw_arch_box(ax, 3.2, 3.65, 1.8, 0.65, "Expected\nConfidence Gain", COLORS["box_memory"], COLORS["green"])
draw_arch_box(ax, 5.4, 3.65, 1.8, 0.65, "Next-Best-View\nSelection", COLORS["box_memory"], COLORS["green"])
draw_arch_box(ax, 7.6, 3.65, 1.8, 0.65, "Risk-Aware\nA* Planning", COLORS["box_memory"], COLORS["green"])
draw_arch_box(ax, 9.8, 3.65, 1.2, 0.65, "Re-\nObserve", COLORS["box_memory"], COLORS["green"])

draw_arch_box(ax, 1.1, 1.75, 1.8, 0.65, "Last-Seen\nMemory Map", COLORS["box_human"], COLORS["orange"])
draw_arch_box(ax, 3.2, 1.75, 1.8, 0.65, "Search Priority\nScore", COLORS["box_human"], COLORS["orange"])
draw_arch_box(ax, 5.4, 1.75, 1.8, 0.65, "Operator\nDashboard", COLORS["box_human"], COLORS["orange"])
draw_arch_box(ax, 7.6, 1.75, 1.8, 0.65, "Human\nVerification", COLORS["box_human"], COLORS["orange"])
draw_arch_box(ax, 9.8, 1.75, 1.2, 0.65, "Decision\nSupport", COLORS["box_human"], COLORS["orange"])

draw_arch_arrow(ax, (2.7, 5.87), (3.0, 5.87), color=COLORS["blue"])
draw_arch_arrow(ax, (4.7, 5.87), (5.0, 5.87), color=COLORS["blue"])
draw_arch_arrow(ax, (6.7, 5.87), (7.0, 5.87), color=COLORS["blue"])
draw_arch_arrow(ax, (8.7, 5.87), (9.0, 5.87), color=COLORS["blue"])

draw_arch_arrow(ax, (9.6, 5.55), (2.0, 4.3), color=COLORS["slate"], lw=1.2, rad=0.08)

draw_arch_arrow(ax, (2.9, 3.97), (3.2, 3.97), color=COLORS["green"])
draw_arch_arrow(ax, (5.0, 3.97), (5.4, 3.97), color=COLORS["green"])
draw_arch_arrow(ax, (7.2, 3.97), (7.6, 3.97), color=COLORS["green"])
draw_arch_arrow(ax, (9.4, 3.97), (9.8, 3.97), color=COLORS["green"])

draw_arch_arrow(ax, (2.0, 3.65), (1.9, 2.4), color=COLORS["slate"], lw=1.2)
draw_arch_arrow(ax, (10.4, 3.65), (10.4, 2.4), color=COLORS["slate"], lw=1.2)

draw_arch_arrow(ax, (2.9, 2.07), (3.2, 2.07), color=COLORS["orange"])
draw_arch_arrow(ax, (5.0, 2.07), (5.4, 2.07), color=COLORS["orange"])
draw_arch_arrow(ax, (7.2, 2.07), (7.6, 2.07), color=COLORS["orange"])
draw_arch_arrow(ax, (9.4, 2.07), (9.8, 2.07), color=COLORS["orange"])

ax.text(
    6,
    0.52,
    "Final identity-related decisions are not automated; the system provides confidence-based support to a human operator.",
    ha="center",
    va="center",
    fontsize=9.5,
    color=COLORS["muted"]
)

fig.tight_layout()
architecture_path = os.path.join(OUTPUT_FIGURES, "architecture_diagram_layered.png")
fig.savefig(architecture_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {architecture_path}")

print("\nDone. Generated final polished evaluation table, graphs, and layered architecture diagram.")