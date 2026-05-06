import os
import csv
import math
import heapq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# NeuroMemory Robot - Advanced Evaluation v2
# Adds:
# 1. Scenario runner
# 2. Ablation study
# 3. Standard A* vs Risk-aware A* comparison
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
    "slate": "#5C677D",
    "light_slate": "#94A3B8",
    "bg": "#F7F9FC",
    "grid": "#CBD5E1",
    "text": "#1F2937",
    "muted": "#6B7280",
    "white": "#FFFFFF",
    "map_bg": "#252A32",
    "obstacle": "#5A544E",
    "smoke": "#9CA3AF",
    "standard_path": "#E76F51",
    "risk_path": "#2A9D8F",
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
# Shared helpers
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def safe_percent(value, reference):
    if reference == 0:
        return 0.0
    return 100.0 * value / reference


# ============================================================
# Part 1 - Scenario runner
# ============================================================

scenario_params = [
    {
        "scenario_id": "SC-01",
        "condition": "Clear visibility",
        "visibility_factor": 0.95,
        "occlusion_factor": 0.00,
        "viewpoint_shift": 0.05,
        "smoke_factor": 0.00,
    },
    {
        "scenario_id": "SC-02",
        "condition": "Smoke",
        "visibility_factor": 0.65,
        "occlusion_factor": 0.15,
        "viewpoint_shift": 0.15,
        "smoke_factor": 0.45,
    },
    {
        "scenario_id": "SC-03",
        "condition": "Low light",
        "visibility_factor": 0.52,
        "occlusion_factor": 0.10,
        "viewpoint_shift": 0.12,
        "smoke_factor": 0.15,
    },
    {
        "scenario_id": "SC-04",
        "condition": "Partial occlusion",
        "visibility_factor": 0.62,
        "occlusion_factor": 0.42,
        "viewpoint_shift": 0.20,
        "smoke_factor": 0.10,
    },
    {
        "scenario_id": "SC-05",
        "condition": "Smoke + occlusion + viewpoint shift",
        "visibility_factor": 0.45,
        "occlusion_factor": 0.55,
        "viewpoint_shift": 0.42,
        "smoke_factor": 0.50,
    },
]


def compute_baseline_confidence(params):
    """
    Basic detector confidence.
    It degrades under low visibility, occlusion, smoke, and viewpoint shift.
    """
    confidence = (
        0.82
        - 0.32 * (1.0 - params["visibility_factor"])
        - 0.24 * params["occlusion_factor"]
        - 0.18 * params["viewpoint_shift"]
        - 0.16 * params["smoke_factor"]
    )
    return clamp(confidence, 0.38, 0.90)


def compute_memory_only_confidence(params):
    """
    Memory-only system.
    It uses a stored reference but does not actively move to improve the observation.
    """
    baseline = compute_baseline_confidence(params)

    memory_boost = (
        0.04
        + 0.10 * (1.0 - params["visibility_factor"])
        + 0.05 * params["smoke_factor"]
        - 0.03 * params["viewpoint_shift"]
    )

    confidence = baseline + memory_boost
    return clamp(confidence, 0.40, 0.86)


def compute_full_neuromemory_confidence(params):
    """
    Full system.
    It combines memory, uncertainty handling, next-best-view,
    and active re-observation.
    """
    memory_only = compute_memory_only_confidence(params)

    nbv_gain = (
        0.05
        + 0.18 * (1.0 - params["visibility_factor"])
        + 0.09 * params["smoke_factor"]
        + 0.05 * params["viewpoint_shift"]
        - 0.035 * params["occlusion_factor"]
    )

    confidence = memory_only + nbv_gain
    return clamp(confidence, 0.45, 0.88)


def compute_priority_score(confidence, params):
    """
    Priority score for operator support.
    Higher confidence and higher degradation make the region more important.
    """
    degradation = (
        0.35 * (1.0 - params["visibility_factor"])
        + 0.30 * params["occlusion_factor"]
        + 0.20 * params["smoke_factor"]
        + 0.15 * params["viewpoint_shift"]
    )

    priority = 0.55 * confidence + 0.45 * degradation
    return clamp(priority, 0.0, 1.0)


def decision_from_confidence(confidence):
    if confidence >= 0.82:
        return "Probable match"
    elif confidence >= 0.70:
        return "Human verification advised"
    elif confidence >= 0.55:
        return "Uncertain but improved"
    else:
        return "Re-observation required"


def run_scenarios():
    rows = []

    for params in scenario_params:
        baseline = compute_baseline_confidence(params)
        memory_only = compute_memory_only_confidence(params)
        full = compute_full_neuromemory_confidence(params)
        priority = compute_priority_score(full, params)

        rows.append({
            "scenario_id": params["scenario_id"],
            "condition": params["condition"],
            "visibility_factor": round(params["visibility_factor"], 3),
            "occlusion_factor": round(params["occlusion_factor"], 3),
            "viewpoint_shift": round(params["viewpoint_shift"], 3),
            "smoke_factor": round(params["smoke_factor"], 3),
            "baseline_confidence_percent": round(baseline * 100, 1),
            "memory_only_confidence_percent": round(memory_only * 100, 1),
            "full_neuromemory_confidence_percent": round(full * 100, 1),
            "priority_score": round(priority, 3),
            "decision_output": decision_from_confidence(full),
        })

    return rows


scenario_rows = run_scenarios()

scenario_runner_csv = os.path.join(OUTPUT_TABLES, "scenario_runner_results.csv")
with open(scenario_runner_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=scenario_rows[0].keys())
    writer.writeheader()
    writer.writerows(scenario_rows)

print(f"Saved table: {scenario_runner_csv}")


# ============================================================
# Part 2 - Ablation study
# ============================================================

def build_ablation_rows():
    rows = []

    for row in scenario_rows:
        baseline = row["baseline_confidence_percent"]
        memory = row["memory_only_confidence_percent"]
        full = row["full_neuromemory_confidence_percent"]

        rows.append({
            "scenario_id": row["scenario_id"],
            "condition": row["condition"],
            "baseline_detection_percent": baseline,
            "memory_only_percent": memory,
            "full_neuromemory_percent": full,
            "memory_gain_percent": round(memory - baseline, 1),
            "nbv_gain_percent": round(full - memory, 1),
            "total_gain_percent": round(full - baseline, 1),
        })

    return rows


ablation_rows = build_ablation_rows()

ablation_csv = os.path.join(OUTPUT_TABLES, "ablation_study.csv")
with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=ablation_rows[0].keys())
    writer.writeheader()
    writer.writerows(ablation_rows)

print(f"Saved table: {ablation_csv}")


scenario_ids = [r["scenario_id"] for r in ablation_rows]
baseline_values = [r["baseline_detection_percent"] for r in ablation_rows]
memory_values = [r["memory_only_percent"] for r in ablation_rows]
full_values = [r["full_neuromemory_percent"] for r in ablation_rows]

x = list(range(len(scenario_ids)))
bar_w = 0.24

fig, ax = plt.subplots(figsize=(10.8, 5.8))
style_axes(ax)

ax.bar(
    [i - bar_w for i in x],
    baseline_values,
    width=bar_w,
    color=COLORS["coral"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Baseline detection"
)

ax.bar(
    x,
    memory_values,
    width=bar_w,
    color=COLORS["blue"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Memory only"
)

ax.bar(
    [i + bar_w for i in x],
    full_values,
    width=bar_w,
    color=COLORS["teal"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Full NeuroMemory"
)

ax.set_xticks(x)
ax.set_xticklabels(scenario_ids)
ax.set_ylim(0, 100)
ax.set_xlabel("Simulation scenario")
ax.set_ylabel("Confidence (%)")
ax.set_title(
    "Ablation Study: Contribution of Memory and Active Re-Observation",
    pad=14,
    fontweight="bold"
)

for i, value in enumerate(full_values):
    ax.text(
        i + bar_w,
        value + 2,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["teal"]
    )

for i, gain in enumerate([r["total_gain_percent"] for r in ablation_rows]):
    ax.text(
        i,
        max(full_values[i], memory_values[i], baseline_values[i]) + 7,
        f"+{gain:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["navy"],
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=COLORS["white"],
            edgecolor=COLORS["grid"],
            alpha=0.9
        )
    )

ax.legend(
    loc="upper right",
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Ablation separates baseline detection, visual memory, and the full system with active next-best-view."
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
ablation_graph = os.path.join(OUTPUT_FIGURES, "ablation_comparison_graph.png")
fig.savefig(ablation_graph, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {ablation_graph}")


# ============================================================
# Part 3 - Standard A* vs Risk-aware A*
# ============================================================

GRID_SIZE = 25
MAP_WIDTH = 850
MAP_HEIGHT = 720
COLS = MAP_WIDTH // GRID_SIZE
ROWS = MAP_HEIGHT // GRID_SIZE

# This specific test is designed so that a shortest-path planner tends
# to cross the low-visibility region, while a risk-aware planner takes
# a longer but safer route around it.
start = (360, 280)
goal = (815, 280)

obstacles = [
    (180, 130, 120, 80),
    (420, 520, 160, 70),
    (600, 120, 120, 95),
    (260, 260, 80, 140),
]

smoke_area = (540, 190, 230, 190)


def point_in_rect(p, rect):
    x, y = p
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def point_in_padded_obstacle(p, padding=20):
    x, y = p
    for rx, ry, rw, rh in obstacles:
        if rx - padding <= x <= rx + rw + padding and ry - padding <= y <= ry + rh + padding:
            return True
    return False


def world_to_grid(p):
    return int(p[0] // GRID_SIZE), int(p[1] // GRID_SIZE)


def grid_to_world(cell):
    return cell[0] * GRID_SIZE + GRID_SIZE // 2, cell[1] * GRID_SIZE + GRID_SIZE // 2


def is_blocked(cell):
    p = grid_to_world(cell)
    x, y = p
    if x < 0 or x >= MAP_WIDTH or y < 0 or y >= MAP_HEIGHT:
        return True
    return point_in_padded_obstacle(p)


def distance_to_rect(p, rect):
    x, y = p
    rx, ry, rw, rh = rect
    cx = clamp(x, rx, rx + rw)
    cy = clamp(y, ry, ry + rh)
    return euclidean(p, (cx, cy))


def cell_cost(cell, risk_aware=False):
    p = grid_to_world(cell)

    if not risk_aware:
        return 1.0

    cost = 1.0

    # Strong smoke penalty makes the planner prefer a longer but safer route.
    if point_in_rect(p, smoke_area):
        cost += 18.0

    for rect in obstacles:
        d = distance_to_rect(p, rect)
        if d < 50:
            cost += (50 - d) / 10.0

    return cost


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start_pos, goal_pos, risk_aware=False):
    start_cell = world_to_grid(start_pos)
    goal_cell = world_to_grid(goal_pos)

    open_set = []
    heapq.heappush(open_set, (0, start_cell))

    came_from = {}
    g_score = {start_cell: 0}

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_cell:
            path = []
            c = current
            while c in came_from:
                path.append(grid_to_world(c))
                c = came_from[c]
            path.append(grid_to_world(start_cell))
            path.reverse()
            return path

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)

            if nxt[0] < 0 or nxt[0] >= COLS or nxt[1] < 0 or nxt[1] >= ROWS:
                continue

            if is_blocked(nxt):
                continue

            tentative = g_score[current] + cell_cost(nxt, risk_aware=risk_aware)

            if nxt not in g_score or tentative < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + heuristic(nxt, goal_cell)
                heapq.heappush(open_set, (f, nxt))

    return [start_pos, goal_pos]


def path_length(path):
    if len(path) < 2:
        return 0.0
    return sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))


def smoke_exposure(path):
    return sum(1 for p in path if point_in_rect(p, smoke_area))


def total_risk_cost(path):
    total = 0.0
    for p in path:
        cell = world_to_grid(p)
        total += cell_cost(cell, risk_aware=True)
    return total


standard_path = astar(start, goal, risk_aware=False)
risk_path = astar(start, goal, risk_aware=True)

path_rows = [
    {
        "planner": "Standard A*",
        "path_nodes": len(standard_path),
        "path_length_px": round(path_length(standard_path), 1),
        "smoke_exposure_nodes": smoke_exposure(standard_path),
        "total_risk_cost": round(total_risk_cost(standard_path), 2),
        "comment": "shorter path with higher exposure"
    },
    {
        "planner": "Risk-aware A*",
        "path_nodes": len(risk_path),
        "path_length_px": round(path_length(risk_path), 1),
        "smoke_exposure_nodes": smoke_exposure(risk_path),
        "total_risk_cost": round(total_risk_cost(risk_path), 2),
        "comment": "longer route with reduced exposure"
    }
]

path_csv = os.path.join(OUTPUT_TABLES, "path_risk_comparison.csv")
with open(path_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=path_rows[0].keys())
    writer.writeheader()
    writer.writerows(path_rows)

print(f"Saved table: {path_csv}")


# ------------------------------------------------------------
# Path risk comparison graph - normalized values
# ------------------------------------------------------------

metrics = ["Path length", "Smoke exposure", "Total risk cost"]

standard_raw = [
    path_rows[0]["path_length_px"],
    path_rows[0]["smoke_exposure_nodes"],
    path_rows[0]["total_risk_cost"],
]

risk_raw = [
    path_rows[1]["path_length_px"],
    path_rows[1]["smoke_exposure_nodes"],
    path_rows[1]["total_risk_cost"],
]

standard_norm = [100.0, 100.0, 100.0]
risk_norm = [
    safe_percent(risk_raw[0], standard_raw[0]),
    safe_percent(risk_raw[1], standard_raw[1]),
    safe_percent(risk_raw[2], standard_raw[2]),
]

x = list(range(len(metrics)))
bar_w = 0.32

fig, ax = plt.subplots(figsize=(9.2, 5.5))
style_axes(ax)

ax.bar(
    [i - bar_w / 2 for i in x],
    standard_norm,
    width=bar_w,
    color=COLORS["coral"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Standard A*"
)

ax.bar(
    [i + bar_w / 2 for i in x],
    risk_norm,
    width=bar_w,
    color=COLORS["teal"],
    edgecolor=COLORS["navy"],
    linewidth=0.6,
    label="Risk-aware A*"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Relative value (% of Standard A*)")
ax.set_ylim(0, max(135, max(risk_norm) + 15))
ax.set_title(
    "Standard A* vs Risk-Aware A* Path Comparison",
    pad=14,
    fontweight="bold"
)

for i, (s_raw, r_raw, r_pct) in enumerate(zip(standard_raw, risk_raw, risk_norm)):
    ax.text(
        i - bar_w / 2,
        104,
        f"{s_raw}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["coral"]
    )
    ax.text(
        i + bar_w / 2,
        r_pct + 4,
        f"{r_raw}",
        ha="center",
        va="bottom",
        fontsize=9,
        color=COLORS["teal"]
    )

ax.legend(
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

add_footnote(
    fig,
    "Bars are normalized to Standard A* = 100%; numeric labels show exact metric values from path_risk_comparison.csv."
)

fig.tight_layout(rect=[0, 0.05, 1, 1])
path_graph = os.path.join(OUTPUT_FIGURES, "path_risk_comparison_graph.png")
fig.savefig(path_graph, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {path_graph}")


# ------------------------------------------------------------
# Path map visualization
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9.2, 7.0))
ax.set_facecolor(COLORS["map_bg"])
ax.set_xlim(0, MAP_WIDTH)
ax.set_ylim(MAP_HEIGHT, 0)
ax.set_aspect("equal")
ax.set_title(
    "Standard A* vs Risk-Aware A* in the Simulated Rescue Map",
    pad=14,
    fontweight="bold",
    color=COLORS["navy"]
)

# Grid
for gx in range(0, MAP_WIDTH, 50):
    ax.axvline(gx, color="#3A404A", linewidth=0.7, alpha=0.7)
for gy in range(0, MAP_HEIGHT, 50):
    ax.axhline(gy, color="#3A404A", linewidth=0.7, alpha=0.7)

# Obstacles
for rect in obstacles:
    rx, ry, rw, rh = rect
    ax.add_patch(
        Rectangle(
            (rx, ry),
            rw,
            rh,
            facecolor=COLORS["obstacle"],
            edgecolor="#B0AAA3",
            linewidth=1.2,
            alpha=0.90
        )
    )
    ax.text(rx + 10, ry + 18, "debris", color="#D0CCC8", fontsize=9)

# Smoke area
sx, sy, sw, sh = smoke_area
ax.add_patch(
    Rectangle(
        (sx, sy),
        sw,
        sh,
        facecolor=COLORS["smoke"],
        edgecolor=COLORS["white"],
        alpha=0.32,
        linewidth=1.2
    )
)
ax.text(sx + 15, sy + 25, "low-visibility region", color=COLORS["white"], fontsize=10)


def plot_path(path, color, label, linestyle="-"):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=2.8, linestyle=linestyle, label=label)
    ax.scatter(xs[0], ys[0], color=color, s=75, edgecolor=COLORS["white"], zorder=5)
    ax.scatter(xs[-1], ys[-1], color=color, s=105, edgecolor=COLORS["white"], marker="*", zorder=5)


plot_path(standard_path, COLORS["standard_path"], "Standard A*", linestyle="--")
plot_path(risk_path, COLORS["risk_path"], "Risk-aware A*", linestyle="-")

ax.text(start[0] - 45, start[1] + 35, "start", color=COLORS["white"], fontsize=10)
ax.text(goal[0] - 105, goal[1] - 25, "target observation", color=COLORS["white"], fontsize=10)

ax.legend(
    loc="lower right",
    frameon=True,
    facecolor=COLORS["white"],
    edgecolor=COLORS["grid"]
)

ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()
path_map = os.path.join(OUTPUT_FIGURES, "path_planning_map_comparison.png")
fig.savefig(path_map, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {path_map}")


print("\nDone. Generated advanced evaluation outputs:")
print("- scenario_runner_results.csv")
print("- ablation_study.csv")
print("- ablation_comparison_graph.png")
print("- path_risk_comparison.csv")
print("- path_risk_comparison_graph.png")
print("- path_planning_map_comparison.png")