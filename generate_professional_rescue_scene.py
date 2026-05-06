from pathlib import Path
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "poster_ready_mission_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(21)
np.random.seed(21)


# ============================================================
# Professional color palette
# ============================================================
BG = "#070A0F"
MAP_BG = "#242B34"
GRID = "#34404C"
BORDER = "#4A5565"

CONCRETE = "#6F675B"
CONCRETE_DARK = "#4C4842"
CONCRETE_LIGHT = "#9A9286"

ROBOT = "#1F6FEB"
ROBOT_DARK = "#0C2F57"
CYAN = "#00D9FF"
TEAL = "#00C2A8"
GREEN = "#00E676"
AMBER = "#FFC247"
ORANGE = "#FF6B35"
RED = "#FF4D4D"
WHITE = "#E9EEF5"
MUTED = "#9AA7B4"
SMOKE = "#B8C0CC"


# ============================================================
# General helpers
# ============================================================
def irregular_polygon(cx, cy, radius, n=8, roughness=0.35, angle=0.0):
    pts = []
    for i in range(n):
        a = angle + 2 * np.pi * i / n
        r = radius * (1 + roughness * (np.random.rand() - 0.5))
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return pts


def rotate_points(points, angle_rad, center=(0, 0)):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = center
    out = []
    for x, y in points:
        x0, y0 = x - cx, y - cy
        out.append((cx + c * x0 - s * y0, cy + s * x0 + c * y0))
    return out


def setup_map_ax(title):
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)
    ax.set_facecolor("#101720")

    ax.add_patch(
        Rectangle(
            (-6.8, -5.1),
            13.6,
            10.2,
            facecolor=MAP_BG,
            edgecolor=BORDER,
            linewidth=1.7,
            zorder=0,
        )
    )

    for x in np.linspace(-6.5, 6.5, 14):
        ax.plot([x, x], [-4.9, 4.9], color=GRID, linewidth=0.45, alpha=0.45, zorder=1)

    for y in np.linspace(-4.8, 4.8, 11):
        ax.plot([-6.6, 6.6], [y, y], color=GRID, linewidth=0.45, alpha=0.45, zorder=1)

    ax.set_title(title, color=WHITE, fontsize=21, fontweight="bold", pad=20)
    ax.set_xlim(-6.9, 6.9)
    ax.set_ylim(-5.2, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax


def draw_rubble_cluster(ax, cx, cy, scale=1.0, label=None, z=8):
    main_count = random.randint(3, 5)

    for _ in range(main_count):
        px = cx + random.uniform(-0.45, 0.45) * scale
        py = cy + random.uniform(-0.40, 0.40) * scale
        r = random.uniform(0.32, 0.62) * scale

        pts = irregular_polygon(
            px,
            py,
            r,
            n=random.randint(6, 10),
            roughness=0.62,
            angle=random.uniform(0, math.pi),
        )

        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor=random.choice([CONCRETE, CONCRETE_DARK, CONCRETE_LIGHT]),
                edgecolor="#2C2A27",
                linewidth=0.7,
                alpha=0.96,
                zorder=z,
            )
        )

    for _ in range(18):
        px = cx + random.uniform(-1.1, 1.1) * scale
        py = cy + random.uniform(-0.95, 0.95) * scale
        r = random.uniform(0.06, 0.18) * scale

        pts = irregular_polygon(
            px,
            py,
            r,
            n=random.randint(4, 7),
            roughness=0.65,
            angle=random.random() * math.pi,
        )

        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor=random.choice([CONCRETE_DARK, CONCRETE, CONCRETE_LIGHT]),
                edgecolor="none",
                alpha=random.uniform(0.62, 0.90),
                zorder=z + 1,
            )
        )

    for _ in range(4):
        x0 = cx + random.uniform(-0.8, 0.8) * scale
        y0 = cy + random.uniform(-0.7, 0.7) * scale
        x1 = x0 + random.uniform(-0.65, 0.65) * scale
        y1 = y0 + random.uniform(-0.45, 0.45) * scale

        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#1A1F25",
            linewidth=2.0,
            alpha=0.85,
            zorder=z + 2,
        )

    if label:
        ax.text(
            cx,
            cy + 1.25 * scale,
            label,
            color=MUTED,
            fontsize=12,
            fontweight="bold",
            ha="center",
            zorder=25,
        )


def draw_robot(ax, x, y, heading=0, label=True):
    body = [(-0.65, -0.42), (0.65, -0.42), (0.65, 0.42), (-0.65, 0.42)]
    body = rotate_points([(px + x, py + y) for px, py in body], heading, (x, y))

    ax.add_patch(
        Polygon(
            body,
            closed=True,
            facecolor=ROBOT_DARK,
            edgecolor=CYAN,
            linewidth=1.6,
            zorder=30,
        )
    )

    top = [(-0.35, -0.22), (0.35, -0.22), (0.35, 0.22), (-0.35, 0.22)]
    top = rotate_points([(px + x, py + y) for px, py in top], heading, (x, y))

    ax.add_patch(
        Polygon(
            top,
            closed=True,
            facecolor=ROBOT,
            edgecolor=WHITE,
            linewidth=0.7,
            zorder=31,
        )
    )

    for wx, wy in [(-0.52, -0.55), (0.52, -0.55), (-0.52, 0.55), (0.52, 0.55)]:
        p = rotate_points([(x + wx, y + wy)], heading, (x, y))[0]
        ax.add_patch(
            Circle(
                p,
                radius=0.16,
                facecolor="#10151B",
                edgecolor="#333A44",
                linewidth=0.6,
                zorder=29,
            )
        )

    front = rotate_points([(x + 0.85, y)], heading, (x, y))[0]
    ax.plot([x, front[0]], [y, front[1]], color=CYAN, linewidth=2.0, zorder=35)

    ax.add_patch(Circle((x, y), radius=0.055, facecolor=GREEN, edgecolor="none", zorder=36))

    if label:
        ax.text(
            x - 0.55,
            y - 0.92,
            "Rescue robot",
            color=CYAN,
            fontsize=14,
            fontweight="bold",
            ha="center",
            zorder=45,
        )


def draw_environment(ax, show_labels=True):
    # Risk zone
    risk_poly = np.array([
        (0.65, -0.75),
        (2.05, -0.95),
        (3.85, -0.65),
        (5.05, 0.25),
        (4.75, 2.05),
        (2.95, 2.55),
        (1.05, 2.05),
        (0.25, 0.65),
    ])

    ax.add_patch(
        Polygon(
            risk_poly,
            closed=True,
            facecolor=AMBER,
            edgecolor="#E2A300",
            alpha=0.20,
            linewidth=2.0,
            zorder=3,
        )
    )

    for radius, alpha in [(1.65, 0.10), (1.15, 0.15), (0.68, 0.23)]:
        ax.add_patch(
            Circle(
                (3.45, 1.15),
                radius=radius,
                facecolor=AMBER,
                edgecolor="none",
                alpha=alpha,
                zorder=4,
            )
        )

    # Low visibility
    smoke_centers = [
        (2.05, 0.65, 1.05),
        (2.65, 1.15, 1.10),
        (3.25, 1.55, 1.00),
        (3.85, 0.65, 0.95),
        (2.65, 2.10, 0.85),
        (4.25, 1.55, 0.75),
    ]

    for sx, sy, sr in smoke_centers:
        ax.add_patch(
            Circle(
                (sx, sy),
                radius=sr,
                facecolor=SMOKE,
                edgecolor="#C9D1DB",
                linewidth=0.3,
                alpha=0.12,
                zorder=5,
            )
        )

    # Debris fields
    draw_rubble_cluster(ax, -4.1, 2.2, scale=0.95, label="Debris field" if show_labels else None)
    draw_rubble_cluster(ax, -2.2, -0.9, scale=0.75)
    draw_rubble_cluster(ax, 0.4, 2.6, scale=0.82)
    draw_rubble_cluster(ax, 4.35, 2.05, scale=0.72)
    draw_rubble_cluster(ax, 3.1, -2.25, scale=0.80)

    # Collapsed walls
    collapsed_walls = [
        ((-3.2, 0.25), 1.7, 0.32, 18),
        ((-0.9, 2.85), 1.25, 0.25, -8),
        ((4.3, -0.35), 1.35, 0.28, 15),
    ]

    for (cx, cy), w, h, rot in collapsed_walls:
        rect = np.array([
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ])

        pts = rotate_points([(cx + x, cy + y) for x, y in rect], math.radians(rot), (cx, cy))

        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor="#7B7266",
                edgecolor="#2F2C28",
                linewidth=0.8,
                alpha=0.93,
                zorder=10,
            )
        )

    if show_labels:
        ax.text(2.55, 0.75, "Risk zone", color=AMBER, fontsize=15, fontweight="bold", ha="center", zorder=40)
        ax.text(3.10, 2.85, "Low-visibility zone", color=WHITE, fontsize=15, fontweight="bold", ha="center", zorder=40)


def draw_survivor(ax, x=3.65, y=1.10, label=True):
    ax.add_patch(
        Circle((x, y), radius=0.34, facecolor=ORANGE, edgecolor=WHITE, linewidth=1.3, zorder=50)
    )
    ax.add_patch(
        Circle((x, y), radius=0.74, facecolor="none", edgecolor=ORANGE, linewidth=2.2, linestyle="--", alpha=0.95, zorder=49)
    )
    ax.add_patch(
        Circle((x, y), radius=1.15, facecolor=ORANGE, edgecolor="none", alpha=0.08, zorder=48)
    )

    if label:
        ax.text(
            x + 0.15,
            y + 0.95,
            "Survivor candidate",
            color=ORANGE,
            fontsize=15,
            fontweight="bold",
            ha="center",
            zorder=60,
        )


def draw_nbv(ax, x=2.05, y=-1.05, label=True):
    ax.add_patch(Circle((x, y), radius=0.34, facecolor=GREEN, edgecolor=WHITE, linewidth=1.1, zorder=52))
    ax.add_patch(Circle((x, y), radius=0.72, facecolor="none", edgecolor=GREEN, linewidth=2.4, zorder=51))
    ax.add_patch(Circle((x, y), radius=1.05, facecolor=GREEN, edgecolor="none", alpha=0.08, zorder=50))

    if label:
        ax.text(
            x - 0.35,
            y - 0.90,
            "Next-best view",
            color=GREEN,
            fontsize=15,
            fontweight="bold",
            ha="center",
            zorder=60,
        )


def draw_paths(ax, robot=(-4.75, -2.95), survivor=(3.65, 1.10), nbv=(2.05, -1.05), show_legend=True):
    direct = np.array([
        [robot[0], robot[1]],
        [-3.0, -1.80],
        [-1.55, -0.95],
        [0.20, -0.05],
        [1.80, 0.55],
        [survivor[0], survivor[1]],
    ])

    risk = np.array([
        [robot[0], robot[1]],
        [-3.40, -2.65],
        [-2.10, -2.42],
        [-0.65, -2.12],
        [0.75, -1.65],
        [nbv[0], nbv[1]],
        [2.72, -0.20],
        [3.20, 0.45],
        [survivor[0], survivor[1]],
    ])

    ax.plot(direct[:, 0], direct[:, 1], color=RED, linewidth=2.6, alpha=0.78, label="Direct path", zorder=44)
    ax.plot(risk[:, 0], risk[:, 1], color=TEAL, linewidth=4.8, alpha=0.97, label="Risk-aware re-observation path", zorder=45)

    ax.scatter(risk[:, 0], risk[:, 1], s=28, color=TEAL, edgecolor="none", zorder=46)

    ax.annotate(
        "",
        xy=(risk[-1, 0], risk[-1, 1]),
        xytext=(risk[-2, 0], risk[-2, 1]),
        arrowprops=dict(arrowstyle="->", color=TEAL, linewidth=3.2),
        zorder=55,
    )

    ax.annotate(
        "",
        xy=(direct[-1, 0], direct[-1, 1]),
        xytext=(direct[-2, 0], direct[-2, 1]),
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=2.2),
        zorder=55,
    )

    if show_legend:
        legend = ax.legend(
            facecolor="#0C111A",
            edgecolor="#263241",
            labelcolor=WHITE,
            loc="lower right",
            fontsize=11,
            framealpha=0.95,
        )

        for line in legend.get_lines():
            line.set_linewidth(4)


def draw_perception_cone(ax, robot=(-4.75, -2.95)):
    rx, ry = robot

    cone = np.array([
        (rx + 0.25, ry + 0.15),
        (-1.3, -1.40),
        (-1.65, 0.25),
    ])

    ax.add_patch(
        Polygon(
            cone,
            closed=True,
            facecolor=CYAN,
            edgecolor=CYAN,
            alpha=0.12,
            linewidth=1.4,
            zorder=22,
        )
    )

    for tx, ty in [(-1.3, -1.4), (-1.65, 0.25), (-0.70, -0.45), (2.05, -1.05)]:
        ax.plot([rx, tx], [ry, ty], color=CYAN, linewidth=1.2, alpha=0.45, zorder=23)

    ax.text(
        -2.35,
        -0.20,
        "camera field-of-view",
        color=CYAN,
        fontsize=11,
        ha="center",
        alpha=0.90,
        zorder=60,
    )


def add_caption(fig, text):
    fig.text(0.5, 0.035, text, ha="center", color=MUTED, fontsize=10.5)


# ============================================================
# Figure 1: complete operator overview
# ============================================================
def save_operator_overview():
    fig, ax = setup_map_ax("NeuroMemory Robot — Operator Mission Overview")

    draw_environment(ax, show_labels=True)
    draw_perception_cone(ax)
    draw_robot(ax, -4.75, -2.95, heading=math.radians(18))
    draw_survivor(ax)
    draw_nbv(ax)
    draw_paths(ax, show_legend=True)

    # Decision layer annotation
    ax.add_patch(
        Rectangle(
            (-0.35, -4.62),
            4.3,
            0.82,
            facecolor="#0C111A",
            edgecolor="#2C3E50",
            linewidth=1.2,
            alpha=0.92,
            zorder=70,
        )
    )

    ax.text(
        1.80,
        -4.18,
        "NeuroMemory decision layer: uncertain identity → active re-observation → safer next-best view",
        color=WHITE,
        fontsize=10.5,
        ha="center",
        va="center",
        zorder=71,
    )

    # Status chips
    chips = [
        ("identity score", "0.72", CYAN),
        ("uncertainty", "0.42", AMBER),
        ("planner gain", "0.31", GREEN),
    ]

    chip_x = -6.35
    chip_y = 4.35

    for i, (k, v, col) in enumerate(chips):
        y = chip_y - i * 0.52

        ax.add_patch(
            Rectangle(
                (chip_x, y - 0.22),
                2.25,
                0.40,
                facecolor="#0C111A",
                edgecolor=col,
                linewidth=1.0,
                alpha=0.92,
                zorder=72,
            )
        )

        ax.text(
            chip_x + 0.10,
            y,
            f"{k}: {v}",
            color=col,
            fontsize=9.6,
            fontweight="bold",
            va="center",
            zorder=73,
        )

    add_caption(
        fig,
        "Operator-style overview showing debris, smoke, uncertain survivor identity, next-best-view selection, and risk-aware active re-observation planning.",
    )

    out = OUT_DIR / "fig_01_operator_overview.png"
    plt.savefig(out, dpi=280, facecolor=BG, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Figure 2: robot perception and sensing focus
# ============================================================
def save_robot_perception_view():
    fig, ax = setup_map_ax("Robot Perception and Active Re-Observation View")

    draw_environment(ax, show_labels=False)
    draw_perception_cone(ax)
    draw_robot(ax, -4.75, -2.95, heading=math.radians(18))
    draw_survivor(ax, label=False)
    draw_nbv(ax, label=False)

    # Highlight sensing targets
    ax.add_patch(Circle((2.05, -1.05), radius=0.92, facecolor=GREEN, edgecolor=GREEN, linewidth=2.0, alpha=0.15, zorder=55))
    ax.add_patch(Circle((3.65, 1.10), radius=1.10, facecolor=ORANGE, edgecolor=ORANGE, linewidth=2.0, alpha=0.13, zorder=55))

    ax.text(-4.75, -3.90, "Rescue robot with camera mast", color=CYAN, fontsize=14, fontweight="bold", ha="center", zorder=80)
    ax.text(-1.65, -0.55, "perception cone", color=CYAN, fontsize=13, fontweight="bold", ha="center", zorder=80)
    ax.text(2.05, -2.10, "candidate viewpoint for re-observation", color=GREEN, fontsize=12, fontweight="bold", ha="center", zorder=80)
    ax.text(3.65, 2.25, "uncertain identity region", color=ORANGE, fontsize=12, fontweight="bold", ha="center", zorder=80)

    # Only show re-observation path
    reobs = np.array([
        [-4.75, -2.95],
        [-3.40, -2.65],
        [-2.10, -2.42],
        [-0.65, -2.12],
        [0.75, -1.65],
        [2.05, -1.05],
    ])

    ax.plot(reobs[:, 0], reobs[:, 1], color=TEAL, linewidth=4.8, alpha=0.97, zorder=60)
    ax.scatter(reobs[:, 0], reobs[:, 1], s=30, color=TEAL, zorder=61)

    add_caption(
        fig,
        "Robot-centered view emphasizing perception direction, camera field-of-view, and active re-observation target selection.",
    )

    out = OUT_DIR / "fig_02_robot_perception_view.png"
    plt.savefig(out, dpi=280, facecolor=BG, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Figure 3: next-best-view and survivor decision focus
# ============================================================
def save_next_best_view_decision():
    fig, ax = setup_map_ax("Next-Best-View Decision and Survivor Candidate Focus")

    draw_environment(ax, show_labels=True)
    draw_robot(ax, -4.75, -2.95, heading=math.radians(18), label=False)
    draw_survivor(ax)
    draw_nbv(ax)
    draw_paths(ax, show_legend=False)

    # Focus area box around survivor/NBV
    focus_poly = np.array([
        (1.10, -1.95),
        (4.95, -1.00),
        (5.25, 2.85),
        (1.25, 3.05),
    ])

    ax.add_patch(
        Polygon(
            focus_poly,
            closed=True,
            facecolor="none",
            edgecolor=GREEN,
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            zorder=80,
        )
    )

    # Information gain arrow
    ax.annotate(
        "",
        xy=(3.65, 1.10),
        xytext=(2.05, -1.05),
        arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=3.2),
        zorder=85,
    )

    ax.text(
        3.05,
        -0.25,
        "expected confidence gain",
        color=GREEN,
        fontsize=12,
        fontweight="bold",
        ha="center",
        zorder=90,
    )

    # Mini decision panel
    ax.add_patch(
        Rectangle(
            (-6.35, 3.25),
            3.65,
            1.35,
            facecolor="#0C111A",
            edgecolor="#2C3E50",
            linewidth=1.2,
            alpha=0.95,
            zorder=90,
        )
    )

    ax.text(-4.52, 4.25, "Decision logic", color=WHITE, fontsize=12, fontweight="bold", ha="center", zorder=91)
    ax.text(-4.52, 3.82, "Low identity certainty", color=AMBER, fontsize=10, ha="center", zorder=91)
    ax.text(-4.52, 3.52, "Active re-observation recommended", color=GREEN, fontsize=10, ha="center", zorder=91)

    add_caption(
        fig,
        "Decision-focused view showing how uncertain identity evidence leads to a next-best-view selection for re-observation.",
    )

    out = OUT_DIR / "fig_03_next_best_view_decision.png"
    plt.savefig(out, dpi=280, facecolor=BG, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Figure 4: before-after path comparison
# ============================================================
def save_before_after_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), facecolor=BG)

    titles = [
        "Baseline Direct Path",
        "NeuroMemory Risk-Aware Re-Observation Path",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor("#101720")

        ax.add_patch(
            Rectangle(
                (-6.8, -5.1),
                13.6,
                10.2,
                facecolor=MAP_BG,
                edgecolor=BORDER,
                linewidth=1.4,
                zorder=0,
            )
        )

        for x in np.linspace(-6.5, 6.5, 12):
            ax.plot([x, x], [-4.9, 4.9], color=GRID, linewidth=0.4, alpha=0.35, zorder=1)
        for y in np.linspace(-4.8, 4.8, 10):
            ax.plot([-6.6, 6.6], [y, y], color=GRID, linewidth=0.4, alpha=0.35, zorder=1)

        draw_environment(ax, show_labels=False)
        draw_robot(ax, -4.75, -2.95, heading=math.radians(18), label=False)
        draw_survivor(ax, label=False)
        draw_nbv(ax, label=False)

        ax.set_title(title, color=WHITE, fontsize=16, fontweight="bold", pad=14)
        ax.set_xlim(-6.9, 6.9)
        ax.set_ylim(-5.2, 5.2)
        ax.set_aspect("equal")
        ax.axis("off")

    robot = (-4.75, -2.95)
    survivor = (3.65, 1.10)
    nbv = (2.05, -1.05)

    direct = np.array([
        [robot[0], robot[1]],
        [-3.0, -1.80],
        [-1.55, -0.95],
        [0.20, -0.05],
        [1.80, 0.55],
        [survivor[0], survivor[1]],
    ])

    risk = np.array([
        [robot[0], robot[1]],
        [-3.40, -2.65],
        [-2.10, -2.42],
        [-0.65, -2.12],
        [0.75, -1.65],
        [nbv[0], nbv[1]],
        [2.72, -0.20],
        [3.20, 0.45],
        [survivor[0], survivor[1]],
    ])

    axes[0].plot(direct[:, 0], direct[:, 1], color=RED, linewidth=4.0, alpha=0.95, zorder=70)
    axes[0].text(1.1, 0.25, "passes through\nrisk/smoke area", color=RED, fontsize=12, fontweight="bold", ha="center", zorder=80)

    axes[1].plot(risk[:, 0], risk[:, 1], color=TEAL, linewidth=4.8, alpha=0.98, zorder=70)
    axes[1].scatter(risk[:, 0], risk[:, 1], s=28, color=TEAL, zorder=71)
    axes[1].text(0.55, -2.35, "reroutes through\nnext-best-view", color=GREEN, fontsize=12, fontweight="bold", ha="center", zorder=80)

    fig.suptitle(
        "Path Planning Comparison: Direct Navigation vs Risk-Aware Active Re-Observation",
        color=WHITE,
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.035,
        "Comparison view for poster use: the proposed NeuroMemory logic selects a safer re-observation route instead of moving directly through the uncertain risk region.",
        ha="center",
        color=MUTED,
        fontsize=10.5,
    )

    out = OUT_DIR / "fig_04_before_after_path_comparison.png"
    plt.savefig(out, dpi=280, facecolor=BG, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"Saved: {out}")


# ============================================================
# Collage
# ============================================================
def make_collage():
    files = [
        ("fig_01_operator_overview.png", "Operator overview"),
        ("fig_02_robot_perception_view.png", "Robot perception"),
        ("fig_03_next_best_view_decision.png", "Next-best-view decision"),
        ("fig_04_before_after_path_comparison.png", "Before/after path comparison"),
    ]

    imgs = []
    for fname, label in files:
        path = OUT_DIR / fname
        im = Image.open(path).convert("RGB")
        imgs.append((im, label))

    tile_w, tile_h = 980, 560
    margin = 30
    title_h = 95
    label_h = 48

    canvas_w = tile_w * 2 + margin * 3
    canvas_h = title_h + (tile_h + label_h) * 2 + margin * 3

    canvas = Image.new("RGB", (canvas_w, canvas_h), (7, 10, 15))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 42)
        label_font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    draw.text(
        (margin, 28),
        "NeuroMemory Robot — Poster-Ready Mission Visualization Set",
        fill=(235, 240, 248),
        font=title_font,
    )

    positions = [
        (margin, title_h + margin),
        (margin * 2 + tile_w, title_h + margin),
        (margin, title_h + margin * 2 + tile_h + label_h),
        (margin * 2 + tile_w, title_h + margin * 2 + tile_h + label_h),
    ]

    for (img, label), (x, y) in zip(imgs, positions):
        img.thumbnail((tile_w, tile_h), Image.LANCZOS)

        bg = Image.new("RGB", (tile_w, tile_h), (10, 14, 21))
        px = (tile_w - img.width) // 2
        py = (tile_h - img.height) // 2
        bg.paste(img, (px, py))

        canvas.paste(bg, (x, y))
        draw.text((x + 18, y + tile_h + 10), label, fill=(235, 240, 248), font=label_font)

    out = OUT_DIR / "poster_ready_mission_figures_collage.png"
    canvas.save(out)
    print(f"Saved: {out}")


def main():
    print("Generating poster-ready mission figures...")

    save_operator_overview()
    save_robot_perception_view()
    save_next_best_view_decision()
    save_before_after_comparison()
    make_collage()

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()