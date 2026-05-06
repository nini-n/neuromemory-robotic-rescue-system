from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import (
    Rectangle, Circle, Polygon, FancyBboxPatch
)
from matplotlib import transforms as mtransforms
import matplotlib.image as mpimg

# ============================================================
# NeuroMemory Robot - Premium Poster-Ready Visualization Set
# ============================================================

OUT_DIR = Path("outputs/premium_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Global style / palette
# -------------------------
COLORS = {
    "bg": "#020812",
    "panel": "#1F2937",
    "panel2": "#111827",
    "panel3": "#0B1220",
    "grid": "#51607A",
    "grid_faint": "#73839B",
    "text": "#E8EDF6",
    "muted": "#AAB5C6",
    "muted2": "#91A0B5",
    "cyan": "#00D5FF",
    "cyan2": "#12C2E9",
    "cyan_soft": "#7DDFFF",
    "green": "#10E676",
    "green2": "#00FF9C",
    "orange": "#FF6B35",
    "orange2": "#FFB347",
    "red": "#FF4D5A",
    "amber": "#FFC845",
    "amber_soft": "#D4A72C",
    "blue_robot": "#2B6FFF",
    "white": "#FFFFFF",
    "smoke": "#C7CBD2",
    "smoke2": "#E1E4EA",
    "shadow": "#000000",
    "debris_dark": "#59544C",
    "debris_mid": "#8B8174",
    "debris_light": "#B6AA98",
    "steel": "#7F8C8D",
    "steel2": "#B8C4D0",
}

plt.rcParams["figure.facecolor"] = COLORS["bg"]
plt.rcParams["axes.facecolor"] = COLORS["bg"]
plt.rcParams["savefig.facecolor"] = COLORS["bg"]
plt.rcParams["font.family"] = "DejaVu Sans"

# -------------------------
# Scene geometry
# -------------------------
ROBOT = np.array([16.0, 22.0])
CANDIDATE = np.array([76.0, 62.0])
NBV = np.array([65.0, 40.0])

DIRECT_PATH = np.array([
    [16.0, 22.0],
    [38.0, 40.0],
    [58.0, 49.0],
    [76.0, 62.0],
])

SAFE_PATH = np.array([
    [16.0, 22.0],
    [25.0, 24.0],
    [34.0, 26.0],
    [46.0, 29.0],
    [58.0, 32.0],
    [65.0, 40.0],
    [72.0, 49.0],
    [76.0, 62.0],
])

LOW_VIS_POLY = np.array([
    [56, 43],
    [79, 45],
    [86, 53],
    [84, 68],
    [69, 72],
    [59, 67],
    [51, 56],
])

DECISION_BOX = np.array([
    [56, 35],
    [86, 38],
    [88, 76],
    [58, 74],
])

DEBRIS_CLUSTERS = [
    (20, 74, 1.10, 0),
    (48, 74, 0.95, 1),
    (79, 69, 1.15, 2),
    (37, 44, 0.95, 3),
    (72, 25, 0.95, 4),
]

SCATTERED_BEAMS = [
    ((26, 55), 12, 12),
    ((42, 69), 9, -7),
    ((82, 53), 7, 13),
]

# -------------------------
# Utilities
# -------------------------
def add_outlined_text(ax, x, y, s, color=COLORS["text"], fontsize=14,
                      weight="bold", ha="center", va="center",
                      alpha=1.0, zorder=20):
    txt = ax.text(
        x, y, s, color=color, fontsize=fontsize, fontweight=weight,
        ha=ha, va=va, alpha=alpha, zorder=zorder
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=4, foreground=COLORS["bg"], alpha=0.95)
    ])
    return txt


def panel_title(fig, main_title, subtitle=None):
    fig.text(
        0.5, 0.965, main_title,
        ha="center", va="top",
        color=COLORS["text"], fontsize=24, fontweight="bold"
    )
    if subtitle:
        fig.text(
            0.5, 0.935, subtitle,
            ha="center", va="top",
            color=COLORS["muted"], fontsize=11
        )


def create_ax(figsize=(12, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.84])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    # Main inner panel
    ax.add_patch(Rectangle(
        (3, 8), 94, 84,
        facecolor=COLORS["panel"],
        edgecolor="#42506A",
        linewidth=2.0,
        zorder=0
    ))

    return fig, ax


def draw_grid(ax):
    # subtle tactical grid
    for x in np.arange(5, 96, 5):
        ax.plot([x, x], [10, 90], color=COLORS["grid"], alpha=0.10, lw=0.8, zorder=1)
    for y in np.arange(10, 91, 5):
        ax.plot([5, 95], [y, y], color=COLORS["grid"], alpha=0.10, lw=0.8, zorder=1)

    # slightly stronger major grid
    for x in np.arange(5, 96, 15):
        ax.plot([x, x], [10, 90], color=COLORS["grid_faint"], alpha=0.08, lw=1.4, zorder=1)
    for y in np.arange(10, 91, 15):
        ax.plot([5, 95], [y, y], color=COLORS["grid_faint"], alpha=0.08, lw=1.4, zorder=1)


def draw_soft_glow_circle(ax, center, radii, color, alphas, zorder=3):
    x, y = center
    for r, a in zip(radii, alphas):
        ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none",
                            alpha=a, zorder=zorder))


def draw_irregular_blob(ax, center, scale=1.0, seed=0, zorder=4):
    rng = np.random.default_rng(seed)
    cx, cy = center

    n = rng.integers(7, 11)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radii = rng.uniform(2.8, 5.8, n) * scale
    pts = np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)]

    # shadow
    shadow_pts = pts + np.array([0.5, -0.5])
    ax.add_patch(Polygon(
        shadow_pts, closed=True, facecolor=COLORS["shadow"], edgecolor="none",
        alpha=0.20, zorder=zorder
    ))

    base_color = rng.choice([
        COLORS["debris_dark"], COLORS["debris_mid"], COLORS["debris_light"]
    ])
    ax.add_patch(Polygon(
        pts, closed=True, facecolor=base_color,
        edgecolor="#403A33", linewidth=1.2, alpha=0.95, zorder=zorder+1
    ))

    # internal fractured facets
    for _ in range(rng.integers(3, 6)):
        n2 = rng.integers(4, 7)
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False)
        rr = rng.uniform(0.9, 2.1, n2) * scale
        ox = rng.uniform(-1.8, 1.8) * scale
        oy = rng.uniform(-1.8, 1.8) * scale
        pts2 = np.c_[cx + ox + rr*np.cos(a2), cy + oy + rr*np.sin(a2)]
        ax.add_patch(Polygon(
            pts2, closed=True,
            facecolor=rng.choice([
                COLORS["debris_light"], COLORS["debris_mid"], COLORS["debris_dark"]
            ]),
            edgecolor="none", alpha=0.65, zorder=zorder+2
        ))

    # small surrounding rubble
    for _ in range(rng.integers(8, 14)):
        rx = cx + rng.uniform(-6.0, 6.0) * scale
        ry = cy + rng.uniform(-6.0, 6.0) * scale
        rr = rng.uniform(0.35, 1.1) * scale
        ax.add_patch(Circle(
            (rx, ry), rr,
            facecolor=rng.choice([
                COLORS["debris_dark"], COLORS["debris_mid"], COLORS["debris_light"]
            ]),
            edgecolor="none", alpha=0.90, zorder=zorder+1
        ))

    # steel bars / beams
    for _ in range(rng.integers(2, 5)):
        x0 = cx + rng.uniform(-4.0, 4.0) * scale
        y0 = cy + rng.uniform(-4.0, 4.0) * scale
        length = rng.uniform(2.8, 5.5) * scale
        theta = rng.uniform(0, 2*np.pi)
        x1 = x0 + length*np.cos(theta)
        y1 = y0 + length*np.sin(theta)
        ax.plot([x0, x1], [y0, y1], color="#1F2630", lw=2.3, alpha=0.85, zorder=zorder+3)


def draw_debris(ax):
    for x, y, scale, seed in DEBRIS_CLUSTERS:
        draw_irregular_blob(ax, (x, y), scale=scale, seed=seed, zorder=5)

    # extra broken beams
    for (cx, cy), length, ang in SCATTERED_BEAMS:
        trans = mtransforms.Affine2D().rotate_deg_around(cx, cy, ang) + ax.transData
        beam = Rectangle(
            (cx - length/2, cy - 1.2), length, 2.4,
            facecolor="#9B8F7F", edgecolor="none", alpha=0.90,
            transform=trans, zorder=5
        )
        ax.add_patch(beam)


def draw_robot(ax, center=ROBOT, angle=18, show_fov=False, label=True, zorder=12):
    cx, cy = center
    trans = mtransforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData

    # shadow
    ax.add_patch(Rectangle(
        (cx - 3.4 + 0.8, cy - 2.2 - 0.8), 6.8, 4.4,
        facecolor=COLORS["shadow"], edgecolor="none", alpha=0.25,
        transform=trans, zorder=zorder-2
    ))

    # chassis
    outer = Rectangle(
        (cx - 3.4, cy - 2.2), 6.8, 4.4,
        facecolor=COLORS["panel3"], edgecolor=COLORS["cyan"], linewidth=2.2,
        transform=trans, zorder=zorder
    )
    inner = Rectangle(
        (cx - 1.9, cy - 1.25), 3.8, 2.5,
        facecolor=COLORS["blue_robot"], edgecolor="#A8C8FF", linewidth=1.2,
        transform=trans, zorder=zorder+1
    )
    ax.add_patch(outer)
    ax.add_patch(inner)

    # wheels
    wheel_offsets = [(-3.2, -2.0), (-3.2, 2.0), (3.2, -2.0), (3.2, 2.0)]
    for wx, wy in wheel_offsets:
        ax.add_patch(Circle(
            (cx + wx, cy + wy), 1.15,
            facecolor="#060B14", edgecolor="#0F1724", linewidth=1.0,
            transform=trans, zorder=zorder-1
        ))

    # camera mast + head
    mast_x0, mast_y0 = cx + 1.3, cy
    mast_x1, mast_y1 = cx + 3.8, cy
    ax.plot(
        [mast_x0, mast_x1], [mast_y0, mast_y1],
        color=COLORS["cyan2"], lw=2.0, transform=trans, zorder=zorder+2
    )
    ax.add_patch(Rectangle(
        (cx + 3.7, cy - 0.6), 1.4, 1.2,
        facecolor=COLORS["steel2"], edgecolor="#D8E2EC", linewidth=0.8,
        transform=trans, zorder=zorder+3
    ))

    # navigation point
    ax.add_patch(Circle(
        (cx, cy), 0.45, facecolor=COLORS["green"], edgecolor="none",
        zorder=zorder+4
    ))

    # robot glow
    draw_soft_glow_circle(ax, center, radii=[3.0, 5.2], color=COLORS["cyan"], alphas=[0.08, 0.035], zorder=zorder-3)

    if show_fov:
        theta = np.deg2rad(angle)
        fwd = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-fwd[1], fwd[0]])

        p0 = center + fwd * 2.6
        p1 = center + fwd * 23 + normal * 7.5
        p2 = center + fwd * 23 - normal * 7.5
        tri = np.vstack([p0, p1, p2])

        ax.add_patch(Polygon(
            tri, closed=True, facecolor=COLORS["cyan"], edgecolor=COLORS["cyan"],
            alpha=0.10, linewidth=1.5, zorder=zorder-4
        ))
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=COLORS["cyan"], alpha=0.45, lw=1.5, zorder=zorder-3
        )
        ax.plot(
            [p0[0], p2[0]], [p0[1], p2[1]],
            color=COLORS["cyan"], alpha=0.45, lw=1.5, zorder=zorder-3
        )
        ax.plot(
            [center[0], center[0] + fwd[0]*23],
            [center[1], center[1] + fwd[1]*23],
            color=COLORS["cyan"], alpha=0.25, lw=1.0, zorder=zorder-3
        )

    if label:
        add_outlined_text(ax, cx - 2.0, cy - 6.0, "Rescue robot",
                          color=COLORS["cyan"], fontsize=18, ha="center", va="top", zorder=zorder+5)


def draw_low_visibility_zone(ax, poly=LOW_VIS_POLY, zorder=6, label=True):
    ax.add_patch(Polygon(
        poly, closed=True,
        facecolor=COLORS["amber"], edgecolor=COLORS["amber"],
        alpha=0.16, linewidth=2.0, zorder=zorder
    ))

    # smoke haze layers
    haze_centers = [
        (70, 63), (77, 64), (82, 56), (67, 56), (75, 70)
    ]
    haze_radii = [10.5, 8.0, 7.0, 6.2, 5.5]
    for (cx, cy), r in zip(haze_centers, haze_radii):
        draw_soft_glow_circle(
            ax, (cx, cy),
            radii=[r, r * 0.72],
            color=COLORS["smoke2"],
            alphas=[0.10, 0.08],
            zorder=zorder+1
        )

    if label:
        add_outlined_text(ax, 72.5, 77.0, "Low-visibility zone",
                          color=COLORS["text"], fontsize=18, zorder=zorder+4)


def draw_risk_zone(ax, center=CANDIDATE, zorder=9, label=True):
    cx, cy = center
    draw_soft_glow_circle(
        ax, (cx, cy),
        radii=[15.0, 11.0, 8.0],
        color=COLORS["orange2"],
        alphas=[0.08, 0.10, 0.12],
        zorder=zorder
    )
    ax.add_patch(Circle(
        (cx, cy), 5.8, facecolor=COLORS["orange"], edgecolor=COLORS["white"],
        linewidth=1.8, alpha=0.92, zorder=zorder+1
    ))
    ax.add_patch(Circle(
        (cx, cy), 8.4, fill=False, edgecolor=COLORS["orange"],
        linewidth=2.0, linestyle="--", alpha=0.95, zorder=zorder+2
    ))
    if label:
        add_outlined_text(ax, cx - 8.5, cy - 2.0, "Risk zone",
                          color=COLORS["amber"], fontsize=17, zorder=zorder+3)


def draw_survivor_candidate(ax, center=CANDIDATE, zorder=10, label=True):
    cx, cy = center

    # halo
    draw_soft_glow_circle(
        ax, (cx, cy), radii=[10.5, 6.8], color=COLORS["orange"], alphas=[0.06, 0.05], zorder=zorder-1
    )

    if label:
        add_outlined_text(ax, cx, cy + 9.0, "Survivor candidate",
                          color=COLORS["orange"], fontsize=19, zorder=zorder+3)


def draw_nbv(ax, center=NBV, zorder=11, label=True):
    cx, cy = center
    draw_soft_glow_circle(ax, (cx, cy), radii=[8.0, 5.4], color=COLORS["green"], alphas=[0.10, 0.08], zorder=zorder-1)
    ax.add_patch(Circle(
        (cx, cy), 3.0, facecolor=COLORS["green"], edgecolor=COLORS["white"],
        linewidth=1.6, alpha=0.98, zorder=zorder
    ))
    ax.add_patch(Circle(
        (cx, cy), 5.2, fill=False, edgecolor=COLORS["green2"],
        linewidth=2.2, alpha=0.95, zorder=zorder+1
    ))
    if label:
        add_outlined_text(ax, cx, cy - 6.5, "Next-best view",
                          color=COLORS["green"], fontsize=18, zorder=zorder+2)


def draw_path(ax, pts, color, lw_main=4.0, lw_glow=8.0, alpha=0.98,
              marker=True, zorder=8, dashed=False):
    x = pts[:, 0]
    y = pts[:, 1]
    ax.plot(x, y, color=color, lw=lw_glow, alpha=0.10, zorder=zorder-1, solid_capstyle="round")
    ax.plot(
        x, y, color=color, lw=lw_main, alpha=alpha,
        linestyle="--" if dashed else "-",
        solid_capstyle="round", zorder=zorder
    )
    if marker:
        ax.scatter(x[1:-1], y[1:-1], s=28, color=color, alpha=0.95, zorder=zorder+1)


def draw_scene_background(ax):
    draw_grid(ax)
    draw_debris(ax)


def draw_common_scene(ax, show_robot_fov=False, show_direct_path=True, show_safe_path=True,
                      label_mode="overview"):
    draw_scene_background(ax)
    draw_low_visibility_zone(ax, label=True)
    draw_risk_zone(ax, label=(label_mode != "perception"))
    draw_survivor_candidate(ax, label=True)
    draw_nbv(ax, label=True)

    if show_direct_path:
        draw_path(ax, DIRECT_PATH, COLORS["red"], lw_main=3.2, lw_glow=7.0, marker=False, zorder=7)
    if show_safe_path:
        draw_path(ax, SAFE_PATH, COLORS["cyan2"], lw_main=4.3, lw_glow=8.8, marker=True, zorder=8)

    draw_robot(ax, ROBOT, angle=18, show_fov=show_robot_fov, label=True)


def metric_box(ax, x, y, text, color, w=15, h=4.2):
    ax.add_patch(Rectangle(
        (x, y), w, h,
        facecolor=(0, 0, 0, 0.25),
        edgecolor=color, linewidth=1.5, zorder=30
    ))
    add_outlined_text(ax, x + 0.8, y + h/2, text, color=color, fontsize=12,
                      ha="left", va="center", zorder=31)


def add_footer_text(fig, text):
    fig.text(
        0.5, 0.03, text,
        ha="center", va="center",
        color=COLORS["muted"], fontsize=10
    )


def add_bottom_callout(ax, text, x=32, y=12, w=45, h=7):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=0.4",
        facecolor=COLORS["panel3"], edgecolor="#28405A",
        linewidth=1.4, alpha=0.92, zorder=25
    )
    ax.add_patch(box)
    add_outlined_text(ax, x + w/2, y + h/2, text, color=COLORS["text"], fontsize=10.5, weight="normal", zorder=26)


def add_legend_paths(ax, x=69, y=8.8):
    # mini legend box
    ax.add_patch(Rectangle(
        (x, y), 27.5, 5.5,
        facecolor=(0, 0, 0, 0.28),
        edgecolor="#28405A", linewidth=1.2, zorder=24
    ))
    ax.plot([x + 2, x + 8], [y + 3.8, y + 3.8], color=COLORS["red"], lw=3, zorder=25)
    ax.plot([x + 2, x + 8], [y + 1.7, y + 1.7], color=COLORS["cyan2"], lw=4, zorder=25)
    ax.text(x + 9, y + 3.8, "Direct path", color=COLORS["text"], fontsize=9, va="center", zorder=25)
    ax.text(x + 9, y + 1.7, "Risk-aware re-observation path", color=COLORS["text"], fontsize=9, va="center", zorder=25)


def add_info_card(ax, x, y, w, h, title, lines, title_color=COLORS["text"], edge="#26415D"):
    card = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.28,rounding_size=0.35",
        facecolor=COLORS["panel3"], edgecolor=edge, linewidth=1.5, alpha=0.95, zorder=22
    )
    ax.add_patch(card)
    add_outlined_text(ax, x + w/2, y + h - 1.4, title, color=title_color,
                      fontsize=13, zorder=23)
    yy = y + h - 4.4
    for txt, col in lines:
        ax.text(x + 1.2, yy, txt, color=col, fontsize=11.2, va="center", zorder=23)
        yy -= 2.6


# ============================================================
# Figure 1: Operator Mission Overview
# ============================================================
def figure_operator_overview():
    fig, ax = create_ax(figsize=(12.5, 10.0))
    panel_title(fig, "NeuroMemory Robot — Operator Mission Overview")
    draw_common_scene(ax, show_robot_fov=True, show_direct_path=True, show_safe_path=True)

    metric_box(ax, 6.0, 84.0, "identity score: 0.72", COLORS["cyan"])
    metric_box(ax, 6.0, 79.0, "uncertainty: 0.42", COLORS["amber"])
    metric_box(ax, 6.0, 74.0, "planner gain: 0.31", COLORS["green"])

    add_outlined_text(ax, 33, 48, "camera field-of-view", color=COLORS["cyan"], fontsize=12, zorder=26)

    add_bottom_callout(
        ax,
        "NeuroMemory decision layer: uncertain identity → active re-observation → safer next-best view",
        x=47, y=12.0, w=31, h=6.3
    )
    add_legend_paths(ax, x=70, y=8.5)

    add_footer_text(
        fig,
        "Operator-style overview showing debris, smoke, uncertain survivor identity, next-best-view selection, and risk-aware active re-observation planning."
    )

    out = OUT_DIR / "01_operator_mission_overview.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Figure 2: Perception / Active Re-Observation View
# ============================================================
def figure_perception_view():
    fig, ax = create_ax(figsize=(12.5, 10.0))
    panel_title(fig, "Robot Perception and Active Re-Observation View")
    draw_scene_background(ax)
    draw_low_visibility_zone(ax, label=True)
    draw_risk_zone(ax, label=False)
    draw_survivor_candidate(ax, label=False)
    draw_nbv(ax, label=False)
    draw_path(ax, SAFE_PATH, COLORS["cyan2"], lw_main=4.4, lw_glow=8.2, marker=True, zorder=9)
    draw_robot(ax, ROBOT, angle=18, show_fov=True, label=True)

    add_outlined_text(ax, 39, 50, "camera field-of-view", color=COLORS["cyan"], fontsize=13)
    add_outlined_text(ax, 43, 46, "perception cone", color=COLORS["cyan"], fontsize=18)

    add_outlined_text(ax, 76, 71, "uncertain identity region", color=COLORS["orange"], fontsize=18)
    add_outlined_text(ax, 67, 39, "candidate viewpoint for re-observation", color=COLORS["green"], fontsize=16, va="top")

    # remove clutter by not drawing some labels and adding focused ones
    add_footer_text(
        fig,
        "Robot-centered view emphasizing perception direction, camera field-of-view, and active re-observation target selection."
    )

    out = OUT_DIR / "02_robot_perception_view.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Figure 3: Next-Best-View Decision Focus
# ============================================================
def figure_decision_view():
    fig, ax = create_ax(figsize=(12.5, 10.0))
    panel_title(fig, "Next-Best-View Decision and Survivor Candidate Focus")
    draw_scene_background(ax)
    draw_low_visibility_zone(ax, label=True)
    draw_risk_zone(ax, label=True)
    draw_survivor_candidate(ax, label=True)
    draw_nbv(ax, label=True)

    draw_path(ax, DIRECT_PATH, COLORS["red"], lw_main=3.0, lw_glow=7.0, marker=False, zorder=7)
    draw_path(ax, SAFE_PATH, COLORS["cyan2"], lw_main=4.2, lw_glow=8.5, marker=True, zorder=8)
    draw_robot(ax, ROBOT, angle=18, show_fov=False, label=False)

    # decision region box
    ax.add_patch(Polygon(
        DECISION_BOX, closed=True, fill=False,
        edgecolor=COLORS["green2"], linewidth=2.2,
        linestyle="--", alpha=0.90, zorder=18
    ))

    add_info_card(
        ax, x=6.2, y=74.0, w=26.0, h=12.0,
        title="Decision logic",
        lines=[
            ("Low identity certainty", COLORS["amber"]),
            ("Active re-observation recommended", COLORS["green2"]),
        ],
        edge="#27496A"
    )

    add_outlined_text(ax, 72, 48, "expected confidence gain", color=COLORS["green"], fontsize=16)
    add_footer_text(
        fig,
        "Decision-focused view showing how uncertain identity evidence leads to a next-best-view selection for re-observation."
    )

    out = OUT_DIR / "03_next_best_view_decision.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Figure 4: Path comparison
# ============================================================
def base_small_scene(ax, show_direct=True, show_safe=False, label_text=None):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((2, 8), 96, 84, facecolor=COLORS["panel"], edgecolor="#42506A", linewidth=1.8, zorder=0))
    draw_grid(ax)
    draw_debris(ax)
    draw_low_visibility_zone(ax, label=False)
    draw_risk_zone(ax, label=False)
    draw_survivor_candidate(ax, label=False)
    draw_nbv(ax, label=False)
    if show_direct:
        draw_path(ax, DIRECT_PATH, COLORS["red"], lw_main=4.0, lw_glow=8.0, marker=False, zorder=8)
    if show_safe:
        draw_path(ax, SAFE_PATH, COLORS["cyan2"], lw_main=4.6, lw_glow=8.8, marker=True, zorder=9)
    draw_robot(ax, ROBOT, angle=18, show_fov=False, label=False)

    if label_text:
        add_outlined_text(ax, 57, 50, label_text[0], color=label_text[1], fontsize=17)


def figure_path_comparison():
    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.text(
        0.5, 0.965,
        "Path Planning Comparison: Direct Navigation vs Risk-Aware Active Re-Observation",
        ha="center", va="top", color=COLORS["text"], fontsize=24, fontweight="bold"
    )

    ax1 = fig.add_axes([0.04, 0.14, 0.44, 0.72])
    ax2 = fig.add_axes([0.52, 0.14, 0.44, 0.72])

    base_small_scene(ax1, show_direct=True, show_safe=False,
                     label_text=("passes through\nrisk/smoke area", COLORS["red"]))
    base_small_scene(ax2, show_direct=False, show_safe=True,
                     label_text=("reroutes through\nnext-best-view", COLORS["green"]))

    fig.text(0.26, 0.83, "Baseline Direct Path", ha="center",
             color=COLORS["text"], fontsize=18, fontweight="bold")
    fig.text(0.74, 0.83, "NeuroMemory Risk-Aware Re-Observation Path", ha="center",
             color=COLORS["text"], fontsize=18, fontweight="bold")

    fig.text(
        0.5, 0.045,
        "Comparison view for poster use: the proposed NeuroMemory logic selects a safer re-observation route instead of moving directly through the uncertain risk region.",
        ha="center", color=COLORS["muted"], fontsize=11
    )

    out = OUT_DIR / "04_path_planning_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Figure 5: Summary collage
# ============================================================
def figure_collage(image_paths):
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.text(
        0.03, 0.97, "NeuroMemory Robot — Poster-Ready Mission Visualization Set",
        ha="left", va="top", color=COLORS["text"], fontsize=22
    )

    layout = [
        [0.04, 0.53, 0.42, 0.35, "Operator overview", image_paths[0]],
        [0.52, 0.53, 0.42, 0.35, "Robot perception", image_paths[1]],
        [0.04, 0.10, 0.42, 0.35, "Next-best-view decision", image_paths[2]],
        [0.52, 0.10, 0.42, 0.35, "Before/after path comparison", image_paths[3]],
    ]

    for left, bottom, width, height, label, img_path in layout:
        ax = fig.add_axes([left, bottom, width, height])
        ax.axis("off")
        img = mpimg.imread(img_path)
        ax.imshow(img)
        fig.text(left, bottom - 0.02, label, color=COLORS["text"], fontsize=16, ha="left")

    out = OUT_DIR / "05_visual_summary_collage.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("Generating premium NeuroMemory poster-ready figures...")
    p1 = figure_operator_overview()
    print(f"Saved: {p1}")

    p2 = figure_perception_view()
    print(f"Saved: {p2}")

    p3 = figure_decision_view()
    print(f"Saved: {p3}")

    p4 = figure_path_comparison()
    print(f"Saved: {p4}")

    p5 = figure_collage([p1, p2, p3, p4])
    print(f"Saved: {p5}")

    print("\nAll premium figures generated successfully.")
    print(f"Output directory: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()