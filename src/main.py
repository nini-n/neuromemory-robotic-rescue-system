import pygame
import sys
import math
import os
import heapq
from datetime import datetime

# ============================================================
# NeuroMemory Robot v6
# Engineering-style simulation-based proof-of-concept
# ============================================================

pygame.init()

WIDTH, HEIGHT = 1200, 720
MAP_WIDTH = 850
PANEL_WIDTH = WIDTH - MAP_WIDTH

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NeuroMemory Robot - Active Visual Memory Simulation")

clock = pygame.time.Clock()
FPS = 60
LOOP_DURATION = 32

# -----------------------------
# Experiment metadata
# -----------------------------
RUN_ID = "NM-RUN-01"
SCENARIO_ID = "SC-05"
SCENARIO_NAME = "Smoke + occlusion + angle shift"
SEED = 42
GRID_SCALE_TEXT = "1 grid = 0.5 m"
SIM_MODE = "Simulation-based proof-of-concept"

# -----------------------------
# Fonts
# -----------------------------
FONT = pygame.font.SysFont("Arial", 22)
SMALL_FONT = pygame.font.SysFont("Arial", 17)
TINY_FONT = pygame.font.SysFont("Arial", 14)
TITLE_FONT = pygame.font.SysFont("Arial", 26, bold=True)

# -----------------------------
# Colors
# -----------------------------
BG = (25, 28, 34)
MAP_BG = (38, 42, 50)
PANEL_BG = (18, 20, 26)
GRID = (55, 60, 70)
WHITE = (235, 235, 235)
GREY = (155, 155, 160)
DARK_GREY = (75, 80, 90)
BLUE = (80, 150, 255)
GREEN = (80, 220, 130)
YELLOW = (245, 205, 70)
RED = (235, 90, 80)
ORANGE = (245, 140, 60)
PURPLE = (160, 110, 255)

# -----------------------------
# World setup
# -----------------------------
GRID_SIZE = 25
COLS = MAP_WIDTH // GRID_SIZE
ROWS = HEIGHT // GRID_SIZE

robot_radius = 16

start_pos = (90, 590)
first_observation_pos = (330, 430)
changed_observation_pos = (650, 250)

obstacles = [
    pygame.Rect(180, 130, 120, 80),
    pygame.Rect(420, 520, 160, 70),
    pygame.Rect(600, 120, 120, 95),
    pygame.Rect(260, 260, 80, 140),
]

smoke_area = pygame.Rect(540, 190, 230, 190)
priority_rect = pygame.Rect(570, 210, 180, 150)

candidate_views = [
    (585, 330),
    (590, 310),
    (720, 315),
    (520, 240),
    (690, 170),
]

# -----------------------------
# Algorithmic visual memory
# -----------------------------
# In a real robotic system, these vectors would be produced by visual
# feature extraction, CNN/SNN embeddings, or event-camera based encoding.
# In this proof-of-concept, they represent compact simulated embeddings.

memory_vector = [1.00, 0.00, 0.00, 0.00, 0.00]

# Degraded observation: smoke + partial occlusion + angle shift
# cosine similarity ≈ 0.64
uncertain_observation_vector = [0.64, 0.768, 0.00, 0.00, 0.00]

# Clearer observation after active next-best-view movement
# cosine similarity ≈ 0.86
clearer_observation_vector = [0.86, 0.510, 0.00, 0.00, 0.00]


# ============================================================
# Phase definitions
# ============================================================

def get_phase(t):
    if t < 3:
        return 1
    elif t < 6:
        return 2
    elif t < 9:
        return 3
    elif t < 12:
        return 4
    elif t < 16:
        return 5
    elif t < 23:
        return 6
    elif t < 27:
        return 7
    else:
        return 8


PHASE_TITLES = {
    1: "Search initialization",
    2: "Initial detection",
    3: "Memory update",
    4: "Degraded observation",
    5: "Similarity evaluation",
    6: "Risk-aware NBV motion",
    7: "Re-observation update",
    8: "Priority map update",
}

EVENT_DESCRIPTIONS = {
    1: [
        "Robot starts scanning the simulated disaster area.",
        "No survivor has been confirmed in the current frame.",
        "The system waits for a valid first observation.",
    ],
    2: [
        "Person A is detected for the first time.",
        "The first visual observation is captured as a reference.",
    ],
    3: [
        "A compact visual-memory vector is stored.",
        "The last-seen position is inserted into the memory map.",
    ],
    4: [
        "A visually similar candidate appears under degraded conditions.",
        "The observation is affected by smoke, occlusion, and angle shift.",
    ],
    5: [
        "Cosine similarity is calculated between memory and observation.",
        "The score is below the confident-match threshold.",
        "The system requests a clearer re-observation.",
    ],
    6: [
        "Candidate viewpoints are scored using expected confidence gain.",
        "A risk-aware A* path is generated toward the selected viewpoint.",
    ],
    7: [
        "The clearer viewpoint improves the similarity score.",
        "The match is probable, but human verification is still advised.",
    ],
    8: [
        "The search-priority score is updated from computed values.",
        "Priority depends on confidence, last-seen relevance, visibility, and cost.",
    ],
}

EVENT_LOGS = {
    1: [
        "[00.0s] Run initialized",
        "[00.8s] Robot scanning area",
    ],
    2: [
        "[03.1s] Person A detected",
        "[04.2s] Reference observation captured",
    ],
    3: [
        "[06.0s] Visual-memory vector stored",
        "[07.1s] Last-seen map updated",
    ],
    4: [
        "[10.0s] Low-visibility region active",
        "[11.2s] Candidate observation appears",
    ],
    5: [
        "[13.0s] Cosine similarity computed",
        "[14.1s] Confidence uncertain; re-check required",
    ],
    6: [
        "[16.0s] Next-best-view selected",
        "[17.0s] Risk-aware A* path generated",
        "[20.0s] Robot moving to re-observe",
    ],
    7: [
        "[23.0s] New observation collected",
        "[24.1s] Similarity score improved",
    ],
    8: [
        "[27.0s] Priority score updated",
        "[28.0s] Human verification advised",
    ],
}


# ============================================================
# Math / algorithms
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_obstacle(p):
    x, y = p
    for rect in obstacles:
        padded = rect.inflate(20, 20)
        if padded.collidepoint(x, y):
            return True
    return False


def world_to_grid(p):
    return int(p[0] // GRID_SIZE), int(p[1] // GRID_SIZE)


def grid_to_world(cell):
    return cell[0] * GRID_SIZE + GRID_SIZE // 2, cell[1] * GRID_SIZE + GRID_SIZE // 2


def is_blocked(cell):
    x, y = grid_to_world(cell)
    if x < 0 or x >= MAP_WIDTH or y < 0 or y >= HEIGHT:
        return True
    return point_in_obstacle((x, y))


def traversal_cost(cell):
    """
    Risk-aware movement cost.
    Normal cell cost = 1.0.
    Smoke / low-visibility regions add cost.
    Areas near debris add smaller local cost.
    """
    x, y = grid_to_world(cell)
    cost = 1.0

    if smoke_area.collidepoint(x, y):
        cost += 3.0

    for rect in obstacles:
        cx = clamp(x, rect.left, rect.right)
        cy = clamp(y, rect.top, rect.bottom)
        d = distance((x, y), (cx, cy))
        if d < 45:
            cost += (45 - d) / 15.0

    return cost


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal):
    start_cell = world_to_grid(start)
    goal_cell = world_to_grid(goal)

    open_set = []
    heapq.heappush(open_set, (0, start_cell))

    came_from = {}
    g_score = {start_cell: 0}

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_cell:
            path = []
            while current in came_from:
                path.append(grid_to_world(current))
                current = came_from[current]
            path.append(grid_to_world(start_cell))
            path.reverse()
            return path

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)

            if nxt[0] < 0 or nxt[0] >= COLS or nxt[1] < 0 or nxt[1] >= ROWS:
                continue

            if is_blocked(nxt):
                continue

            tentative = g_score[current] + traversal_cost(nxt)

            if nxt not in g_score or tentative < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative
                f = tentative + heuristic(nxt, goal_cell)
                heapq.heappush(open_set, (f, nxt))

    return [start, goal]


def visibility_score(viewpoint, target):
    """
    Expected observation quality from a candidate viewpoint.
    Higher score means the candidate should provide a clearer view.
    """
    base = 1.0

    if smoke_area.collidepoint(viewpoint):
        base -= 0.35

    if point_in_obstacle(viewpoint):
        base -= 0.60

    d = distance(viewpoint, target)
    dist_term = clamp(1.0 - d / 350.0, 0.0, 1.0)

    if d < 45:
        dist_term -= 0.15

    return clamp(0.60 * base + 0.40 * dist_term, 0.0, 1.0)


def obstacle_penalty(viewpoint):
    if point_in_obstacle(viewpoint):
        return 1.0

    penalty = 0.0
    for rect in obstacles:
        cx = clamp(viewpoint[0], rect.left, rect.right)
        cy = clamp(viewpoint[1], rect.top, rect.bottom)
        d = distance(viewpoint, (cx, cy))
        if d < 50:
            penalty += (50 - d) / 50

    if smoke_area.collidepoint(viewpoint):
        penalty += 0.25

    return clamp(penalty, 0.0, 1.0)


def select_next_best_view(current_pos, target_pos):
    """
    Candidate viewpoints are scored using:
    - expected confidence gain
    - visibility score
    - distance cost
    - obstacle / risk penalty
    """
    scored = []

    uncertain = cosine_similarity(memory_vector, uncertain_observation_vector)
    clearer = cosine_similarity(memory_vector, clearer_observation_vector)

    for vp in candidate_views:
        vis = visibility_score(vp, target_pos)
        dist_cost = clamp(distance(current_pos, vp) / 500.0, 0.0, 1.0)
        obs = obstacle_penalty(vp)

        expected_confidence = uncertain + (clearer - uncertain) * vis
        expected_gain = expected_confidence - uncertain

        score = (
            0.45 * expected_gain
            + 0.35 * vis
            - 0.12 * dist_cost
            - 0.08 * obs
        )

        scored.append({
            "point": vp,
            "visibility": vis,
            "distance_cost": dist_cost,
            "obstacle_penalty": obs,
            "expected_confidence": expected_confidence,
            "expected_gain": expected_gain,
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[0], scored


def priority_score(match_confidence, last_seen_pos, current_pos, zone_center):
    """
    Search-priority score used to select the next region of interest.
    """
    last_seen_relevance = clamp(1.0 - distance(last_seen_pos, zone_center) / 500.0, 0.0, 1.0)
    distance_cost = clamp(distance(current_pos, zone_center) / 650.0, 0.0, 1.0)
    risk_penalty = 0.20 if smoke_area.collidepoint(zone_center) else 0.05
    visibility = visibility_score(current_pos, zone_center)

    score = (
        0.55 * match_confidence
        + 0.25 * last_seen_relevance
        + 0.20 * visibility
        - 0.07 * distance_cost
        - 0.03 * risk_penalty
    )

    return clamp(score, 0.0, 1.0), {
        "match": match_confidence,
        "last_seen": last_seen_relevance,
        "visibility": visibility,
        "distance_cost": distance_cost,
        "risk_penalty": risk_penalty,
    }


# Precomputed algorithm outputs
uncertain_confidence = cosine_similarity(memory_vector, uncertain_observation_vector)
clear_confidence = cosine_similarity(memory_vector, clearer_observation_vector)

start_for_nbv = (560, 310)
best_view, all_view_scores = select_next_best_view(start_for_nbv, changed_observation_pos)
path_to_best_view = astar(start_for_nbv, best_view["point"])

zone_center = priority_rect.center
final_priority, priority_components = priority_score(
    clear_confidence,
    first_observation_pos,
    best_view["point"],
    zone_center,
)


# ============================================================
# Drawing helpers
# ============================================================

def draw_text(surface, text, x, y, font=SMALL_FONT, color=WHITE):
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


def draw_lines(surface, lines, x, y, font=SMALL_FONT, color=WHITE, line_gap=23):
    for i, line in enumerate(lines):
        draw_text(surface, line, x, y + i * line_gap, font, color)


def draw_grid():
    for x in range(0, MAP_WIDTH, 50):
        pygame.draw.line(screen, GRID, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, GRID, (0, y), (MAP_WIDTH, y), 1)


def draw_dashed_line(surface, start, end, color, width=2, dash_length=12):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    if dist == 0:
        return

    dash_count = max(1, int(dist // dash_length))
    for i in range(dash_count):
        if i % 2 == 0:
            s = (x1 + dx * i / dash_count, y1 + dy * i / dash_count)
            e = (x1 + dx * (i + 1) / dash_count, y1 + dy * (i + 1) / dash_count)
            pygame.draw.line(surface, color, s, e, width)


def save_screenshot():
    os.makedirs("outputs/frames", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("outputs", "frames", f"neuromemory_frame_{timestamp}.png")
    pygame.image.save(screen, path)
    print(f"Saved screenshot: {path}")


def interpolate_path(path, progress):
    if not path:
        return start_for_nbv

    if len(path) == 1:
        return path[0]

    progress = clamp(progress, 0.0, 1.0)
    total_segments = len(path) - 1
    pos = progress * total_segments
    idx = int(math.floor(pos))
    idx = min(idx, total_segments - 1)
    local = pos - idx

    a = path[idx]
    b = path[idx + 1]

    return (
        a[0] + (b[0] - a[0]) * local,
        a[1] + (b[1] - a[1]) * local,
    )


# ============================================================
# Drawing scene elements
# ============================================================

def draw_scenario_banner(current_time):
    banner = pygame.Rect(18, 18, MAP_WIDTH - 36, 62)
    pygame.draw.rect(screen, (20, 23, 30), banner, border_radius=12)
    pygame.draw.rect(screen, (70, 75, 90), banner, 2, border_radius=12)

    draw_text(
        screen,
        f"Scenario {SCENARIO_ID}: Survivor re-identification under degraded visibility",
        banner.x + 18,
        banner.y + 9,
        FONT,
        WHITE,
    )

    subtitle = (
        f"{RUN_ID} | sim time: {current_time:04.1f}s | seed: {SEED} | "
        "memory → similarity → uncertainty → risk-aware A* → priority update"
    )
    draw_text(screen, subtitle, banner.x + 18, banner.y + 36, TINY_FONT, GREY)


def draw_metadata_box(current_time):
    box = pygame.Rect(585, 92, 245, 78)
    pygame.draw.rect(screen, (20, 23, 30), box, border_radius=10)
    pygame.draw.rect(screen, (70, 75, 90), box, 1, border_radius=10)

    lines = [
        f"mode: {SIM_MODE}",
        f"planner: risk-aware A*",
        f"similarity: cosine | NBV candidates: {len(candidate_views)}",
    ]

    draw_lines(screen, lines, box.x + 12, box.y + 12, TINY_FONT, GREY, line_gap=19)


def draw_environment(phase, current_time):
    pygame.draw.rect(screen, MAP_BG, (0, 0, MAP_WIDTH, HEIGHT))
    draw_grid()

    for rect in obstacles:
        pygame.draw.rect(screen, (85, 80, 75), rect, border_radius=4)
        pygame.draw.rect(screen, (128, 122, 115), rect, 2, border_radius=4)
        draw_text(screen, "debris", rect.x + 10, rect.y + 8, TINY_FONT, (170, 165, 155))

    if phase >= 4:
        smoke_surface = pygame.Surface((smoke_area.width, smoke_area.height), pygame.SRCALPHA)
        smoke_surface.fill((180, 180, 180, 66))
        screen.blit(smoke_surface, (smoke_area.x, smoke_area.y))
        pygame.draw.rect(screen, GREY, smoke_area, 2)
        draw_text(screen, "low-visibility region", smoke_area.x + 18, smoke_area.y + 15, SMALL_FONT, WHITE)

    if phase >= 8:
        priority_surface = pygame.Surface((priority_rect.width, priority_rect.height), pygame.SRCALPHA)
        priority_surface.fill((80, 220, 130, 45))
        screen.blit(priority_surface, (priority_rect.x, priority_rect.y))
        pygame.draw.rect(screen, GREEN, priority_rect, 3)
        draw_text(
            screen,
            f"priority region: {final_priority * 100:.1f}%",
            priority_rect.x - 5,
            priority_rect.y - 28,
            SMALL_FONT,
            GREEN,
        )

    draw_scenario_banner(current_time)
    draw_metadata_box(current_time)


def draw_robot(pos):
    x, y = int(pos[0]), int(pos[1])

    pygame.draw.circle(screen, (35, 70, 130), (x, y), robot_radius + 7)
    pygame.draw.circle(screen, BLUE, (x, y), robot_radius)
    pygame.draw.circle(screen, WHITE, (x, y), robot_radius, 2)

    pygame.draw.line(screen, WHITE, (x, y), (x + 24, y - 12), 3)
    pygame.draw.circle(screen, WHITE, (x + 28, y - 14), 4)

    draw_text(screen, "robot", x - 20, y + 22, SMALL_FONT, WHITE)


def draw_person(pos, label, changed=False):
    x, y = pos
    color = ORANGE if not changed else YELLOW

    pygame.draw.circle(screen, color, (x, y - 13), 9)
    pygame.draw.rect(screen, color, pygame.Rect(x - 8, y - 5, 16, 28), border_radius=5)

    if changed:
        pygame.draw.circle(screen, RED, (x + 10, y - 10), 5)
        pygame.draw.circle(screen, RED, (x + 14, y + 2), 4)

    draw_text(screen, label, x - 35, y + 28, SMALL_FONT, WHITE)


def draw_memory_marker(pos):
    x, y = pos
    pygame.draw.circle(screen, GREEN, (x, y), 28, 3)
    pygame.draw.line(screen, GREEN, (x - 20, y - 20), (x + 20, y + 20), 2)
    pygame.draw.line(screen, GREEN, (x + 20, y - 20), (x - 20, y + 20), 2)
    draw_text(screen, "last-seen", x - 36, y - 50, SMALL_FONT, GREEN)


def draw_candidate_views(phase):
    if phase < 6:
        return

    for item in all_view_scores:
        p = item["point"]
        score = item["score"]
        color = PURPLE if p == best_view["point"] else DARK_GREY

        pygame.draw.circle(screen, color, p, 14, 2)
        draw_text(screen, f"{score:.2f}", p[0] - 14, p[1] + 18, TINY_FONT, color)

    bx, by = best_view["point"]
    pygame.draw.circle(screen, PURPLE, (bx, by), 23, 3)
    pygame.draw.circle(screen, PURPLE, (bx, by), 5)
    draw_text(screen, "selected NBV", bx - 45, by - 42, SMALL_FONT, PURPLE)


def draw_astar_path(phase):
    if phase < 6:
        return

    if len(path_to_best_view) > 1:
        for i in range(len(path_to_best_view) - 1):
            pygame.draw.line(screen, PURPLE, path_to_best_view[i], path_to_best_view[i + 1], 3)

    draw_dashed_line(screen, start_for_nbv, best_view["point"], PURPLE, 1, 12)


def draw_connection_lines(phase):
    if phase >= 5:
        draw_dashed_line(screen, first_observation_pos, changed_observation_pos, GREEN, 2, 16)


def draw_context_labels(phase):
    if phase >= 4:
        label_box = pygame.Rect(575, 405, 245, 52)
        pygame.draw.rect(screen, (20, 23, 30), label_box, border_radius=8)
        pygame.draw.rect(screen, (95, 90, 70), label_box, 1, border_radius=8)
        draw_text(screen, "degraded observation", label_box.x + 12, label_box.y + 8, TINY_FONT, YELLOW)
        draw_text(screen, "smoke + occlusion + viewpoint shift", label_box.x + 12, label_box.y + 28, TINY_FONT, WHITE)

    if phase >= 8:
        label_box = pygame.Rect(455, 465, 340, 70)
        pygame.draw.rect(screen, (20, 23, 30), label_box, border_radius=8)
        pygame.draw.rect(screen, (70, 110, 85), label_box, 1, border_radius=8)
        draw_text(screen, "computed priority score", label_box.x + 12, label_box.y + 7, TINY_FONT, GREEN)
        draw_text(screen, f"score = {final_priority:.3f}", label_box.x + 12, label_box.y + 27, TINY_FONT, WHITE)
        draw_text(screen, "confidence + last-seen + visibility - costs", label_box.x + 12, label_box.y + 47, TINY_FONT, WHITE)


def draw_map_legend():
    legend_x, legend_y = 24, HEIGHT - 112
    box = pygame.Rect(legend_x, legend_y, 270, 88)
    pygame.draw.rect(screen, (20, 23, 30), box, border_radius=10)
    pygame.draw.rect(screen, (70, 75, 90), box, 1, border_radius=10)

    draw_text(screen, "map legend", legend_x + 12, legend_y + 8, TINY_FONT, GREY)

    items = [
        (GREEN, "stored last-seen position"),
        (PURPLE, "risk-aware A* / NBV action"),
        (YELLOW, "degraded candidate observation"),
    ]

    for i, (color, label) in enumerate(items):
        y = legend_y + 31 + i * 18
        pygame.draw.circle(screen, color, (legend_x + 17, y + 5), 5)
        draw_text(screen, label, legend_x + 30, y, TINY_FONT, WHITE)


# ============================================================
# Dashboard
# ============================================================

def draw_panel_box(x, y, w, h, title=None):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (24, 27, 35), rect, border_radius=10)
    pygame.draw.rect(screen, (65, 70, 82), rect, 2, border_radius=10)
    if title:
        draw_text(screen, title, x + 14, y + 10, SMALL_FONT, GREY)
    return rect


def current_confidence_for_phase(phase, t):
    if phase < 5:
        return 0.0

    if phase == 5:
        return uncertain_confidence

    if phase == 6:
        u = clamp((t - 16) / 7, 0.0, 1.0)
        return uncertain_confidence + (clear_confidence - uncertain_confidence) * u

    if phase >= 7:
        return clear_confidence

    return 0.0


def draw_confidence_bar(x, y, confidence):
    bar_w, bar_h = 280, 22
    pygame.draw.rect(screen, (65, 70, 82), (x, y, bar_w, bar_h), border_radius=8)

    fill_w = int(bar_w * confidence)
    bar_color = RED if confidence < 0.55 else YELLOW if confidence < 0.80 else GREEN
    pygame.draw.rect(screen, bar_color, (x, y, fill_w, bar_h), border_radius=8)


def draw_algorithm_values(x, y, confidence, phase):
    box = draw_panel_box(x, y, PANEL_WIDTH - 50, 112, title="Algorithm values")

    if phase < 5:
        lines = [
            "similarity: waiting for new observation",
            "NBV: not selected",
            "priority: not updated",
        ]
    elif phase < 7:
        lines = [
            f"cosine similarity: {confidence:.3f}",
            f"expected confidence gain: {best_view['expected_gain']:.3f}",
            f"selected NBV score: {best_view['score']:.3f}",
            f"risk-aware A* nodes: {len(path_to_best_view)}",
        ]
    else:
        lines = [
            f"cosine similarity: {clear_confidence:.3f}",
            f"expected confidence gain: {best_view['expected_gain']:.3f}",
            f"risk-aware A* nodes: {len(path_to_best_view)}",
            f"priority score: {final_priority:.3f}",
        ]

    draw_lines(screen, lines, box.x + 14, box.y + 38, TINY_FONT, WHITE, line_gap=17)


def draw_event_log(x, y, phase):
    box = draw_panel_box(x, y, PANEL_WIDTH - 50, 86, title="Event log")

    logs = EVENT_LOGS.get(phase, [])
    for i, log in enumerate(logs[:4]):
        draw_text(screen, log, box.x + 14, box.y + 34 + i * 15, TINY_FONT, WHITE)


def draw_panel(phase, confidence):
    pygame.draw.rect(screen, PANEL_BG, (MAP_WIDTH, 0, PANEL_WIDTH, HEIGHT))

    x = MAP_WIDTH + 25

    draw_text(screen, "NeuroMemory Robot", x, 22, TITLE_FONT, WHITE)
    draw_text(screen, "Rescue Operator Dashboard", x, 56, SMALL_FONT, GREY)

    pygame.draw.line(screen, (70, 75, 85), (MAP_WIDTH + 20, 85), (WIDTH - 20, 85), 2)

    draw_text(screen, "Phase:", x, 105, FONT, GREY)
    draw_text(screen, PHASE_TITLES[phase], x, 132, SMALL_FONT, YELLOW)

    event_box = draw_panel_box(x, 165, PANEL_WIDTH - 50, 123, title="Event description")
    draw_lines(screen, EVENT_DESCRIPTIONS[phase], event_box.x + 14, event_box.y + 36, TINY_FONT, WHITE, line_gap=18)

    draw_algorithm_values(x, 304, confidence, phase)

    draw_text(screen, "Re-identification status:", x, 435, SMALL_FONT, GREY)

    if phase < 2:
        status = "no current candidate"
        status_color = GREY
    elif phase < 4:
        status = "reference memory stored"
        status_color = GREEN
    elif phase == 5:
        status = "uncertain candidate"
        status_color = YELLOW
    elif phase == 6:
        status = "re-observing via risk-aware A*"
        status_color = PURPLE
    elif phase >= 7:
        status = "probable match"
        status_color = GREEN
    else:
        status = "processing"
        status_color = WHITE

    draw_text(screen, status, x, 460, SMALL_FONT, status_color)

    draw_text(screen, f"Confidence: {confidence * 100:.1f}%", x, 492, FONT, WHITE)
    draw_confidence_bar(x, 525, confidence)

    if phase >= 7:
        recommendation = "human verification advised"
    elif phase == 6:
        recommendation = "collecting clearer observation"
    elif phase == 5:
        recommendation = "uncertain: trigger re-observation"
    elif phase >= 3:
        recommendation = "continue search with stored memory"
    else:
        recommendation = "scanning"

    draw_text(screen, "Decision support:", x, 565, SMALL_FONT, GREY)
    draw_text(screen, recommendation, x, 590, TINY_FONT, WHITE)

    draw_event_log(x, 618, phase)


# ============================================================
# Robot motion
# ============================================================

def lerp(a, b, u):
    return a + (b - a) * u


def move_towards(start, end, u):
    return (lerp(start[0], end[0], u), lerp(start[1], end[1], u))


def compute_robot_position(t, phase):
    if phase <= 2:
        u = clamp(t / 6, 0, 1)
        return move_towards(start_pos, (290, 455), u)

    elif phase <= 5:
        u = clamp((t - 6) / 6, 0, 1)
        return move_towards((290, 455), start_for_nbv, u)

    elif phase == 6:
        u = clamp((t - 16) / 7, 0, 1)
        return interpolate_path(path_to_best_view, u)

    else:
        return best_view["point"]


# ============================================================
# Main drawing
# ============================================================

def draw_scene(phase, confidence, robot_pos, current_time):
    screen.fill(BG)

    draw_environment(phase, current_time)
    draw_connection_lines(phase)

    if 2 <= phase <= 3:
        draw_person(first_observation_pos, "Person A", changed=False)

    if phase >= 3:
        draw_memory_marker(first_observation_pos)

    if phase >= 4:
        draw_person(changed_observation_pos, "Person A?", changed=True)

    if phase >= 6:
        draw_astar_path(phase)
        draw_candidate_views(phase)

    draw_context_labels(phase)
    draw_map_legend()

    draw_robot(robot_pos)

    draw_panel(phase, confidence)


# ============================================================
# Main loop
# ============================================================

start_ticks = pygame.time.get_ticks()

running = True
while running:
    clock.tick(FPS)

    elapsed = (pygame.time.get_ticks() - start_ticks) / 1000
    t = elapsed % LOOP_DURATION

    phase = get_phase(t)
    confidence = current_confidence_for_phase(phase, t)
    robot_pos = compute_robot_position(t, phase)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                save_screenshot()
            elif event.key == pygame.K_r:
                start_ticks = pygame.time.get_ticks()

    draw_scene(phase, confidence, robot_pos, t)

    pygame.display.flip()

pygame.quit()
sys.exit()