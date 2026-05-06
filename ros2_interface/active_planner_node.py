"""
NeuroMemory Robot - ROS 2 Active Planner Node

Purpose:
    This node represents the active planning layer of the NeuroMemory system.

Role in the robotic pipeline:
    /neuromemory/identity_score
    /neuromemory/uncertainty
    /neuromemory/last_seen
        -> active_planner_node
        -> /neuromemory/next_best_view
        -> /neuromemory/risk_aware_path
        -> /neuromemory/planner_status

Important:
    This is a ROS 2-ready integration interface.
    The current project is simulation-based. This file shows how the
    uncertainty-aware next-best-view and risk-aware planning logic can be
    transferred to a mobile robot or drone platform.

Notes:
    - The node does not perform final identity decisions.
    - It decides whether re-observation is recommended.
    - It publishes next-best-view and risk-aware path information for a robot controller.
"""

import math
import heapq
from dataclasses import dataclass
from typing import Tuple, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String
    from geometry_msgs.msg import Point, PoseArray, Pose

    ROS2_AVAILABLE = True

except ImportError:
    ROS2_AVAILABLE = False

    class Node:
        pass

    class Float32:
        def __init__(self):
            self.data = 0.0

    class String:
        def __init__(self):
            self.data = ""

    class Point:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class Pose:
        def __init__(self):
            self.position = Point()

    class PoseArray:
        def __init__(self):
            self.poses = []


@dataclass
class PlannerState:
    identity_score: float = 0.0
    uncertainty: float = 1.0
    last_seen: Tuple[float, float] = (0.0, 0.0)
    has_last_seen: bool = False


class RiskAwareGridPlanner:
    """
    Lightweight risk-aware A* planner for ROS-ready interface demonstration.

    In real deployment:
        - occupancy map would come from SLAM or costmap,
        - risk layer could include smoke, debris, low visibility, or unsafe zones,
        - output path would be sent to navigation stack.
    """

    def __init__(self, width=30, height=20):
        self.width = width
        self.height = height

        # Example obstacles and risk zones in grid coordinates.
        self.obstacles = {
            (10, 8), (11, 8), (12, 8),
            (17, 12), (18, 12), (19, 12),
            (20, 5), (20, 6), (20, 7),
        }

        self.risk_cells = {
            (14, 7), (15, 7), (16, 7),
            (14, 8), (15, 8), (16, 8),
            (14, 9), (15, 9), (16, 9),
        }

    def is_valid(self, cell):
        x, y = cell

        if x < 0 or x >= self.width:
            return False

        if y < 0 or y >= self.height:
            return False

        if cell in self.obstacles:
            return False

        return True

    def cell_cost(self, cell):
        cost = 1.0

        if cell in self.risk_cells:
            cost += 5.0

        return cost

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def plan(self, start, goal):
        start = self.world_to_grid(start)
        goal = self.world_to_grid(goal)

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)

                if not self.is_valid(nxt):
                    continue

                tentative = g_score[current] + self.cell_cost(nxt)

                if nxt not in g_score or tentative < g_score[nxt]:
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f_score = tentative + self.heuristic(nxt, goal)
                    heapq.heappush(open_set, (f_score, nxt))

        return []

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return [self.grid_to_world(c) for c in path]

    def world_to_grid(self, point):
        x, y = point
        return int(round(x)), int(round(y))

    def grid_to_world(self, cell):
        x, y = cell
        return float(x), float(y)


class NextBestViewSelector:
    """
    Selects a next-best-view around the last-seen location.

    The score combines:
        - expected confidence gain,
        - distance cost,
        - risk penalty.
    """

    def __init__(self):
        self.candidate_offsets = [
            (2.0, 0.0),
            (-2.0, 0.0),
            (0.0, 2.0),
            (0.0, -2.0),
            (2.0, 2.0),
            (-2.0, 2.0),
            (2.0, -2.0),
            (-2.0, -2.0),
        ]

    def select(self, robot_position, last_seen, uncertainty):
        candidates = []

        for dx, dy in self.candidate_offsets:
            candidate = (last_seen[0] + dx, last_seen[1] + dy)

            distance = math.hypot(candidate[0] - robot_position[0], candidate[1] - robot_position[1])

            expected_gain = uncertainty * (1.0 / (1.0 + 0.15 * distance))
            distance_penalty = 0.05 * distance

            score = expected_gain - distance_penalty

            candidates.append({
                "candidate": candidate,
                "score": score,
                "expected_gain": expected_gain,
                "distance": distance,
            })

        best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]

        return best


class ActivePlannerNode(Node):
    """
    ROS 2-ready active planning node.

    Subscribes:
        /neuromemory/identity_score
        /neuromemory/uncertainty
        /neuromemory/last_seen

    Publishes:
        /neuromemory/next_best_view
        /neuromemory/risk_aware_path
        /neuromemory/planner_status
    """

    def __init__(self):
        if ROS2_AVAILABLE:
            super().__init__("active_planner_node")

        self.state = PlannerState()
        self.robot_position = (2.0, 2.0)

        self.nbv_selector = NextBestViewSelector()
        self.grid_planner = RiskAwareGridPlanner()

        if ROS2_AVAILABLE:
            self.identity_score_sub = self.create_subscription(
                Float32,
                "/neuromemory/identity_score",
                self.on_identity_score,
                10,
            )

            self.uncertainty_sub = self.create_subscription(
                Float32,
                "/neuromemory/uncertainty",
                self.on_uncertainty,
                10,
            )

            self.last_seen_sub = self.create_subscription(
                Point,
                "/neuromemory/last_seen",
                self.on_last_seen,
                10,
            )

            self.next_best_view_pub = self.create_publisher(
                Point,
                "/neuromemory/next_best_view",
                10,
            )

            self.path_pub = self.create_publisher(
                PoseArray,
                "/neuromemory/risk_aware_path",
                10,
            )

            self.status_pub = self.create_publisher(
                String,
                "/neuromemory/planner_status",
                10,
            )

            self.timer = self.create_timer(1.0, self.planning_step)

            self.get_logger().info("ActivePlannerNode started.")

        else:
            print("[ROS2 unavailable] ActivePlannerNode initialized in standalone interface mode.")

    def on_identity_score(self, msg: Float32):
        self.state.identity_score = float(msg.data)

    def on_uncertainty(self, msg: Float32):
        self.state.uncertainty = float(msg.data)

    def on_last_seen(self, msg: Point):
        self.state.last_seen = (float(msg.x), float(msg.y))
        self.state.has_last_seen = True

    def planning_step(self):
        """
        Decides whether active re-observation is required.
        """

        if not self.state.has_last_seen:
            self._publish_status("waiting_for_last_seen_memory")
            return

        if self.state.identity_score >= 0.85 and self.state.uncertainty <= 0.20:
            self._publish_status("identity_confidence_sufficient_monitoring_only")
            return

        if self.state.uncertainty < 0.30:
            self._publish_status("uncertainty_low_no_active_reobservation")
            return

        nbv_result = self.nbv_selector.select(
            robot_position=self.robot_position,
            last_seen=self.state.last_seen,
            uncertainty=self.state.uncertainty,
        )

        next_best_view = nbv_result["candidate"]

        path = self.grid_planner.plan(
            start=self.robot_position,
            goal=next_best_view,
        )

        self._publish_next_best_view(next_best_view)
        self._publish_path(path)

        self._publish_status(
            f"active_reobservation_recommended "
            f"nbv=({next_best_view[0]:.2f},{next_best_view[1]:.2f}) "
            f"path_nodes={len(path)} "
            f"expected_gain={nbv_result['expected_gain']:.3f}"
        )

    def _publish_next_best_view(self, point):
        if ROS2_AVAILABLE:
            msg = Point()
            msg.x = float(point[0])
            msg.y = float(point[1])
            msg.z = 0.0
            self.next_best_view_pub.publish(msg)
        else:
            print(f"[standalone] next_best_view={point}")

    def _publish_path(self, path):
        if ROS2_AVAILABLE:
            msg = PoseArray()

            for x, y in path:
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = 0.0
                msg.poses.append(pose)

            self.path_pub.publish(msg)
        else:
            print(f"[standalone] risk_aware_path={path}")

    def _publish_status(self, status):
        if ROS2_AVAILABLE:
            msg = String()
            msg.data = status
            self.status_pub.publish(msg)
            self.get_logger().info(status)
        else:
            print(f"[planner_status] {status}")


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS 2 is not installed in this Python environment.")
        print("This file is still valid as a ROS-ready interface specification.")
        print("Install ROS 2 and rclpy to run it as a live ROS node.")
        return

    rclpy.init(args=args)

    node = ActivePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ActivePlannerNode stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()