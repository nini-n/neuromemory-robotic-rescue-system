"""
NeuroMemory Robot - ROS 2 Operator Status Node

Purpose:
    This node represents the human-operator decision-support layer.

Role in the robotic pipeline:
    /neuromemory/identity_score
    /neuromemory/uncertainty
    /neuromemory/last_seen
    /neuromemory/next_best_view
    /neuromemory/planner_status
        -> operator_status_node
        -> /neuromemory/operator_summary
        -> /neuromemory/priority_score
        -> /neuromemory/operator_alert

Important:
    This is a ROS 2-ready integration interface.
    The current project is simulation-based. This file shows how the dashboard
    and human-in-the-loop decision support can be transferred to a robot/drone
    operator pipeline.

Notes:
    - This node does not make final identity decisions.
    - It summarizes confidence, uncertainty, re-observation status, and priority.
    - It is designed to support a human operator under degraded visibility.
"""

import time
import math
from dataclasses import dataclass
from typing import Tuple

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, String
    from geometry_msgs.msg import Point

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


@dataclass
class OperatorState:
    identity_score: float = 0.0
    uncertainty: float = 1.0
    last_seen: Tuple[float, float] = (0.0, 0.0)
    next_best_view: Tuple[float, float] = (0.0, 0.0)
    planner_status: str = "waiting"
    last_update_time: float = 0.0


class OperatorDecisionLogic:
    """
    Converts NeuroMemory signals into operator-facing summaries.

    This layer is deliberately conservative:
        - it does not output final identity decisions,
        - it increases priority when uncertainty and risk are high,
        - it recommends human verification for ambiguous cases.
    """

    @staticmethod
    def compute_priority_score(identity_score: float, uncertainty: float, planner_status: str) -> float:
        """
        Priority increases when:
            - identity score is moderately high,
            - uncertainty remains high,
            - active re-observation is recommended.

        This reflects a rescue scenario where a possible survivor candidate
        should be prioritized but still verified by a human operator.
        """

        identity_component = 0.45 * identity_score
        uncertainty_component = 0.40 * uncertainty

        planner_component = 0.0
        if "active_reobservation_recommended" in planner_status:
            planner_component = 0.15
        elif "waiting" in planner_status:
            planner_component = 0.05

        priority = identity_component + uncertainty_component + planner_component

        return max(0.0, min(1.0, priority))

    @staticmethod
    def alert_level(identity_score: float, uncertainty: float, priority_score: float) -> str:
        if priority_score >= 0.75 and uncertainty >= 0.35:
            return "high_priority_human_verification"

        if priority_score >= 0.60:
            return "medium_priority_monitor_and_reobserve"

        if identity_score >= 0.85 and uncertainty <= 0.20:
            return "probable_match_review_required"

        return "low_priority_continue_search"

    @staticmethod
    def summary_text(
        identity_score: float,
        uncertainty: float,
        priority_score: float,
        last_seen: Tuple[float, float],
        next_best_view: Tuple[float, float],
        planner_status: str,
        alert: str,
    ) -> str:
        return (
            f"NeuroMemory operator summary | "
            f"identity_score={identity_score:.3f}, "
            f"uncertainty={uncertainty:.3f}, "
            f"priority={priority_score:.3f}, "
            f"last_seen=({last_seen[0]:.2f},{last_seen[1]:.2f}), "
            f"next_best_view=({next_best_view[0]:.2f},{next_best_view[1]:.2f}), "
            f"planner_status={planner_status}, "
            f"operator_alert={alert}"
        )


class OperatorStatusNode(Node):
    """
    ROS 2-ready operator status node.

    Subscribes:
        /neuromemory/identity_score
        /neuromemory/uncertainty
        /neuromemory/last_seen
        /neuromemory/next_best_view
        /neuromemory/planner_status

    Publishes:
        /neuromemory/priority_score
        /neuromemory/operator_alert
        /neuromemory/operator_summary
    """

    def __init__(self):
        if ROS2_AVAILABLE:
            super().__init__("operator_status_node")

        self.state = OperatorState()
        self.logic = OperatorDecisionLogic()

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

            self.next_best_view_sub = self.create_subscription(
                Point,
                "/neuromemory/next_best_view",
                self.on_next_best_view,
                10,
            )

            self.planner_status_sub = self.create_subscription(
                String,
                "/neuromemory/planner_status",
                self.on_planner_status,
                10,
            )

            self.priority_pub = self.create_publisher(
                Float32,
                "/neuromemory/priority_score",
                10,
            )

            self.alert_pub = self.create_publisher(
                String,
                "/neuromemory/operator_alert",
                10,
            )

            self.summary_pub = self.create_publisher(
                String,
                "/neuromemory/operator_summary",
                10,
            )

            self.timer = self.create_timer(1.0, self.publish_operator_status)

            self.get_logger().info("OperatorStatusNode started.")

        else:
            print("[ROS2 unavailable] OperatorStatusNode initialized in standalone interface mode.")

    def on_identity_score(self, msg: Float32):
        self.state.identity_score = float(msg.data)
        self.state.last_update_time = time.time()

    def on_uncertainty(self, msg: Float32):
        self.state.uncertainty = float(msg.data)
        self.state.last_update_time = time.time()

    def on_last_seen(self, msg: Point):
        self.state.last_seen = (float(msg.x), float(msg.y))
        self.state.last_update_time = time.time()

    def on_next_best_view(self, msg: Point):
        self.state.next_best_view = (float(msg.x), float(msg.y))
        self.state.last_update_time = time.time()

    def on_planner_status(self, msg: String):
        self.state.planner_status = str(msg.data)
        self.state.last_update_time = time.time()

    def publish_operator_status(self):
        priority_score = self.logic.compute_priority_score(
            identity_score=self.state.identity_score,
            uncertainty=self.state.uncertainty,
            planner_status=self.state.planner_status,
        )

        alert = self.logic.alert_level(
            identity_score=self.state.identity_score,
            uncertainty=self.state.uncertainty,
            priority_score=priority_score,
        )

        summary = self.logic.summary_text(
            identity_score=self.state.identity_score,
            uncertainty=self.state.uncertainty,
            priority_score=priority_score,
            last_seen=self.state.last_seen,
            next_best_view=self.state.next_best_view,
            planner_status=self.state.planner_status,
            alert=alert,
        )

        if ROS2_AVAILABLE:
            priority_msg = Float32()
            priority_msg.data = float(priority_score)
            self.priority_pub.publish(priority_msg)

            alert_msg = String()
            alert_msg.data = alert
            self.alert_pub.publish(alert_msg)

            summary_msg = String()
            summary_msg.data = summary
            self.summary_pub.publish(summary_msg)

            self.get_logger().info(summary)

        else:
            print(f"[operator_status] {summary}")

    def standalone_demo_step(
        self,
        identity_score: float,
        uncertainty: float,
        last_seen: Tuple[float, float],
        next_best_view: Tuple[float, float],
        planner_status: str,
    ):
        """
        Allows this file to be tested without ROS 2.
        """

        self.state.identity_score = identity_score
        self.state.uncertainty = uncertainty
        self.state.last_seen = last_seen
        self.state.next_best_view = next_best_view
        self.state.planner_status = planner_status
        self.state.last_update_time = time.time()

        self.publish_operator_status()


def standalone_demo():
    """
    Lightweight demonstration for non-ROS environments.
    """

    print("Running OperatorStatusNode standalone demo...")

    node = OperatorStatusNode()

    demo_cases = [
        {
            "identity_score": 0.88,
            "uncertainty": 0.18,
            "last_seen": (4.5, 3.0),
            "next_best_view": (4.5, 3.0),
            "planner_status": "identity_confidence_sufficient_monitoring_only",
        },
        {
            "identity_score": 0.72,
            "uncertainty": 0.42,
            "last_seen": (6.0, 4.5),
            "next_best_view": (8.0, 4.5),
            "planner_status": "active_reobservation_recommended nbv=(8.00,4.50) path_nodes=9 expected_gain=0.311",
        },
        {
            "identity_score": 0.48,
            "uncertainty": 0.67,
            "last_seen": (7.2, 5.0),
            "next_best_view": (9.2, 7.0),
            "planner_status": "active_reobservation_recommended nbv=(9.20,7.00) path_nodes=14 expected_gain=0.428",
        },
    ]

    for case in demo_cases:
        node.standalone_demo_step(**case)


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS 2 is not installed in this Python environment.")
        print("This file is still valid as a ROS-ready operator interface specification.")
        print("Running standalone demonstration instead.\n")
        standalone_demo()
        return

    rclpy.init(args=args)

    node = OperatorStatusNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("OperatorStatusNode stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()