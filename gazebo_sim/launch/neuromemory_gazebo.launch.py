"""
NeuroMemory Robot - Gazebo Launch Skeleton

Purpose:
    Launch file skeleton for future ROS 2 + Gazebo integration.

Notes:
    This file documents how the NeuroMemory Gazebo world would be launched
    in a ROS 2 workspace. It is provided as a robot-ready integration layer.
"""

from pathlib import Path

try:
    from launch import LaunchDescription
    from launch.actions import ExecuteProcess, SetEnvironmentVariable
    from launch.substitutions import EnvironmentVariable

    LAUNCH_AVAILABLE = True

except ImportError:
    LAUNCH_AVAILABLE = False


def generate_launch_description():
    if not LAUNCH_AVAILABLE:
        raise RuntimeError(
            "ROS 2 launch libraries are not installed. "
            "This file is a ROS 2 launch skeleton and should be used inside a ROS 2 workspace."
        )

    project_root = Path(__file__).resolve().parents[2]
    world_path = project_root / "gazebo_sim" / "worlds" / "neuromemory_rescue_world.sdf"
    model_path = project_root / "gazebo_sim" / "models"

    return LaunchDescription([
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=str(model_path),
        ),

        ExecuteProcess(
            cmd=[
                "gz",
                "sim",
                "-r",
                str(world_path),
            ],
            output="screen",
        ),
    ])