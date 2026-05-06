# NeuroMemory Robot - Gazebo Simulation Skeleton

## Purpose

This folder provides a Gazebo-ready simulation skeleton for the NeuroMemory Robot project.

The current project is mainly implemented as a lightweight 2D simulation and operator dashboard. This Gazebo layer shows how the same rescue-robot concept can be transferred toward a robotics-grade 3D simulation environment.

This is not yet a full ROS 2 navigation deployment. It is a structured Gazebo-ready extension.

---

## Folder Structure

```text
gazebo_sim/
├── worlds/
│   └── neuromemory_rescue_world.sdf
├── models/
│   └── rescue_robot/
│       ├── model.config
│       └── model.sdf
├── launch/
│   └── neuromemory_gazebo.launch.py
└── README_gazebo.md
```

---

## Included Components

### 1. Rescue World

`worlds/neuromemory_rescue_world.sdf`

Includes:

- ground plane,
- simple debris blocks,
- low-visibility / smoke-like zone,
- survivor candidate placeholder,
- next-best-view marker,
- rescue robot spawn.

### 2. Rescue Robot Model

`models/rescue_robot/model.sdf`

Includes:

- differential-drive style robot body,
- left and right wheels,
- front caster,
- camera mast,
- forward-facing camera sensor placeholder,
- `/camera/image_raw` camera topic.

### 3. ROS 2 Launch Skeleton

`launch/neuromemory_gazebo.launch.py`

Provides a future ROS 2 launch structure for starting the Gazebo world.

---

## Relation to NeuroMemory Pipeline

```text
Gazebo camera sensor
        ↓
/camera/image_raw
        ↓
perception_memory_node.py
        ↓
/neuromemory/identity_score
/neuromemory/uncertainty
/neuromemory/last_seen
        ↓
active_planner_node.py
        ↓
/neuromemory/next_best_view
/neuromemory/risk_aware_path
        ↓
operator_status_node.py
        ↓
/neuromemory/operator_summary
```

---

## Current Status

This folder is intended as a Gazebo-ready extension.

It does not replace the current Pygame simulation. Instead, it documents how the NeuroMemory method can be migrated to a robotics-grade simulation stack.

---

## Future Work

A complete Gazebo/ROS 2 deployment would require:

1. ROS 2 package structure,
2. Gazebo ROS bridge,
3. camera topic bridge,
4. robot controller,
5. navigation stack integration,
6. costmap/risk-layer integration,
7. real perception model connection,
8. RViz visualization.

---

## Suggested Project Description

The current system uses a lightweight 2D simulator for algorithm development and operator visualization. A Gazebo-ready rescue world and robot model are additionally provided to show how the same perception-memory-planning architecture could be extended toward robotics-grade simulation and future deployment.
