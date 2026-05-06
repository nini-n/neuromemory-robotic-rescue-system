# NeuroMemory Robot — ROS 2 Interface Specification

## Purpose

This folder defines a ROS 2-ready interface layer for the NeuroMemory Robot project.

The current implementation is a simulation-based prototype. However, the perception, memory, planning, and operator decision-support logic is structured so that it can be transferred to a future mobile robot or drone platform.

This interface does not claim full hardware deployment. It defines how the system would communicate inside a robotic software stack.

---

## System Pipeline

```text
/camera/image_raw
        ↓
perception_memory_node.py
        ↓
/neuromemory/identity_score
/neuromemory/uncertainty
/neuromemory/last_seen
/neuromemory/status
        ↓
active_planner_node.py
        ↓
/neuromemory/next_best_view
/neuromemory/risk_aware_path
/neuromemory/planner_status
        ↓
operator_status_node.py
        ↓
/neuromemory/priority_score
/neuromemory/operator_alert
/neuromemory/operator_summary
```

---

## Node 1 — `perception_memory_node.py`

### Role

This node represents the perception and visual-memory layer.

It receives camera frames, extracts a visual feature representation, compares it with stored visual memory, and publishes identity confidence and uncertainty values.

### Subscribed Topic

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera frame from robot or drone |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/neuromemory/identity_score` | `std_msgs/Float32` | Similarity/confidence score between current observation and visual memory |
| `/neuromemory/uncertainty` | `std_msgs/Float32` | Uncertainty score derived from confidence |
| `/neuromemory/last_seen` | `geometry_msgs/Point` | Estimated last-seen position of the candidate |
| `/neuromemory/status` | `std_msgs/String` | Perception-memory status message |

### Notes

The current file uses a lightweight placeholder feature extractor. In a real deployment, this can be replaced by:

- explainable body-region HSV features,
- OSNet ReID embedding,
- hybrid NeuroMemory similarity score.

The node does not make final identity decisions.

---

## Node 2 — `active_planner_node.py`

### Role

This node represents the active planning layer.

It receives confidence, uncertainty, and last-seen information. If uncertainty is high, it selects a next-best-view candidate and produces a risk-aware path.

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/neuromemory/identity_score` | `std_msgs/Float32` | Current identity confidence |
| `/neuromemory/uncertainty` | `std_msgs/Float32` | Current uncertainty level |
| `/neuromemory/last_seen` | `geometry_msgs/Point` | Last-seen candidate position |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/neuromemory/next_best_view` | `geometry_msgs/Point` | Recommended viewpoint for re-observation |
| `/neuromemory/risk_aware_path` | `geometry_msgs/PoseArray` | Planned path avoiding risk zones where possible |
| `/neuromemory/planner_status` | `std_msgs/String` | Planner status and explanation |

### Planning Logic

The planner uses:

- uncertainty-aware re-observation trigger,
- next-best-view candidate selection,
- risk-aware A* style path planning.

A real robotic deployment could replace the internal grid planner with a ROS 2 navigation stack or costmap-based planner.

---

## Node 3 — `operator_status_node.py`

### Role

This node represents the human-in-the-loop decision-support layer.

It receives identity score, uncertainty, last-seen position, next-best-view, and planner status. It then produces an operator-facing priority score, alert level, and textual summary.

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/neuromemory/identity_score` | `std_msgs/Float32` | Current identity confidence |
| `/neuromemory/uncertainty` | `std_msgs/Float32` | Current uncertainty score |
| `/neuromemory/last_seen` | `geometry_msgs/Point` | Last-seen candidate position |
| `/neuromemory/next_best_view` | `geometry_msgs/Point` | Recommended re-observation viewpoint |
| `/neuromemory/planner_status` | `std_msgs/String` | Active planner status |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/neuromemory/priority_score` | `std_msgs/Float32` | Operator priority score |
| `/neuromemory/operator_alert` | `std_msgs/String` | Alert category for the operator |
| `/neuromemory/operator_summary` | `std_msgs/String` | Human-readable decision-support summary |

### Operator Logic

The operator layer is deliberately conservative.

It does not output final identity decisions. Instead, it produces:

- probable match review required,
- medium priority monitor and re-observe,
- high priority human verification,
- low priority continue search.

---

## Human-in-the-Loop Safety

The NeuroMemory system is designed as a decision-support framework.

It does **not** autonomously decide that two observations belong to the same person. Instead, it provides:

- similarity score,
- uncertainty score,
- re-observation recommendation,
- risk-aware path suggestion,
- operator priority score.

The final identity-related decision remains with the human operator.

---

## Relation to Current Simulation

The current Pygame-based environment is used as a lightweight 2D rescue-environment simulator and operator dashboard.

The ROS 2 interface shows how the same method can be mapped to a future robotic architecture:

```text
Pygame simulation component → ROS 2 deployment equivalent

Visual detection           → /camera/image_raw + perception_memory_node
Visual memory vector       → /neuromemory/identity_score
Uncertainty check          → /neuromemory/uncertainty
Last-seen map              → /neuromemory/last_seen
Next-best-view             → /neuromemory/next_best_view
Risk-aware A*              → /neuromemory/risk_aware_path
Operator dashboard         → /neuromemory/operator_summary
```

---

## Future Deployment Path

A complete robotic deployment would require:

1. ROS 2 installation and package setup,
2. real or simulated camera topic,
3. object/person detector,
4. visual feature extractor,
5. robot localization or odometry,
6. map or costmap layer,
7. navigation controller,
8. operator interface.

Possible future simulation platforms:

- ROS 2 + Gazebo,
- Isaac Sim,
- real mobile robot or drone platform.

---

## Current Status

This folder currently provides a ROS 2-ready interface specification and Python node skeletons.

The files are not required to run the current Pygame simulation. They document and prototype how the NeuroMemory logic can be migrated into a robotics middleware architecture.
