# NeuroMemory Robotic Rescue System

Simulation-based robotic rescue and active visual-memory prototype for uncertain search-and-rescue environments.

This repository contains the code and simulation assets for a NeuroMemory-inspired rescue robot concept. The project demonstrates how a robot can combine visual-memory scoring, uncertainty-aware re-observation, risk-aware path planning, and operator-facing decision support in a structured simulation workflow.

## Project Overview

The project is built as a simulation-first research prototype. It does not claim hardware deployment. Instead, it focuses on showing the full logic of a robotic rescue pipeline: initial detection, memory update, active re-observation, next-best-view planning, risk-aware motion, and operator-level status reporting.

The repository includes:

- a Python/Pygame simulation and operator dashboard,
- active re-observation and risk-aware planning logic,
- evaluation scripts for threshold sensitivity, ablation, and failure analysis,
- public person re-identification dataset validation scripts,
- Gazebo-ready simulation skeleton,
- ROS 2 interface specification and example nodes,
- Isaac/USD rescue-scene assets,
- poster-ready technical figures and 3D visualization scripts.

## Repository Structure

```text
neuromemory_robot/
├── src/                         # Main simulation and evaluation scripts
├── ros2_interface/              # ROS 2 interface specification and example nodes
├── gazebo_sim/                  # Gazebo world, robot model, and launch skeleton
├── isaac_scene/                 # Isaac Sim / USD rescue scene asset
├── figures/                     # Selected project figures
├── docs/                        # Project documentation
├── generate_neuromemory_3d_views.py
├── generate_neuromemory_premium_visuals_v2.py
├── generate_professional_rescue_scene.py
├── make_3d_render_collage.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Components

### 1. NeuroMemory Simulation

The main simulation is implemented in `src/main.py`. It visualizes a rescue scenario where the robot evaluates memory confidence, uncertainty, risk, and active re-observation decisions.

```bash
python src/main.py
```

### 2. Evaluation and Analysis

The evaluation scripts generate tables and figures for the project explanation. They include threshold sensitivity, feature analysis, failure-case analysis, ablation studies, and final result summaries.

```bash
python src/evaluation_outputs.py
python src/advanced_evaluation.py
python src/feature_threshold_failure_analysis.py
python src/final_results_summary.py
```

### 3. Public ReID Validation

Some scripts validate the visual-memory idea using public person re-identification data. This is not face recognition. The system uses feature-level person re-identification style validation, and the final interpretation is intended to remain human-supervised.

```bash
python src/dataset_feature_validation.py
python src/pretrained_embedding_validation.py
python src/reid_specific_embedding_validation.py
python src/reid_query_gallery_validation.py
```

The dataset itself is not included in the repository. Place the required sample data under:

```text
data/sample_reid/
data/market1501_extracted/
```

### 4. Gazebo Simulation Skeleton

The `gazebo_sim/` folder contains a Gazebo-ready rescue world and a simple rescue robot model. It is included as a structured extension layer, not as a complete autonomous navigation deployment.

See:

```text
gazebo_sim/README_gazebo.md
```

### 5. ROS 2 Interface Layer

The `ros2_interface/` folder defines how the system can be mapped into a ROS 2-style robotic software stack. It includes example planner and operator-status nodes.

See:

```text
ros2_interface/neuromemory_messages.md
```

### 6. Isaac / USD Scene

The `isaac_scene/` folder contains a USD rescue-scene asset that can be used as a starting point for Isaac Sim visualization.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies such as ROS 2, Gazebo, Isaac Sim, and torchreid may require separate installation depending on the environment.

## Example Commands

Run the main simulation:

```bash
python src/main.py
```

Generate poster-ready rescue-scene figures:

```bash
python generate_professional_rescue_scene.py
python generate_neuromemory_premium_visuals_v2.py
```

Generate 3D visualizations:

```bash
python generate_neuromemory_3d_views.py
python make_3d_render_collage.py
```

Run the full evaluation pipeline:

```bash
python src/run_full_evaluation.py
```

## Notes

This project is a research-oriented prototype. The current version focuses on simulation, visualization, evaluation, and robotics-system structuring. It is not presented as a deployed rescue robot or a clinically validated identification system.

## License

This project is released under the MIT License.
