### Mouse Maze

This repository contains the code and models for the study on mouse exploration strategies in a complex labyrinth, as presented in:

* **Singla, U. \& Mattar, M., 2026 (bioRxiv)**: *Temporal Abstraction Explains Mouse Exploration in a Complex Maze*
* **Singla, U. \& Mattar, M., 2024 (CogSci)**: *Temporal Persistence Explains Mice Exploration in a Labyrinth*

#### Overview

This project models the exploration behavior of mice freely navigating a complex binary-tree maze (originally described by Rosenberg et al., 2021). We demonstrate that a hierarchical Reinforcement Learning (RL) agent using temporally-extended $\epsilon$-greedy exploration (**$\epsilon z$-greedy**, Dabney et al., 2020) is able to capture the efficiency and turning biases of animals.

The codebase allows for:
1.  Simulating various agents ($\epsilon z$-greedy, Random Walk, Biased Walk, Uncertainty-based).
2.  Comparing simulated trajectories against animal behavioral data.
3.  Reproducing the figures presented in the paper.

#### Directory Structure

```text
├── src/
│   ├── sample_agent.py         # Entry point: runs a model, generates trajectories, dumps model, and plots figures
│   ├── BaseModel/              # Base model class to implement a variety of models
│   ├── TeAltOptions_model.py   # The main temporally-extended model (Ez-greedy)
│   ├── Te*_model.py            # Variations of models built on TeAltOptions_model
│   ├── figure_*.ipynb           # Jupyter notebooks to generate all paper figures
│   ├── utils.py                 # General utilities
│   ├── plot_utils.py            # General plotting utilities
│   ├── MM_Maze_Utils.py         # Utilities to plot maze and handle data structures (from original study)
│   ├── MM_Traj_Utils.py         # Mouse Trajectory processing utilities (from original study)
│   ├── MM_Plot_Utils.py         # Maze plotting utilities (from original study)
│   └── MM_Models.py             # Markov chain models from the original study
└── model_nbs/                  # Jupyter notebooks for older/experimental models (not maintained)
```

#### Installation
This project has minimal dependencies and works with Python 3.9+.

#### Running a Sample Agent

The main entry point for running a simulation is `sample_agent.py`. This script initializes the maze environment, runs the $\epsilon z$-greedy agent (or others), generates trajectories, and saves the output.

```bash
python sample_agent.py
```

#### Acknowledgments
Data and original maze utilities adapted from Rosenberg et al. _eLife_ (2021).

