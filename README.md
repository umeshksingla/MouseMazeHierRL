## Mouse Maze

This repository contains the code and models for the study on mouse exploration strategies in a complex labyrinth, as presented in:

* **Singla et al., 2026 (bioRxiv)**: *Temporal Abstraction Explains Mouse Exploration in a Complex Maze*
* **Singla et al., 2024 (CogSci)**: *Temporal Persistence Explains Mice Exploration in a Labyrinth*

[**Read the Paper (PDF)**](https://escholarship.org/content/qt0s4241rf/qt0s4241rf.pdf)

## Overview

This project models the exploration behavior of mice freely navigating a complex binary-tree maze (originally described by Rosenberg et al., 2021). We demonstrate that a hierarchical Reinforcement Learning (RL) agent using **temporally-extended $\epsilon$-greedy exploration ($\epsilon z$-greedy)** captures the efficiency and turning biases of real mice better than random walks or simple Markovian models.

The codebase allows for:
1.  Simulating various agents (Random Walk, Biased Walk, Ez-greedy, Uncertainty-based).
2.  Comparing simulated trajectories against animal behavioral data.
3.  Reproducing the figures presented in the paper.

## Directory Structure

```text
├── src/
│   ├── BaseModel/           # Base model class to implement a variety of models
│   ├── TeAltOptions_model.py # The main temporally-extended model (Ez-greedy)
│   └── Te*_model.py         # Variations of models built on TeAltOptions_model
├── model_nbs/               # Jupyter notebooks for older/experimental models (not maintained)
├── figure_*.ipynb           # Jupyter notebooks to generate all paper figures
├── utils.py                 # General utilities
├── plot_utils.py            # General plotting utilities
├── MM_Maze_Utils.py         # Utilities to plot maze and handle data structures (from orig. study)
├── MM_Traj_Utils.py         # Trajectory processing utilities (from orig. study)
├── MM_Plot_Utils.py         # Maze plotting utilities (from orig. study)
├── MM_Models.py             # Markov chain models from the original study
└── sample_agent.py          # Entry point: runs a model, generates trajectories, dumps model, and plots figures