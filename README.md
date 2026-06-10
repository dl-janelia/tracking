# Exercise 8 - Transformers and tracking

This exercise was created by Benjamin Gallusser and Albert Dominguez Mantes,
updated for 2024/2025 by Caroline Malin-Mayor, and extended for 2026 by
Albert Dominguez Mantes with a transformers-from-scratch part.

<img src="figures/tracking.gif" width="500"/>

## Objectives
- Build the core components of a transformer from scratch and train one on a toy sequence task.
- Write a pipeline that takes in cell detections and links them across time to obtain lineage trees.
- Plug a small, trainable transformer into the tracking ILP as an edge scorer, and compare it against the pretrained `trackastra` model.


## Setup
1. Go into the folder with this repo and run
    ```
    source setup.sh
    ```
    to set up the environment for this exercise. This will take a few minutes.
2. Open the corresponding notebook in VSCode and select the `tracking` kernel:
    - `01-Transformers/exercise.ipynb`: transformers from scratch on a toy sequence task.
    - `02-Tracking/exercise.ipynb`: ILP-based tracking with motile, with a learned transformer edge scorer and a comparison against pretrained `trackastra`.

## Overview

### Introduction to transformers for 1D sequences

In this part, you will build the core components of a transformer from scratch using `torch`, understand their fundamental properties, and apply them to a simple sequence classification task.

### Tracking by detection with an integer linear program (ILP)

In this part, you will write an ILP-based pipeline that takes in cell detections and links them across time to obtain lineage trees. You will then train a small transformer-based model to predict linking scores for pairs of cells in adjacent time points to be used in the ILP, and use a pre-trained DL model, `trackastra`, to predict linking scores as well.

#### Methods/Tools:

- **`networkx`**: To represent the tracking inputs and outputs as graphs. Tracking is often framed
    as a graph optimization problem. Nodes in the graph represent detections, and edges represent links
    across time. The "tracking" task is then framed as selecting the correct edges to link your detections.
- **`motile`**: To set up and solve an Integer Lineage Program (ILP) for tracking.
    ILP-based methods frame tracking as a constrained optimization problem. The task is to select a subset of nodes/edges from a "candidate graph" of all possible nodes/edges. The subset must minimize user-defined costs (e.g. edge distance), while also satisfying a set of tracking constraints (e.g. each cell is linked to at most one cell in the previous frame). Note: this tracking approach is not inherently using
    "deep learning" - the costs and constraints are usually hand-crafted to encode biological and data-based priors, although cost features can also be learned from data.
- **`trackastra`**: To predict linking scores for cells in adjacent time points. `trackastra` is a transformer-based deep learning model, 
    with published pre-trained models that work on many types of input.
- **`napari`**: To visualize tracking inputs and outputs. Qualitative analysis is crucial for tuning the 
    weights of the objective function and identifying data-specific costs and constraints.
- **`traccuracy`**: To evaluate tracking results. Metrics such as accuracy can be misleading for tracking,
    because rare events such as divisions are much harder than the common linking tasks, and might
    be more biologically relevant for downstream analysis. Therefore, it is important to evaluate on
    a wide range of error metrics and determine which are most important for your use case.

#### Bonus: Tracking with two-step Linear Assignment Problem (LAP)

There is a bonus notebook showing how to use a two-step linking algorithm implemented in the Fiji plugin TrackMate. We will not go over this in the exercise time, but it is available for those who are interested in learning on their own. In the bonus you will learn how to use **Trackmate**, a versatile ready-to-go implementation of two-step LAP tracking and other algorithms in `ImageJ/Fiji`.
