# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 09-tracking
#     language: python
#     name: 09-tracking
# ---

# %% [markdown]
# # Exercise 8: Tracking-by-detection with an integer linear program (ILP)
#
# Objective:
# - Write a pipeline that takes in cell detections and links them across time to obtain lineage trees
#
# Methods/Tools:
#
# - **`networkx`**: To represent the tracking inputs and outputs as graphs. Tracking is often framed
#     as a graph optimization problem. Nodes in the graph represent detections, and edges represent links
#     across time. The "tracking" task is then framed as selecting the correct edges to link your detections.
# - **`motile`**: To set up and solve an Integer Linear Program (ILP) for tracking.
#     ILP-based methods frame tracking as a constrained optimization problem. The task is to select a subset of nodes/edges from a "candidate graph" of all possible nodes/edges. The subset must minimize user-defined costs (e.g. edge distance), while also satisfying a set of tracking constraints (e.g. each cell is linked to at most one cell in the previous frame). Note: this tracking approach is not inherently using
#     "deep learning" - the costs and constraints are usually hand-crafted to encode biological and data-based priors, although cost features can also be learned from data.
# - **`napari`**: To visualize tracking inputs and outputs. Qualitative analysis is crucial for tuning the
#     weights of the objective function and identifying data-specific costs and constraints.
# - **`trackastra`**: To predict linking scores for cells in adjacent time points. `trackastra` is a transformer-based deep learning model, 
#     with published pre-trained models that work on many types of input.
# - **`traccuracy`**: To evaluate tracking results. Metrics such as accuracy can be misleading for tracking,
#     because rare events such as divisions are much harder than the common linking tasks, and might
#     be more biologically relevant for downstream analysis. Therefore, it is important to evaluate on
#     a wide range of error metrics and determine which are most important for your use case.
#
# After running through the full tracking pipeline, from loading to evaluation, we will learn how to **incorporate custom costs** based on dataset-specific prior information and deep learning models.
#
# <div class="alert alert-danger">
# Set your python kernel to <code>08-tracking</code>
# </div>
#
# Places where you are expected to write code are marked with
# ```
# ### YOUR CODE HERE ###
# ```
#
# This notebook was originally written by Benjamin Gallusser, and was edited for 2024 and 2025 by Caroline Malin-Mayor.

# %% [markdown]
# ## Section 0: Setup
# The setup.sh script already installed dependencies and downloaded the dataset we will be using. However, since we will be using `napari` to interactively visualize our tracking results, if you are running this notebook on a remote machine, we need to set up a few things.

# %% [markdown]
# ### Set up NoMachine with port forwarding
# 1. From VSCode connected to your remote machine, forward a port (e.g. `4000`) to your local machine.
#     - Open you command palette in VSCode (usually CMD-Shift-P) and type "forward a port"
#     - Then type in the desired port number `4000` and hit enter
#     - From the "PORTS" tab, you should see port 4000 listed as a forwarded port
# 2. Download and install [NoMachine](https://www.nomachine.com/download) on your local machine if it is not already installed.
# 3. Enter the server address in host, set the port to match the port you forwarded in step 1 and protocol as NX. Feel free to enter any name you would like.
# 4. Click on the configuration tab on the left.
# 5. Choose "Use key-based authentication with a key you provide" and hit the "Modify" button.
# 6. Provide the path to your ssh key .pem file.
# 7. Finally hit connect (or Add).
# 8. If you are asked to create a desktop, click yes.
# 9. You should then see a time and date, hitting enter should let you enter your username and access the desktop. The first login may be slow.
# 10. Still in NoMachine, open a shell window. Hit the application button in the bottom left corner and launch "Konsole"
# 11. From the shell, run `echo $DISPLAY`. Copy the output. It should be something like `:1005`
# 12. Return to your notebook in VSCode, and proceed with the exercise.
# 13. Modify the cell below to input the DISPLAY port you retrieved in step 11

# %%
import os
os.environ["DISPLAY"] = "TODO"

# %% [markdown]
# ### Import packages

# %%
import skimage
import numpy as np
import napari
import networkx as nx
import scipy

import motile

import zarr
import geff
import trackastra.model
from motile_tracker.motile.backend import MotileRun, graph_to_nx
from motile_tracker.data_views.views.tree_view.tree_widget import TreeWidget
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer
import traccuracy
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.matchers import IOUMatcher
from csv import DictReader

from tqdm.auto import tqdm

from typing import Iterable, Any

# %% [markdown]
# ## Section 1: Visualize the data

# %% [markdown]
# ### Load the dataset and inspect it in napari

# %% [markdown]
# For this exercise we will be working with a fluorescence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA). It is similar to the dataset at https://zenodo.org/record/4034976#.YwZRCJPP1qt. The raw data, pre-computed segmentations, detection probabilities, and ground truth tracks are saved in a zarr. The segmentation was generated with a pre-trained StartDist model, so there may be some segmentation errors which can affect the tracking process. The detection probabilities also come from StarDist, and are downsampled in x and y by 2 compared to the detections and raw data.

# %% [markdown]
# Here we load the raw image data, segmentation, and probabilities from the zarr, and view them in napari.

# %%
data_path = "./data/breast_cancer_fluo.zarr"
data_root = zarr.open(data_path, 'r')
image_data = data_root["raw"][:]
segmentation = data_root["seg"][:]
probabilities = data_root["probs"][:]

# %%
viewer = napari.Viewer()

# %% [markdown]
# Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) to visualize the data. Napari is a wonderful viewer for imaging data that you can interact with in python, even directly out of jupyter notebooks. If you've never used napari, you might want to take a few minutes to go through [this tutorial](https://napari.org/stable/tutorials/fundamentals/viewer.html). Here we visualize the raw data, the predicted segmentations, and the predicted probabilities as separate layers. You can toggle each layer on and off in the layers list on the left.

# %%

viewer.add_image(probabilities, name="probs", scale=(1, 2, 2))
viewer.add_image(image_data, name="raw")
viewer.add_labels(segmentation, name="seg")

# %% [markdown]
# After running the previous cell, open NoMachine and check for an open napari window.

# %% [markdown]
# ### Read in the ground truth graph and inspect it in napari
# In addition to the image data and segmentations, we also have a ground truth tracking solution.
# The ground truth tracks are stored in a [`geff`](http://liveimagetrackingtools.org/geff/latest/) (Graph Exchange File Format) group in the zarr.
# This is a new format that is in the process of being adopted by the tracking community, with support for import/export from a variety of
# common tools, such as TrackMate, `trackastra`, `napari`, and `traccuracy`.
#
# Each node in the graph represents a detection, and has properties `t`, `y`, and `x` holding the location of that detection.
# Edges in the graph link detected cells between time frames: edges go from a detection in time `t` to the same cell (or its daughter) detected in time `t + 1`.
# Note that there are no ground truth segmentations - each detection is just a point representing the center of a cell.
#
# Here we load the graph using the `geff` API into a `networkx` graph, a common library for working with graphs in Python. Specifically, we load it into a [`nx.DiGraph`](https://networkx.org/documentation/stable/reference/classes/digraph.html), since our edges are directed.
#

# %%
gt_tracks, metadata = geff.read_nx("data/breast_cancer_fluo.zarr/gt_tracks.geff")
print(f"The ground truth tracks have {gt_tracks.number_of_nodes()} nodes and {gt_tracks.number_of_edges()} edges")

# %% [markdown]
# Here we set up a napari widget for visualizing the tracking results. This is part of the motile tracker, not part of core napari.
# If you get a napari error that the viewer window is closed, please re-run the previous visualization cell to re-open the viewer window.

# %%
widget = TreeWidget(viewer)
viewer.window.add_dock_widget(widget, name="Lineage View", area="right")
tracks_viewer = TracksViewer.get_instance(viewer)

# %% [markdown]
# Here we add a "MotileRun" to the napari tracking visualization widget. A MotileRun includes a name, a graph, and optionally a segmentation. The tracking visualization widget will add:
# - a Points layer with the points in the tracks
# - a Tracks layer to display the track history as a "tail" behind the point in the current time frame
# - a Labels layer, if a segmentation was provided
# - a Lineage View widget, which displays an abstract graph representation of all the solution tracks
#
# These views are synchronized such that every element is colored by the track ID of the element. Clicking on a node in the Lineage View will navigate to that cell in the data, and vice versa.
#
# Hint - if your screen is too small, you can "pop out" the lineage tree view into a separate window using the icon that looks like two boxes in the top left of the lineage tree view. You can also close the tree view with the x just above it, and open it again from the menu bar: Plugins -> Motile Tracker -> Lineage View (then re-run the below cell to add the data to the lineage view).

# %%
ground_truth_run = MotileRun(
    graph=gt_tracks,
    segmentation=None,
    run_name="ground_truth",
    time_attr="t",
    pos_attr=("x", "y"),
    scale=[1, 1, 1],
)

tracks_viewer.update_tracks(ground_truth_run, "ground_truth")

# %% [markdown]
# ## Section 2 (Task 1) Build a candidate graph
#
# To set up our tracking problem, we will create a "candidate graph" - a DiGraph that contains all possible detections (graph nodes) and links (graph edges) between them.
#
# Then we use an optimization method called an integer linear program (ILP) to select the best nodes and edges from the candidate graph to generate our final tracks.
#
# To create our candidate graph, we will use the provided StarDist segmentations.
# Each node in the candidate graph represents one segmentation, and each edge represents a potential link between segmentations. This candidate graph will also contain features that will be used in the optimization task, such as position on nodes and, later, customized scores on edges.


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 1: Extract candidate nodes from the predicted segmentations</h3>
# First we need to turn each segmentation into a node in a <a href=https://networkx.org/documentation/stable/reference/classes/digraph.html>`networkx.DiGraph`</a>.
# Use <a href=https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops>skimage.measure.regionprops</a> to extract properties from each segmentation, and create a candidate graph with nodes only. Check out this networkx <a href=https://networkx.org/documentation/stable/tutorial.html>tutorial</a> for examples of creating a networkx graph.
#
# Here are the requirements for the output graph:
# <ol>
#     <li>Each detection (unique label id) in the segmentation becomes a node in the graph</li>
#     <li>The node id is the label of the detection</li>
#     <li>Each node has an integer "t" attribute, based on the index into the first dimension of the input segmentation array</li>
#     <li>Each node has float "x" and "y" attributes containing the "x" and "y" values from the centroid of the detection region</li>
#     <li>Each node has a "score" attribute containing the probability score output from StarDist. The probability map is at half resolution, so you will need to divide the centroid by 2 before indexing into the probability score.</li>
#     <li>The graph has no edges (yet!)</li>
# </ol>
# </div>

# %% tags=["task"]
def nodes_from_segmentation(segmentation: np.ndarray) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x).

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    print("Extracting nodes from segmentation")
    for t in tqdm(range(len(segmentation))):
        seg_frame = segmentation[t]
        props = skimage.measure.regionprops(seg_frame)
        for regionprop in props:
            ### YOUR CODE HERE ###

    return cand_graph

cand_graph = nodes_from_segmentation(segmentation)


# %% tags=["solution"]
def nodes_from_segmentation(segmentation: np.ndarray) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x).

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    print("Extracting nodes from segmentation")
    for t in tqdm(range(len(segmentation))):
        seg_frame = segmentation[t]
        props = skimage.measure.regionprops(seg_frame)
        for regionprop in props:
            node_id = regionprop.label
            x = float(regionprop.centroid[0])
            y = float(regionprop.centroid[1])
            attrs = {
                "t": t,
                "x": x,
                "y": y,
                "score": float(probabilities[t, int(x // 2), int(y // 2)]),
            }
            assert node_id not in cand_graph.nodes
            cand_graph.add_node(node_id, **attrs)
    return cand_graph

cand_graph = nodes_from_segmentation(segmentation)

# %%
# run this cell to test your implementation of the candidate graph
assert cand_graph.number_of_nodes() == 6123, f"Found {cand_graph.number_of_nodes()} nodes, expected 6123"
assert cand_graph.number_of_edges() == 0, f"Found {cand_graph.number_of_edges()} edges, expected 0"
for node, data in cand_graph.nodes(data=True):
    assert type(node) == int, f"Node id {node} has type {type(node)}, expected 'int'"
    assert "t" in data, f"'t' attribute missing for node {node}"
    assert type(data["t"]) == int, f"'t' attribute has type {type(data['t'])}, expected 'int'"
    assert "x" in data, f"'x' attribute missing for node {node}"
    assert type(data["x"]) == float, f"'x' attribute has type {type(data['x'])}, expected 'float'"
    assert "y" in data, f"'y' attribute missing for node {node}"
    assert type(data["y"]) == float, f"'y' attribute has type {type(data['y'])}, expected 'float'"
    assert "score" in data, f"'score' attribute missing for node {node}"
    assert type(data["score"]) == float, f"'score' attribute has type {type(data['score'])}, expected 'float'"
print("Your candidate graph passed all the tests!")

# %% [markdown]
# We can visualize our candidate points using the napari Points layer. You should see one point in the center of each segmentation when we display it using the below cell.

# %%
points_array = np.array([[data["t"], data["x"], data["y"]] for node, data in cand_graph.nodes(data=True)])
cand_points_layer = napari.layers.Points(data=points_array, name="cand_points")
viewer.add_layer(cand_points_layer)


# %% [markdown]
# ### Adding Candidate Edges
#
# After extracting the nodes, we need to add candidate edges. The `add_cand_edges` function below adds candidate edges to a nodes-only graph by connecting all nodes in adjacent frames that are closer than a given max_edge_distance.
#
# Note: At the bottom of the cell, we add edges to our candidate graph with max_edge_distance=50. This is the maximum number of pixels that a cell centroid will be able to move between frames. If you want longer edges to be possible, you can increase this distance, but solving may take longer.

# %%
def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["t"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict

def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [[cand_graph.nodes[node]["x"], cand_graph.nodes[node]["y"]] for node in node_ids]
    return scipy.spatial.KDTree(positions)

def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    print("Extracting candidate edges")
    node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree

add_cand_edges(cand_graph, max_edge_distance=50)

# %% [markdown]
# Visualizing the candidate edges in napari is, unfortunately, not yet possible. However, we can print out the number of candidate nodes and edges, and compare it to the ground truth nodes and edgesedges. We should see that we have a few more candidate nodes than ground truth (due to false positive detections) and many more candidate edges than ground truth - our next step will be to use optimization to pick a subset of the candidate nodes and edges to generate our solution tracks.

# %%
print(f"Our candidate graph has {cand_graph.number_of_nodes()} nodes and {cand_graph.number_of_edges()} edges")
print(f"Our ground truth track graph has {gt_tracks.number_of_nodes()} nodes and {gt_tracks.number_of_edges()}")


# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 1</h2>
#     We have visualized our data in napari and set up a candidate graph with all possible detections and links that we could select with our optimization task.
#
# We will now together go through the `motile` <a href=https://funkelab.github.io/motile/quickstart.html#sec-quickstart>quickstart</a> example before you actually set up and run your own motile optimization. If you reach this checkpoint early, feel free to start reading through the quickstart and think of questions you want to ask!
# </div>

# %% [markdown]
# ## Section 3 (Task 2): Set up the tracking optimization problem

# %% [markdown]
# As hinted earlier, our goal is to prune the candidate graph. More formally we want to find a graph $\tilde{G}=(\tilde{V}, \tilde{E})$ whose vertices $\tilde{V}$ are a subset of the candidate graph vertices $V$ and whose edges $\tilde{E}$ are a subset of the candidate graph edges $E$.
#
# Finding a good subgraph $\tilde{G}=(\tilde{V}, \tilde{E})$ can be formulated as an [integer linear program (ILP)](https://en.wikipedia.org/wiki/Integer_programming) (also, refer to the tracking lecture slides), where we assign a binary variable $x$ and a cost $c$ to each vertex and edge in $G$, and then computing $min_x c^Tx$.
#
# We can add linear costs for selecting nodes or edges. For example, the EdgeDistance cost $C_d(\tilde{E})$ of a particular selection $\tilde{E}$ is a linear equation $C_d(\tilde{E}) = \sum_{e \in E} x_e (w_d * d_e + c_d)$, where $w_es$ is a manually set weight, $C_es$ is a manually set constant, and $d_e$ is the distance of the two endpoints of that edge.
#
# A set of linear constraints ensures that the solution will be a feasible cell tracking graph. For example, if an edge is part of $\tilde{G}$, both its incident nodes have to be part of $\tilde{G}$ as well.
#
# `motile` ([docs here](https://funkelab.github.io/motile/)), makes it easy to link with an ILP in python by implementing common linking constraints and costs.

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 2: Set up a basic motile tracking pipeline</h3>
# <p>Use the motile <a href=https://funkelab.github.io/motile/quickstart.html#sec-quickstart>quickstart</a> example to set up a basic motile pipeline for our task.
#
# Here are some key similarities and differences between the quickstart and our task:
# <ul>
#     <li>We do not have scores on our edges. However, we can use the edge distance as a cost, so that longer edges are more costly than shorter edges. Instead of using the <code>EdgeSelection</code> cost, we can use the <a href=https://funkelab.github.io/motile/api.html#edgedistance><code>EdgeDistance</code></a> cost with <code>position_attribute=("x", "y")</code>. You will want a positive weight, since higher distances should be more costly, unlike in the example when higher scores were good and so we inverted them with a negative weight.</li>
#     <li>Because distance is always positive, and you want a positive weight, you will want to include a negative constant on the <code>EdgeDistance</code> cost. If there are no negative selection costs, the ILP will always select nothing, because the cost of selecting nothing is zero.</li>
#     <li>We want to allow divisions. So, we should pass in 2 to our <code>MaxChildren</code> constraint. The <code>MaxParents</code> constraint should have 1, the same as the quickstart, because neither task allows merging.</li>
#     <li>You should include an <code>Appear</code> cost and a <code>NodeSelection</code> cost similar to the one in the quickstart.</li>
# </ul>
#
# Once you have set up the basic motile optimization task in the function below, you will probably need to adjust the weight and constant values on your costs until you get a solution that looks reasonable.
#
# </p>
# </div>
#

# %% tags=["task"]
def solve_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    ### YOUR CODE HERE ###
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())

    return solution_graph


# %% tags=["solution"]
def solve_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-1.0, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeDistance(weight=1, constant=-20, position_attribute=("x", "y"))
    )
    solver.add_cost(motile.costs.Appear(constant=2.0))
    solver.add_cost(motile.costs.Split(constant=1.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


# %% [markdown]
# Here is a utility function to gauge some statistics of a solution.

# %%
def print_graph_stats(graph, name):
    print(f"{name}\t\t{graph.number_of_nodes()} nodes\t{graph.number_of_edges()} edges\t{len(list(nx.weakly_connected_components(graph)))} tracks")


# %% [markdown]
# Here we actually run the optimization, and compare the found solution to the ground truth.
#
# <div class="alert alert-block alert-warning"><h4>Gurobi license error</h4>
# Please ignore the warning `Could not create Gurobi backend ...`.
#
#
# Our integer linear program (ILP) tries to use the proprietary solver Gurobi. You probably don't have a license, in which case the ILP will fall back to the open source solver SCIP.
#
# SCIP is slower than Gurobi - to deal with this, we add a 120 second timeout to the solve call, which should approximate the truly optimal solution. For larger problems, or cases where getting the most optimal solution is crucial, one could increase the timeout or get a Gurobi license (recommended).
# </div>

# %%
# run this cell to actually run the solving and get a solution
solution_graph = solve_basic_optimization(cand_graph)

# then print some statistics about the solution compared to the ground truth
print_graph_stats(solution_graph, "solution")
print_graph_stats(gt_tracks, "gt tracks")


# %% [markdown]
# If you haven't selected any nodes or edges in your solution, try adjusting your weight and/or constant values. Make sure you have some negative costs or selecting nothing will always be the best solution!

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 1: Interpret your results based on statistics</h3>
# <p>
# What do these printed statistics tell you about your solution? What else would you like to know?
# </p>
# </div>

# %% [markdown]
# ## Section 4: Visualize the Result
# Rather than just looking at printed statistics about our solution, let's visualize it in `napari`.
#
# Before we can create our MotileRun, we need to create an output segmentation from our solution. Not all candidate nodes will be selected in our solution graph, so we need to filter the masks corresponding to the un-selected candidate detections out of the output segmentation. The motile tracker widget will handle coloring each cell by its track ID automatically.
#
# Note that bad tracking results at this point does not mean that you implemented anything wrong! We still need to customize our costs and constraints to the task before we can get good results. As long as your pipeline selects something, and you can kind of interepret why it is going wrong, that is all that is needed at this point.

# %%
def filter_segmentation(
    solution_nx_graph: nx.DiGraph,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Filter a segmentation to only include detections in the solution graph.

    Args:
        solution_nx_graph (nx.DiGraph): Networkx graph with the solution to use
            for filtering. Nodes not in graph will be removed from seg.
        segmentation (np.ndarray): Original segmentation with dimensions (t,y,x)

    Returns:
        np.ndarray: Filtered segmentation array with shape (t,y,x)
    """
    solution_nodes = set(solution_nx_graph.nodes())
    filtered = segmentation.copy()
    for t in range(len(filtered)):
        mask = np.isin(filtered[t], list(solution_nodes), invert=True)
        filtered[t][mask] = 0
    return filtered

solution_seg = filter_segmentation(solution_graph, segmentation)

# %%
basic_run = MotileRun(
    graph=solution_graph,
    segmentation=solution_seg,
    run_name="basic_solution",
    time_attr="t",
    pos_attr=("x", "y"),
)

tracks_viewer.update_tracks(basic_run, "basic_solution")

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 2: Interpret your results based on visualization</h3>
# <p>
# How is your solution based on looking at the visualization? When is it doing well? When is it doing poorly?
# </p>
# </div>
#

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 2</h2>
# We will discuss the exercise up to this point as a group shortly, and then give a brief overview of quantitative tracking evaluation. If you reach this checkpoint early, you can start looking at the tracking metrics described in the next section.
# </div>

# %% [markdown]
# ## Section 5 (Task 3): Evaluation Metrics
#
# We were able to understand via visualizing the predicted tracks on the images that the basic solution is far from perfect for this problem. Additionally, we would also like to quantify this. We will use the package [`traccuracy`](https://traccuracy.readthedocs.io/en/latest/) to learn about and compute cell tracking metrics. While we looked at the basic documentation together during the checkpoint, take some time now to do a deeper dive into the matchers and metrics available, considering which metrics you might want to focus on for different biological analyses.


# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 3: Matchers and Metrics</h3>
# <p>
# <ul>
#   <li>The example code we provide uses an <a href=https://traccuracy.readthedocs.io/en/latest/matchers/matchers.html>IOU Matcher</a>, which has a hyperparameter of "iou_threshold". How could changing the IOU threshold influence the quantitative output? What other matchers could we have chosen?</li>
#   <li>What metrics would you like to know about for algorithm development? What about for downstream biological analysis?</li>
# </ul>
# </p>
# </div>
#

# %% [markdown]
# The example code below uses an IOU matcher and computes the following metrics:
#
# - **TRA**: TRA is a metric established by the [Cell Tracking Challenge](http://celltrackingchallenge.net). It compares your solution graph to the ground truth graph and measures how many changes to edges and nodes would need to be made in order to make the graphs identical. TRA ranges between 0 and 1 with 1 indicating a perfect match between the solution and the ground truth. While TRA is convenient to use in that it gives us a single number, it doesn't tell us what type of mistakes are being made in our solution.
# - **Node Errors**: We can look at the number of false positive and false negative nodes in our solution which tells us how how many cells are being incorrectly included or excluded from the solution.
# - **Edge Errors**: Similarly, the number of false positive and false negative edges in our graph helps us assess what types of mistakes our solution is making when linking cells between frames.
# - **Division Errors**: Finally, as biologists we are often very interested in the division events that occur and want to ensure that they are being accurately identified. We can look at the number of true positive, false positive and false negative divisions to assess how our solution is capturing these important events.
#


# %% [markdown]
# The metrics we want to compute require a ground truth segmentation. Since we do not have a ground truth segmentation, we can make one by drawing a circle around each ground truth detection. While not perfect, it will be good enough to match ground truth to predicted detections in order to compute metrics.

# %%
from skimage.draw import disk
def make_gt_detections(data_shape, gt_tracks, radius):
    segmentation = np.zeros(data_shape, dtype="uint32")
    frame_shape = data_shape[1:]
    # make frame with one cell in center with label 1
    for node, data in gt_tracks.nodes(data=True):
        pos = (data["x"], data["y"])
        time = data["t"]
        gt_tracks.nodes[node]["label"] = node
        rr, cc = disk(center=pos, radius=radius, shape=frame_shape)
        segmentation[time][rr, cc] = node
    return segmentation

gt_dets = make_gt_detections(data_root["raw"].shape, gt_tracks, 10)

# %% [markdown]
# We then construct two `traccuracy.TrackingGraph` objects, one for ground truth and one for prediction, and then call `traccuracy.run_metrics` with our chosen matcher and metrics. The output is saved into a pandas data frame for easy comparison when we compute multiple solutions.

# %%
import pandas as pd

def get_metrics(gt_graph, labels, run, results_df):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        labels (np.ndarray): Ground truth detections.
        run (MotileRun): Instance of Motilerun
        results_df (pd.DataFrame): Dataframe containing any prior results

    Returns:
        results (pd.DataFrame): Dataframe of evaluation results
    """
    gt_graph = traccuracy.TrackingGraph(
        graph=nx.DiGraph(gt_graph.copy()),
        frame_key="t",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=labels,
    )

    pred_nx = run.graph.copy()
    for node in pred_nx.nodes():
        pred_nx.nodes[node]["label"] = node
    pred_graph = traccuracy.TrackingGraph(
        graph=pred_nx,
        frame_key="t",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=run.segmentation,
    )

    results, _ = traccuracy.run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=IOUMatcher(iou_threshold=0.3, one_to_one=True),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )
    columns = ["fp_nodes", "fn_nodes", "fp_edges", "fn_edges", "TRA", "True Positive Divisions", "False Positive Divisions", "False Negative Divisions", "Wrong Children Divisions"]
    results_filtered = {}
    results_filtered.update(results[0]["results"])
    results_filtered.update(results[1]["results"]["Frame Buffer 0"])
    results_filtered["name"] = run.run_name
    current_result = pd.DataFrame(results_filtered, index=[0])[["name"] + columns]

    if results_df is None:
        results_df = current_result
    else:
        results_df = pd.concat([results_df, current_result])

    return results_df


# %%
results_df = None
results_df = get_metrics(gt_tracks, gt_dets, basic_run, results_df)
results_df


# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 4: Interpret your results based on metrics</h3>
# <p>
# What additional information, if any, do the metrics give you compared to the statistics and the visualization?
# </p>
# </div>
#

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 3 (Optional): Adapt the above code to use your preferred matchers and metrics</h3>
# <p>If there are additional metrics you found interesting from the documentation, you can try to add them now! The <a href=https://traccuracy.readthedocs.io/en/latest/metrics/ctc.html#ctc-bio-metrics>CTC Bio metrics</a> are an easy option to add, since they require the same matching we are already doing. You can also vary the IOUThreshold and see how it affects the scores reported. If you want a challenge, you can try using a different <a href=https://traccuracy.readthedocs.io/en/latest/matchers/matchers.html>matcher</a> or adding the <a href=https://traccuracy.readthedocs.io/en/latest/metrics/track_overlap.html>TrackOverlap</a> or <a href=https://traccuracy.readthedocs.io/en/latest/metrics/chota.html>CHOTA</a> metrics.</p>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 3</h2>
# If you reach this checkpoint with extra time, think about what kinds of improvements you could make to the costs and constraints to fix the issues that you are seeing. You can try tuning your weights and constants, or adding or removing motile Costs and Constraints, and seeing how that changes the output. We have added a convenience function in the box below where you can copy your solution from above, adapt it, and run the whole pipeline including visualization and metrics computation.
#
# Do not get frustrated if you cannot get good results yet! Try to think about why and what custom costs we might add.
# </div>

# %% tags=["task"]
def adapt_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    ### YOUR CODE HERE ###
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())

    return solution_graph

def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = adapt_basic_optimization(cand_graph)
    solution_seg = filter_segmentation(solution_graph, segmentation)
    run = MotileRun(
        graph=solution_graph,
        segmentation=solution_seg,
        run_name=run_name,
        time_attr="t",
        pos_attr=("x", "y"),
    )
    tracks_viewer.update_tracks(run, run_name)
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run below, so you can tell them apart in the results table
results_df = run_pipeline(cand_graph, "basic_solution_2", results_df)
results_df


# %% tags=["solution"]
def adapt_basic_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-5.0, constant=2.5, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeDistance(weight=1, constant=-20, position_attribute=("x", "y"))
    )
    solver.add_cost(motile.costs.Appear(constant=20.0))
    solver.add_cost(motile.costs.Split(constant=15.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())

    return solution_graph

def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = adapt_basic_optimization(cand_graph)
    solution_seg = filter_segmentation(solution_graph, segmentation)
    run = MotileRun(
        graph=solution_graph,
        segmentation=solution_seg,
        run_name=run_name,
        time_attr="t",
        pos_attr=("x", "y"),
    )
    tracks_viewer.update_tracks(run, run_name)
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

results_df = run_pipeline(cand_graph, "basic_solution_2", results_df)
results_df


# %% [markdown]
# ## Section 6 (Task 4): Incorporating prior knowledge
#
# There 3 main ways to encode prior knowledge about your task into the motile tracking pipeline.
# 1. Add an attribute to the candidate graph and incorporate it with an existing cost
# 2. Change the structure of the candidate graph
# 3. Add a new type of cost or constraint
#
# The first way is the most common, and is quite flexible, so we will focus on an example of this type of customization.

# %% [markdown]
# ### Incorporating Known Direction of Motion
#
# So far, we have been using motile's EdgeDistance as an edge selection cost, which penalizes longer edges by computing the Euclidean distance between the endpoints. However, in our dataset we see a trend of upward motion in the cells, and the false detections at the top are not moving. If we penalize movement based on what we expect, rather than Euclidean distance, we can select more correct cells and penalize the non-moving artefacts at the same time.
#
#

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4a: Add a drift distance attribute</h3>
# <p> For this task, we need to determine the "expected" amount of motion, then add an attribute to our candidate edges that represents distance from the expected motion direction. Look at the dataset in `napari` and see how much the cells move on average, and in which direction, to get the expected "drift" quantity. The average edge direction in the ground truth annotations could also be used to verify the drift distance, but since we are evaluating on this dataset as well, that could be considered "cheating."</p>
# <p>Notes on the networkx API: An edge with an id (0, 1) indicates that it connects nodes 0 and 1. You can access the attributes of those nodes using cand_graph.nodes[0] and cand_graph.nodes[1]. When we added candidate nodes to the graph earlier, we gave each node the properties x, y, and score. Checkout the networkx <a href=https://networkx.org/documentation/stable/tutorial.html#adding-attributes-to-graphs-nodes-and-edges>tutorial</a> if you need some examples of how to manipulate networkx graphs.
# </p>
# </div>

# %% tags=["task"]
x_drift = ...  ### YOUR CODE HERE ###
y_drift = ...  ### YOUR CODE HERE ###
drift = np.array([y_drift, x_drift])

def add_drift_dist_attr(cand_graph, drift):
    for edge in cand_graph.edges():
        ### YOUR CODE HERE ###
        # get the location of the endpoints of the edge
        # then compute the distance between the expected movement and the actual movement
        # and save it in the "drift_dist" attribute (below)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

add_drift_dist_attr(cand_graph, drift)

# %% tags=["solution"]
drift = np.array([-10, 0])

def add_drift_dist_attr(cand_graph, drift):
    for edge in cand_graph.edges():
        source, target = edge
        source_data = cand_graph.nodes[source]
        source_pos = np.array([source_data["x"], source_data["y"]])
        target_data = cand_graph.nodes[target]
        target_pos = np.array([target_data["x"], target_data["y"]])
        expected_target_pos = source_pos + drift
        drift_dist = np.linalg.norm(expected_target_pos - target_pos)
        cand_graph.edges[edge]["drift_dist"] = drift_dist

add_drift_dist_attr(cand_graph, drift)


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 4b: Add a drift distance attribute</h3>
# <p> Now, we set up yet another solving pipeline. This time, we will replace our EdgeDistance
# cost with an EdgeSelection cost using our new "drift_dist" attribute. The weight should be positive, since a higher distance from the expected drift should cost more, similar to our prior EdgeDistance cost. Also similarly, we need a negative constant to make sure that the overall cost of selecting tracks is negative.</p>
# </div>

# %% tags=["task"]
def solve_drift_optimization(cand_graph):
    """Set up and solve the network flow problem.

    Args:
        cand_graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)

    ### YOUR CODE HERE ###

    solver.solve(timeout=120)

    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = solve_drift_optimization(cand_graph)
    solution_seg = filter_segmentation(solution_graph, segmentation)
    run = MotileRun(
        graph=solution_graph,
        segmentation=solution_seg,
        run_name=run_name,
        time_attr="t",
        pos_attr=("x", "y"),
    )
    tracks_viewer.update_tracks(run, run_name)
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run if you re-run this cell!
results_df = run_pipeline(cand_graph, "drift_dist", results_df)
results_df


# %% tags=["solution"]
def solve_drift_optimization(cand_graph):
    """Set up and solve the ILP

    Args:
        cand_graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-100, constant=75, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeSelection(weight=1.0, constant=-30, attribute="drift_dist")
    )
    solver.add_cost(motile.costs.Appear(constant=40.0))
    solver.add_cost(motile.costs.Split(constant=45.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = solve_drift_optimization(cand_graph)
    solution_seg = filter_segmentation(solution_graph, segmentation)
    run = MotileRun(
        graph=solution_graph,
        segmentation=solution_seg,
        run_name=run_name,
        time_attr="t",
        pos_attr=("x", "y"),
    )
    tracks_viewer.update_tracks(run, run_name)
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run if you re-run this cell!
results_df = run_pipeline(cand_graph, "drift_dist", results_df)
results_df

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 5</h3>
# If you have picked good weights, this approach should generally do better than the previous distance based approach. Don't forget to look at the results visually to qualitatively evaluate your solutions!
# <ul>
#   <li>On what metrics is it better? On what metrics might it be worse? What do you see visually about where your model does well or poorly?</li>
#   <li>What are other situations where you can incorporate prior knowledge about the dataset into the costs and weights you choose to add? In what situations is this difficult?</li>
# </ul>
# </div>

# %% [markdown]
# ## Section 7 (Task 5) - Incorporating Trackastra Scores
#
# [Trackastra](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf) is a transformer-based method for cell tracking. The method trains a transformer to predict an association score for each possible edge in the candidate graph, and then uses that score combined with distance to perform linking. The [trackastra package](https://github.com/weigertlab/trackastra) has published a general 2D model trained on a variety of datasets, which we will use to predict edge scores for our candidate graph. We will then incorporate the scores into our ILP in a similar fashion to our hand-crafted drift distance.

# %%
# download the pretrained model
model = trackastra.model.Trackastra.from_pretrained("general_2d", device="automatic")
# predict
predictions = model._predict(image_data, segmentation)
trackastra_nodes = predictions["nodes"]
trackastra_scores = predictions["weights"]
# show representative outputs
print("Example node output:", trackastra_nodes[0])
print("Example score output", trackastra_scores[0])


# %% [markdown]
# You can see that the trackastra model prediction outputs a set of nodes. Each node has:
# - `id`, the trackastra defined identifier for the node
# - `coords`, the location of the node in space
# - `time`, the time frame of the node
# - `label`, the label of the segmentation that was used to create the node
#
# Trackastra likely does something very similar to the regionprops approach you used in Task 1 to get the nodes from the segmentation labels. However, there is an obvious difference: we used the label as the node id, because we knew our segmentation labels did not repeat across time. Since Trackastra does not assume this, it assigns a new `id` for each node.
#
# Trackastra also outputs a list of scores. Each score has:
# - A tuple of node IDs, corresponding to the `id` field in the nodes list
# - A float association score between 0 and 1, with higher values indicating that the model believes the nodes are the same or mother/daughter cells, and a low value indicating the model does not think the cells are associated. By default, scores below 0.05 are not included, although this setting can be changed.
#
# The code below adds a "trackastra_score" attribute to each edge in our candidate graph.

# %%
def add_trackastra_score_attr(cand_graph: nx.DiGraph, trackastra_nodes, trackastra_scores):
    # create a mapping from trackastra node ids to our node ids (which are the segmentation label)
    node_id_map = {
        node["id"]: int(node["label"]) for node in trackastra_nodes
    }
    # find the candidate edge for each predicted score and add the attribute
    for edge, score in trackastra_scores:
        source, target = edge
        cand_source = node_id_map[int(source)]
        cand_target = node_id_map[int(target)]
        if cand_graph.has_edge(cand_source, cand_target):
            cand_graph.edges[(cand_source, cand_target)]["trackastra_score"] = score

    # add a score of 0 to all edges that were not included in the trackastra predictions
    for source, target, data in cand_graph.edges(data=True):
        if "trackastra_score" not in data:
            cand_graph.edges[(source, target)]["trackastra_score"] = 0

# run the function to add the predicted trackastra scores to our candidate graph
add_trackastra_score_attr(cand_graph, trackastra_nodes, trackastra_scores)


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 5: Solve with trackastra scores</h3>
# <p> Now that our candidate graph contains trackastra scores, we set up our final solving pipeline! You should include an EdgeSelection cost based on the "trackastra_score" attribute. Trackastra scores are between 0 and 1, with higher scores being better. Should the weight be positive or negative? Remember, we are minimizing the total cost, so we will pick the edges that have the smallest/most negative cost. </p>
# <p>You can choose what other costs (if any) to combine with the trackastra score, and how to weight them against each other. </p>
# </div>

# %% tags=["task"]
def solve_trackastra_optimization(cand_graph):
    """Set up and solve the ILP

    Args:
        cand_graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)

    ### YOUR CODE HERE ###
    
    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = solve_trackastra_optimization(cand_graph)
    solution_seg = filter_segmentation(solution_graph, segmentation)
    run = MotileRun(
        graph=solution_graph,
        segmentation=solution_seg,
        run_name=run_name,
        time_attr="t",
        pos_attr=("x", "y"),
    )
    tracks_viewer.update_tracks(run, run_name)
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run if you re-run this cell!
results_df = run_pipeline(cand_graph, "trackastra", results_df)
results_df


# %% tags=["solution"]
def solve_trackastra_optimization(cand_graph):
    """Set up and solve the ILP

    Args:
        cand_graph (nx.DiGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelection(weight=-100, constant=75, attribute="score")
    )
    solver.add_cost(
        motile.costs.EdgeSelection(weight=1.0, constant=-30, attribute="drift_dist"), name="drift"
    )
    solver.add_cost(
        motile.costs.EdgeSelection(weight=-50, constant=25, attribute="trackastra_score"), name="trackastra"
    )
    solver.add_cost(motile.costs.Appear(constant=40.0))
    solver.add_cost(motile.costs.Split(constant=45.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.solve(timeout=120)
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


def run_pipeline(cand_graph, run_name, results_df):
    solution_graph = solve_trackastra_optimization(cand_graph)
    solution_seg = filter_segmentation(solution_graph, segmentation)
    run = MotileRun(
        graph=solution_graph,
        segmentation=solution_seg,
        run_name=run_name,
        time_attr="t",
        pos_attr=("x", "y"),
    )
    tracks_viewer.update_tracks(run, run_name)
    results_df = get_metrics(gt_tracks, gt_dets, run, results_df)
    return results_df

# Don't forget to rename your run if you re-run this cell!
results_df = run_pipeline(cand_graph, "trackastra", results_df)
results_df

# %% [markdown]
# <div class="alert alert-block alert-warning"><h3>Question 6</h3>
# <ul>
#   <li>How do the learned Trackastra scores compare to your hand crafted scores? </li>
#   <li>What types of mistakes does your best model make?</li>
#   <li>How could you improve the results even further? </li>
# </ul>
# </div>

# %% [markdown]
# <div class="alert alert-block alert-success"><h2>Checkpoint 4</h2>
# That is the end of the main exercise! If you have extra time, feel free to keep tuning your costs, examining the metrics and visualization tools, or try the bonus trackmate exercise.

# %%
