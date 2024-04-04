from __future__ import annotations
from numpy import ndarray

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def draw_graph_from_ndarray(
    array: ndarray,
    graph_type: str = "auto",
    rename_nodes: list | None = None,
    testing_text: str | None = None,
    save_fig: bool = False,
    saving_path: str | None = None
):
    """
    Draw the directed or undirected (causal) graph that is in form of adjacency matrix
    (implementation based on NetworkX).

    Parameters:
        array: the causal graph (directed) or causal skeleton (indirected) in form of adjacency matrix.
        graph_type: use `directed` to forcedly plot a directed graph.
        rename_nodes: Rename the nodes consisting with the column of dataset (n * d).
        Default as "X1,...Xd".
        testing_text: Add simple text to the figure.
        save_fig: Specify saving a figure or not. Make sure to enter the saving path if you specify `save_fig=True`.
        saving_path: The image saving path along with your image file name. e.g. ../file_location/image_file_name.
    """

    # ensure / convert into nx.graph
    if not isinstance(array, np.ndarray):
        raise TypeError("The expected type of input object should be numpy array.")

    if graph_type == 'auto':
        if (array == array.T).all():
            directed = False
        else:
            directed = True
    elif graph_type == 'directed':
        directed = True
    else:
        raise ValueError("Choose the graph type as 'auto' or 'directed'.")

    # directed graph
    if directed:
        # conventional causal direction / transpose of matrix
        G = nx.from_numpy_array(array.T, create_using=nx.DiGraph)
    # undirected graph
    else:
        # conventional causal direction / transpose of matrix
        G = nx.from_numpy_array(array.T)

    # rename graph node name
    if rename_nodes is None:
        node_id = 0
        for node in G.nodes():
            rename_node = f'X{node_id + 1}'
            G = nx.relabel_nodes(G, mapping={node: rename_node})
            node_id += 1
    else:
        for node, rename_node in zip(G.nodes(), rename_nodes):
            rename_node = str(rename_node)
            G = nx.relabel_nodes(G, mapping={node: rename_node})

    # fix position
    pos = nx.circular_layout(G)
    # setting parameters
    plt.figure(figsize=(5, 2.7), dpi=120)

    if directed:
        nx.draw(
            G=G,
            pos=pos,
            with_labels=True,
            node_color='black',
            font_color='white',
            font_size=15,
            width=1.25,
            arrowsize=20,
            node_size=500
        )

    else:
        nx.draw(
            G=G,
            pos=pos,
            with_labels=True,
            node_color='black',
            font_color='white',
            font_size=15,
            width=1.25,
            node_size=500
        )

    if testing_text is not None:
        plt.text(x=0.5, y=0.5, s=testing_text, fontsize=12, color='red')
        print('* Figure Label: ', testing_text)

    if save_fig:
        plt.savefig(
            saving_path + ".png",
            dpi=200,
            transparent=False,
            bbox_inches="tight"
        )

