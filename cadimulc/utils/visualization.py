import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_array


def draw_graph_from_ndarray(array, graph_type="auto", save_fig=False, testing_text=None):
    """ implementation / networkx graph

    Notes
    -----
    * input: ndarray, dataframe, or networkx graph
    * nodes name: X1,...,Xd
    * Auto direct / indirect
    """

    # ensure / convert into nx.graph
    if not isinstance(array, np.ndarray):
        raise TypeError("The expected type of input object should be numpy array.")

    if graph_type == 'auto':
        if (array == array.T).all():
            directed = False
        else:
            directed = True
    else:
        directed = graph_type

    # directed graph
    if directed:
        # conventional causal direction / transpose of matrix
        G = nx.from_numpy_array(array.T, create_using=nx.DiGraph)

        # rename graph node name
        node_id = 0
        for node in G.nodes():
            rename_node = f'X{node_id + 1}'
            G = nx.relabel_nodes(G, mapping={node: rename_node})
            node_id += 1

        # fix position
        pos = nx.circular_layout(G)
        # setting parameters
        plt.figure(figsize=(5, 2.7), dpi=120)
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

        if testing_text is not None:
            plt.text(x=0.5, y=0.5, s=testing_text, fontsize=12, color='red')
            print('* Figure Label: ', testing_text)

    # undirected graph
    else:
        # conventional causal direction / transpose of matrix
        G = nx.from_numpy_array(array.T)

        # rename graph node name
        node_id = 0
        for node in G.nodes():
            rename_node = f'X{node_id + 1}'
            G = nx.relabel_nodes(G, mapping={node: rename_node})
            node_id += 1

        # fix position
        pos = nx.circular_layout(G)
        # setting parameters
        plt.figure(figsize=(5, 2.7), dpi=120)
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
        plt.savefig()