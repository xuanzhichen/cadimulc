"""Description"""


# ### DEVELOPMENT NOTES (LEAST) ############################################
# * None


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Retesting dag generation for randomly fixing issues.      12th.Jan, 2024
#
# * Fixed bug of null values of test_data_generation()        11th.Dec, 2023
#
# * test_data_generation() has entered the runnable phase where null values
#   exist in the non-examined dataset.                         7th.Dec, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# None
#
# Done:
# TODO: Integrate testing logics of dag generation and data generation. Rea-
#       dy to go back to test_hybrid_algorithms.py.
# TODO: Find out the reason of null values on the non-examined dataset.


# import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from cadimulc.utils.visualization import draw_graph_from_ndarray
from cadimulc.utils.extensive_modules import convert_graph_type

from cadimulc.utils.generation import Generator


# #########################################################################
# ### AUXILIARY FUNCTION(S) ###############################################
# #########################################################################

# ### SUBORDINATE COMPONENT(S) ############################################
# test_data_generation()
def check_data_generation(graph_node_num, sample, dag, random_seed):
    """
    Write down some descriptions here.

    Parameters
    ----------
    graph_node_num : int
    sample : int
    dag : ndarray
    random_seed : int

    Return
    ------
    data_examined : ndarray
    """

    np.random.seed(random_seed)

    generator = Generator(
        graph_node_num=graph_node_num,
        sample=sample,
    )
    generator.dag = dag

    generator._generate_data()
    data_examined = generator.data

    return data_examined


# #########################################################################
# ### TEST SECTION ########################################################
# #########################################################################

ACTIVATION_0x1 = {
    'testing_part_one':   False,
    'testing_part_two':   True,
    'testing_part_three': False,
    'testing_part_four':  False,
}


# ### CORRESPONDING TEST ##################################################
# generation.py > _generate_dag()
def test_dag_generation():
    """
    Construct a diamond-shaped toy skeleton, then ...
    display the graph orientation procedure.

    Namely, perform two times permutation to ensure random graph creation.
    """

    # === PART ONE ========================================================

    if ACTIVATION_0x1['testing_part_one']:
        # fix permutation
        np.random.seed(42)

        # Construct a diamond-shaped toy skeleton.
        undigraph = nx.Graph()
        undigraph.add_edge(u_of_edge='X1', v_of_edge='X2')
        undigraph.add_edge(u_of_edge='X1', v_of_edge='X3')
        undigraph.add_edge(u_of_edge='X2', v_of_edge='X4')
        undigraph.add_edge(u_of_edge='X3', v_of_edge='X4')

        # convert / numerical computation
        print("\n\nPreview: The diamond-shaped toy skeleton")
        undigraph = convert_graph_type(undigraph, np.ndarray)
        draw_graph_from_ndarray(array=undigraph, testing_text='SKELETON')
        print('figure label: SKELETON \n')

        # Generate a permutation matrix in preparation for the undirected graph
        # represented as a matrix.
        permu_mat = np.random.permutation(np.eye(undigraph.shape[0]))

        # first permutation (after an undirected graph)
        print("Step one: Display the graph after the first permutation.")
        print("\n\nStep one: Display the graph after the first permutation.")
        graph_temp1 = permu_mat.T @ undigraph @ permu_mat
        draw_graph_from_ndarray(array=graph_temp1, testing_text='PERMUTATION-1')
        print("figure label: PERMUTATION-1 \n")

        # extract the lower triangle part of graph_temp1
        # by excluding the main diagonal and elements above it
        print("Step two: undetermined")
        graph_temp2 = np.tril(graph_temp1, -1)
        draw_graph_from_ndarray(array=graph_temp2, testing_text='LOWER TRIANGLE')
        print("figure label: LOWER TRIANGLE \n")

        # second permutation (after a directed graph)
        print("Step three: Display the graph after the second permutation.")
        permu_mat2 = np.random.permutation(np.eye(undigraph.shape[0]))
        graph_temp3 = permu_mat2.T @ graph_temp2 @ permu_mat2
        draw_graph_from_ndarray(array=graph_temp3, testing_text='PERMUTATION-2')
        print("figure label: PERMUTATION-2 \n")

        plt.show()

    # === PART TWO ========================================================

    if ACTIVATION_0x1['testing_part_two']:
        np.random.seed(42)
        random.seed(42)

        generator = Generator(
            graph_node_num=5,
            sample=42,
        )

        generator._generate_dag(sparsity=0.7)
        skeleton = generator.skeleton
        dag = generator.dag

        draw_graph_from_ndarray(skeleton)
        draw_graph_from_ndarray(dag)

        plt.show()

    # === PART THREE ======================================================

    if ACTIVATION_0x1['testing_part_three']:
        # np.random.seed(42)

        random.seed(42)

        generator = Generator(
            graph_node_num=5,
            sample=42
        )

        undigraph = generator._get_undigraph(
            graph_node_num=generator.graph_node_num,
            sparsity=0.7
        )

        draw_graph_from_ndarray(undigraph)
        plt.show()

    # === PART FOUR =======================================================

    if ACTIVATION_0x1['testing_part_four']:
        # np.random.seed(42)

        random.seed(42)

        undigraph_nx = nx.random_graphs.erdos_renyi_graph(
            n=5,
            p=0.7
        )
        undigraph_np = nx.to_numpy_array(undigraph_nx)

        draw_graph_from_ndarray(undigraph_np)
        plt.show()


# ### CORRESPONDING TEST ##################################################
# generation.py > _generate_data()

# ### AUXILIARY COMPONENT(S) ##############################################
# check_data_generation()
def test_data_generation():
    """
    Construct a diamond-shaped toy skeleton, then ...

    Write down some descriptions here.
    """

    random_seed = 42

    # ### PART ONE ########################################################
    np.random.seed(random_seed)

    digraph = nx.DiGraph()
    graph_node_num = 4
    sample = 100
    data = np.zeros([sample, graph_node_num])

    digraph.add_edge(u_of_edge='X1', v_of_edge='X2')
    digraph.add_edge(u_of_edge='X1', v_of_edge='X3')
    digraph.add_edge(u_of_edge='X2', v_of_edge='X4')
    digraph.add_edge(u_of_edge='X3', v_of_edge='X4')

    topo_order = list(nx.topological_sort(digraph))

    for child_index, child_var in enumerate(topo_order):
        parent_vars = list(digraph.predecessors(child_var))
        parent_indexes = [topo_order.index(var) for var in parent_vars]

        print()
        print("Child index: ", child_index)
        print("Child variable", child_var)
        print("Parent indexe(s): ", parent_indexes)
        print("Parent variable(s)", parent_vars)

        if child_var == 'X1':
            assert parent_vars == []
        elif (child_var == 'X2') or (child_var == 'X3'):
            assert parent_vars == ['X1']
        else:
            assert parent_vars == ['X2', 'X3']

        # 'Non-Gaussian'
        data[:, child_index] += np.random.uniform(low=np.negative(10), high=10, size=sample)
        print("\nVariable {}: Additive noise done.".format(child_var))

        if len(parent_vars) > 0:
            for parent_var, parent_index in zip(parent_vars, parent_indexes):
                corr = round(np.random.uniform(low=0.3, high=0.5), 3)

                nonlinear_label = np.random.randint(low=1, high=3 + 1)
                if nonlinear_label == 1:
                    data[:, child_index] += corr * np.sin(data[:, parent_index])
                elif nonlinear_label == 2:
                    data[:, child_index] += corr * np.sqrt(np.abs(data[:, parent_index]))
                else:
                    data[:, child_index] += corr * np.power(data[:, parent_index], 3)

                print("Variable {}: Non-linear effects (label={}) from parent(s) {} done."
                      .format(child_var, nonlinear_label, parent_var))

    data = data / np.std(data, axis=0)

    # ### PART TWO ########################################################
    digraph = convert_graph_type(origin=digraph, target=np.ndarray)
    data_examined = check_data_generation(
        graph_node_num=graph_node_num,
        sample=sample,
        dag=digraph.T,
        random_seed=random_seed
    )

    # ### PART THREE ######################################################
    np.testing.assert_equal(actual=data, desired=data_examined)





