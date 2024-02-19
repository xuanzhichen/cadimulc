"""Description"""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License
# Test: See each methods if test is necessary

# ### DEVELOPMENT NOTES (LEAST) ############################################
# * None

# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# *

# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: conduct_ind_test() add branch 'HSIC-Fisher' (for MLC-LiNGAM)
#
# Done:
# None


import numpy as np
import pandas as pd
import networkx as nx
import copy as cp

from causallearn.search.FCMBased.lingam.hsic2 import hsic_gam
from causallearn.utils.KCI.KCI import KCI_UInd


def check_1dim_array(X):
    n = X.shape[0]
    return X.reshape(n, 1) if len(X.shape) == 1 else X


def copy_and_rename(object):
    """ literal function
    """
    return cp.copy(object)


def display_test_section_symbols(testing_mark=None):
    total_length = len(
        '============================= test session starts '
        '============================='
    )

    if testing_mark is not None:
        mark_length = len(testing_mark)
        marginal_length = total_length - mark_length - len('  ')
        equal_signs = '=' * (marginal_length // 2)

        print('\n' + equal_signs + ' ' + testing_mark.upper() + ' ' + equal_signs + '\n')

    else:
        print('\n' + ('-' * (total_length - len('  '))) + '\n')


def cancel_test_duplicates():
    """
    Function an empty jump to cancel duplicates, usually occurs in test.
    """
    pass


def get_medium_num(input_list):
    sorted_list = sorted(input_list)
    list_length = len(sorted_list)
    if list_length % 2 == 0:
        medium_number = (sorted_list[list_length // 2 - 1] + sorted_list[list_length // 2]) / 2
    else:
        medium_number = sorted_list[list_length // 2]
    return medium_number


def convert_graph_type(origin, target):
    """ convert into a targeted data type

    Notes
    -----
    * Typical data types representing a causal graph involves ndarray, dataframe,
      and networkx graph.
    * nodes name: X1,...,Xd

    Parameters
    ----------
    origin : object
    target : class
        np.ndarray, pd.DataFrame, and nx.Graph

    Returns
    -------
    converted_graph : object
        targeted object / ndarray, dataframe, and networkx graph
    """

    def convert_graph_from_np_to_df():
        pass

    def convert_graph_from_np_to_nx():
        pass

    def convert_graph_from_df_to_np():
        pass

    def convert_graph_from_df_to_nx():
        pass

    def convert_graph_from_nx_to_np(original_graph):
        """ Convert the networkx graph to a ndarray.
        """
        converted_graph = nx.to_numpy_array(original_graph)
        return converted_graph

    def convert_graph_from_nx_to_df():
        pass

    if isinstance(origin, np.ndarray) and (target == pd.DataFrame):
        original_graph = copy_and_rename(origin)
        return convert_graph_from_np_to_df(original_graph)

    if isinstance(origin, np.ndarray) and (target == nx.Graph):
        original_graph = copy_and_rename(origin)
        return convert_graph_from_np_to_nx(original_graph)

    if isinstance(origin, pd.DataFrame) and (target == np.ndarray):
        original_graph = copy_and_rename(origin)
        return convert_graph_from_df_to_np(original_graph)

    if isinstance(origin, pd.DataFrame) and (target == nx.Graph):
        original_graph = copy_and_rename(origin)
        return convert_graph_from_df_to_nx(original_graph)

    if isinstance(origin, nx.Graph) and (target == np.ndarray):
        original_graph = copy_and_rename(origin)
        return convert_graph_from_nx_to_np(original_graph)

    if isinstance(origin, nx.Graph) and (target == pd.DataFrame):
        original_graph = copy_and_rename(origin)
        return convert_graph_from_nx_to_df(original_graph)


def get_skeleton_from_adjmat(adjacency_matrix):
    """
    both ndarray, 0/1 adjacency matrix
    """

    dim = adjacency_matrix.shape[0]
    skeleton = cp.copy(adjacency_matrix)

    for i in range(dim):
        for j in range(dim):
            if (adjacency_matrix[i][j] == 1) and \
               (adjacency_matrix[j][i] == 0):
                skeleton[j][i] = 1

    return skeleton