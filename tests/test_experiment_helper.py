"""Description"""

# ### DEVELOPMENT NOTES ####################################################
# * None


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Continuing a final temporal test of encapsulation.        22nd.Dec, 2023
#
# * Testing is end, backing to exp.helper's primary function. 18th.Dec, 2023
#
# * Testing is nearing its end.                               14th.Dec, 2023
#
# * Corresponding test for run_generation_procedure() is in progress while
#   corresponding content of programming refers to "part-2".  12th.Dec, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: Complete part-2 and testing synchronously according to the current
#       workflow.
# TODO: Summary common testing logistic, including dividing the explicit
#       display of information and global argument. (Optional)
#
# Done:
# TODO: For the final testing stage, check the graph representation by visu-
#       alizing


import numpy as np
import pandas as pd
import scipy.io
import copy as cp
import matplotlib.pyplot as plt

from cadimulc.utils.extensive_modules import (
    display_test_section_symbols,
    copy_and_rename
)
from cadimulc.utils.visualization import draw_graph_from_ndarray

from cadimulc.utils.experiment_helper import ExperimentHelper


ACTIVATION = {
    'testing-1': False,
    'testing-2': False,
    'testing-3': False,
    'testing-4': True,
}


# ### CORRESPONDING TEST ##################################################
# experiment_helper.py > run_generation_procedure()
def test_run_generation_procedure():
    """
    Descriptions:

    Prerequisites:

    * testing-1：none
    * testing-3：none
    """

    # === TEST FOR LOADING FMRI DATASET ===================================

    # --- SEPARATION ------------------------------------------------------

    file_name = '../paper_2023/simulated_fmri/dataset_netsim/sim2.mat'
    # file_name = '../paper_2023/simulated_fmri/dataset_netsim/sim3.mat'

    mat = scipy.io.loadmat(file_name=file_name)

    # --- SEPARATION ------------------------------------------------------

    if ACTIVATION['testing-1']:
        display_test_section_symbols()

        print("\ndisplay information")

        print("mat[Nsubjects]:   ", mat["Nsubjects"][0][0])
        print("mat[Nnodes]:      ", mat["Nnodes"][0][0])
        print("mat[Ntimepoints]: ", mat["Ntimepoints"][0][0])
        # adjacency matrix
        print("mat[net] (info):  ", "type: ", type(mat["net"]), " shape: ", mat["net"].shape)
        # dataset
        print("mat[ts] (info):   ", "type: ", type(mat["ts"]), " shape: ", mat["ts"].shape)

    # === TEST FOR (DATA PROCESSING) ======================================

    # --- SEPARATION ------------------------------------------------------

    # Since there are 200 pieces of sample (namely, time points) for one
    # individual (subject), sample size should be a multiple of 200.
    sample = 1000
    subjects_num = mat["Nsubjects"][0][0]
    timepoints_num = mat["Ntimepoints"][0][0]

    subjects_selected = np.random.randint(
        low=0,
        high=subjects_num - 1,
        size=int(sample / timepoints_num)
    )

    fragments_list = []
    for subject_index in subjects_selected:
        offset = timepoints_num * subject_index
        fmri_data_fragment = pd.DataFrame(mat["ts"]).iloc[offset:offset + timepoints_num, :]
        fragments_list.append(fmri_data_fragment)

    original_dataset = pd.concat(fragments_list)

    # --- SEPARATION ------------------------------------------------------

    if ACTIVATION['testing-2']:
        display_test_section_symbols()

        print("Selected subjects: ", subjects_selected)
        print("Offset: ", subjects_selected[0] * timepoints_num)
        print()
        print(original_dataset.head())

    # === TEST FOR (DATA PROCESSING) ======================================

    # --- SEPARATION ------------------------------------------------------

    adjacency_matrix = mat["net"]
    original_fmri_dag = (adjacency_matrix[0] > 0).astype(int).T

    # latent_var_id = [0, 5]
    latent_var_id = []

    if len(latent_var_id) == 0:
        processed_dataset = cp.copy(original_dataset.values)
        processed_fmri_dag = cp.copy(original_fmri_dag)
    else:
        processed_dataset = cp.copy(
            original_dataset.drop(columns=latent_var_id).values
        )
        processed_fmri_dag_temp = np.delete(original_fmri_dag, obj=latent_var_id, axis=0)
        processed_fmri_dag = np.delete(processed_fmri_dag_temp, obj=latent_var_id, axis=1)

    # --- SEPARATION ------------------------------------------------------

    if ACTIVATION['testing-3']:
        display_test_section_symbols(testing_mark='testing-3')

        print(original_dataset.head())
        draw_graph_from_ndarray(array=original_fmri_dag, testing_text='original_dag')

        display_test_section_symbols()

        print(pd.DataFrame(processed_dataset).head())
        draw_graph_from_ndarray(array=processed_fmri_dag, testing_text='processed_dag')

        plt.show()

    # === TEST FOR (DATA PROCESSING) ======================================

    if ACTIVATION['testing-4']:
        experiment_helper = ExperimentHelper(
            fmri_dataset_path=file_name,
            latent_var_set=('X0', 'X5', 'X10'),
            sample=sample
        )

        processed_dataset_examined, processed_fmri_dag_examined = \
            experiment_helper.run_generation_procedure(
                hidden_num=2
            ).unpack()

        # --- SEPARATION ------------------------------------------------------

        display_test_section_symbols(testing_mark='testing-4')

        processed_fmri_dag_expected = copy_and_rename(processed_fmri_dag)
        processed_dataset_expected = copy_and_rename(processed_dataset)

        assert processed_fmri_dag_examined == processed_fmri_dag_expected
        np.testing.assert_equal(
            actual=processed_dataset_expected,
            desired=processed_dataset_examined
        )

