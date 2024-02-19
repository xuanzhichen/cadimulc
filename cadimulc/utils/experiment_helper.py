"""Experiment helper"""

# Author: Xuanzhi CHEN <xuanzhichen.42@gmail.com>
# License: MIT License

# ### DEVELOPMENT NOTES ####################################################
# * None


# ### DEVELOPMENT PROGRESS (LEAST) #########################################
# * Continuing a final temporal test of encapsulation.        22nd.Dec, 2023
#
# * Testing are running into the final stage                  14th.Dec, 2023
#
# * Divide the content of programming into 1) loading fMRI dataset and 2)
#   processing dataset. For the latter, testing is going on.  12th.Dec, 2023


# ### TO-DO LIST (LEAST) ###################################################
# Required (Optional):
# TODO: Complete part-2 and testing synchronously according to the current
#       workflow.
#
# Done:
# None


import numpy as np
import pandas as pd
import scipy.io
import copy as cp


class ExperimentHelper(object):
    def __init__(self,
                 fmri_dataset_path,
                 latent_var_set,
                 sample,
                 ):
        """
        Parameters
        ----------
        fmri_dataset_path : string
            Write down some descriptions here.

        latent_var_set : set (element: string)
            Write down some descriptions here.

        sample : int
            Write down some descriptions here.

        Attributes
        ----------
        processed_dataset : ndarray
            Write down some descriptions here.

        fmri_dag : ndarray
            Write down some descriptions here.
        """

        self.fmri_dataset_path = fmri_dataset_path
        self.latent_var_set = latent_var_set
        self.sample = sample

        self.original_dataset = None
        self.processed_dataset = None
        self.original_fmri_dag = None
        self.processed_fmri_dag = None

    # ### CORRESPONDING TEST ##################################################
    # test_experiment_helper.py > test_run_generation_procedure()

    # ### AUXILIARY COMPONENT(S) ##############################################
    # _clear()
    # _load_fmri_dataset()
    # _get_latent_var_id()
    def run_generation_procedure(self, hidden_num):
        """ load and process fMRI dataset

        Write down some descriptions here.

        Parameters
        ----------
        hidden_num : int

        Arguments
        ---------
        self.fmri_dataset_path : string (parameter)
        self.sample : int (parameter)
        self.latent_var_set : set (parameter)

        Returns
        -------
        self : object

        self.original_dataset (updated) : ndarray
        self.processed_dataset (updated) : dataframe
        self.original_fmri_dag (updated) : ndarray
        self.processed_fmri_dag (updated) : ndarray
        """

        self._clear()

        self._load_fmri_dataset()

        latent_var_id = self._get_latent_var_id(hidden_num)

        if len(latent_var_id) == 0:
            self.processed_dataset = cp.copy(self.original_dataset.values)
            self.processed_fmri_dag = cp.copy(self.original_fmri_dag)
        else:
            self.processed_dataset = cp.copy(
                self.original_dataset.drop(columns=latent_var_id).values
            )
            processed_fmri_dag_temp = np.delete(
                self.original_fmri_dag,
                obj=latent_var_id,
                axis=0
            )
            self.processed_fmri_dag = np.delete(
                processed_fmri_dag_temp,
                obj=latent_var_id,
                axis=1
            )

        return self

    # ### SUBORDINATE COMPONENT(S) ############################################
    # run_generation_procedure()
    def _clear(self):
        self.original_dataset = None
        self.processed_dataset = None
        self.original_fmri_dag = None
        self.processed_fmri_dag = None

        return self

    # ### SUBORDINATE COMPONENT(S) ############################################
    # run_generation_procedure()
    def _load_fmri_dataset(self):
        """
        Write down some descriptions here.

        Arguments
        ---------
        self.fmri_dataset_path : string (parameter)
        self.sample : int (parameter)

        Returns
        -------
        self : object

        self.original_fmri_dag : ndarray
        self.original_dataset : dataframe
        """

        mat = scipy.io.loadmat(file_name=self.fmri_dataset_path)

        adjacency_matrix = mat["net"]
        self.original_fmri_dag = (adjacency_matrix[0] > 0).astype(int).T

        subjects_num = mat["Nsubjects"][0][0]
        timepoints_num = mat["Ntimepoints"][0][0]
        subjects_selected = np.random.randint(
            low=0,
            high=subjects_num - 1,
            size=int(self.sample / timepoints_num)
        )

        fragments_list = []
        for subject_index in subjects_selected:
            offset = timepoints_num * subject_index
            fmri_data_fragment = pd.DataFrame(mat["ts"]).iloc[offset:offset + timepoints_num, :]
            fragments_list.append(fmri_data_fragment)

        self.original_dataset = pd.concat(fragments_list)

        return self

    # ### SUBORDINATE COMPONENT(S) ############################################
    # run_generation_procedure()
    def _get_latent_var_id(self, hidden_num):
        """
        Write down some descriptions here.

        Parameters
        ----------
        hidden_num : int

        Arguments
        ---------
        self.latent_var_set : set (parameter)

        Return
        ------
        latent_var_id : list
        """

        latent_var_id = []
        for var_string in self.latent_var_set[:hidden_num]:
            var_letter = var_string[0]
            var_id = int(var_string[1])
            if (var_letter != 'X') and (type(var_id) != int):
                raise ValueError()
            else:
                latent_var_id.append(var_id)

        return latent_var_id

    # ### SUBORDINATE COMPONENT(S) ############################################
    # run_generation_procedure()
    def unpack(self):
        """
        Write down some descriptions here.
        """
        return self.processed_fmri_dag, self.processed_dataset

    def run_algorithm(self, model):
        if model == 'Nonlinear-MLC':
            self._run_nonlinear_mlc()

        elif model == 'CAM-UV':
            self._run_cam_uv()

        else:
            raise ValueError('testing')

        return self

    def run_evaluation_procedure(self):
        pass

    def _get_exogenous_var(self):
        pass

    def _run_nonlinear_mlc(self):
        pass

    def _run_cam_uv(self):
        pass


