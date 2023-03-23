# -*- coding: utf-8 -*-
"""
Created March 2023

This script recreates the results of the paper
"Spatial discontinuous Galerkin spectral element method in CADET"
published in computers and chemical engineering, 2023.

@author: Jan Michael Breuer
"""

import GRM_linear
import LRM_linear
import LRMP_linear
import GRM_SMA
import LRMP_SMA
import LRM_SMA
import LRM_Langmuir

import os

# =============================================================================
# Setup as follows:
#
# 1) Set the variable _files_root_folder_ to the root folder to which the
#   simulations and results shall be stored.
#
# 2) Execute this script, which sets up the required folder structure.
#   Place the exact solution files for the linear GRM
#   (e.g. "conv-grm1d-dyn-ref.csv", computed using CADET-semi-analytic)
#   in the root/GRM_verification_linear folder.
#
# 3) Set the variable _cadet_path_ to the cadet executable
#
# 4) Choose which simulations and respective evaluations should be run by
#   setting the corrsponding variable (e.g. _run_GRM_linear_simulations_) to 1.
#
# 5) Note that we reran the simulations multiple times (10-100) to establish
#   fair comparisons. A single run however does not change the results
#   significantly. If multiple runs are desired, specify the _mult_sim_rerun_
#   variable in the respective function call in this file.
#
# =============================================================================


# Folder path where simulations and results are stored
_files_root_folder_ = 'C:/Users/jmbr/JupyterNotebooks/Paper CADET-DG/Paper CADET-DG_'
# Path to CADET executable
_cadet_path_ = 'C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE_befSens/bin/cadet-cli.exe'

# Specify which simulations shall be run
_run_GRM_linear_simulations_ = 0
_run_LRMP_linear_simulations_ = 0
_run_LRM_linear_simulations_ = 0
_run_GRM_SMA_simulations_ = 0
_run_LRMP_SMA_simulations_ = 0
_run_LRM_SMA_simulations_ = 0
_run_LRM_Langmuir_simulations_ = 0

# Specify which evaluations shall be run
_run_GRM_linear_evaluation_ = 1
_run_LRMP_linear_evaluation_ = 1
_run_LRM_linear_evaluation_ = 1
_run_GRM_SMA_evaluation_ = 1
_run_LRMP_SMA_evaluation_ = 1
_run_LRM_SMA_evaluation_ = 1
_run_LRM_Langmuir_evaluation_ = 1


# =============================================================================
# The following code executes everything necessary to recreate the results of
# the paper "Spatial discontinuous Galerkin spectral element method in CADET".
# =============================================================================

def create_folder_structure(path):
    os.makedirs(os.path.join(path, 'GRM_verification_linear'), exist_ok=True)
    os.makedirs(os.path.join(path, 'LRMP_verification_linear'), exist_ok=True)
    os.makedirs(os.path.join(path, 'LRM_verification_linear'), exist_ok=True)
    os.makedirs(os.path.join(path, 'GRM_SMA'), exist_ok=True)
    os.makedirs(os.path.join(path, 'LRM_SMA'), exist_ok=True)
    os.makedirs(os.path.join(path, 'LRMP_SMA'), exist_ok=True)
    os.makedirs(os.path.join(path, 'LRM_Langmuir'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'GRM_Linear'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'LRM_Linear'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'LRMP_Linear'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'GRM_SMA'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'LRM_SMA'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'LRMP_SMA'), exist_ok=True)
    os.makedirs(os.path.join(path, 'results', 'LRM_Langmuir'), exist_ok=True)


create_folder_structure(_files_root_folder_)


GRM_linear.eval_GRM_linear(run_sim=_run_GRM_linear_simulations_,
                           run_eval=_run_GRM_linear_evaluation_,
                           _file_path_=_files_root_folder_,
                           _cadet_path_=_cadet_path_,
                           # _mult_sim_rerun_ specifies how many times
                           # simulations shall be rerun to get minimal
                           # simulation time for benchmarks
                           _mult_sim_rerun_=0
                           )

LRMP_linear.eval_LRMP_linear(run_sim=_run_LRMP_linear_simulations_,
                             run_eval=_run_LRMP_linear_evaluation_,
                             _file_path_=_files_root_folder_,
                             _cadet_path_=_cadet_path_,
                             # _mult_sim_rerun_ specifies how many times
                             # simulations shall be rerun to get minimal
                             # simulation time for benchmarks
                             _mult_sim_rerun_=0
                             )

LRM_linear.eval_LRM_linear(run_sim=_run_LRM_linear_simulations_,
                           run_eval=_run_LRM_linear_evaluation_,
                           _file_path_=_files_root_folder_,
                           _cadet_path_=_cadet_path_,
                           # _mult_sim_rerun_ specifies how many times
                           # simulations shall be rerun to get minimal
                           # simulation time for benchmarks
                           _mult_sim_rerun_=0
                           )

GRM_SMA.eval_GRM_SMA(run_sim=_run_GRM_SMA_simulations_,
                     run_eval=_run_GRM_SMA_evaluation_,
                     _file_path_=_files_root_folder_,
                     _cadet_path_=_cadet_path_,
                     # _mult_sim_rerun_ specifies how many times
                     # simulations shall be rerun to get minimal
                     # simulation time for benchmarks
                     _mult_sim_rerun_=0
                     )

LRMP_SMA.eval_LRMP_SMA(run_sim=_run_LRMP_SMA_simulations_,
                       run_eval=_run_LRMP_SMA_evaluation_,
                       _file_path_=_files_root_folder_,
                       _cadet_path_=_cadet_path_,
                       # _mult_sim_rerun_ specifies how many times
                       # simulations shall be rerun to get minimal
                       # simulation time for benchmarks
                       _mult_sim_rerun_=0
                       )

LRM_SMA.eval_LRM_SMA(run_sim=_run_LRM_SMA_simulations_,
                     run_eval=_run_LRM_SMA_evaluation_,
                     _file_path_=_files_root_folder_,
                     _cadet_path_=_cadet_path_,
                     # _mult_sim_rerun_ specifies how many times
                     # simulations shall be rerun to get minimal
                     # simulation time for benchmarks
                     _mult_sim_rerun_=0
                     )

LRM_Langmuir.eval_LRM_Langmuir(run_sim=_run_LRM_Langmuir_simulations_,
                               run_eval=_run_LRM_Langmuir_evaluation_,
                               _file_path_=_files_root_folder_,
                               _cadet_path_=_cadet_path_,
                               # _mult_sim_rerun_ specifies how many times
                               # simulations shall be rerun to get minimal
                               # simulation time for benchmarks
                               _mult_sim_rerun_=0
                               )

