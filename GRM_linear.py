# -*- coding: utf-8 -*-
"""
Created March 2023

This script implements numerical evaluations for the GRM linear setting.

@author: Jan Michael Breuer
"""

import convergence
import Paper_settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# =============================================================================
# Evaluation for a GRM with linear binding and one-component,
# and with optional surface diffusion and binding kinetics
# =============================================================================


def eval_GRM_linear(run_sim, run_eval, _file_path_, _cadet_path_,
                    _mult_sim_rerun_=0):

    tmp_path = _file_path_ + "\\GRM_verification_linear\\"

    _recalculate_results_ = run_eval
    _plot_benchmark_ = run_eval
    _plot_benchmark_collocation_DGSEM_ = run_eval
    _export_results_ = run_eval
    _save_path_ = _file_path_ + "\\results\\GRM_Linear\\"

    _models_ = ["GRM_dynLin_1Comp", "GRM_reqLin_1Comp",
                "GRMsd_dynLin_1Comp", "GRMsd_reqLin_1Comp"]

    _image_names_ = ['GRM with kinetic binding',
                     'GRM with rapid-eq. binding',
                     'GRM with surface diffusion, kinetic binding',
                     'GRM with surface diffusion, rapid-eq. binding']

    exact_solution_names = ["conv-grm1d-dyn-ref.csv",
                            "conv-grm1d-req-ref.csv",
                            "conv-grm1d-dyn-sd-ref.csv",
                            "conv-grm1d-req-sd-ref.csv"]

    exact_solution_path = tmp_path
    _exact_solutions_ = []

    if run_eval:
        for model_idx in range(0, len(_models_)):
    
            exact_solution_times = np.loadtxt(
                exact_solution_path+exact_solution_names[model_idx],
                delimiter=",")[:, 0]
            reference = np.zeros(len(exact_solution_times) + 1)
            reference[1:] = np.loadtxt(
                exact_solution_path + exact_solution_names[model_idx],
                delimiter=",")[:, 1]  # time, unit0, unit1 <-> time, GRM, inletUOP,
    
            _exact_solutions_.append(reference)

    # =========================================================================
    # Setup discretizations
    # =========================================================================

    # Discretizations
    ax_methods = [0, 1, 2, 3, 4, 5]
    ax_cells = [
        [4, 8, 16, 32, 64, 128, 256],
        [4, 8, 16, 32, 64, 128],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32],
        [4, 8, 16]
    ]
    par_methods = [0, 1, 2, 3, 4, 5]
    par_cells = [
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8],
        [1, 2, 4]
    ]

    # =========================================================================
    # Run Simulations
    # =========================================================================

    if run_sim:

        for model in _models_:
            is_kinetic = bool(re.search("dyn", model))
            has_surf_diff = bool(re.search("sd", model))

            for method_idx in range(0, len(ax_methods)):
                for cell_idx in range(0, len(ax_cells[method_idx])):

                    Paper_settings.GRMlinear1Comp_VerificationSetting(
                        method_=ax_methods[method_idx],
                        colCells_=ax_cells[method_idx][cell_idx],
                        parMethod_=par_methods[method_idx],
                        parCells_=par_cells[method_idx][cell_idx],
                        isKinetic_=is_kinetic, surfDiff=has_surf_diff,
                        tolerance=1e-12,
                        plot_=False, run_=True,
                        save_path=tmp_path, cadet_path=_cadet_path_)

                    if ax_methods[method_idx] != 0:
                        Paper_settings.GRMlinear1Comp_VerificationSetting(
                            method_=-ax_methods[method_idx],
                            colCells_=ax_cells[method_idx][cell_idx],
                            parMethod_=par_methods[method_idx],
                            parCells_=par_cells[method_idx][cell_idx],
                            isKinetic_=is_kinetic, surfDiff=has_surf_diff,
                            tolerance=1e-12,
                            plot_=False, run_=True,
                            save_path=tmp_path, cadet_path=_cadet_path_)

        # Optionally rerun simulations multiple times for benchmarking
        if _mult_sim_rerun_:
            if _mult_sim_rerun_ > 0:

                convergence.mult_sim_rerun(tmp_path, _cadet_path_,
                                           n_wdh=_mult_sim_rerun_)

    if _recalculate_results_:
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            par_methods=par_methods, par_cells=par_cells,
            exact_names=_exact_solutions_,
            unit='001', save_path_=_save_path_
        )
        ax_methods = np.array(ax_methods)*(-1)  # exact integration DGSEM
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            par_methods=par_methods, par_cells=par_cells,
            exact_names=_exact_solutions_,
            unit='001', save_path_=_save_path_
        )
        # set methods back to collocation DGSEM (default)
        ax_methods = np.array(ax_methods)*(-1)

    # =========================================================================
    # Generate output from results
    # =========================================================================
    # =========================================================================
    # 1) Read data
    # =========================================================================
    tables_cDGSEM = {}
    tables_DGSEM = {}

    def read_data():
        for modelIdx in range(0, len(_models_)):

            # read first method for current model

            if ax_methods[0] != 0:
                result = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_DG_P' +
                    str(int(abs(ax_methods[0]))) + '.csv',
                    delimiter=',')

                result2 = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_DGexInt_P' +
                    str(int(abs(ax_methods[0]))) + '.csv',
                    delimiter=',')
            else:
                result = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_FV' + '.csv',
                    delimiter=',')

                result2 = result

            # read remaining methods for current model and add to result
            for m in range(1, len(ax_methods)):

                if ax_methods[m] != 0:
                    result = pd.concat(
                        (result,
                         pd.read_csv(
                             _save_path_ + _models_[modelIdx] + '_DG_P' +
                             str(int(abs(ax_methods[m]))) + '.csv',
                             delimiter=',')
                         )
                    )
                    result2 = pd.concat(
                        (result2,
                         pd.read_csv(
                             _save_path_ + _models_[modelIdx] + '_DGexInt_P' +
                             str(int(abs(ax_methods[m]))) + '.csv',
                             delimiter=',')
                         )
                    )
                else:
                    result = pd.concat(
                        (result,
                         pd.read_csv(
                             tmp_path + _models_[modelIdx] + '_FV' + '.csv',
                             delimiter=',')
                         )
                    )
                    result2 = pd.concat(
                        (result2,
                         pd.read_csv(
                             _save_path_ + _models_[modelIdx] + '_FV' + '.csv',
                             delimiter=',')
                         )
                    )

            result['$N_e^z$'] = result['$N_e^z$'].round().astype(int)
            result['DoF'] = result['DoF'].round().astype(int)
            result2['$N_e^z$'] = result2['$N_e^z$'].round().astype(int)
            result2['DoF'] = result2['DoF'].round().astype(int)

            tables_cDGSEM[_models_[modelIdx]] = result
            tables_DGSEM[_models_[modelIdx]] = result2

    if run_eval:
        read_data()

    # =========================================================================
    # 2) Create Latex convergence tables
    # =========================================================================

    # merge_columns = ['$N_d$', '$N_e^z$', '$N_e^r$']
    # latex_columns = merge_columns + ['Max. error', 'Max. EOC']

    # latex_GRM_conv = pd.merge(
    # tables_cDGSEM['GRM_dynLin_1Comp'][latex_columns],
    # tables_cDGSEM['GRM_reqLin_1Comp'][latex_columns],
    # on=merge_columns)

    # print(convergence.std_latex_table(latex_GRM_conv,
    #                                   latex_GRM_conv.columns))
    # print(convergence.std_latex_table(latex_GRMsd_conv,
    #                                   latex_GRMsd_conv.columns))

    # =========================================================================
    # 3) Benchmark images
    # =========================================================================

    def benchmark_DGFV_GRMlinear(with_title=False):

        plot_args = {'shape': [10, 10],
                     'y_label': 'Max. error in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])
        line_args = {'linestyle': 'dashed'}

        for m in range(0, len(_models_)):

            FV_simulations = convergence.generate_simulation_names(
                tmp_path+_models_[m],
                [0], [4, 8, 16, 32, 64, 128, 256],
                [0], [1, 2, 4, 8, 16, 32, 64]
            )
            exact_solution_times = np.loadtxt(
                exact_solution_path+exact_solution_names[m],
                delimiter=",")[:, 0]
            reference = np.zeros(len(exact_solution_times) + 1)
            reference[1:] = np.loadtxt(
                exact_solution_path + exact_solution_names[m],
                delimiter=",")[:, 1]  # time, unit0, unit1 <-> time, GRM, inlet

            FV_errors = convergence.calculate_all_max_errors(
                FV_simulations, reference)
            FV_sim_times = convergence.get_compute_times(FV_simulations)
            FV_DoF = convergence.calculate_DOFs(
                [[4, 8, 16, 32, 64, 128, 256],
                 [1, 2, 4, 8, 16, 32, 64]],
                [0, 0],
                full_DOFs=True
            )

            if with_title:
                plot_args['title'] = _image_names_[m]
            plot_args['y_lim'] = [1e-12, 1e-1]

            table_P1 = tables_cDGSEM[_models_[
                m]].loc[tables_cDGSEM[_models_[m]]['$N_d$'] == 1]
            table_P2 = tables_cDGSEM[_models_[
                m]].loc[tables_cDGSEM[_models_[m]]['$N_d$'] == 2]
            table_P3 = tables_cDGSEM[_models_[
                m]].loc[tables_cDGSEM[_models_[m]]['$N_d$'] == 3]
            table_P4 = tables_cDGSEM[_models_[
                m]].loc[tables_cDGSEM[_models_[m]]['$N_d$'] == 4]

            convergence.std_plot(FV_sim_times, FV_errors, label='FV',
                                 color='purple', marker='s', **line_args)
            convergence.std_plot(
                table_P1['Sim. time'], table_P1['Max. error'], label='P1', **line_args)
            convergence.std_plot(
                table_P2['Sim. time'], table_P2['Max. error'], label='P2', **line_args)
            convergence.std_plot(
                table_P3['Sim. time'], table_P3['Max. error'], label='P3', **line_args)
            convergence.std_plot(
                table_P4['Sim. time'], table_P4['Max. error'], label='P4', **line_args)

            plot_args['x_label'] = 'Compute time in seconds'
            plot_args['x_lim'] = [0.0, 1e3]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_time.png',
                            bbox_inches='tight')
            plt.show()

            convergence.std_plot(FV_DoF, FV_errors, label='FV',
                                 color='purple', marker='s', **line_args)
            convergence.std_plot(
                table_P1['DoF'], table_P1['Max. error'], label='P1', **line_args)
            convergence.std_plot(
                table_P2['DoF'], table_P2['Max. error'], label='P2', **line_args)
            convergence.std_plot(
                table_P3['DoF'], table_P3['Max. error'], label='P3', **line_args)
            convergence.std_plot(
                table_P4['DoF'], table_P4['Max. error'], label='P4', **line_args)

            plot_args['x_label'] = 'Degrees of freedom'
            plot_args['x_lim'] = [1, 1e5]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_benchmark_:
        benchmark_DGFV_GRMlinear(with_title=_export_results_)

    def benchmark_collocationDGSEM_GRMlinear(model=_models_[1],
                                             methods=[ax_methods[0]],
                                             cells=[ax_cells[0]],
                                             title=None,
                                             save_path=None):

        colours = ['blue', 'red', 'green',
                   'orange', 'brown', 'grey', 'magenta']

        plot_args = {'shape': [10, 10],
                     'y_label': 'Max. error in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])
        if title is not None:
            plot_args['title'] = title
        else:
            if model == _models_[0]:
                plot_args['title'] = 'GRM with kinetic linear binding'
            elif model == _models_[1]:
                plot_args['title'] = 'GRM with rapid-eq. linear binding'
            elif model == _models_[2]:
                plot_args['title'] = 'GRM with surf. diff. and kinetic linear binding'
            elif model == _models_[3]:
                plot_args['title'] = 'GRM with surf. diff and rapid-eq. linear binding'
        if _export_results_:
            plot_args.pop("title", None)

        line_args_DG = {'linestyle': 'dotted',
                        'markerfacecolor': 'white'}
        line_args_cDG = {'linestyle': 'dashed',
                         'markerfacecolor': 'black'}

        for i in range(0, len(methods)):

            table_DG = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_DGSEM[model]['$N_d$'] == methods[i])
            ]

            table_cDG = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_cDGSEM[model]['$N_d$'] == methods[i])
            ]

            convergence.std_plot(table_DG['Sim. time'], table_DG['Max. error'],
                                 label='DG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_DG)
            convergence.std_plot(table_cDG['Sim. time'], table_cDG['Max. error'],
                                 label='cDG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_cDG)

        plot_args['x_label'] = 'Compute time in seconds'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = "P" + str(methods[0])
            if len(methods) > 1:
                save_m += "-P" + str(methods[len(methods)-1])
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+save_m+'_time.png',
                        bbox_inches='tight')

        plt.show()

        for i in range(0, len(methods)):
            table_DG = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_DGSEM[model]['$N_d$'] == methods[i])
            ]

            table_cDG = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_cDGSEM[model]['$N_d$'] == methods[i])
            ]

            convergence.std_plot(table_DG['DoF'], table_DG['Max. error'],
                                 label='DG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_DG)
            convergence.std_plot(table_cDG['DoF'], table_cDG['Max. error'],
                                 label='cDG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_cDG)

        plot_args['x_label'] = 'Degrees of freedom'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = "P" + str(methods[0])
            if len(methods) > 1:
                save_m += "-P" + str(methods[len(methods)-1])
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+save_m+'_DOF.png',
                        bbox_inches='tight')
        plt.show()

    if _plot_benchmark_collocation_DGSEM_:

        ax_methods = np.array([1, 2, 3, 4, 5])
        for model_idx in range(0, len(_models_)):
            for method_idx in range(0, ax_methods.size):

                benchmark_collocationDGSEM_GRMlinear(
                    model=_models_[model_idx],
                    methods=[ax_methods[method_idx]],
                    cells=[ax_cells[method_idx]],
                    save_path=_save_path_
                )

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
