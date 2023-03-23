# -*- coding: utf-8 -*-
"""
Created March 2023

This script implements numerical evaluations for the LRM linear setting.

@author: Jan Michael Breuer
"""

import convergence
import Paper_settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# LRM with linear binding and one-component and optional binding kinetics
# =============================================================================


def eval_LRM_linear(run_sim, run_eval, _file_path_, _cadet_path_,
                    _mult_sim_rerun_=0):

    tmp_path = _file_path_ + "\\LRM_verification_linear\\"

    _models_ = ["LRM_dyn_1comp", "LRM_req_1comp"]

    _recalculate_results_ = run_eval
    _plot_benchmark_ = run_eval
    _plot_benchmark_collocation_DGSEM_ = run_eval
    _export_results_ = run_eval
    _save_path_ = _file_path_+'\\results\\LRM_Linear\\'

    # use paper tolerances, i.e. discard discretizations corrupted by
    # approximation tolerances (Those that already reached
    # approximation error ~ error tolerance)
    _paper_tolerances_ = 1

    # =========================================================================
    # Setup discretizations
    # =========================================================================

    ax_methods = np.array([0, 1, 2, 3, 4, 5, 6])

    ax_cells = [
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8, 16]
    ]

    # =========================================================================
    # Run Simulations
    # =========================================================================

    if run_sim:
        for isKinetic_ in [True, False]:
            for method_idx in range(0, len(ax_methods)):
                for cell_idx in range(0, len(ax_cells[method_idx])):

                    Paper_settings.LRMlinear1Comp_VerificationSetting(
                        method_=ax_methods[method_idx],
                        colCells_=ax_cells[method_idx][cell_idx],
                        isKinetic_=isKinetic_,
                        tolerance=1e-12, plot_=False, run_=True,
                        save_path=tmp_path, cadet_path=_cadet_path_
                    )
                    # Exact integration DGSEM
                    if ax_methods[method_idx] != 0:
                        Paper_settings.LRMlinear1Comp_VerificationSetting(
                            method_=-ax_methods[method_idx],
                            colCells_=ax_cells[method_idx][cell_idx],
                            isKinetic_=isKinetic_,
                            tolerance=1e-12, plot_=False, run_=True,
                            save_path=tmp_path, cadet_path=_cadet_path_
                        )

            # Compute high accuracy solution
            Paper_settings.LRMlinear1Comp_VerificationSetting(
                method_=-6, colCells_=300,
                isKinetic_=isKinetic_,
                tolerance=1e-12, plot_=False, run_=True,
                save_path=tmp_path, cadet_path=_cadet_path_
            )

    # =========================================================================
    # Calculate results (tables)
    # =========================================================================

    if _recalculate_results_:
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            exact_names=['LRM_dyn_1comp_DGexInt_P6Z300.h5',
                         'LRM_req_1comp_DGexInt_P6Z300.h5'],
            unit='001',
            save_path_=_save_path_)
        ax_methods *= -1  # also calculate exact integration DGSEM results
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            exact_names=['LRM_dyn_1comp_DGexInt_P6Z300.h5',
                         'LRM_req_1comp_DGexInt_P6Z300.h5'],
            unit='001',
            save_path_=_save_path_)
        ax_methods *= -1  # set default back to collocation DGSEM

        # Optionally rerun simulations multiple times for benchmarking
        if _mult_sim_rerun_:
            if _mult_sim_rerun_ > 0:

                convergence.mult_sim_rerun(tmp_path, _cadet_path_,
                                           n_wdh=_mult_sim_rerun_)

    # =========================================================================
    # Generate output from results
    # =========================================================================
    # =========================================================================
    # 1) Read data
    # =========================================================================

    tables_cDGSEM = {}
    tables_DGSEM = {}

    if _plot_benchmark_ or _plot_benchmark_collocation_DGSEM_:

        for modelIdx in range(0, len(_models_)):

            # read first method for current model

            if ax_methods[0] != 0:
                result = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_DG_P' +
                    str(int(abs(ax_methods[0]))) + '.csv', delimiter=','
                    )
                result2 = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_DGexInt_P' +
                    str(int(abs(ax_methods[0]))) + '.csv', delimiter=','
                    )
            else:
                result = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_FV' + '.csv',
                    delimiter=','
                    )
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
                             _save_path_ + _models_[modelIdx] + '_FV' + '.csv',
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

    # =========================================================================
    # 2) Create Latex convergence tables
    # =========================================================================

    # merge_columns = ['$N_d$', '$N_e^z$']
    # latex_columns = merge_columns + ['Max. error', 'Max. EOC']#, 'Sim. time']

    # latex_LRM_conv = pd.merge(tables_cDG['LRM_dyn_1comp'][latex_columns],
    #                           tables_cDG['LRM_req_1comp'][latex_columns],
    #                           on=merge_columns)

    # =========================================================================
    # 3) Benchmark DGSEM vs FV
    # =========================================================================

    def benchmark_LRMlinear(tables=tables_cDGSEM):
        # same order as _models_[]
        image_names = ['LRM with kinetic binding',
                       'LRM with rapid-eq. binding']

        plot_args = {'shape': [10, 10],
                     'y_label': 'Max. error in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (10, 10)

        line_args = {'linestyle': 'dashed'}

        for m in range(0, len(_models_)):

            FV_simulations = convergence.generate_simulation_names(
                tmp_path+_models_[m],
                [0], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            )
            reference = convergence.get_outlet(
                tmp_path+_models_[m]+'_DGexInt_P6Z300.h5')

            FV_errors = convergence.calculate_all_max_errors(
                FV_simulations, reference)
            FV_sim_times = convergence.get_compute_times(FV_simulations)
            FV_DoF = convergence.calculate_DOFs(
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                [0],
                full_DOFs=True
            )

            if not _export_results_:
                plot_args['title'] = image_names[m]
            plot_args['y_lim'] = [1e-11, 2e0]

            table_P1 = tables[_models_[m]].loc[tables[_models_[m]]['$N_d$']
                                               == 1]
            table_P2 = tables[_models_[m]].loc[tables[_models_[m]]['$N_d$']
                                               == 2]
            table_P3 = tables[_models_[m]].loc[tables[_models_[m]]['$N_d$']
                                               == 3]
            table_P4 = tables[_models_[m]].loc[tables[_models_[m]]['$N_d$']
                                               == 4]

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
            plot_args['x_lim'] = [1e-3, 1e1]
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
            plot_args['x_lim'] = [1, 3e3]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_benchmark_:
        benchmark_LRMlinear()

    # =========================================================================
    # Benchmark collocation DGSEM vs DGSEM
    # =========================================================================

    def benchmark_collocationDGSEM_LRMlinear(model=_models_[1],
                                             methods=[1],
                                             cells=[[1, 2, 4, 8, 16, 32,
                                                     64, 128, 256, 512, 1024]],
                                             title=None):

        colours = ['blue', 'red', 'green',
                   'orange', 'brown', 'grey', 'magenta']

        plot_args = {'shape': [10, 10],
                     'y_label': '$L^\infty$ error in mol $ m^{-3}$'}
        plt.rcParams["figure.figsize"] = (10, 10)
        if title is not None:
            plot_args['title'] = title
        else:
            if model == _models_[1]:
                plot_args['title'] = 'LRM with rapid-eq. linear binding'
            elif model == _models_[0]:
                plot_args['title'] = 'LRM with kinetic linear binding'

        if _export_results_:
            plot_args.pop("title", None)

        line_args_DG = {'linestyle': 'dotted',
                        'markerfacecolor': 'white'}
        line_args_cDG = {'linestyle': 'dashed',
                         'markerfacecolor': 'black'}
        line_args_FV = {'linestyle': 'solid'}

        FV_simulations = convergence.generate_simulation_names(
            tmp_path+model,
            [0], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        )
        reference = convergence.get_outlet(
            tmp_path+model+'_DGexInt_P6Z300.h5')

        FV_errors = convergence.calculate_all_max_errors(
            FV_simulations, reference)
        FV_sim_times = convergence.get_compute_times(FV_simulations)
        FV_DoF = convergence.calculate_DOFs(
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            [0],
            full_DOFs=True
        )

        # plot_args['y_lim'] = [1e-11, 2e0]

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
            save_m = 'DGexInt_P' if methods[i] < 0 else 'DG_P'
            if len(methods) > 1:
                save_m += str(abs(methods[0]))+'-' + \
                    str(abs(methods[len(methods)-1]))
            else:
                save_m += str(abs(methods[i]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_time.png',
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
            save_m = 'DGexInt_P' if methods[i] < 0 else 'DG_P'
            if len(methods) > 1:
                save_m += str(abs(methods[0]))+'-' + \
                    str(abs(methods[len(methods)-1]))
            else:
                save_m += str(abs(methods[i]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_DOF.png',
                        bbox_inches='tight')
        plt.show()

    if _paper_tolerances_:
        ax_cells_ = [
            [],  # FV empty
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            [1, 2, 4, 8, 16, 32, 64, 128],
            [1, 2, 4, 8, 16, 32, 64],
            [1, 2, 4, 8, 16, 32, 64],
            [1, 2, 4, 8, 16],
            [1, 2, 4, 8, 16]
        ]
    else:
        ax_cells_ = ax_cells

    if _plot_benchmark_collocation_DGSEM_:

        for model in _models_:
            # benchmark_collocationDGSEM_LRMlinear(model=model,
            #                                       methods=[1, 3, 5],
            #                                       cells=[ax_cells[0],
            #                                             ax_cells[2],
            #                                             ax_cells[4]
            #                                             ],
            #                                       title=None
            #                                       )
            # benchmark_collocationDGSEM_LRMlinear(model=model,
            #                                       methods=[1,4],
            #                                       cells=[
            #                                             ax_cells[0],
            #                                             ax_cells[3]
            #                                             ],
            #                                       title=None
            #                                       )
            # benchmark_collocationDGSEM_LRMlinear(model=model,
            #                                       methods=[2,5],
            #                                       cells=[
            #                                             ax_cells[1],
            #                                             ax_cells[4]
            #                                             ],
            #                                       title=None
            #                                       )
            # benchmark_collocationDGSEM_LRMlinear(model=model,
            #                                       methods=[3,6],
            #                                       cells=[
            #                                             ax_cells[2],
            #                                             ax_cells[5]
            #                                             ],
            #                                       title=None
            #                                       )

            for method_idx in range(0, ax_methods.size):
                if ax_methods[method_idx] == 0:
                    continue

                benchmark_collocationDGSEM_LRMlinear(
                    model=model,
                    methods=[ax_methods[method_idx]],
                    cells=[ax_cells_[method_idx]]
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
