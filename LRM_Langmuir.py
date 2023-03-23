# -*- coding: utf-8 -*-
"""
Created March 2023

This script implements numerical evaluations for the LRM Langmuir setting.

@author: Jan Michael Breuer
"""

import convergence
import Paper_settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# =============================================================================
# LRM with Langmuir binding and two dispersion parameters D_ax in [1e-4, 1e-5]
# =============================================================================


def eval_LRM_Langmuir(run_sim, run_eval, _file_path_, _cadet_path_,
                      _mult_sim_rerun_=0):

    tmp_path = _file_path_ + "\\LRM_Langmuir\\"

    models = ["LRMdisp0.0001_reqLangmuir_2comp",
              "LRMdisp1e-5_reqLangmuir_2comp"]
    _exact_names_ = ['LRM2CompLangmuirD0.0001_DG_P10Z300.h5',
                     'LRM2CompLangmuirD1e-05_DG_P10Z300.h5']

    _recalculate_results_ = run_eval
    _plot_benchmark_ = run_eval
    _plot_neg_conc_ = run_eval
    _plot_oscillations_ = run_eval
    _plot_benchmark_collocation_DGSEM_ = run_eval
    _export_results_ = run_eval
    _save_path_ = _file_path_ + '\\results\\LRM_Langmuir\\'

    # =========================================================================
    # Setup discretizations
    # =========================================================================

    # Recalculate
    ax_methods = np.array([
        0,
        3, 4, 5])
    ax_cells_e4 = [  # axial dispersion D_ax=1e-4
        [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [8, 16, 32, 64, 128, 256],
        [8, 16, 32, 64, 128],
        [8, 16, 32, 64, 128]
    ]
    ax_cells_e5 = [  # axial dispersion D_ax=1e-5
        [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        [8, 16, 32, 64, 128, 256, 512, 1024],
        [8, 16, 32, 64, 128, 256, 512, 1024],
        [8, 16, 32, 64, 128, 256, 512, 1024]
    ]

    # =========================================================================
    # Run Simulations
    # =========================================================================

    def rerun_simulations(ax_methods, ax_cells, d_ax,
                          save_path=tmp_path,
                          cadet_path=_cadet_path_):

        for method_idx in range(0, len(ax_methods)):
            for ncells in ax_cells[method_idx]:

                Paper_settings.LRM_langmuir_oscillations(
                    method=ax_methods[method_idx], ncells=ncells,
                    D_ax=d_ax,
                    save_path=tmp_path,
                    cadet_path=cadet_path)

    if run_sim:

        rerun_simulations(ax_methods, ax_cells_e4, 1e-4,
                          save_path=tmp_path, cadet_path=_cadet_path_)
        rerun_simulations(ax_methods, ax_cells_e5, 1e-5,
                          save_path=tmp_path, cadet_path=_cadet_path_)
        
        ax_methods = ax_methods * (-1.0) # exact integration DGSEM
        rerun_simulations(ax_methods, ax_cells_e4, 1e-4,
                          save_path=tmp_path, cadet_path=_cadet_path_)
        rerun_simulations(ax_methods, ax_cells_e5, 1e-5,
                          save_path=tmp_path, cadet_path=_cadet_path_)
        ax_methods = ax_methods * (-1.0) # back to collocation DGSEM

        # High accuracy reference solutions
        Paper_settings.LRM_langmuir_oscillations(
            method=10, ncells=300,
            D_ax=1e-4,
            save_path=tmp_path,
            cadet_path=_cadet_path_)

        Paper_settings.LRM_langmuir_oscillations(
            method=10, ncells=300,
            D_ax=1e-5,
            save_path=tmp_path,
            cadet_path=_cadet_path_)

        # Optionally rerun simulations multiple times for benchmarking
        if _mult_sim_rerun_:
            if _mult_sim_rerun_ > 0:

                convergence.mult_sim_rerun(tmp_path, _cadet_path_,
                                           n_wdh=_mult_sim_rerun_)

    # =========================================================================
    # Calculate results (tables)
    # =========================================================================

    if _recalculate_results_:
        save_path_ = tmp_path if _export_results_ == 0 else _save_path_

        convergence.recalculate_results(
            file_path=tmp_path, models=[models[0]],
            ax_methods=ax_methods, ax_cells=ax_cells_e4,
            exact_names=[_exact_names_[0]],
            unit='001', incl_min_val=True, save_path_=save_path_
            )

        convergence.recalculate_results(
            file_path=tmp_path, models=[models[1]],
            ax_methods=ax_methods, ax_cells=ax_cells_e5,
            exact_names=[_exact_names_[1]],
            unit='001', incl_min_val=True, save_path_=save_path_
            )

        ax_methods *= -1  # exInt DGSEM
        convergence.recalculate_results(
            file_path=tmp_path, models=[models[0]],
            ax_methods=ax_methods, ax_cells=ax_cells_e4,
            exact_names=[_exact_names_[0]],
            unit='001', incl_min_val=True, save_path_=save_path_
            )

        convergence.recalculate_results(
            file_path=tmp_path, models=[models[1]],
            ax_methods=ax_methods, ax_cells=ax_cells_e5,
            exact_names=[_exact_names_[1]],
            unit='001', incl_min_val=True, save_path_=save_path_
            )
        ax_methods *= -1  # return to collocation DGSEM

    # =========================================================================
    # Generate output from results
    # =========================================================================
    # =========================================================================
    # 1) Read data
    # =========================================================================

    tables_cDGSEM = {}
    tables_DGSEM = {}

    def read_data():
        for modelIdx in range(0, len(models)):

            # read first method for current model

            if ax_methods[0] != 0:
                result = pd.read_csv(
                    _save_path_ + models[modelIdx] + '_DG_P' +
                    str(int(abs(ax_methods[0]))) + '.csv', delimiter=','
                    )
                result2 = pd.read_csv(
                    _save_path_ + models[modelIdx] + '_DGexInt_P' +
                    str(int(abs(ax_methods[0]))) + '.csv', delimiter=','
                    )
            else:
                result = pd.read_csv(
                    _save_path_ + models[modelIdx] + '_FV' + '.csv',
                    delimiter=','
                    )
                result2 = result

            # read remaining methods for current model and add to result
            for m in range(1, len(ax_methods)):

                if ax_methods[m] != 0:
                    result = pd.concat(
                        (result,
                         pd.read_csv(
                             _save_path_ + models[modelIdx] + '_DG_P' +
                             str(int(abs(ax_methods[m]))) + '.csv',
                             delimiter=',')
                         )
                    )
                    result2 = pd.concat(
                        (result2,
                         pd.read_csv(
                             _save_path_ + models[modelIdx] + '_DGexInt_P' +
                             str(int(abs(ax_methods[m]))) + '.csv',
                             delimiter=',')
                         )
                    )
                else:
                    result = pd.concat(
                        (result,
                         pd.read_csv(
                             _save_path_ + models[modelIdx] + '_FV' + '.csv',
                             delimiter=',')
                         )
                    )
                    result2 = pd.concat(
                        (result2,
                         pd.read_csv(
                             _save_path_ + models[modelIdx] + '_FV' + '.csv',
                             delimiter=',')
                         )
                    )

            result['$N_e^z$'] = result['$N_e^z$'].round().astype(int)
            result['DoF'] = result['DoF'].round().astype(int)
            result2['$N_e^z$'] = result2['$N_e^z$'].round().astype(int)
            result2['DoF'] = result2['DoF'].round().astype(int)

            tables_cDGSEM[models[modelIdx]] = result
            tables_DGSEM[models[modelIdx]] = result2

    # =========================================================================
    # 2) Create Latex convergence tables
    # =========================================================================

    # merge_columns = ['$N_d$', '$N_e^z$']
    # latex_columns = merge_columns + \
    #     ['Max. error', 'Max. EOC', 'Sim. time', 'Min. value']

    # latex_LRM_conv = pd.merge(tables_cDGSEM[models[0]][latex_columns],
    # tables_cDGSEM[models[1]][latex_columns],
    # on=merge_columns)
    # print(convergence.std_latex_table(
    #     latex_LRM_conv, latex_LRM_conv.columns)
    #     )

    # print(convergence.std_latex_table(
    #     tables_cDGSEM[models[0]], latex_columns)
    #     )
    # print(convergence.std_latex_table(
    #     tables_cDGSEM[models[1]], latex_columns)
    #     )

    # =========================================================================
    # 3) Benchmark DGSEM vs FV
    # =========================================================================

    def benchmark_LRMlangmuir(models_=models, tables_=tables_cDGSEM):

        read_data()

        # same order as models[]
        image_names = ['LRM with kinetic Langmuir binding and $D_{ax}=1e-4$',
                       'LRM with kinetic Langmuir binding and $D_{ax}=1e-5$']

        plot_args = {'shape': [10, 10],
                     'y_label': 'Max. error in mol $/ m^3$',
                     'y_lim': [2e-6, 20]}

        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])

        line_args = {'linestyle': 'dashed'}

        for m in range(0, len(models_)):

            if not _export_results_:
                plot_args['title'] = image_names[m]

            for method in tables_[models_[m]]['$N_d$'].unique():
                if method != 0:
                    table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                                     == method]
                    convergence.std_plot(
                        table_['Sim. time'],
                        table_['Max. error'],
                        label='DG P' + re.search('\d+', str(method)).group(0),
                        **line_args)

            table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                             == 0]
            convergence.std_plot(
                table_['Sim. time'], table_['Max. error'], label='FV',
                color='purple', marker='s', **line_args)

            plot_args['x_label'] = 'Compute time in seconds'
            # plot_args['x_lim'] = [1e-3, 1e1]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+models_[m]+'_time.png',
                            bbox_inches='tight')
            plt.show()

            for method in tables_[models_[m]]['$N_d$'].unique():
                if method != 0:
                    table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                                     == method]
                    convergence.std_plot(
                        table_['DoF'],
                        table_['Max. error'],
                        label='DG P' + re.search('\d+', str(method)).group(0),
                        **line_args)

            table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                             == 0]
            convergence.std_plot(
                table_['DoF'], table_['Max. error'], label='FV',
                color='purple', marker='s', **line_args)

            plot_args['x_label'] = 'Degrees of freedom'
            # plot_args['x_lim'] = [1, 2e3]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_benchmark_:
        benchmark_LRMlangmuir(tables_=tables_cDGSEM)

    # =========================================================================
    # Analyze Oscillations
    # =========================================================================

    def oscillations_LRMlangmuir(models_=models, tables_=tables_cDGSEM):

        # same order as models[]
        image_names = ['LRM with kinetic Langmuir binding and $D_{ax}=1e-4$',
                       'LRM with kinetic Langmuir binding and $D_{ax}=1e-5$']

        plot_args = {'shape': [10, 10],
                     'y_label': 'Min. value in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])

        line_args = {'linestyle': 'dashed'}

        for m in range(0, len(models_)):

            if not _export_results_:
                plot_args['title'] = image_names[m]
            plot_args['y_lim'] = [1e-22, 1e1]

            for method in tables_[models_[m]]['$N_d$'].unique():
                if method != 0:
                    table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                                     == method]
                    convergence.std_plot(
                        table_['DoF'],
                        np.where(np.array(table_['Min. value']) < 0,
                                 np.abs(np.array(table_['Min. value'])),
                                 np.finfo(np.float64).eps),
                        label='DG P' + str(method),
                        **line_args)

            table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                             == 0]
            convergence.std_plot(
                table_['DoF'],
                np.where(np.array(table_['Min. value']) < 0,
                         np.abs(np.array(table_['Min. value'])),
                         np.finfo(np.float64).eps),
                label='FV', color='purple', marker='s', **line_args)

            plot_args['x_label'] = 'Degrees of freedom'
            # plot_args['x_lim'] = [1, 2e3]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_neg_conc_:
        read_data()
        oscillations_LRMlangmuir(tables_=tables_cDGSEM)

    # =========================================================================
    # Benchmark collocation DGSEM vs. DGSEM
    # =========================================================================

    def benchmark_collocationDGSEM_LRMlinear(model=models[1],
                                             methods=[1],
                                             cells=ax_cells_e5,
                                             title=None):

        colours = ['blue', 'red', 'green',
                   'orange', 'brown', 'grey', 'magenta']

        plot_args = {'shape': [10, 10],
                     'y_label': 'Max. error in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])

        # same order as models[]
        if model == models[0]:
            title = 'LRMP with kinetic Langmuir binding and $D_{ax}=1e-4$'
        elif model == models[1]:
            title = 'LRMP with kinetic Langmuir binding and $D_{ax}=1e-5$'
        if title is not None:
            plot_args['title'] = title

        line_args_DG = {'linestyle': 'dotted',
                        'markerfacecolor': 'white'}
        line_args_cDG = {'linestyle': 'dashed',
                         'markerfacecolor': 'black'}
        line_args_FV = {'linestyle': 'solid'}

        if _export_results_:
            plot_args.pop("title", None)
        # plot_args['y_lim'] = [1e-11, 2e0]

        for i in range(0, len(methods)):

            table_DGSEM = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_DGSEM[model]['$N_d$'] == methods[i])
            ]

            table_cDGSEM = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_cDGSEM[model]['$N_d$'] == methods[i])
            ]

            convergence.std_plot(table_DGSEM['Sim. time'],
                                 table_DGSEM['Max. error'],
                                 label='DG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_DG)
            convergence.std_plot(table_cDGSEM['Sim. time'],
                                 table_cDGSEM['Max. error'],
                                 label='cDG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_cDG)

        plot_args['x_label'] = 'Compute time in seconds'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = 'DGexInt_P' if methods[i] < 0 else 'DG_P'
            save_m += str(abs(methods[i]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_time.png',
                        bbox_inches='tight')

        plt.show()

        for i in range(0, len(methods)):
            table_DGSEM = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_DGSEM[model]['$N_d$'] == methods[i])
            ]

            table_cDGSEM = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_cDGSEM[model]['$N_d$'] == methods[i])
            ]

            convergence.std_plot(table_DGSEM['DoF'], table_DGSEM['Max. error'],
                                 label='DG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_DG)
            convergence.std_plot(table_cDGSEM['DoF'], table_cDGSEM['Max. error'],
                                 label='cDG P'+str(int(methods[i])),
                                 color=colours[i],
                                 **line_args_cDG)

        plot_args['x_label'] = 'Degrees of freedom'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = 'DGexInt_P' if methods[i] < 0 else 'DG_P'
            save_m += str(abs(methods[i]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_DOF.png',
                        bbox_inches='tight')
        plt.show()

    if _plot_benchmark_collocation_DGSEM_:
        read_data()
        # benchmark_collocationDGSEM_LRMlinear()
        for method_idx in range(1, ax_methods.size):
            benchmark_collocationDGSEM_LRMlinear(model=models[0],
                                                 methods=[
                                                     ax_methods[method_idx]],
                                                 cells=[
                                                     ax_cells_e4[method_idx]]
                                                 )
        for method_idx in range(1, ax_methods.size):
            benchmark_collocationDGSEM_LRMlinear(model=models[1],
                                                 methods=[
                                                     ax_methods[method_idx]],
                                                 cells=[
                                                     ax_cells_e5[method_idx]]
                                                 )

    if _plot_oscillations_:

        kwargs = {'marker': ''
                  }
        plt_args = {'shape': [10, 10],
                    'x_scale': 'linear',
                    'y_scale': 'linear',
                    'x_label': 'Time in seconds',
                    'y_label': 'Concentration in mol $/ m^3$',
                    'y_lim': [-2.0, 18.0]}

        for model_idx in range(0, len(models)):

            exact_ = convergence.get_outlet(
                tmp_path+_exact_names_[model_idx], unit='001')
            solution_times = convergence.get_solution_times(
                tmp_path+_exact_names_[0])
            if not _export_results_:
                plt_args['title'] = 'LRM Langmuir D_ax=' + \
                    re.search('(LRMdisp)([^_]+)', models[model_idx]).group(2)
            convergence.std_plot(
                solution_times,
                exact_[:, 0],
                **kwargs, color='orange', label='comp 1')
            convergence.std_plot(
                solution_times,
                exact_[:, 1],
                color='blue', label='comp 2', **kwargs)

            convergence.std_plot_prep(**plt_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'outlet_exact_'+models[model_idx]+'.png',
                            bbox_inches='tight')
            plt.show()

            for FVne in [64, 128, 256]:
                # if not _export_results_:
                plt_args['title'] = '$N_e = $' + str(FVne)

                name_ = models[model_idx]+'_FV_Z'+str(FVne)+'.h5'
                convergence.std_plot(
                    convergence.get_solution_times(tmp_path+name_),
                    convergence.get_outlet(tmp_path+name_, unit='001')[:, 0],
                    **kwargs, color='red', label='FV')
                convergence.std_plot(
                    convergence.get_solution_times(tmp_path+name_),
                    convergence.get_outlet(tmp_path+name_, unit='001')[:, 1],
                    color='red', **kwargs)

                convergence.std_plot(
                    solution_times,
                    exact_[:, 0],
                    **kwargs, color='blue', label='Exact')
                convergence.std_plot(
                    solution_times,
                    exact_[:, 1],
                    color='blue', **kwargs)

                convergence.std_plot_prep(**plt_args)
                if _export_results_:
                    plt.savefig(_save_path_ +
                                'oscillations_FVZ' +
                                str(FVne)+'_'+models[model_idx]+'.png',
                                bbox_inches='tight')
                plt.show()

            for DGne in [32, 64, 128]:
                # if not _export_results_:
                plt_args['title'] = '$N_e = $' + str(DGne)

                name_ = models[model_idx]+'_DG_P3Z'+str(DGne)+'.h5'
                convergence.std_plot(
                    convergence.get_solution_times(tmp_path+name_),
                    convergence.get_outlet(tmp_path+name_, unit='001')[:, 0],
                    **kwargs, color='red', label='DG P3')
                convergence.std_plot(
                    convergence.get_solution_times(tmp_path+name_),
                    convergence.get_outlet(tmp_path+name_, unit='001')[:, 1],
                    color='red', **kwargs)

                convergence.std_plot(
                    solution_times,
                    exact_[:, 0],
                    **kwargs, color='blue', label='Exact')
                convergence.std_plot(
                    solution_times, exact_[:, 1],
                    color='blue', **kwargs)

                convergence.std_plot_prep(**plt_args)
                if _export_results_:
                    plt.savefig(_save_path_ +
                                'oscillations_DGP3Z' +
                                str(DGne)+'_'+models[model_idx]+'.png',
                                bbox_inches='tight')
                plt.show()
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
