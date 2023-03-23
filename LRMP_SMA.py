# -*- coding: utf-8 -*-
"""
Created March 2023

This script implements numerical evaluations for the LRMP SMA setting.

@author: Jan Michael Breuer
"""

import convergence
import Paper_settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# LRMP with SMA binding and four-components
# =============================================================================


def eval_LRMP_SMA(run_sim, run_eval, _file_path_, _cadet_path_,
                  _mult_sim_rerun_=0):

    _with_flux_ = False  # old version with flux state variable

    tmp_path = _file_path_ + "\\LRMP_SMA\\"
    if _with_flux_:
        tmp_path += "with_flux\\"

    _recalculate_results_ = run_eval
    _plot_benchmark_ = run_eval
    _plot_benchmark_collocation_DGSEM_ = run_eval
    _export_results_ = run_eval
    _save_path_ = _file_path_ + '\\results\\LRMP_SMA\\'

    _models_ = ["LRMP_SMA_4comp"]
    _image_names_ = ['LRMP with rapid equilibrium SMA binding']

    # =========================================================================
    # Setup discretizations
    # =========================================================================

    ax_methods = [0, 1, 2, 3, 4, 5]
    ax_cells = [
        [4, 8, 16, 32, 64, 128, 256, 512],
        [4, 8, 16, 32, 64, 128, 256],
        [4, 8, 16, 32, 64, 128],
        [4, 8, 16, 32],
        [4, 8, 16],
        [4, 8]
    ]

    if run_sim:

        for m_idx in range(0, len(ax_methods)):
            for nCells in ax_cells[m_idx]:

                Paper_settings.LWE_setting(
                    ax_methods[m_idx], nCells,
                    par_method=0, nParCells=0,
                    save_path=tmp_path,
                    transport_model='LUMPED_RATE_MODEL_WITH_PORES',
                    is_kinetic=0,
                    idas_tolerance=1E-12,
                    cadet_path=_cadet_path_,
                    run_sim=True)

                # Exact integration DGSEM
                if ax_methods[m_idx] != 0:
                    Paper_settings.LWE_setting(
                        -ax_methods[m_idx], nCells,
                        par_method=0, nParCells=0,
                        save_path=tmp_path,
                        transport_model='LUMPED_RATE_MODEL_WITH_PORES',
                        is_kinetic=0,
                        idas_tolerance=1E-12,
                        cadet_path=_cadet_path_,
                        run_sim=True)

            # Compute high accuracy solution
            Paper_settings.LWE_setting(
                -5, 128,
                par_method=0, nParCells=0,
                save_path=tmp_path,
                transport_model='LUMPED_RATE_MODEL_WITH_PORES',
                is_kinetic=0,
                idas_tolerance=1E-12,
                cadet_path=_cadet_path_,
                run_sim=True)

        # Optionally rerun simulations multiple times for benchmarking
        if _mult_sim_rerun_:
            if _mult_sim_rerun_ > 0:

                convergence.mult_sim_rerun(tmp_path, _cadet_path_,
                                           n_wdh=_mult_sim_rerun_)

    if _recalculate_results_:
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            exact_names=['LRMP_SMA_4comp_DGexInt_P5Z128.h5'],
            unit='000', save_path_=_save_path_
            )
        ax_methods = np.array(ax_methods)*(-1)  # exact integration DGSEM
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            exact_names=['LRMP_SMA_4comp_DGexInt_P5Z128.h5'],
            unit='000', save_path_=_save_path_
            )
        # back to collocation DGSEM (default)
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
                    delimiter=','
                    )
                result2 = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_DGexInt_P' +
                    str(int(abs(ax_methods[0]))) + '.csv',
                    delimiter=','
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

    if run_eval:
        read_data()

    # =========================================================================
    # 2) Create Latex convergence tables
    # =========================================================================

    # read_data(path_=tmp_path)
    # merge_columns = ['$N_d$', '$N_e^z$']
    # latex_columns = ['$N_d$', '$N_e^z$', 'max error', 'max EOC', 'Sim. time']

    # print(convergence.std_latex_table(
    #    tables['LRMP_SMA_4comp'][latex_columns],
    #    latex_columns))

    # =========================================================================
    # 3) Benchmark DGSEM vs FV
    # =========================================================================

    def benchmark_LRMP_SMA(tables=tables_cDGSEM):

        plot_args = {'shape': [10, 10],
                     'y_label': '$L^\infty$ error in mol $/ m^3$'}

        line_args = {'linestyle': 'dashed'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])

        for m in range(0, len(_models_)):

            if not _export_results_:
                plot_args['title'] = _image_names_[m]
            # plot_args['y_lim'] = [1e-11, 2e0]

            for method_idx in range(0, len(ax_methods)):

                table = tables[_models_[m]].loc[tables[_models_[m]]['$N_d$']
                                                == ax_methods[method_idx]]
                if ax_methods[method_idx] == 0:
                    convergence.std_plot(
                        table['Sim. time'], table['Max. error'], label='FV',
                        color='purple', marker='s', **line_args)
                else:
                    convergence.std_plot(
                        table['Sim. time'], table['Max. error'],
                        label='P'+str(ax_methods[method_idx]), **line_args)

            plot_args['x_label'] = 'Compute time in seconds'
            # plot_args['x_lim'] = [1e-3, 1e1]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_time.png',
                            bbox_inches='tight')
            plt.show()

            for method_idx in range(0, len(ax_methods)):

                table = tables[_models_[m]].loc[tables[_models_[m]]['$N_d$']
                                                == ax_methods[method_idx]]
                if ax_methods[method_idx] == 0:
                    convergence.std_plot(table['DoF'], table['Max. error'],
                                         label='FV', color='purple',
                                         marker='s', **line_args)
                else:
                    convergence.std_plot(
                        table['DoF'], table['Max. error'],
                        label='P'+str(ax_methods[method_idx]), **line_args)

            plot_args['x_label'] = 'Degrees of freedom'
            # plot_args['x_lim'] = [1e-3, 1e1]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_benchmark_:
        benchmark_LRMP_SMA()

    # =========================================================================
    # Benchmark collocation DGSEM vs DGSEM
    # =========================================================================

    def benchmark_collocationDGSEM_LRMP_SMA(model=_models_[0],
                                            methods=ax_methods[1],
                                            cells=[ax_cells[0]],
                                            title=None,
                                            save_path=tmp_path):

        colours = ['blue', 'red', 'green',
                   'orange', 'brown', 'grey', 'magenta']

        plot_args = {'shape': [10, 10],
                     'y_label': '$L^\infty$ error in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])

        line_args_DG = {'linestyle': 'dotted',
                        'markerfacecolor': 'white'}
        line_args_cDG = {'linestyle': 'dashed',
                         'markerfacecolor': 'black'}

        if title is not None:
            plot_args['title'] = title
        elif not _export_results_ and model == _models_[0]:
            plot_args['title'] = _image_names_[0]

        # plot_args['y_lim'] = [1e-11, 2e0]

        plot_args['x_label'] = 'Compute time in seconds'

        for method_idx in range(0, len(methods)):

            table_DG = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[method_idx])) &
                (tables_DGSEM[model]['$N_d$'] == methods[method_idx])
            ]

            table_cDG = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[method_idx])) &
                (tables_cDGSEM[model]['$N_d$'] == methods[method_idx])
            ]

            convergence.std_plot(table_DG['Sim. time'], table_DG['Max. error'],
                                 label='DG P'+str(methods[method_idx]),
                                 color=colours[method_idx],
                                 **line_args_DG
                                 )
            convergence.std_plot(table_cDG['Sim. time'],
                                 table_cDG['Max. error'],
                                 label='cDG P'+str(methods[method_idx]),
                                 color=colours[method_idx],
                                 **line_args_cDG
                                 )
        convergence.std_plot_prep(**plot_args)

        if _export_results_:
            save_m = 'DGexInt_P' if methods[method_idx] < 0 else 'DG_P'
            save_m += str(abs(methods[method_idx]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_time.png',
                        bbox_inches='tight')
        plt.show()

        for method_idx in range(0, len(methods)):
            table_DG = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[method_idx])) &
                (tables_DGSEM[model]['$N_d$'] == methods[method_idx])
            ]

            table_cDG = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[method_idx])) &
                (tables_cDGSEM[model]['$N_d$'] == methods[method_idx])
            ]

            convergence.std_plot(table_DG['DoF'], table_DG['Max. error'],
                                 label='DG P'+str(methods[method_idx]),
                                 color=colours[method_idx],
                                 **line_args_DG)
            convergence.std_plot(table_cDG['DoF'], table_cDG['Max. error'],
                                 label='cDG P'+str(methods[method_idx]),
                                 color=colours[method_idx],
                                 **line_args_cDG)

        plot_args['x_label'] = 'DoF'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = 'DGexInt_P' if methods[method_idx] < 0 else 'DG_P'
            save_m += str(abs(methods[method_idx]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_DOF.png',
                        bbox_inches='tight')
        plt.show()

    if _plot_benchmark_collocation_DGSEM_:

        ax_methods = np.array([1, 2, 3, 4, 5])

        for method_idx in range(0, ax_methods.size):

            benchmark_collocationDGSEM_LRMP_SMA(model=_models_[0],
                                                methods=[
                                                    ax_methods[method_idx]],
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
