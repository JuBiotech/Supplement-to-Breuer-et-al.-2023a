# -*- coding: utf-8 -*-
"""
Created March 2023

This script implements numerical evaluations for the LRM SMA setting.

@author: Jan Michael Breuer
"""

import convergence
import Paper_settings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# LRM with SMA binding and four-components
# =============================================================================


def eval_LRM_SMA(run_sim, run_eval, _file_path_, _cadet_path_,
                 _mult_sim_rerun_=0):

    tmp_path = _file_path_ + '\\LRM_SMA\\'

    _models_ = ["LRM_SMA_4comp"]

    _recalculate_results_ = run_eval
    _plot_benchmark_ = run_eval
    _plot_benchmark_collocation_DGSEM_ = run_eval
    _export_results_ = run_eval
    _save_path_ = _file_path_ + '\\results\\LRM_SMA\\'

    if not _export_results_:
        _save_path_ = tmp_path

    # =========================================================================
    # Setup discretizations
    # =========================================================================

    ax_methods = [
        0,
        1, 2, 3, 4, 5, 6]

    ax_cells = [
        [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [4, 8, 16, 32, 64, 128, 256],
        [4, 8, 16, 32, 64, 128, 256],
        [4, 8, 16, 32, 64, 128],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32]
    ]

    # =========================================================================
    # Run Simulations
    # =========================================================================

    if run_sim:

        for m_idx in range(0, len(ax_methods)):
            for nCells in ax_cells[m_idx]:

                Paper_settings.LWE_setting(
                    ax_methods[m_idx], nCells,
                    par_method=0, nParCells=0,
                    save_path=tmp_path,
                    transport_model='LUMPED_RATE_MODEL_WITHOUT_PORES',
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
                        transport_model='LUMPED_RATE_MODEL_WITHOUT_PORES',
                        is_kinetic=0,
                        idas_tolerance=1E-12,
                        cadet_path=_cadet_path_,
                        run_sim=True)

            # Compute high accuracy solution
            Paper_settings.LWE_setting(
                -6, 256,
                par_method=0, nParCells=0,
                save_path=tmp_path,
                transport_model='LUMPED_RATE_MODEL_WITHOUT_PORES',
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
            exact_names=['LRM_SMA_4comp_DGexInt_P6Z256.h5'],
            unit='000', save_path_=_save_path_
            )

        ax_methods = np.array(ax_methods)*(-1)  # exact integration DGSEM
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=ax_methods, ax_cells=ax_cells,
            exact_names=['LRM_SMA_4comp_DGexInt_P6Z256.h5'],
            unit='000', save_path_=_save_path_
            )
        ax_methods = np.array(ax_methods)*(-1)  # back to collocation DGSEM

    # =============================================================================
    # 1) Read data
    # =============================================================================

    tables_cDGSEM = {}
    tables_DGSEM = {}

    def read_tables():
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

            # read remaining methods for current model and concatenate to result
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
        read_tables()

    # =========================================================================
    # 2) Create Latex convergence tables
    # =========================================================================

    # latex_columns =
    # ['$N_d$', '$N_e^z$', 'Max. error', 'Max. EOC', 'Sim. time']#,'Min. value'

    # =========================================================================
    # 3) Benchmark DGSEM vs FV
    # =========================================================================

    def benchmark_LRM_LWE(tables=tables_cDGSEM,
                          methods=[0, 1, 2, 3, 4, 5],
                          cells_=ax_cells):

        # same order as models[]
        image_names = ['LRM with rapid-eq. SMA binding']

        plot_args = {'shape': [10, 10],
                     'y_label': '$L^\infty$ error in mol $m^{-3}$'}
        plt.rcParams["figure.figsize"] = (plot_args['shape'][0],
                                          plot_args['shape'][1])

        line_args = {'linestyle': 'dashed'}

        for m in range(0, len(_models_)):

            model = _models_[m]

            if not _export_results_:
                plot_args['title'] = image_names[m]
            # plot_args['y_lim'] = [1e-11, 2e0]

            for method_idx in range(0, len(methods)):
                table = tables[model][
                    (tables[model]['$N_e^z$'].isin(cells_[method_idx])) &
                    (tables[model]['$N_d$'] == methods[method_idx])
                ]

                if methods[method_idx] == 0:
                    convergence.std_plot(
                        table['Sim. time'], table['Max. error'],
                        label='FV', color='purple', marker='s',
                        **line_args)
                else:
                    convergence.std_plot(
                        table['Sim. time'], table['Max. error'],
                        label='P'+str(abs(methods[method_idx])), **line_args)

            plot_args['x_label'] = 'Compute time in seconds'
            # plot_args['x_lim'] = [1e-3, 1e1]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_time.png',
                            bbox_inches='tight')
            plt.show()

            for method_idx in range(0, len(methods)):
                table = tables[model][
                    (tables[model]['$N_e^z$'].isin(cells_[method_idx])) &
                    (tables[model]['$N_d$'] == methods[method_idx])
                ]
                if methods[method_idx] == 0:
                    convergence.std_plot(
                        table['DoF'], table['Max. error'],
                        label='FV', color='purple', marker='s',
                        **line_args)
                else:
                    convergence.std_plot(
                        table['DoF'], table['Max. error'],
                        label='P'+str(abs(methods[method_idx])), **line_args)

            plot_args['x_label'] = 'Degrees of Freedom'
            # plot_args['x_lim'] = [1e-3, 1e1]
            convergence.std_plot_prep(**plot_args)
            if _export_results_:
                plt.savefig(_save_path_ +
                            'benchmark_'+_models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_benchmark_:
        benchmark_LRM_LWE()

    # =========================================================================
    # Benchmark collocation DGSEM vs DGSEM
    # =========================================================================

    def benchmark_collocationDGSEM_LRM_LWE(model=_models_[0],
                                           methods=[1, 2, 3, 4, 5],
                                           cells=ax_cells[1:],
                                           title='LRM with rapid-eq. SMA binding',
                                           **plt_args):
        # same order as _models_[]
        if title == None:
            title = ""
        elif model == _models_[0]:
            title = 'LRM with rapid-eq. SMA binding'

        colours = ['blue', 'red', 'green',
                   'orange', 'brown', 'grey', 'magenta']

        plot_args = {'shape': [10, 10],
                     'y_label': 'Max. error in mol $m^{-3}$'}
        plot_args.update(plt_args)

        line_args_DG = {'linestyle': 'dotted',
                        'markerfacecolor': 'white'}
        line_args_cDG = {'linestyle': 'dashed',
                         'markerfacecolor': 'black'}
        line_args_FV = {'linestyle': 'solid'}

        if not _export_results_:
            plot_args['title'] = title
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
        # plot_args['x_lim'] = [1e-3, 1e1]
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = 'DGexInt_P' if methods[i] < 0 else 'DG_P'
            save_m += str(abs(methods[i]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_time.png',
                        bbox_inches='tight')
        plt.show()

        # convergence.std_plot(FV_DoF, FV_errors, label='FV',
        #                       color='purple', marker='s', **line_args_FV)
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

        plot_args['x_label'] = 'DoF'
        # plot_args['x_lim'] = [1, 2e3]
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = 'DGexInt_P' if methods[i] < 0 else 'DG_P'
            save_m += str(abs(methods[i]))
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_DOF.png',
                        bbox_inches='tight')
        plt.show()

    if _plot_benchmark_collocation_DGSEM_:

        for model in _models_:

            benchmark_collocationDGSEM_LRM_LWE(model=model,
                                               methods=[1, 3, 5],
                                               cells=[ax_cells[1],
                                                      ax_cells[3],
                                                      ax_cells[5]])

            benchmark_collocationDGSEM_LRM_LWE(model=model,
                                               methods=[2, 4, 6],
                                               cells=[ax_cells[2],
                                                      ax_cells[4],
                                                      ax_cells[6]])

            for i in range(1, len(ax_methods)):
                benchmark_collocationDGSEM_LRM_LWE(
                    model=model,
                    methods=[ax_methods[i]],
                    cells=[ax_cells[i]]
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
