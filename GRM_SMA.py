# -*- coding: utf-8 -*-
"""
Created March 2023

This script implements numerical evaluations for the GRM SMA setting.

@author: Jan Michael Breuer
"""

import convergence
import Paper_settings

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import re

# =============================================================================
# Evaluation for a GRM with SMA binding and four-components
# Note that we compare two refinement strategies:
# naive h-refinement,
# optimal particle refinement
# =============================================================================


def eval_GRM_SMA(run_sim, run_eval, _file_path_, _cadet_path_,
                 _mult_sim_rerun_=0):

    tmp_path = _file_path_ + '\\GRM_SMA\\'

    _models_ = ["GRM_SMA_4comp"]

    _rerun_simulations_ = run_sim
    _rerun_optimal_refinement_ = run_sim
    _rerun_find_refinement_optimal_strategy_ = False  # runs all p-refinements

    _recalculate_results_ = run_eval
    _paper_error_range_ = 1  # limit benchmarks to a more relevant error range
    _plot_h_refinement_benchmark_ = run_eval
    _plot_optimal_refinement_benchmark_ = run_eval
    _plot_L1_benchmark_ = False
    # include multithread FV simulations
    _optimal_refinement_benchmark_nthread = run_eval
    _plot_benchmark_collocation_DGSEM_ = run_eval
    _export_results_ = run_eval
    _optimal_FV_refinement_ = False  # tests optimal FV refinement

    _save_path_ = _file_path_ + "\\results\\GRM_SMA\\"

    # =========================================================================
    # Setup discretizations
    # =========================================================================

    methods = [0, 1, 2, 3, 4, 5]
    ax_cells = [
        [16, 32, 64, 128, 256, 512],
        [4, 8, 16, 32, 64, 128, 256],
        [4, 8, 16, 32, 64, 128],
        [4, 8, 16, 32, 64],
        [4, 8, 16, 32],
        [4, 8, 16, 32]
    ]
    par_cells = [
        [4, 8, 16, 32, 64, 128],
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16],
        [1, 2, 4, 8],
        [1, 2, 4, 8]
    ]
    ax_methods = methods

    # =========================================================================
    # Run Simulations
    # =========================================================================

    if run_sim:

        for m_idx in range(0, len(methods)):
            for cell_idx in range(0, len(ax_cells[m_idx])):

                Paper_settings.LWE_setting(
                    methods[m_idx], ax_cells[m_idx][cell_idx],
                    par_method=methods[m_idx], nParCells=par_cells[m_idx][cell_idx],
                    save_path=tmp_path,
                    transport_model='GENERAL_RATE_MODEL',
                    is_kinetic=0,
                    idas_tolerance=1E-8,
                    cadet_path=_cadet_path_,
                    run_sim=True)

                # Exact integration DGSEM
                if methods[m_idx] != 0:
                    Paper_settings.LWE_setting(
                        -methods[m_idx], ax_cells[m_idx][cell_idx],
                        par_method=methods[m_idx], nParCells=par_cells[m_idx][cell_idx],
                        save_path=tmp_path,
                        transport_model='GENERAL_RATE_MODEL',
                        is_kinetic=0,
                        idas_tolerance=1E-8,
                        cadet_path=_cadet_path_,
                        run_sim=True)

                # Run multi-thread FV
                else:
                    for n_threads in [2, 6]:
                        Paper_settings.LWE_setting(
                            methods[m_idx], ax_cells[m_idx][cell_idx],
                            par_method=methods[m_idx], nParCells=par_cells[m_idx][cell_idx],
                            save_path=tmp_path,
                            transport_model='GENERAL_RATE_MODEL',
                            is_kinetic=0,
                            idas_tolerance=1E-8,
                            cadet_path=_cadet_path_,
                            run_sim=True,
                            n_threads=n_threads)

        # Compute high accuracy solution
        Paper_settings.LWE_setting(
            -5, 64,
            par_method=5, nParCells=16,
            save_path=tmp_path,
            transport_model='GENERAL_RATE_MODEL',
            is_kinetic=0,
            idas_tolerance=1E-8,
            cadet_path=_cadet_path_,
            run_sim=True)

        # run optimal refinement simulations
        if _rerun_optimal_refinement_:
            for ax_p, ax_cellss, par_ps in [
                    (3, 1, 1), (3, 2, 1), (3, 3, 1), (3, 4, 2), (3, 5, 5),
                    (3, 6, 6), (3, 7, 7), (3, 8, 8), (3, 9, 10), (3, 10, 11),
                    (4, 1, 1), (4, 2, 2), (4, 3, 6),
                    (4, 4, 8), (4, 5, 8), (4, 6, 11),
                    (5, 1, 1), (5, 2, 2), (5, 3, 6), (5, 4, 11)
            ]:

                Paper_settings.LWE_setting(
                    ax_p, ax_cellss,
                    par_method=par_ps, nParCells=1,
                    save_path=tmp_path,
                    transport_model='GENERAL_RATE_MODEL',
                    is_kinetic=0,
                    idas_tolerance=1E-8,
                    cadet_path=_cadet_path_,
                    run_sim=True)

        # run optimal refinement exploration simulations
        if _rerun_find_refinement_optimal_strategy_:
            # Explore bulk disc
            for poly_degs, ax_cellss in [(4, 8), (3, 10), (5, 8)]:

                Paper_settings.LWE_setting(
                    poly_degs, ax_cellss,
                    par_method=15, nParCells=1,
                    save_path=tmp_path,
                    transport_model='GENERAL_RATE_MODEL',
                    is_kinetic=0,
                    idas_tolerance=1E-8,
                    cadet_path=_cadet_path_,
                    run_sim=True)
            # Explore particle disc
            for par_poly_deg in range(1, 16):
                Paper_settings.LWE_setting(
                    5, 12,
                    par_method=par_poly_deg, nParCells=1,
                    save_path=tmp_path,
                    transport_model='GENERAL_RATE_MODEL',
                    is_kinetic=0,
                    idas_tolerance=1E-8,
                    cadet_path=_cadet_path_,
                    run_sim=True)

        # Optionally rerun simulations multiple times for benchmarking
        if _mult_sim_rerun_:
            if _mult_sim_rerun_ > 0:

                convergence.mult_sim_rerun(tmp_path, _cadet_path_,
                                           n_wdh=_mult_sim_rerun_)

    # =========================================================================
    # Recalculate convergence results
    # =========================================================================

    if _recalculate_results_:

        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=methods, ax_cells=ax_cells,
            par_methods=methods, par_cells=par_cells,
            exact_names=['GRM_SMA_4comp_DGexInt_P5Z64parP5parZ16.h5'],
            unit='000',
            save_path_=_save_path_
        )
        convergence.recalculate_results(
            file_path=tmp_path, models=_models_,
            ax_methods=np.array(methods)*(-1), ax_cells=ax_cells,
            par_methods=methods, par_cells=par_cells,
            exact_names=['GRM_SMA_4comp_DGexInt_P5Z64parP5parZ16.h5'],
            unit='000',
            save_path_=_save_path_
        )

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
                    str(int(abs(ax_methods[0]))) + '.csv', delimiter=',')
                result2 = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_DGexInt_P' +
                    str(int(abs(ax_methods[0]))) + '.csv', delimiter=',')
            else:
                result = pd.read_csv(
                    _save_path_ + _models_[modelIdx] + '_FV' + '.csv',
                    delimiter=',')

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
        read_data()
    # =============================================================================
    # 2) Create Latex convergence tables
    # =============================================================================

    # merge_columns = ['Method', '$N_e^z$']
    # latex_columns = ['Method', '$N_e^z$', 'Max. error', 'Max. EOC', 'Sim. time']

    # print(convergence.std_latex_table(tables['GRM_LWE_4comp'][latex_columns],
    #                                   latex_columns))

    # =============================================================================
    # 3) Benchmark DGSEM vs FV
    # =============================================================================

    def benchmark_GRM_LWE(models_=_models_, tables_=tables_cDGSEM,
                          ax_cells_=None):

        # same order as models[]
        image_names = ['GRM with rapid-equilibrium SMA binding']

        plot_args = {'shape': [10, 10],
                     'y_label': '$L^\infty$ error in mol $/ m^3$'}
        plt.rcParams["figure.figsize"] = (
            plot_args['shape'][0], plot_args['shape'][1])

        line_args = {'linestyle': 'dashed'}

        for m in range(0, len(models_)):

            if not _export_results_:
                plot_args['title'] = image_names[m]

            for method in tables_[models_[m]]['$N_d$'].unique():

                if method != 0:  # DG plots first

                    if _paper_error_range_ and ax_cells_ is not None:
                        table_ = tables_[models_[m]][
                            (tables_[models_[m]]['$N_e^z$'].isin(ax_cells_[method])) &
                            (tables_[models_[m]]['$N_d$'] == method)
                        ]
                    else:
                        table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                                         == method]
                    convergence.std_plot(
                        table_['Sim. time'],
                        table_['Max. error'],
                        label='P' + re.search('\d+', str(method)).group(0),
                        **line_args)

            # FV plot
            if _paper_error_range_ and ax_cells_ is not None:
                table_ = tables_[models_[m]][
                    (tables_[models_[m]]['$N_e^z$'].isin(ax_cells_[0])) &
                    (tables_[models_[m]]['$N_d$'] == 0)
                ]
            else:
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
                            'benchmark_hRefinement_'+models_[m]+'_time.png',
                            bbox_inches='tight')
            plt.show()

            # DoF plots
            for method in tables_[models_[m]]['$N_d$'].unique():
                if method != 0:
                    if _paper_error_range_ and ax_cells_ is not None:
                        table_ = tables_[models_[m]][
                            (tables_[models_[m]]['$N_e^z$'].isin(ax_cells_[method])) &
                            (tables_[models_[m]]['$N_d$'] == method)
                        ]
                    else:
                        table_ = tables_[models_[m]].loc[tables_[models_[m]]['$N_d$']
                                                         == method]
                    convergence.std_plot(
                        table_['DoF'],
                        table_['Max. error'],
                        label='P' + re.search('\d+', str(method)).group(0),
                        **line_args)

                    if _paper_error_range_ and ax_cells_ is not None:
                        table_ = tables_[models_[m]][
                            (tables_[models_[m]]['$N_e^z$'].isin(ax_cells_[0])) &
                            (tables_[models_[m]]['$N_d$'] == 0)
                        ]
                    else:
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
                            'benchmark_hRefinement_'+models_[m]+'_DOF.png',
                            bbox_inches='tight')
            plt.show()

    if _plot_h_refinement_benchmark_:
        benchmark_GRM_LWE(tables_=tables_cDGSEM,
                          ax_cells_={
                              0: [16, 32, 64, 128, 256, 512],
                              1: [4, 8, 16, 32, 64, 128, 256],
                              2: [4, 8, 16, 32, 64],
                              3: [4, 8, 16, 32],
                              4: [4, 8, 16],
                              5: [4, 8, 16]
                          })

    # =============================================================================
    # Optimal refinement strategy simulations:
    # =============================================================================

    def find_refinement_strategy(rerun_bulk=False, rerun_par=False):

        reference = convergence.get_outlet(
            tmp_path+_models_[0]+'_DGexInt_P5Z64parP5parZ16.h5',
            unit='000')

        # explore particle discretization errors
        for poly_degs, ax_nSim in [(4, 8), (3, 10), (5, 8)]:
            par_nSim = 15
            ax_ps = [5]*par_nSim
            ax_cellss = [[12]]*par_nSim
            par_ps = np.linspace(1, par_nSim, par_nSim, dtype=int)
            par_cellss = [[1]]*par_nSim
            if rerun_par:
                for m_idx in range(0, len(ax_ps)):
                    convergence.recalculate_results(
                        file_path=tmp_path, models=_models_,
                        ax_methods=ax_ps[m_idx], ax_cells=ax_cellss[m_idx],
                        par_methods=par_ps[m_idx], par_cells=par_cellss[m_idx],
                        exact_names=[
                            'GRM_SMA_4comp_DGexInt_P5Z64parP5parZ16.h5'],
                        unit='000',
                        save_path_=_save_path_
                    )
                    convergence.recalculate_results(
                        file_path=tmp_path, models=_models_,
                        ax_methods=-ax_ps[m_idx], ax_cells=ax_cellss[m_idx],
                        par_methods=par_ps[m_idx], par_cells=par_cellss[m_idx],
                        exact_names=[
                            'GRM_SMA_4comp_DGexInt_P5Z64parP5parZ16.h5'],
                        unit='000',
                        save_path_=_save_path_
                    )

            sim_names = convergence.generate_simulation_names(
                tmp_path+_models_[0],
                ax_ps, ax_cellss, par_ps, par_cellss)

            par_pref = convergence.calculate_all_max_errors(
                sim_names, reference, unit='000')

            # explore bulk discretization errors
            ax_ps = [poly_degs]
            ax_cellss = [np.linspace(1, ax_nSim, ax_nSim, dtype=int)]
            par_ps = [15]
            par_cellss = [[1]*ax_nSim]
            if rerun_bulk:
                for m_idx in range(0, len(ax_ps)):
                    convergence.recalculate_results(
                        file_path=tmp_path, models=_models_,
                        ax_methods=ax_ps[m_idx], ax_cells=ax_cellss[m_idx],
                        par_methods=par_ps[m_idx], par_cells=par_cellss[m_idx],
                        exact_names=[
                            'GRM_SMA_4comp_DGexInt_P5Z64parP5parZ16.h5'],
                        unit='000',
                        save_path_=_save_path_
                    )
                    convergence.recalculate_results(
                        file_path=tmp_path, models=_models_,
                        ax_methods=-ax_ps[m_idx], ax_cells=ax_cellss[m_idx],
                        par_methods=par_ps[m_idx], par_cells=par_cellss[m_idx],
                        exact_names=[
                            'GRM_SMA_4comp_DGexInt_P5Z64parP5parZ16.h5'],
                        unit='000',
                        save_path_=_save_path_
                    )

            ax_cellss = np.linspace(1, ax_nSim, ax_nSim, dtype=int)
            par_cellss = [1]*ax_nSim
            sim_names = convergence.generate_simulation_names(
                tmp_path+_models_[0],
                ax_ps, ax_cellss, par_ps, par_cellss)

            ax_href = convergence.calculate_all_max_errors(
                sim_names, reference, unit='000')

            print("P"+str(poly_degs)+" particle p-ref:\n", par_pref)
            print("P"+str(poly_degs)+" axial/bulk h-ref:\n", ax_href)

    if _rerun_find_refinement_optimal_strategy_:
        find_refinement_strategy(rerun_bulk=_rerun_find_refinement_optimal_strategy_,
                                 rerun_par=_rerun_find_refinement_optimal_strategy_)

    def benchmark_optimal_refinement(FV_ax_cells, FV_par_cells,
                                     error_type="max", bench_dof=False
                                     ):

        reference = convergence.get_outlet(
            tmp_path+_models_[0]+'_DGexInt_P5Z64parP5parZ16.h5',
            unit='000')

        # optimal refinement simulations and plot
        for ax_p, ax_cellss, par_ps in [
                (3,
                 [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
                 [1, 1, 1, 2, 5, 6, 7, 8, 10, 11]),
                (4,
                 [[1], [2], [3], [4], [5], [6]],
                 [1, 2, 6, 8, 8, 11]),
                (5,
                 [[1], [2], [3], [4]], [1, 2, 6, 11])
        ]:

            plot_args = {'shape': [10, 10],
                         'y_label': '$L^\infty$-error in mol $/ m^3$',
                         'x_label': 'Compute time in seconds'}
            plt.rcParams["figure.figsize"] = (
                plot_args['shape'][0], plot_args['shape'][1])

            if bench_dof:
                plot_args['x_label'] = 'Degrees of freedom'
            if not re.search('max|Linf', error_type):
                if re.search('L1', error_type):
                    plot_args['y_label'] = '$L^1$-error in mol $/ m^3$'

            line_args = {'linestyle': 'dashed'}

            sim_names = convergence.generate_simulation_names(
                tmp_path+_models_[0],
                [ax_p]*len(ax_cellss), ax_cellss,
                par_ps, [[1]]*len(ax_cellss)
            )

            if re.search('max|Linf', error_type):
                errors = convergence.calculate_all_max_errors(sim_names,
                                                              reference,
                                                              unit='000')
                hmpf = ''
            elif re.search('simpsonL1', error_type):
                errors = np.array([0.0]*len(sim_names))
                for m_idx in range(0, len(sim_names)):
                    # abs. value (negative oscillations might be substracted)
                    abs_comp_sum = np.sum(
                        abs(reference -
                            convergence.get_outlet(sim_names[m_idx], unit='000')),
                        axis=1
                    )
                    errors[m_idx] = scipy.integrate.simpson(
                        y=abs_comp_sum
                    )
                hmpf = 'SimpsonL1'
            elif re.search('L1', error_type):
                errors = convergence.calculate_all_L1_errors(
                    sim_names, reference, unit='000', comps=[0, 1, 2, 3]
                )
                hmpf = 'L1'
            else:
                raise ValueError("Error type " + str(error_type) + " unknown.")

            DoF = np.empty((len(ax_cellss)), dtype=float)
            sim_times = np.empty((len(ax_cellss)), dtype=float)
            for i in range(0, len(ax_cellss)):
                DoF[i] = convergence.calculate_DOFs(
                    [ax_cellss[i], [1]], [ax_p, par_ps[i]], nComp=4,
                    full_DOFs=True
                )
                sim_times[i] = convergence.get_compute_time(
                    convergence.generate_GRM_name(
                        tmp_path +
                        _models_[0], ax_p, ax_cellss[i][0], par_ps[i], 1
                    )
                )

            if bench_dof:
                convergence.std_plot(DoF, errors, label='DG P' +
                                     str(ax_p), **line_args)
            else:
                convergence.std_plot(
                    sim_times, errors, label='DG P'+str(ax_p), **line_args)

        # FV simulations
        if _optimal_refinement_benchmark_nthread:
            FV_prefixes = ['', '2thread', '6thread']  # , '4thread'
            FV_labels = ['FV', 'FV 2 threads',
                         'FV 6 threads']  # , 'FV 4 threads'
            markerfacecolors = ['purple', 'blue', 'green']
        else:
            FV_prefixes = ['']
            FV_labels = ['FV']
            markerfacecolors = ['purple']

        for FV_prefix_idx in range(0, len(FV_prefixes)):
            sim_names = convergence.generate_simulation_names(
                tmp_path+FV_prefixes[FV_prefix_idx]+_models_[0],
                [0], FV_ax_cells, [0], FV_par_cells)

            if re.search('max|Linf', error_type):
                errors = convergence.calculate_all_max_errors(sim_names,
                                                              reference,
                                                              unit='000')
                hmpf = ''
            elif re.search('simpsonL1', error_type):
                errors = np.array([0.0]*len(sim_names))
                for m_idx in range(0, len(sim_names)):
                    # abs value (negative oscillations might be substracted)
                    abs_comp_sum = np.sum(
                        abs(reference -
                            convergence.get_outlet(sim_names[m_idx], unit='000')),
                        axis=1
                    )
                    errors[m_idx] = scipy.integrate.simpson(
                        y=abs_comp_sum
                    )
                hmpf = 'SimpsonL1_'
            elif re.search('L1', error_type):
                errors = convergence.calculate_all_L1_errors(sim_names,
                                                             reference,
                                                             unit='000',
                                                             comps=[0, 1, 2, 3])
                hmpf = 'L1_'
            else:
                raise ValueError("Error type " + str(error_type) + " unknown.")

            sim_times = convergence.get_compute_times(sim_names)
            DoF = convergence.calculate_DOFs([FV_ax_cells,
                                              FV_par_cells],
                                             [0, 0],
                                             nComp=4,
                                             full_DOFs=True)

            if bench_dof and FV_prefix_idx == 0:
                convergence.std_plot(DoF, errors, label='FV',
                                     color='purple', marker='s', **line_args)
                hmpf += 'DOF'
            elif not bench_dof:
                convergence.std_plot(sim_times, errors, label=FV_labels[FV_prefix_idx],
                                     markerfacecolor=markerfacecolors[FV_prefix_idx],
                                     color='purple', marker='s', **line_args)
                hmpf += 'time'

        convergence.std_plot_prep(**plot_args)

        if _export_results_:
            name_pref = 'benchmark_FVthread_DGoptRefinement_' if _optimal_refinement_benchmark_nthread else 'benchmark_optRefinement_'
            plt.savefig(_save_path_ +
                        name_pref+_models_[0]+'_'+hmpf+'.png',
                        bbox_inches='tight')
        plt.show()

    if _plot_optimal_refinement_benchmark_:
        benchmark_optimal_refinement(FV_ax_cells=ax_cells[0],
                                     FV_par_cells=par_cells[0],
                                     rerun=_rerun_optimal_refinement_,
                                     error_type="max", bench_dof=False)
        benchmark_optimal_refinement(FV_ax_cells=ax_cells[0],
                                     FV_par_cells=par_cells[0],
                                     rerun=False,
                                     error_type="max", bench_dof=True)
    if _plot_L1_benchmark_:
        benchmark_optimal_refinement(FV_ax_cells=ax_cells[0],
                                     FV_par_cells=par_cells[0],
                                     rerun=_rerun_optimal_refinement_,
                                     error_type="simpsonL1", bench_dof=False)
        benchmark_optimal_refinement(FV_ax_cells=ax_cells[0],
                                     FV_par_cells=par_cells[0],
                                     rerun=False,
                                     error_type="simpsonL1", bench_dof=True)
        # benchmark_optimal_refinement(FV_ax_cells=[64, 128, 256, 512],
        #                              FV_par_cells=[16, 32, 64, 128],
        #                              rerun=_rerun_optimal_refinement_,
        #                              error_type="simpsonL1", bench_dof=False)
        # benchmark_optimal_refinement(FV_ax_cells=[64, 128, 256, 512],
        #                              FV_par_cells=[16, 32, 64, 128],
        #                              rerun=False,
        #                              error_type="simpsonL1", bench_dof=True)

    # =========================================================================
    # Optimal FV strategy
    # =========================================================================

    def plot_optimal_FV_strategy(FV_ax_cells, FV_par_cells, error_type,
                                 reference=tmp_path+'GRM_LWE_4comp_DG_P5Z64parP5parZ16.h5',
                                 bench_dof=False):

        # FV simulations
        for i in range(0, len(FV_ax_cells)):

            sim_names = convergence.generate_simulation_names(tmp_path+_models_[0],
                                                              [0], FV_ax_cells[i],
                                                              [0], FV_par_cells[i])
            if re.search('max|Linf', error_type):
                errors = convergence.calculate_all_max_errors(sim_names,
                                                              reference,
                                                              unit='000')
            elif re.search('simpsonL1', error_type):
                errors = np.array([0.0]*len(sim_names))
                for m_idx in range(0, len(sim_names)):
                    # abs value (negative oscillations might be substracted)
                    abs_comp_sum = np.sum(
                        abs(reference -
                            convergence.get_outlet(sim_names[m_idx], unit='000')),
                        axis=1
                    )
                    errors[m_idx] = scipy.integrate.simpson(
                        y=abs_comp_sum
                    )
            elif re.search('L1', error_type):
                errors = convergence.calculate_all_L1_errors(sim_names,
                                                             reference,
                                                             unit='000',
                                                             comps=[0, 1, 2, 3])
            else:
                raise ValueError("Error type " + str(error_type) + " unknown.")

            sim_times = convergence.get_compute_times(sim_names)
            DoF = convergence.calculate_DOFs([FV_ax_cells[i],
                                              FV_par_cells[i]],
                                             [0, 0],
                                             nComp=4,
                                             full_DOFs=True)

            line_args = {'linestyle': 'dashed', 'marker': 's'}

            if bench_dof:
                convergence.std_plot(
                    DoF, errors,
                    label='FV axZ/parZ: ' +
                    str(int(FV_ax_cells[i][0]/FV_par_cells[i][0])),
                    **line_args)
                hmpf = 'time'
            else:
                convergence.std_plot(sim_times, errors,
                                     label='FV axZ/parZ: ' +
                                     str(int(
                                         FV_ax_cells[i][0]/FV_par_cells[i][0])),
                                     **line_args)
                hmpf = 'DOF'

        plot_args = {'shape': [10, 10],
                     'y_label': error_type+' error in mol $/ m^3$'}
        plot_args['x_label'] = 'Degrees of freedom' if bench_dof else 'Compute time'
        convergence.std_plot_prep(**plot_args)

        if _export_results_:
            plt.savefig(_save_path_ +
                        'benchmark_FV_optRefinement_' +
                        _models_[0]+'_'+hmpf+'.png',
                        bbox_inches='tight')
        plt.show()

    if _optimal_FV_refinement_:
        plot_optimal_FV_strategy(
            FV_ax_cells=[[16, 32, 64, 128, 256, 512],
                         [16, 32, 64, 128, 256, 512]],
            FV_par_cells=[[8, 16, 32, 64, 128, 256],
                          [4, 8, 16, 32, 64, 128]],
            error_type='max', bench_dof=False)
        plot_optimal_FV_strategy(
            FV_ax_cells=[[16, 32, 64, 128, 256, 512],
                         [16, 32, 64, 128, 256, 512]],
            FV_par_cells=[[8, 16, 32, 64, 128, 256],
                          [4, 8, 16, 32, 64, 128]],
            error_type='max', bench_dof=True)

    # =========================================================================
    # Benchmark collocation DGSEM vs DGSEM
    # =========================================================================

    def benchmark_collocationDGSEM_GRM_LWE(model=_models_[0],
                                           polydegs=[methods[1]],
                                           cells=[ax_cells[1]],
                                           title=None):

        colours = ['blue', 'red', 'green',
                   'orange', 'brown', 'grey', 'magenta']

        plot_args = {'shape': [10, 10],
                     'y_label': '$L^\infty$ error in mol $/ m^3$'}
        # same order as _models_[]
        if model == _models_[0]:
            title = 'GRM with rapid equilibrium SMA binding'
        if title is not None:
            plot_args['title'] = title

        line_args_DG = {'linestyle': 'dotted',
                        'markerfacecolor': 'white'}
        line_args_cDG = {'linestyle': 'dashed',
                         'markerfacecolor': 'black'}

        plot_args['title'] = title
        # plot_args['y_lim'] = [1e-11, 2e0]

        for i in range(0, len(polydegs)):

            table_DG = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_DGSEM[model]['$N_d$'] == polydegs[i])
            ]

            table_cDG = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_cDGSEM[model]['$N_d$'] == polydegs[i])
            ]

            convergence.std_plot(table_DG['Sim. time'], table_DG['Max. error'],
                                 label='DG P'+str(int(polydegs[i])),
                                 color=colours[i],
                                 **line_args_DG)
            convergence.std_plot(table_cDG['Sim. time'], table_cDG['Max. error'],
                                 label='cDG P'+str(int(polydegs[i])),
                                 color=colours[i],
                                 **line_args_cDG)

        plot_args['x_label'] = 'Compute time in seconds'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = "P" + str(polydegs[0])
            if len(polydegs) > 1:
                save_m += "-P" + str(polydegs[len(polydegs)-1])
            plt.savefig(
                _save_path_ + 'benchmark_cDG_'+model+"_"+save_m+'_time.png',
                bbox_inches='tight')
        plt.show()

        for i in range(0, len(polydegs)):
            table_DG = tables_DGSEM[model][
                (tables_DGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_DGSEM[model]['$N_d$'] == polydegs[i])
            ]

            table_cDG = tables_cDGSEM[model][
                (tables_cDGSEM[model]['$N_e^z$'].isin(cells[i])) &
                (tables_cDGSEM[model]['$N_d$'] == polydegs[i])
            ]

            convergence.std_plot(table_DG['DoF'], table_DG['Max. error'],
                                 label='DG P'+str(int(polydegs[i])),
                                 color=colours[i],
                                 **line_args_DG)
            convergence.std_plot(table_cDG['DoF'], table_cDG['Max. error'],
                                 label='cDG P'+str(int(polydegs[i])),
                                 color=colours[i],
                                 **line_args_cDG)

        plot_args['x_label'] = 'DoF'
        convergence.std_plot_prep(**plot_args)
        if _export_results_:
            save_m = "P" + str(polydegs[0])
            if len(polydegs) > 1:
                save_m += "-P" + str(polydegs[len(polydegs)-1])
            plt.savefig(_save_path_ +
                        'benchmark_cDG_'+model+"_"+save_m+'_DoF.png',
                        bbox_inches='tight')
        plt.show()

    if _plot_benchmark_collocation_DGSEM_:

        ax_methods = np.array([1, 2, 3, 4, 5])
        if _paper_error_range_ == True:
            ax_cells_ = {
                0: [16, 32, 64, 128, 256, 512],
                1: [4, 8, 16, 32, 64, 128, 256],
                2: [4, 8, 16, 32, 64],
                3: [4, 8, 16, 32],
                4: [4, 8, 16],
                5: [4, 8, 16]
            }
        else:
            ax_cells_ = {
                0: [16, 32, 64, 128, 256, 512],
                1: [4, 8, 16, 32, 64, 128, 256],
                2: [4, 8, 16, 32, 64, 128],
                3: [4, 8, 16, 32, 64],
                4: [4, 8, 16, 32],
                5: [4, 8, 16, 32]
            }
        for method_idx in range(0, ax_methods.size):

            benchmark_collocationDGSEM_GRM_LWE(
                model=_models_[0],
                polydegs=[ax_methods[method_idx]],
                cells=[ax_cells_[ax_methods[method_idx]]]
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
