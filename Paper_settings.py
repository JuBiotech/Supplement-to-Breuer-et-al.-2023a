# -*- coding: utf-8 -*-
"""
Created March 2023

This script defines model settings considered in
"Spatial discontinuous Galerkin spectral element method in CADET"
published in computers and chemical engineering, 2023.

@author: Jan Michael Breuer
"""

import convergence
from cadet import Cadet
import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
# All simulations should be saved for the evaluation!
# 
# Naming of files follows the logic: prefix + "_" + transport model + "_" +
# binding model + "_" + nComp + "_" + siscretization
# Example: heyho_GRMsurfDiff_reqLinear_1Comp_DG_P3Z5parP4Z1.h5
# =============================================================================


def GRMlinear1Comp_VerificationSetting(
        method_=0, colCells_=5,
        parMethod_=0, parCells_=1,
        isKinetic_=True, surfDiff=False,
        tolerance=1e-12,
        plot_=False, run_=True,
        save_path="C:/Users/jmbr/JupyterNotebooks/",
        cadet_path="C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE/bin/cadet-cli.exe"
):

    Cadet.cadet_path = cadet_path

    # =========================================================================
    # PARAMETERS
    # =========================================================================
    velocity = 5.75E-4    # m / s
    film_diffusion = 6.9e-6    # m / s
    length = 0.014    # m
    adsorption = 3.55    # m^3 / (mol * s)   (mobile phase)
    desorption = 0.1    # 1 / s (desorption)
    isKinetic = isKinetic_
    D_ax = 5.75e-8    # m^2 / s (interstitial volume)
    par_diffusion = 6.07E-11
    if(surfDiff):
        par_surfdiffusion = 1.0E-11
    else:
        par_surfdiffusion = 0.0
    porosity = 0.37
    par_porosity = 0.75
    par_radius = 4.5e-5    # m
    inlet = 1.0    # mol / m^3 = (1/1000) mol / liter
    t_end = 1500    # s

    solution_times_factor = 4
    idas_tolerance = tolerance
    #########################

    model = Cadet()
    # SYSTEM
    model.root.input.model.nunits = 3
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1
        0, 1, -1, -1, 60/1e6,
        1, 2, -1, -1, 60/1e6]
    model.root.input.solver.sections.nsec = 2
    model.root.input.solver.sections.section_times = [0.0, 10, t_end]
    model.root.input.solver.sections.section_continuity = [0]

    # SOLVER
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    model.root.input.solver.nthreads = 1
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = idas_tolerance
    model.root.input.solver.time_integrator.algtol = idas_tolerance*100
    model.root.input.solver.time_integrator.reltol = idas_tolerance*100
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # UNIT OPERATIONS - INLET
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = 1
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.root.input.model.unit_000.sec_000.const_coeff = [
        inlet,]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.const_coeff = [
        0.0,]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_001.lin_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.quad_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.cube_coeff = [0.0,]

    # UNIT OPERATIONS - MODEL
    if(method_ == 0):
        model.root.input.model.unit_001.unit_type = 'GENERAL_RATE_MODEL'
    else:
        model.root.input.model.unit_001.unit_type = 'GENERAL_RATE_MODEL_DG'
    model.root.input.model.unit_001.ncomp = 1
    model.root.input.model.unit_001.adsorption_model = 'LINEAR'
    model.root.input.model.unit_001.adsorption_model_multiplex = 1
    # NO REACTIONS
    model.root.input.model.unit_001.adsorption.is_kinetic = isKinetic
    model.root.input.model.unit_001.adsorption.lin_ka = [
        adsorption,]      # m^3 / (mol * s)   (mobile phase)
    model.root.input.model.unit_001.adsorption.lin_kd = [
        desorption,]      # 1 / s (desorption)
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_c = [0.0,]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_cp = [0.0,]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_q = [0.0,]
    model.root.input.model.unit_001.col_dispersion = D_ax
    model.root.input.model.unit_001.col_length = length
    model.root.input.model.unit_001.col_porosity = porosity
    model.root.input.model.unit_001.film_diffusion = film_diffusion
    model.root.input.model.unit_001.film_diffusion_multiplex = 0
    model.root.input.model.unit_001.par_porosity = par_porosity
    model.root.input.model.unit_001.par_radius = par_radius
    model.root.input.model.unit_001.par_coreradius = 0.0

    model.root.input.model.unit_001.par_diffusion = par_diffusion
    model.root.input.model.unit_001.PAR_SURFDIFFUSION = par_surfdiffusion

    model.root.input.model.unit_001.velocity = velocity
    model.root.input.model.unit_001.par_type_volfrac = 1

    # Discretization
    model.root.input.model.unit_001.discretization.ncol = colCells_

    model.root.input.model.unit_001.discretization.npartype = 1
    if(method_ == 0):
        model.root.input.model.unit_001.discretization.npar = parCells_
    else:
        model.root.input.model.unit_001.discretization.polyDeg = abs(
            int(method_))     # polynomial degree
        model.root.input.model.unit_001.discretization.exact_integration = 0 if method_ > 0 else 1
        model.root.input.model.unit_001.discretization.parPolyDeg = abs(
            int(parMethod_))  # polynomial degree
        model.root.input.model.unit_001.discretization.nparcell = parCells_
        model.root.input.model.unit_001.discretization.par_exact_integration = 1 if parMethod_ > 0 else 0
    model.root.input.model.unit_001.discretization.nbound = [1]
    model.root.input.model.unit_001.discretization.par_geom = [
        "SPHERE"]      # particle geometry (sphere, cylinder, slab)
    model.root.input.model.unit_001.discretization.par_disc_type = [
        "EQUIDISTANT_PAR"]  # EQUIDISTANT_PAR, EQUIVOLUME_PAR, USER_DEFINED_PAR
    model.root.input.model.unit_001.discretization.par_boundary_order = 2
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.gs_type = 0
    model.root.input.model.unit_001.discretization.max_krylov = 10
    model.root.input.model.unit_001.discretization.max_restarts = 100
    model.root.input.model.unit_001.discretization.schur_safety = 0.1
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 3

    # UNIT OPERATIONS - OUTLET
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = 1

    # RETURN DATA
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 0
    model.root.input['return'].unit_001.write_solution_bulk = 0
    model.root.input['return'].unit_001.write_solution_inlet = 0
    model.root.input['return'].unit_001.write_solution_particle = 0
    model.root.input['return'].unit_001.write_solution_outlet = 1
    model.root.input['return'].unit_001.WRITE_COORDINATES = 0

    # Copy inlet return settings to the outlet unit operation
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0, t_end, t_end*solution_times_factor + 1)

    prefix = save_path+"GRM"
    if surfDiff:
        prefix += "sd"
    prefix += "_dynLin" if isKinetic else "_reqLin"
    prefix += "_1comp"
    model.filename = convergence.generate_GRM_name(
        prefix, method_, colCells_, parMethod_, parCells_, '.h5')

    model.save()

    # RUN
    if(run_):
        data = model.run()
        if data.returncode == 0:
            print(model.filename + " simulation completed successfully")
            model.load()
        else:
            print(data)
            raise Exception(model.filename + " simulation failed")

        # PLOT
        if(plot_):
            solution_times = model.root.output.solution.solution_times
            c_outlet = model.root.output.solution.unit_001.solution_outlet

            plt.plot()
            plt.plot(solution_times, c_outlet)
            plt.xlabel('$time~/~min$')
            plt.ylabel('$Outlet~concentration~/~mol \cdot m^{-3} $')
            plt.show()


def LRMlinear1Comp_VerificationSetting(
        method_=0, colCells_=5,
        isKinetic_=True,
        tolerance=1e-12, plot_=False, run_=True,
        save_path="C:\\Users\\jmbr\\JupyterNotebooks\\",
        cadet_path="C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE/bin/cadet-cli.exe"
):

    Cadet.cadet_path = cadet_path

    # =========================================================================
    #     PARAMETERS
    # =========================================================================
    velocity = 2.0/60.0                # m / s = 6000 cm / min
    length = 1.0                       # m
    adsorption = 1.0                   # m^3 / (mol * s)   (mobile phase)
    desorption = 1.0                   # 1 / s (desorption)
    D_ax = 1e-4                        # m^2 / s (interstitial volume)
    porosity = 0.6
    inlet = 1.0                        # mol / m^3 = (1/1000) mol / liter
    nCells = colCells_
    t_end = 130
    #########################

    model = Cadet()
    # SYSTEM
    model.root.input.model.nunits = 3
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1
        0, 1, -1, -1, 60/1e6,
        1, 2, -1, -1, 60/1e6]
    model.root.input.solver.sections.nsec = 2
    model.root.input.solver.sections.section_times = [0.0, 60, t_end]   # s
    model.root.input.solver.sections.section_continuity = [0]

    # SOLVER
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    model.root.input.solver.nthreads = 1
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = tolerance
    model.root.input.solver.time_integrator.algtol = tolerance*100
    model.root.input.solver.time_integrator.reltol = tolerance*100
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # UNIT OPERATIONS - INLET
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = 1
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.root.input.model.unit_000.sec_000.const_coeff = [
        inlet,]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.const_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.lin_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.quad_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.cube_coeff = [0.0,]

    # UNIT OPERATIONS - MODEL
    if method_ == 0:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    else:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES_DG'
    model.root.input.model.unit_001.ncomp = 1
    model.root.input.model.unit_001.adsorption_model = 'LINEAR'
    model.root.input.model.unit_001.adsorption.is_kinetic = int(
        isKinetic_)    # Kinetic binding
    model.root.input.model.unit_001.adsorption.lin_ka = [adsorption,]
    model.root.input.model.unit_001.adsorption.lin_kd = [desorption,]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_c = [0.0,]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_q = [0.0,]
    model.root.input.model.unit_001.col_dispersion = D_ax
    model.root.input.model.unit_001.col_length = length
    model.root.input.model.unit_001.total_porosity = porosity
    # m / s = 6000 cm / min
    model.root.input.model.unit_001.velocity = velocity
    # Discretization
    model.root.input.model.unit_001.discretization.ncol = nCells
    if method_ != 0:
        model.root.input.model.unit_001.discretization.polydeg = int(
            abs(method_))
        model.root.input.model.unit_001.discretization.exact_integration = 0 if method_ > 0 else 1
    model.root.input.model.unit_001.discretization.nbound = [1]  # Bound states
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 3

    # UNIT OPERATIONS - OUTLET
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = 1

    # RETURN DATA
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 0
    model.root.input['return'].unit_001.write_solution_bulk = 0
    model.root.input['return'].unit_001.write_solution_inlet = 0
    model.root.input['return'].unit_001.write_solution_outlet = 1
    model.root.input['return'].unit_001.WRITE_COORDINATES = 0

    # Copy inlet return settings to the outlet unit operation
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0, t_end, t_end + 1)

    prefix = save_path + 'LRM'
    if isKinetic_:
        prefix += '_dyn'
    else:
        prefix += '_req'
    prefix += '_1comp'

    model.filename = convergence.generate_1D_name(prefix, method_, nCells)
    model.save()

    # RUN
    if run_:
        data = model.run()
        if data.returncode == 0:
            print("method " + str(int(method_)) + "Z" +
                  str(int(nCells)) + " simulation completed successfully")
            model.load()
        else:
            print(data)
            raise Exception("method " + str(int(method_)) +
                            "Z" + str(int(nCells)) + " simulation failed")

    if plot_:
        solution_times = model.root.output.solution.solution_times
        c_outlet = model.root.output.solution.unit_001.solution_outlet

        plt.plot()
        plt.plot(solution_times, c_outlet)
        plt.xlabel('$time~/~min$')
        plt.ylabel('$Outlet~concentration~/~mol \cdot m^{-3} $')
        plt.show()


def LRMPlinear1Comp_VerificationSetting(
        method_=0, colCells_=5,
        isKinetic_=True,
        tolerance=1e-12, plot_=False, run_=True,
        save_path="C:\\Users\\jmbr\\JupyterNotebooks\\",
        cadet_path="C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE/bin/cadet-cli.exe"
):

    Cadet.cadet_path = cadet_path

    ## PARAMETERS ###########
    velocity = 2.0/60.0                # m / s
    film_diffusion = [velocity / 10]     # m / s
    length = 1.0                       # m
    # m^3 / (mol * s)   (mobile phase)
    adsorption = [1.0]
    desorption = [1.0]                  # 1 / s (desorption)
    D_ax = [1e-4]                        # m^2 / s (interstitial volume)
    porosity = 0.6
    par_porosity = 0.2
    par_radius = 0.0001               # m
    inlet = 1.0                       # mol / m^3 = (1/1000) mol / liter
    nCells = colCells_
    t_end = 130
    #########################

    model = Cadet()
    # SYSTEM
    model.root.input.model.nunits = 3
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1
        0, 1, -1, -1, 60/1e6,
        1, 2, -1, -1, 60/1e6]
    model.root.input.solver.sections.nsec = 2
    model.root.input.solver.sections.section_times = [0.0, 60, t_end]   # s
    model.root.input.solver.sections.section_continuity = [0]

    # SOLVER
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    model.root.input.solver.nthreads = 1
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-12
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-10
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # UNIT OPERATIONS - INLET
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = 1
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.root.input.model.unit_000.sec_000.const_coeff = [
        inlet, inlet]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0]
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0]
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0]
    model.root.input.model.unit_000.sec_001.const_coeff = [
        0.0]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_001.lin_coeff = [0.0]
    model.root.input.model.unit_000.sec_001.quad_coeff = [0.0]
    model.root.input.model.unit_000.sec_001.cube_coeff = [0.0]

    # UNIT OPERATIONS - MODEL
    if method_ == 0:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES'
    else:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITH_PORES_DG'
    model.root.input.model.unit_001.ncomp = 1
    model.root.input.model.unit_001.adsorption_model = 'LINEAR'
    model.root.input.model.unit_001.adsorption_model_multiplex = 1
    model.root.input.model.unit_001.adsorption.is_kinetic = int(
        isKinetic_)    # Kinetic binding
    # m^3 / (mol * s)   (mobile phase)
    model.root.input.model.unit_001.adsorption.lin_ka = adsorption
    # 1 / s (desorption)
    model.root.input.model.unit_001.adsorption.lin_kd = desorption
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_c = [0.0]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_cp = [0.0]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_q = [0.0]
    model.root.input.model.unit_001.col_dispersion = D_ax
    model.root.input.model.unit_001.col_length = length
    model.root.input.model.unit_001.col_porosity = porosity
    model.root.input.model.unit_001.film_diffusion = film_diffusion
    model.root.input.model.unit_001.film_diffusion_multiplex = 1
    model.root.input.model.unit_001.par_porosity = par_porosity
    model.root.input.model.unit_001.par_radius = par_radius
    model.root.input.model.unit_001.velocity = velocity
    model.root.input.model.unit_001.par_type_volfrac = 1
    # Discretization
    if method_ != 0:
        model.root.input.model.unit_001.discretization.exact_integration = 0 if method_ > 0 else 1     # cells
        model.root.input.model.unit_001.discretization.polydeg = int(
            abs(method_))     # cells
    model.root.input.model.unit_001.discretization.ncol = nCells

    model.root.input.model.unit_001.discretization.npartype = 1
    model.root.input.model.unit_001.discretization.nbound = [1]
    model.root.input.model.unit_001.discretization.par_geom = [
        "SPHERE"]      # particle geometry (sphere, cylinder, slab)
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.gs_type = 0
    model.root.input.model.unit_001.discretization.max_krylov = 10
    model.root.input.model.unit_001.discretization.max_restarts = 100
    model.root.input.model.unit_001.discretization.schur_safety = 0.1
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 3

    # UNIT OPERATIONS - OUTLET
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = 1

    # RETURN DATA
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 0
    model.root.input['return'].unit_001.write_solution_bulk = 1
    model.root.input['return'].unit_001.write_solution_inlet = 1
    model.root.input['return'].unit_001.write_solution_outlet = 1
    model.root.input['return'].unit_001.WRITE_COORDINATES = 1
    # Copy inlet return settings to the outlet unit operation
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000
    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0, t_end, t_end + 1)
    prefix = save_path + 'LRMP'
    if isKinetic_:
        prefix += '_dyn'
    else:
        prefix += '_req'
    prefix += '_1comp'

    model.filename = convergence.generate_1D_name(prefix, method_, nCells)
    model.save()

    # RUN
    if(run_):
        data = model.run()
        if data.returncode == 0:
            print("method " + str(int(method_)) + "Z" +
                  str(int(nCells)) + " simulation completed successfully")
            model.load()
        else:
            print(data)
            raise Exception("method " + str(int(method_)) +
                            "Z" + str(int(nCells)) + " simulation failed")

    if(plot_):
        solution_times = model.root.output.solution.solution_times
        c_outlet = model.root.output.solution.unit_001.solution_outlet

        plt.plot()
        plt.plot(solution_times, c_outlet)
        plt.xlabel('$time~/~min$')
        plt.ylabel('$Outlet~concentration~/~mol \cdot m^{-3} $')
        plt.show()


def LRMlinear1Comp_noBind(
        method_=0, colCells_=5,
        tolerance=1e-12,
        plot_=False, run_=True,
        save_path="C:\\Users\\jmbr\\JupyterNotebooks\\",
        cadet_path="C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE/bin/cadet-cli.exe"
):

    Cadet.cadet_path = cadet_path

    ## PARAMETERS ###########
    velocity = 2.0/60.0                # m / s = 6000 cm / min
    length = 1.0                       # m
    D_ax = 5e-4                        # m^2 / s (interstitial volume)
    porosity = 0.6
    inlet = 1.0                       # mol / m^3 = (1/1000) mol / liter
    nCells = colCells_
    t_end = 130
    #########################

    model = Cadet()
    # SYSTEM
    model.root.input.model.nunits = 3
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1
        0, 1, -1, -1, 60/1e6,
        1, 2, -1, -1, 60/1e6]
    model.root.input.solver.sections.nsec = 2
    model.root.input.solver.sections.section_times = [0.0, 60, t_end]   # s
    model.root.input.solver.sections.section_continuity = [0]

    # SOLVER
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    model.root.input.solver.nthreads = 1
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = tolerance
    model.root.input.solver.time_integrator.algtol = tolerance*100
    model.root.input.solver.time_integrator.reltol = tolerance*100
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # UNIT OPERATIONS - INLET
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = 1
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.root.input.model.unit_000.sec_000.const_coeff = [
        inlet,]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.const_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.lin_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.quad_coeff = [0.0,]
    model.root.input.model.unit_000.sec_001.cube_coeff = [0.0,]

    # UNIT OPERATIONS - MODEL
    if method_ == 0:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    else:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES_DG'
    model.root.input.model.unit_001.ncomp = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_c = [0.0,]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_q = [0.0,]
    model.root.input.model.unit_001.col_dispersion = D_ax
    model.root.input.model.unit_001.col_length = length
    model.root.input.model.unit_001.total_porosity = porosity
    model.root.input.model.unit_001.velocity = velocity
    # Discretization
    model.root.input.model.unit_001.discretization.ncol = nCells
    if method_ != 0:
        model.root.input.model.unit_001.discretization.polydeg = int(
            abs(method_))
        model.root.input.model.unit_001.discretization.exact_integration = 0 if method_ > 0 else 1
    model.root.input.model.unit_001.discretization.nbound = [1]  # Bound states
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 3

    # UNIT OPERATIONS - OUTLET
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = 1

    # RETURN DATA
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 0
    model.root.input['return'].unit_001.write_solution_bulk = 0
    model.root.input['return'].unit_001.write_solution_inlet = 0
    model.root.input['return'].unit_001.write_solution_outlet = 1
    model.root.input['return'].unit_001.WRITE_COORDINATES = 0
    # Copy inlet return settings to the outlet unit operation
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000
    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0, t_end, t_end + 1)

    prefix = save_path + 'LRM_noBind'
    prefix += '_1comp'

    model.filename = convergence.generate_1D_name(prefix, method_, nCells)
    model.save()

    # RUN
    if(run_):
        data = model.run()
        if data.returncode == 0:
            print("method " + str(int(method_)) + "Z" +
                  str(int(nCells)) + " simulation completed successfully")
            model.load()
        else:
            print(data)
            raise Exception("method " + str(int(method_)) +
                            "Z" + str(int(nCells)) + " simulation failed")

    if(plot_):
        solution_times = model.root.output.solution.solution_times
        c_outlet = model.root.output.solution.unit_001.solution_outlet

        plt.plot()
        plt.plot(solution_times, c_outlet)
        plt.xlabel('$time~/~min$')
        plt.ylabel('$Outlet~concentration~/~mol \cdot m^{-3} $')
        plt.show()


def LWE_setting(
        method, nCells,
        par_method, nParCells,
        save_path,
        transport_model='GENERAL_RATE_MODEL',
        is_kinetic=0,
        idas_tolerance=1E-8,
        cadet_path="C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE/bin/cadet-cli.exe",
        rad_method=0, nRadCells=3,
        run_sim=False,
        n_threads=1):

    Cadet.cadet_path = cadet_path

    # =========================================================================
    #     PARAMETERS
    # =========================================================================
    polyDeg = int(abs(method))
    parPolyDeg = int(abs(par_method))

    velocity = 5.75e-4    # m / s
    length = 0.014    # m
    D_ax = 5.75e-8    # m^2 / s (interstitial volume)
    col_porosity = 0.37
    par_porosity = 0.75
    total_porosity = 0.8425

    film_diffusion = [6.9E-6, 6.9E-6, 6.9E-6, 6.9E-6]
    par_radius = 4.5E-5
    par_diffusion = [70.0E-11, 6.07E-11, 6.07E-11, 6.07E-11]
    surface_diffusion = [0.0, 0.0, 0.0, 0.0]

    binding = "STERIC_MASS_ACTION"
    adsorption = [0.0, 35.5, 1.59, 7.7]    # m^2 / s
    desorption = [0.0, 1000.0, 1000.0, 1000.0]    # 1 / s
    charge = [0.0, 4.7, 5.29, 3.7]
    steric_factor = [0.0, 11.83, 10.6, 10.0]
    n_binding_sites = 1200.0

    init_cb_cp = [50, 0.0, 0.0, 0.0]
    init_cs = [1200, 0.0, 0.0, 0.0]

    inlet1 = [50.0, 1.0, 1.0, 1.0]    # mol / m^3 = (1/1000) mol / liter
    inlet2 = [50.0, 0.0, 0.0, 0.0]
    inlet3 = [100.0, 0.0, 0.0, 0.0]
    inlet3_lin = [0.2, 0.0, 0.0, 0.0]

    nComp = 4
    time_sections = [0, 12, 40]
    solution_resolution = 1.0
    section_times = [0.0, 10.0, 90.0, 1500.0]
    #########################

    model = Cadet()
    # SYSTEM
    model.root.input.model.nunits = 2
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.CONNECTIONS_INCLUDE_PORTS = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 1.0
    ]  # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1
    model.root.input.solver.sections.nsec = 3
    model.root.input.solver.sections.section_times = section_times   # s
    model.root.input.solver.sections.section_continuity = [0, 0]

    # SOLVER
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = n_threads
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = idas_tolerance
    model.root.input.solver.time_integrator.algtol = idas_tolerance*100
    model.root.input.solver.time_integrator.reltol = idas_tolerance*100
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 10000

    # UNIT OPERATIONS - INLET
    model.root.input.model.unit_001.unit_type = 'INLET'
    model.root.input.model.unit_001.ncomp = nComp
    model.root.input.model.unit_001.inlet_type = 'PIECEWISE_CUBIC_POLY'
    # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_001.sec_000.const_coeff = inlet1
    model.root.input.model.unit_001.sec_000.lin_coeff = [0.0, 0.0, 0.0, 0.0]
    model.root.input.model.unit_001.sec_000.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    model.root.input.model.unit_001.sec_000.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_001.sec_001.const_coeff = inlet2
    model.root.input.model.unit_001.sec_001.lin_coeff = [0.0, 0.0, 0.0, 0.0]
    model.root.input.model.unit_001.sec_001.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    model.root.input.model.unit_001.sec_001.cube_coeff = [0.0, 0.0, 0.0, 0.0]
    # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_001.sec_002.const_coeff = inlet3
    model.root.input.model.unit_001.sec_002.lin_coeff = inlet3_lin
    model.root.input.model.unit_001.sec_002.quad_coeff = [0.0, 0.0, 0.0, 0.0]
    model.root.input.model.unit_001.sec_002.cube_coeff = [0.0, 0.0, 0.0, 0.0]

    # UNIT OPERATIONS - MODEL
    if method == 0:
        model.root.input.model.unit_000.unit_type = transport_model
    else:
        model.root.input.model.unit_000.unit_type = transport_model + '_DG'
    model.root.input.model.unit_000.ncomp = nComp

    model.root.input.model.unit_000.adsorption_model = binding
    model.root.input.model.unit_000.adsorption.is_kinetic = is_kinetic
    # m^3 / (mol * s)   (mobile phase)
    model.root.input.model.unit_000.adsorption.SMA_KA = adsorption
    # 1 / s (desorption)
    model.root.input.model.unit_000.adsorption.SMA_KD = desorption
    model.root.input.model.unit_000.adsorption.SMA_LAMBDA = n_binding_sites
    model.root.input.model.unit_000.adsorption.SMA_NU = charge
    model.root.input.model.unit_000.adsorption.SMA_SIGMA = steric_factor
    model.root.input.model.unit_000.par_radius = par_radius
    model.root.input.model.unit_000.par_coreradius = 0.0

    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_000.init_c = init_cb_cp
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_000.init_q = init_cs

    # m / s = 6000 cm / min
    model.root.input.model.unit_000.velocity = velocity
    model.root.input.model.unit_000.col_dispersion = D_ax
    model.root.input.model.unit_000.par_diffusion = par_diffusion
    model.root.input.model.unit_000.par_surfdiffusion = surface_diffusion
    model.root.input.model.unit_000.film_diffusion = film_diffusion
    model.root.input.model.unit_000.col_dispersion_radial = 1.0E-6

    model.root.input.model.unit_000.col_length = length
    model.root.input.model.unit_000.col_radius = 0.01
    model.root.input.model.unit_000.col_porosity = col_porosity
    model.root.input.model.unit_000.par_porosity = par_porosity
    model.root.input.model.unit_000.total_porosity = total_porosity
    # Discretization
    model.root.input.model.unit_000.discretization.ncol = nCells
    model.root.input.model.unit_000.discretization.nrad = nRadCells
    if method != 0:
        model.root.input.model.unit_000.discretization.polydeg = polyDeg
        model.root.input.model.unit_000.discretization.exact_integration = 1 if method < 0 else 0
        model.root.input.model.unit_000.discretization.par_exact_integration = 0 if par_method < 0 else 1
        model.root.input.model.unit_000.discretization.nparcell = nParCells
        model.root.input.model.unit_000.discretization.parpolydeg = parPolyDeg
    else:
        model.root.input.model.unit_000.discretization.npar = nParCells
    model.root.input.model.unit_000.discretization.nbound = [1, 1, 1, 1]
    # model.root.input.model.unit_000.discretization.par_geom = [
    #     "SPHERE"]      # particle geometry (sphere, cylinder, slab)
    model.root.input.model.unit_000.discretization.par_disc_type = [
        "EQUIDISTANT_PAR"]  # EQUIDISTANT_PAR, EQUIVOLUME_PAR, USER_DEFINED_PAR
    model.root.input.model.unit_000.discretization.radial_disc_type = [
        "EQUIDISTANT"]  # EQUIDISTANT_PAR, EQUIVOLUME_PAR, USER_DEFINED_PAR

    model.root.input.model.unit_000.discretization.use_analytic_jacobian = 1
    # model.root.input.model.unit_000.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_000.discretization.gs_type = 1
    model.root.input.model.unit_000.discretization.max_krylov = 0
    model.root.input.model.unit_000.discretization.max_restarts = 10
    model.root.input.model.unit_000.discretization.schur_safety = 1E-8
    model.root.input.model.unit_000.discretization.weno.boundary_model = 0
    model.root.input.model.unit_000.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_000.discretization.weno.weno_order = 3

    # RETURN DATA
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].write_solution_times = 1
    model.root.input['return'].unit_000.WRITE_SENS_BULK = 0
    model.root.input['return'].unit_000.WRITE_SOLUTION_SOLID = 0
    model.root.input['return'].unit_000.WRITE_COORDINATES = 0
    model.root.input['return'].unit_000.WRITE_SOLUTION_BULK = 0
    model.root.input['return'].unit_000.WRITE_SOLUTION_PARTICLE = 0
    model.root.input['return'].unit_000.WRITE_SOLUTION_INLET = 0
    model.root.input['return'].unit_000.WRITE_SOLUTION_OUTLET = 1
    model.root.input['return'].unit_000.WRITE_SOLUTION_FLUX = 0
    model.root.input['return'].unit_000.WRITE_SENS_OUTLET = 1
    model.root.input['return'].unit_000.WRITE_SENS_FLUX = 0
    model.root.input['return'].unit_000.WRITE_SENS_PARTICLE = 0
    model.root.input['return'].unit_000.WRITE_SENS_INLET = 0

    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        section_times[0],
        section_times[3],
        int(section_times[3] * solution_resolution) + 1)

    name = save_path
    if n_threads != 1:
        name += str(n_threads) + 'thread'
    if transport_model == 'GENERAL_RATE_MODEL':
        model.filename = convergence.generate_GRM_name(
            name+'GRM_SMA_4comp',
            method, nCells, par_method, nParCells, '.h5')
    elif transport_model == 'LUMPED_RATE_MODEL_WITH_PORES':
        model.filename = convergence.generate_1D_name(
            name+'LRMP_SMA_4comp', method, nCells, '.h5')
    elif transport_model == 'LUMPED_RATE_MODEL_WITHOUT_PORES':
        model.filename = convergence.generate_1D_name(
            name+'LRM_SMA_4comp', method, nCells, '.h5')

    model.save()

    # RUN
    if run_sim:
        data = model.run()
        if data.returncode == 0:
            print(model.filename + " simulation completed successfully")
            model.load()
        else:
            print(data)
            raise Exception(model.filename + " simulation failed")


def LRM_langmuir_oscillations(
        method, ncells,
        save_path,
        D_ax=1e-4,
        is_kinetic=False,
        cadet_path="C:/Users/jmbr/Cadet/code/out/install/MS_MKL_RELEASE/bin/cadet-cli.exe",
        run_=True):

    Cadet.cadet_path = cadet_path

    ## PARAMETERS ###########
    polyDeg = int(abs(method))
    nCells = ncells

    velocity = 0.1                # m / s = 6000 cm / min
    length = 1.0                       # m
    adsorption = [0.1, 0.05]
    desorption = [1.0, 1.0]
    qmax = [10.0, 10.0]
    binding = "MULTI_COMPONENT_LANGMUIR"
    D_ax = D_ax  # 1e-4                        # m^2 / s (interstitial volume)
    porosity = 0.4
    inlet1 = 10.0                       # mol / m^3 = (1/1000) mol / liter
    inlet2 = 10.0                       # mol / m^3 = (1/1000) mol / liter
    nComp = 2
    time_sections = [0, 12, 40]
    solution_resolution = 10
    time_sections = [0, 12, 40]
    #########################

    model = Cadet()
    # SYSTEM
    model.root.input.model.nunits = 3
    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        # [unit_000, unit_001, all components, all components, Q/ m^3*s^-1
        0, 1, -1, -1, 60/1e6,
        1, 2, -1, -1, 60/1e6]  # [unit_001, unit_002, all components, all components, Q/ m^3*s^-1
    model.root.input.solver.sections.nsec = 2
    model.root.input.solver.sections.section_times = time_sections   # s
    model.root.input.solver.sections.section_continuity = [0]

    # SOLVER
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8
    model.root.input.solver.nthreads = 1
    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-10
    model.root.input.solver.time_integrator.algtol = 1e-8
    model.root.input.solver.time_integrator.reltol = 1e-8
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # UNIT OPERATIONS - INLET
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = nComp
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'
    model.root.input.model.unit_000.sec_000.const_coeff = [
        inlet1, inlet2]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_000.lin_coeff = [0.0, 0.0]
    model.root.input.model.unit_000.sec_000.quad_coeff = [0.0, 0.0]
    model.root.input.model.unit_000.sec_000.cube_coeff = [0.0, 0.0]
    model.root.input.model.unit_000.sec_001.const_coeff = [
        0.0, 0.0]  # mol / m^3 = (1/1000) mol / liter
    model.root.input.model.unit_000.sec_001.lin_coeff = [0.0, 0.0]
    model.root.input.model.unit_000.sec_001.quad_coeff = [0.0, 0.0]
    model.root.input.model.unit_000.sec_001.cube_coeff = [0.0, 0.0]

    # UNIT OPERATIONS - MODEL
    if method == 0:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES'
    else:
        model.root.input.model.unit_001.unit_type = 'LUMPED_RATE_MODEL_WITHOUT_PORES_DG'
    model.root.input.model.unit_001.ncomp = nComp
    model.root.input.model.unit_001.adsorption_model = binding
    model.root.input.model.unit_001.adsorption.is_kinetic = is_kinetic
    # m^3 / (mol * s)   (mobile phase)
    model.root.input.model.unit_001.adsorption.MCL_KA = adsorption
    # 1 / s (desorption)
    model.root.input.model.unit_001.adsorption.MCL_KD = desorption
    model.root.input.model.unit_001.adsorption.MCL_QMAX = qmax
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_c = [0.0, 0.0]
    # mol / m^3 =  (1/1000) mol / liter
    model.root.input.model.unit_001.init_q = [0.0, 0.0]
    model.root.input.model.unit_001.col_dispersion = D_ax
    model.root.input.model.unit_001.col_length = length
    model.root.input.model.unit_001.total_porosity = porosity
    # m / s = 6000 cm / min
    model.root.input.model.unit_001.velocity = velocity
    # Discretization
    model.root.input.model.unit_001.discretization.ncol = nCells
    if method != 0:
        model.root.input.model.unit_001.discretization.polydeg = polyDeg
        model.root.input.model.unit_001.discretization.exact_integration = 1 if method < 0 else 0
    model.root.input.model.unit_001.discretization.nbound = [1, 1]
    model.root.input.model.unit_001.discretization.use_analytic_jacobian = 1
    # @todo: needed in convDispOp FV
    model.root.input.model.unit_001.discretization.reconstruction = 'WENO'
    model.root.input.model.unit_001.discretization.weno.boundary_model = 0
    model.root.input.model.unit_001.discretization.weno.weno_eps = 1e-10
    model.root.input.model.unit_001.discretization.weno.weno_order = 3

    # UNIT OPERATIONS - OUTLET
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = nComp

    # RETURN DATA
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 0
    model.root.input['return'].unit_000.write_solution_inlet = 0
    model.root.input['return'].unit_000.write_solution_outlet = 0
    model.root.input['return'].unit_001.write_solution_bulk = 1
    model.root.input['return'].unit_001.write_solution_inlet = 1
    model.root.input['return'].unit_001.write_solution_outlet = 1
    model.root.input['return'].unit_001.WRITE_COORDINATES = 1
    # Copy inlet return settings to the outlet unit operation
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000
    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(
        0,
        time_sections[len(time_sections)-1],
        solution_resolution*time_sections[len(time_sections)-1] + 1)

    prefix = save_path
    if D_ax == 1e-4:
        prefix += "LRMdisp0.0001_"
    elif D_ax == 1e-5:
        prefix += "LRMdisp1e-5_"
    else:
        prefix += str(D_ax)

    prefix += 'dyn' if is_kinetic else 'req'
    prefix += 'Langmuir_2comp'

    model.filename = convergence.generate_1D_name(prefix, method, nCells)
    model.save()

    # RUN
    if(run_):
        data = model.run()
        if data.returncode == 0:
            print(model.filename + " simulation completed successfully")
            model.load()
        else:
            print(data)
            raise Exception(model.filename + " simulation failed")
