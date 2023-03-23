# -*- coding: utf-8 -*-
"""
Created January 2023

This script implements evaluation functionalities related to numerical 
convergence analysis for CADET simulations.

@author: Jan Michael Breuer
"""

from cadet import Cadet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import h5py
import pytest
import shutil
from colorama import Fore


def get_simulation(simulation):
    """Get CADET object from h5 file.

    Parameters
    ----------
    simulation : Either string specifying an h5 file (with path)
    or CADET object

    Returns
    -------
    CADET object
    """
    if isinstance(simulation, str):
        sim = Cadet()
        sim.filename = simulation
        sim.load()
        return sim

    elif not isinstance(simulation, Cadet):
        raise ValueError(
            "Data provided is neither string to specify an h5 file nor CADET object, but " +
            str(type(simulation))
        )

    return simulation


def sim_go_to(dictionary, keys):
    """Move in Cadet object or h5 groups via dict

    Parameters
    ----------
    dictionary : dict
         Specifies the structure of Cadet object or h5.
    unit : array of strings
        Specifies the path to take.

    Returns
    -------
    dict or any
        Data or dict or dict at the and of path.
    """
    keys = np.array(keys)

    for key in keys:
        if key in dictionary.keys():
            dictionary = dictionary[key]
        else:
            raise ValueError(
                "Simulation does not contain group: " + key
            )

    return dictionary

# TODO? currently only allows split_components_data = false


def get_solution(simulation, unit='unit_001', which='outlet'):
    """Get solution from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """

    sol = np.squeeze(
        sim_go_to(get_simulation(simulation).root,
                  ['output',
                   'solution',
                   unit,
                   'solution_' + which]
                  )
    )

    return sol


def get_outlet(simulation, unit='001'):
    """Get outlet from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Outlet vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'outlet')


def get_bulk(simulation, unit='001'):
    """Get bulk solution from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Bulk solution vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'bulk')


def get_solid(simulation, unit='001'):
    """Get solid (particle) solution from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'solid')


def get_particle(simulation, unit='001'):
    """Get liquid particle solution from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    return get_solution(simulation, 'unit_' + unit, 'particle')


def get_solution_times(simulation):
    """Get solution times from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'solution',
                                              'solution_times']
        )
    )


def get_compute_time(simulation):
    """Get compute time from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest

    Returns
    -------
    np.array
        Solution vector.
    """
    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['meta',
                                              'time_sim']
        )
    )


def get_compute_times(simulations):

    computeTimes = np.zeros(len(simulations))

    for simulation in range(0, len(simulations)):

        computeTimes[simulation] = get_compute_time(simulations[simulation])

    return np.squeeze(computeTimes)


def get_axial_coordinates(simulation, unit='001'):
    """Get axial coordinates from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'coordinates',
                                              'unit_' + unit,
                                              'axial_coordinates']
        )
    )


def get_particle_coordinates(simulation, unit='001'):
    """Get particle coordinates from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    unit_type = sim_go_to(
        get_simulation(simulation).root, ['input',
                                          'model',
                                          'unit_' + unit,
                                          'unit_type']
    )
    if re.search("LUMPED_RATE_MODEL_WITHOUT_PORES", unit_type):
        raise ValueError(
            "LRM does not have particle coordinates!"
        )

    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'solution',
                                              'coordinates',
                                              'unit_' + unit,
                                              'particle_coordinates']
        )
    )


def get_radial_coordinates(simulation, unit='001'):
    """Get radial coordinates from simulation

    Parameters
    ----------
    simulation : string or CADET object
        specifies the simulation of interest
    unit : string
        specifies the unit of onterest (000-999)


    Returns
    -------
    np.array
        Solution vector.
    """
    unit_type = sim_go_to(
        get_simulation(simulation).root, ['input',
                                          'model',
                                          'unit_' + unit,
                                          'unit_type']
    )
    if not re.search("GENERAL_RATE_MODEL_2D", unit_type):
        raise ValueError(
            "Simulation does not have radial coordinates!"
        )

    return np.squeeze(
        sim_go_to(
            get_simulation(simulation).root, ['output',
                                              'solution',
                                              'coordinates',
                                              'unit_' + unit,
                                              'radial_coordinates']
        )
    )


def calculate_all_L1_errors(simulations, reference,
                            unit='001', which='outlet',
                            weights=None,
                            comps=-1):
    """Calculate L1 errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999.
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...
    weights : np.array
        L1 error weights
    comps : list
        components that are to be considered. defaults to -1,
        i.e. all components are considered

    Returns
    -------
    np.array
        L1 errors.
    """

    abs_errors = np.array(calculate_all_abs_errors(
        simulations, reference, unit, which))

    if comps == -1:
        comps = range(0, abs_errors.shape[2])

    if len(abs_errors.shape) == 3:  # shape: nMethods, error domain, nComponents
        if weights is None:
            weights = [1.0] * abs_errors.shape[1]

        errors = np.zeros((abs_errors.shape[0]))

        for method_idx in range(0, abs_errors.shape[0]):
            for comp_idx in comps:
                errors[method_idx] += np.sum(np.multiply(
                    abs_errors[method_idx, :, comp_idx], weights))
    # @todo methods und components unterscheidung
    else:
        raise ValueError("not implemented yet for this nMethods, nComp")

    # shape: nMethods, error domain
    return errors


def calculate_all_min_vals(simulations, unit='001', which='outlet'):
    """Calculate minimal values of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    unit : string
        Unit ID 000-999
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...

    Returns
    -------
    np.array
        Minimal values errors.
    """
    errors = []

    for simulation in simulations:

        errors.append(np.min(get_solution(simulation, 'unit_'+unit, which)))

    return np.array(errors)


def calculate_abs_error(solution, reference):
    """Calculate absolute error of solution.

    Parameters
    ----------
    solution : np.array
        Solution of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Absolute error.
    """
    return np.abs(solution - reference)


def calculate_all_abs_errors(simulations, reference, unit='001', which='outlet'):
    """Calculate absolute errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...

    Returns
    -------
    np.array
        Absolute errors.
    """
    errors = []

    if isinstance(reference, str) or isinstance(reference, Cadet):

        for simulation in simulations:

            errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which),
                                              get_solution(reference, 'unit_'+unit, which)))

    elif isinstance(reference, np.ndarray):

        for simulation in simulations:

            errors.append(calculate_abs_error(get_solution(simulation, 'unit_'+unit, which),
                                              reference))
    else:
        raise ValueError(
            "Reference is neither np.ndarray nor Cadet object nor string specifying h5 file name."
        )

    return np.array(errors)


def calculate_sse(solution, reference=-1):
    """Calculate sum squared error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        SSE.
    """
    if(reference == -1):
        np.sum((solution) ** 2, axis=0)
    else:
        return np.sum((solution - reference) ** 2, axis=0)


def calculate_max_error(solution, reference=-1):
    """Calculate max error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Max Error.
    """
    solution = np.array(solution)
    if(reference == -1):
        return np.max(np.abs(solution))
    else:
        solution = np.array(reference)
        return np.max(calculate_abs_error(solution, reference))


def calculate_all_max_errors(simulations, reference, unit='001', which='outlet'):
    """Calculate absolute errors of multiple simulations.

    Parameters
    ----------
    simulations : np.ndarray
        Simulations either Cadet objects or specified by h5 names.
    reference : Cadet object or string
        (exact) reference solution specified by Cadet object or h5 name.
    unit : string
        Unit ID 000-999.
    which : string
        Specifies the solution array of interest, i.e. outlet, bulk, ...

    Returns
    -------
    np.array
        Absolute errors.
    """
    abs_errors = np.array(calculate_all_abs_errors(
        simulations, reference, unit, which))
    if len(abs_errors.shape) == 3:  # shape: nMethods, error domain, nComponents
        abs_errors = np.max(abs_errors, axis=len(abs_errors.shape)-1)
    # shape: nMethods, error domain
    return np.max(abs_errors, axis=len(abs_errors.shape)-1)


def calculate_Linf_error(solution, reference=-1):
    """Calculate L_infinity (i.e. maximal) error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Max Error.
    """
    return calculate_max_error(solution, reference)


def calculate_weighted_error(solution, reference=-1, weights=1.0):
    """Calculate weighted error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (~exact) reference solution.
    weights : np.array or scalar
        weights

    Returns
    -------
    np.array
        Max Error.
    """
    if(reference == -1):
        return np.sum(np.abs(solution - reference) * weights)
    else:
        return np.sum(np.abs(solution) * weights)


def calculate_solution_times_sizes(simulation):
    """Calculate the solution times step sizes

    Parameters
    ----------
    simulation : string or Cadet object
        simulation.

    Returns
    -------
    np.array
        Time step sizes.
    """

    solution_times = get_solution_times(simulation)

    return solution_times[1:] - solution_times[:-1]


def calculate_L1_error(solution, reference=-1, weights=1.0):
    """Calculate L_1 error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Max Error.
    """
    return calculate_weighted_error(solution, reference, weights)


def calculate_average_error(solution, reference):
    """Calculate average error of solution.

    Parameters
    ----------
    solution : np.array
        Solution or errors of simulation.
    reference : np.array
        (exact) reference solution.

    Returns
    -------
    np.array
        Average Error.
    """
    return np.calculate_weighted_error(solution, reference, 1.0)


def generate_1D_name(prefix, axP, axCells, suffix='.h5'):
    """Generate simulation name for bulk discretized models (LRMP, LRM).

    Parameters
    ----------
    prefix : string
        Name prefix.
    axP : int
        axial polynomial degree.
    axCells : int
        axial number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File name.
    """
    if int(axCells) <= 0:
        return None

    if int(axP) == 0:
        return prefix + "_FV_Z" + str(int(axCells)) + suffix

    elif int(axP) > 0:
        return prefix + "_DG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + suffix

    elif int(axP) < 0:
        return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + suffix

    else:
        return None


def generate_simulation_names_1D(prefix, methods, disc, suffix='.h5'):
    """Generate simulation names for bulk discretized models (LRMP, LRM).

    Parameters
    ----------
    prefix : string
        Name prefix.
    methods : np.array<int>
        axial polynomial degrees (0 for FV, -int for exInt DGSEM).
    disc : np.array<int>
        axial number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File names.
    """
    methods = np.array(methods)
    disc = np.array(disc)
    _simulation_names = []
    nMethods = len(methods)
    # Check input parameters and infer data
    if nMethods > 1 and nMethods != disc.shape[0]:
        disc = disc.transpose()
        if nMethods != disc.shape[0]:
            raise ValueError(
                "Method and discretization must have feasible dimensionalities, look up description."
            )

    if nMethods > 1:
        nDisc = disc.shape[1]
    else:
        nDisc = disc.size
        _disc = disc

    # Main loop, create names
    for m in range(0, nMethods):
        if nMethods > 1:
            _disc = disc[m]

        for d in range(0, nDisc):

            name = generate_1D_name(prefix, methods[m], _disc[d], suffix)
            if name is not None:
                _simulation_names.append(name)

    return _simulation_names


def test_generate_simulation_names_1D():

    prefix = "LinearLRM1Comp"
    methods = [1]
    disc = [1, 2, 4]
    names = np.array(generate_simulation_names_1D(prefix, methods, disc))
    expected_names = np.array([prefix+"_DG_P1Z1",
                              prefix+"_DG_P1Z2",
                              prefix+"_DG_P1Z4"])
    np.testing.assert_array_equal(names, expected_names)

    methods = [2, -3]
    disc = [[2, 4, 8], [1, 2, 4]]
    names = np.array(generate_simulation_names_1D(prefix, methods, disc))
    expected_names = np.array([prefix+"_DG_P2Z2",
                              prefix+"_DG_P2Z4",
                              prefix+"_DG_P2Z8",
                              prefix+"_DGexInt_P3Z1",
                              prefix+"_DGexInt_P3Z2",
                              prefix+"_DGexInt_P3Z4"
                               ])
    np.testing.assert_array_equal(names, expected_names)


def generate_GRM_name(prefix, axP, axCells, parP, parCells, suffix='.h5'):
    """Generate simulation name for bulk and particle discretized GRM.

    Parameters
    ----------
    prefix : string
        Name prefix.
    axP : int
        axial polynomial degree.
    axCells : int
        axial number of cells.
    parP : int
        particle polynomial degree.
    parCells : int
        particle number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File name.
    """
    if int(axCells) <= 0 or int(parCells) <= 0:
        return None

    if int(axP) == 0 and int(parP) == 0:
        return prefix + "_FV_Z" + str(int(axCells)) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) > 0 and int(parP) > 0:
        return prefix + "_DG_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) < 0 and int(parP) > 0:
        return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) > 0 and int(parP) < 0:
        return prefix + "_DGinexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    elif int(axP) < 0 and int(parP) < 0:
        return prefix + "_DGexInt_P" + str(int(abs(axP))) + "Z" + str(int(axCells)) + "_DGinexInt_parP" + str(int(abs(parP))) + "parZ" + str(int(parCells)) + suffix

    else:
        return None

# TODO generate_simulation_names: scalar values as methods input!


def generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells, suffix='.h5'
):
    """Generate simulation names for bulk and particle discretized GRM.

    Parameters
    ----------
    prefix : string
        Name prefix.
    ax_methods : np.array<int>
        axial polynomial degrees (0 for FV, -int for exInt DGSEM).
    ax_cells : int
        axial number of cells.
    par_methods : np.array<int>
        particle polynomial degrees (0 for FV, -int for collocation DGSEM).
    par_cells : int
        particle number of cells.
    suffix : string
        Name suffix, including filetype .h5.

    Returns
    -------
    string
        File names.
    """
    ax_methods = np.array(ax_methods)
    ax_cells = np.array(ax_cells)
    par_methods = np.array(par_methods)
    par_cells = np.array(par_cells)

    _simulation_names = []
    nMethods = len(ax_methods)
    # Check input parameters and infer data
    if nMethods > 1 and nMethods != ax_cells.shape[0]:
        ax_cells = ax_cells.transpose()
        par_cells = par_cells.transpose()
        if nMethods != ax_cells.shape[0] or ax_methods.shape != par_methods.shape:
            raise ValueError(
                "Methods and discretizations must have feasible dimensionalities, look up description."
            )
    if ax_cells.shape != par_cells.shape:
        raise ValueError(
            "Axial and particle discretizations must have same dimensionalities."
        )

    if(nMethods > 1):
        nDisc = ax_cells.shape[1]
    else:
        nDisc = len(ax_cells)
        _ax_cells = ax_cells
        _par_cells = par_cells

    # Main loop, create names
    for m in range(0, nMethods):
        if(nMethods > 1):
            _ax_cells = ax_cells[m]
            _par_cells = par_cells[m]

        for d in range(0, nDisc):

            name = generate_GRM_name(
                prefix, ax_methods[m], _ax_cells[d], par_methods[m], _par_cells[d], suffix)
            if name is not None:
                _simulation_names.append(name)

    return _simulation_names


def test_generate_simulation_names_GRM():

    prefix = "LinearGRM2Comp"
    ax_methods = [2]
    par_methods = [1]
    ax_cells = [8]
    par_cells = [1]
    names = np.array(generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells, suffix='.hmpf'))
    expected_names = np.array([prefix+"_DG_P2Z8parP1parZ1.hmpf"])
    np.testing.assert_array_equal(names, expected_names)

    ax_methods = [2]
    par_methods = [1]
    ax_cells = [8, 16, 32]
    par_cells = [1, 2, 4]
    names = np.array(generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells))
    expected_names = np.array([prefix+"_DG_P2Z8parP1parZ1.h5",
                              prefix+"_DG_P2Z16parP1parZ2.h5",
                              prefix+"_DG_P2Z32parP1parZ4.h5"])
    np.testing.assert_array_equal(names, expected_names)

    ax_methods = [0, 0,  -2, 3]
    par_methods = [-1, 0,  2, 3]
    ax_cells = [[8, 16, 32], [8, 16, 32], [8, 16, 32], [8, 16, 32]]
    par_cells = [[1, 2, 4], [1, 2, 4], [1, 2, 4], [1, 2, 4]]
    names = np.array(generate_simulation_names_GRM(
        prefix, ax_methods, ax_cells, par_methods, par_cells))
    expected_names = np.array([
        prefix+"_FV_Z8parZ1.h5",
        prefix+"_FV_Z16parZ2.h5",
        prefix+"_FV_Z32parZ4.h5",
        prefix+"_DGexInt_P2Z8parP2parZ1.h5",
        prefix+"_DGexInt_P2Z16parP2parZ2.h5",
        prefix+"_DGexInt_P2Z32parP2parZ4.h5",
        prefix+"_DG_P3Z8parP3parZ1.h5",
        prefix+"_DG_P3Z16parP3parZ2.h5",
        prefix+"_DG_P3Z32parP3parZ4.h5",
    ])

    np.testing.assert_array_equal(names, expected_names)


def generate_simulation_names(
        prefix, ax_methods, ax_cells, par_methods=None, par_cells=None,
        suffix='.h5'):
    """Generates simulation names.

    Parameters
    ----------
    prefix : string
        Prefix of simulation name.
    ax_methods : np.array
        Specifies axial discretization method as DG -> polynomial degree; FV -> 0.
    ax_cells : np.array
        Specifies axial number of cells.
    par_methods : np.array
        Specifies particle discretization method as DG -> polynomial degree; FV -> 0.
    par_cells : np.array
        Specifies particle number of cells.

    Returns
    -------
    np.array
        Simulation names for all discretizations.
    """
    if(par_methods is None or par_cells is None):
        if(par_methods == par_cells):
            return generate_simulation_names_1D(prefix, ax_methods, ax_cells, suffix)
        else:
            raise ValueError(
                "Particle discretization and methods must be either None or both specified."
            )

    else:
        return generate_simulation_names_GRM(
            prefix, ax_methods, ax_cells, par_methods, par_cells, suffix)


def test_generate_simulation_names():

    with pytest.raises(ValueError):
        names = np.array(generate_simulation_names("prefix", [0], [2, 4], [1]))

    methods = [2, -3]
    disc = [[2, 4, 8], [1, 2, 4]]
    prefix = "prefix"
    names = np.array(generate_simulation_names(prefix, methods, disc))
    expected_names = np.array([prefix+"_DG_P2Z2.h5",
                              prefix+"_DG_P2Z4.h5",
                              prefix+"_DG_P2Z8.h5",
                              prefix+"_DGexInt_P3Z1.h5",
                              prefix+"_DGexInt_P3Z2.h5",
                              prefix+"_DGexInt_P3Z4.h5"
                               ])
    np.testing.assert_array_equal(names, expected_names)

    methods = [0]
    disc = [2, 4, 8]
    prefix = "prefix"
    names = np.array(generate_simulation_names(prefix, methods, disc))
    expected_names = np.array([prefix+"_FV_Z2.h5",
                              prefix+"_FV_Z4.h5",
                              prefix+"_FV_Z8.h5"
                               ])
    np.testing.assert_array_equal(names, expected_names)


def std_name_prefix(transport_model, binding_model, dyn=None, n_comp=None):
    """Generate uniform name prefices depending on transport and binding model.

    Parameters
    ----------
    transport_model : string
        Applied transport model, i.e. LRMP, GRM, LRM, GRMsd
    binding_model : string
        Applied transport model, e.g. Linear, SMA, ...
    dyn : boolean
        Binding kinetics.
    n_comp : int
        Number of applied components.

    Returns
    -------
    string
        Simulation name prefix.

    """
    tm = ""
    bnd = ""

    if re.search("General|GRM", transport_model):
        tm = "GRM"
        if re.search("surface diffusion|surf|diff|sd", transport_model, re.IGNORECASE):
            tm += "sd"
    elif re.search("LRMP|with", transport_model, re.IGNORECASE):
        tm = "LRMP"
    elif re.search("LRM|without", transport_model, re.IGNORECASE):
        tm = "LRM"
    else:
        raise ValueError(
            "Unrecognized transport model: " + str(transport_model)
        )
    if re.search("2D", transport_model, re.IGNORECASE):
        tm += "2D"

    if dyn is not None:
        bnd = "dyn" if dyn else "req"

    if re.search("linear", binding_model, re.IGNORECASE):
        bnd += "Lin"
    elif re.search("langmuir", binding_model, re.IGNORECASE):
        bnd += "Langmuir"
    elif re.search("sma", binding_model, re.IGNORECASE):
        bnd += "SMA"
    else:
        bnd += binding_model

    if n_comp is not None:
        return tm + "_" + bnd + "_" + str(int(n_comp)) + "comp"
    else:
        return tm + "_" + bnd


def calculate_DOFs(discretization, method=np.array([3]), nComp=1,
                   full_DOFs=False, model='LRMP', nBound=None):
    """Calculate number of degrees of freedom.

    Parameters
    ----------
    discretization : np.array
        Discretization steps (GRM stored as [[bulk], [particle]]).
    method : np.array
        Discretization method, i.e. polynomial degree(s) of method with FV -> 0.
        Multiple polynomial degrees must be specified for GRM, i.e. [bulk, particle].
    model : string
        Chromatography model (LRMP, LRM, GRM), only required for LRM and LRMP
    nComp : int
        Number of components
    nBound : int
        Number of bound states

    Returns
    -------
    np.array
        Number of Degrees of freedom.

    Raises
    -------
    ValueError
        If any discretization value is smaller than 0.
        If methods dimensionality is larger than two or does not match discretizations dimensionality.

    """
    method = np.array(method)
    discretization = np.array(discretization)

    if np.any(discretization <= 0):
        raise ValueError(
            "Discretization must be larger than 0."
        )
    if discretization.ndim not in [1, 2] or method.ndim not in [0, 1]:
        if not (discretization.ndim == 0 and method.ndim == 0):
            raise ValueError(
                "Method and discretization must have feasible dimensionalities, look up description."
            )

    if nBound == None:
        nBound = nComp

    inlet_dof = nComp if full_DOFs else 0
    bulk_dof = 0
    par_dof = 0
    flux_dof = 0

    if discretization.ndim == 2:  # GRM
        bulk_dof = (abs(method[0]) + 1) * discretization[0, :] * nComp
        if full_DOFs:
            flux_dof = bulk_dof
            par_dof = (
                (abs(method[0]) + 1) * discretization[0, :]
                * ((abs(method[1]) + 1) * discretization[1, :] * (nComp + nBound))
            )

    elif discretization.ndim in [0, 1]:  # LRM or LRMP
        bulk_dof = (abs(method) + 1) * discretization * nComp
        if full_DOFs:
            if model == "LRMP":
                par_dof = (nComp + nBound) * abs(method+1) * discretization
                if method == 0:  # add flux states for FV discretization
                    flux_dof = bulk_dof
            elif model == "LRM":
                par_dof = nBound * abs(method+1) * discretization
            else:
                raise ValueError(
                    "Unknown transport model \"" + model +
                    "\" or wrong dimensionality of input."
                )

    return inlet_dof + bulk_dof + flux_dof + par_dof


def test_calculate_DOFs():

    # test GRM
    disc = [[4, 8, 10, 32], [1, 2, 3, 5]]
    method = [3, 2]
    nComp = 3
    nBound = nComp

    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc[0]) * nComp
    fluxDOFs = bulkDOFs
    parDOFs = np.multiply(
        (nComp + nBound) * (method[1] + 1) * np.array(disc[1]),
        (method[0] + 1) * np.array(disc[0])
    )
    DOFs = inletDOFs + bulkDOFs + fluxDOFs + parDOFs

    np.testing.assert_array_equal(
        bulkDOFs,
        calculate_DOFs(disc, method, nComp=nComp, full_DOFs=False)
    )
    np.testing.assert_array_equal(
        DOFs,
        calculate_DOFs(disc, method, nComp=nComp, full_DOFs=True)
    )

    # test LRM
    disc = [4, 8, 10, 32]
    method = [1]
    nComp = 3
    nBound = 4

    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc) * nComp
    parDOFs = (method[0] + 1) * np.array(disc) * nBound
    DOFs = inletDOFs + bulkDOFs + parDOFs

    np.testing.assert_array_equal(
        bulkDOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=False,
                       model="LRM")
    )
    np.testing.assert_array_equal(
        DOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=True,
                       model="LRM")
    )

    # test LRMP
    disc = [4, 8, 10, 32]
    method = [1]
    nComp = 3
    nBound = 4

    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc) * nComp
    parDOFs = (method[0] + 1) * np.array(disc) * (nComp + nBound)
    DOFs = inletDOFs + bulkDOFs + parDOFs

    np.testing.assert_array_equal(
        bulkDOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=False,
                       model="LRMP")
    )
    method = [0]  # FV, i.e. with flux
    inletDOFs = np.array([1, 1, 1, 1]) * nComp
    bulkDOFs = (method[0] + 1) * np.array(disc) * nComp
    fluxDOFs = bulkDOFs
    parDOFs = (method[0] + 1) * np.array(disc) * (nComp + nBound)
    DOFs = inletDOFs + bulkDOFs + fluxDOFs + parDOFs
    np.testing.assert_array_equal(
        DOFs,
        calculate_DOFs(disc, method, nComp=nComp, nBound=nBound,
                       full_DOFs=True,
                       model="LRMP")
    )


def calculate_eoc(discretization, error):
    """Calculate experimental order of convergence.

    Parameters
    ----------
    discretization : np.array
        Discretization steps.
    error : np.array
        Errors.

    Returns
    -------
    np.array
        Experimental order of convergence

    Raises
    -------
    ValueError
        If any discretization value is smaller than 0.
        If any error value is smaller than 0.

    """
    error = np.asarray(error)
    discretization = np.asarray(discretization)

    if np.any(discretization <= 0):
        raise ValueError(
            "Discretization must be larger than 0."
        )
    if np.any(error < 0):
        raise ValueError(
            "Error must be larger or equal than 0 (must be provided as absolute error)."
        )
    if np.any(error < np.finfo(float).eps):
        error[error < np.finfo(float).eps] = np.finfo(float).eps

    ratio_discretization = discretization[:-1]/discretization[1:]
    ratio_error = error[1:]/error[:-1]

    return np.log(ratio_error)/np.log(ratio_discretization)


def test_eoc():
    dof = [1, 2, 4, 8, 16]
    error = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    eoc_expected = [1, 1, 1, 1]
    eoc = calculate_eoc(dof, error)

    np.testing.assert_almost_equal(eoc, eoc_expected)

    with pytest.raises(ValueError):
        dof_zero = [1, 2, 4, 8, 0]
        eoc = calculate_eoc(dof_zero, error)

    with pytest.raises(ValueError):
        error_smaller_zero = [-0.1, 0.05, 0.025, 0.0125, 0.00625]
        eoc = calculate_eoc(dof, error_smaller_zero)

    # Test eps
    error_zero = [0.1, 0.05, 0.025, 0.0125, 0]

    eoc_expected = [1, 1, 1, 45.67807191]
    eoc = calculate_eoc(dof, error_zero)
    np.testing.assert_almost_equal(eoc, eoc_expected)


"""
 TODO Function: Automatic refinement until error tolerance is reached
 -> convergence table

"""


def convergency_table(method,
                      disc,
                      abs_errors,
                      sim_names=None,
                      error_types=np.array(["max"]),
                      full_DOFs=False):
    """Calculate convergency table for one spatial method.

    Parameters
    ----------
    method : np.array
        Discretization method, i.e. polynomial degree(s) of method with FV -> 0.
        Multiple polynomial degrees must be specified for GRM, i.e. [bulk, particle].
    disc : np.array
        Discretization steps (GRM stored as [[bulk], [particle]]).
    sim_names : np.array
        String with simulation names, if compute times should be included.
    abs_errors : np.array
        Absolute errors of simulations corresponding to discretization array.
    error_types : np.array
        Contains error types for which convergency is computed.
    full_DOFs : boolean
        Default recommended! Specifies whether or not particle DOFs are considered for convergency.

    Returns
    -------
    np.array
        Header of convergency table
    np.array
        Convergency table

    Raises
    -------
    ValueError
        If discretization is not one or two dimensional.
        If dimensionality of errors and discretization does not match.
        If error type is unknown.

    """
    disc = np.array(disc)
    error_types = np.array(error_types)
    abs_errors = np.abs(np.array(abs_errors))
    header = []
    header.append("$N_e^z$")
    table = []
    # Infer dimensionality of table
    offRow = 1
    if (disc.ndim == 2):    # GRM
        nDisc = disc.shape[1]
        offCol = 2
        header.append("$N_e^r$")
        table.append(disc[0])
        table.append(disc[1])
    elif (disc.ndim == 1):   # LRMP, LRM
        nDisc = len(disc)
        offCol = 1
        table.append(disc)
    else:
        raise ValueError(
            "Discretization must be array or n x 2 matrix."
        )
    if disc.any() <= 0:
        raise ValueError(
            "Number of cells must be >= 1."
        )
    # Check input
    if (nDisc != abs_errors.shape[0]):
        abs_errors = abs_errors.transpose()
        if (nDisc != abs_errors.shape[0]):
            raise ValueError(
                "Not the same number of discretizations as errors."
            )

    DOFs = calculate_DOFs(disc, method, full_DOFs=full_DOFs)

    # Main loop: calculate all errors and EOC's
    for error_type in range(0, len(error_types)):

        current_errors = np.zeros(nDisc)

        if(re.match("max", error_types[error_type], re.IGNORECASE)
           or re.match("Linf", error_types[error_type], re.IGNORECASE)
           or re.match("Linfty", error_types[error_type], re.IGNORECASE)):

            table_error_name = "Max."
            for d in range(0, nDisc):
                current_errors[d] = calculate_Linf_error(abs_errors[d, :])
        elif(re.match("L1", error_types[error_type], re.IGNORECASE)
             or re.match("L^1", error_types[error_type], re.IGNORECASE)):

            table_error_name = "$L^1$"
            for d in range(0, nDisc):
                current_errors[d] = calculate_L1_error(abs_errors[d, :])

        else:
            raise ValueError(
                "Unknown error Type " + error_type + "."
            )

        table.append(current_errors)
        header.append(table_error_name + " error")
        # calculate EOC's from errors
        # TODO should be NAN or '-', not 0.0
        table.append(
            np.insert(
                calculate_eoc(DOFs, current_errors), 0, 0.0
            )
        )
        header.append(table_error_name + " EOC")

    if sim_names is not None:
        header.append('Sim. time')
        table.append(get_compute_times(sim_names))

    return header, np.array(table).transpose()


def test_convergency_table():

    # GRM test for L1/max error
    expected_order = 3
    method = np.array([1, 1]) * (expected_order - 1)
    discretizations = [[4, 8, 16], [1, 2, 4]]
    full_DOFs = False

    error_type = ["max"]
    initial_errors = np.random.rand(100)
    error_factors = np.zeros(len(discretizations[0]))
    for i in range(0, len(error_factors)):
        error_factors[i] = 2 ** (expected_order * i)
    abs_errors = np.zeros([len(error_factors), len(initial_errors)])

    for factor in range(0, len(error_factors)):
        abs_errors[factor] = initial_errors / error_factors[factor]

    header, table = convergency_table(method=method, disc=discretizations, abs_errors=abs_errors,
                                      error_types=error_type, full_DOFs=full_DOFs)

    np.testing.assert_array_equal(header, np.array(
        ["$N_e^z$", "$N_e^r$", 'Max. error', 'Max. EOC']))
    np.testing.assert_almost_equal(
        table[:, 0], discretizations[0])  # check axial disc
    np.testing.assert_almost_equal(
        table[:, 1], discretizations[1])  # check particle disc
    np.testing.assert_almost_equal(
        table[:, 2], np.amax(abs_errors, axis=1))  # check errors
    np.testing.assert_almost_equal(table[1:, 3], np.ones(
        len(discretizations[0])-1) * expected_order)  # check EOC

    # LRMP/LRM test for L1/max error
    expected_order = 3
    method = (expected_order - 1)
    discretizations = [4, 8, 16]
    full_DOFs = False

    error_type = ["max"]
    initial_errors = np.random.rand(100)
    error_factors = np.zeros(len(discretizations))
    for i in range(0, len(error_factors)):
        error_factors[i] = 2 ** (expected_order * i)
    abs_errors = np.zeros([len(error_factors), len(initial_errors)])

    for factor in range(0, len(error_factors)):
        abs_errors[factor] = initial_errors / error_factors[factor]

    header, table = convergency_table(method=method, disc=discretizations, abs_errors=abs_errors,
                                      error_types=error_type, full_DOFs=full_DOFs)

    np.testing.assert_array_equal(header, np.array(
        ["$N_e^z$", 'Max. error', 'Max. EOC']))
    np.testing.assert_almost_equal(
        table[:, 0], discretizations)  # check axial disc
    np.testing.assert_almost_equal(
        table[:, 1], np.amax(abs_errors, axis=1))  # check errors
    np.testing.assert_almost_equal(table[1:, 2], np.ones(
        len(discretizations)-1) * expected_order)  # check EOC


def calculate_convergence_tables_from_files(
        prefix, reference,
        ax_methods, ax_cells,
        par_methods=None, par_cells=None,
        unit='001', error_types=['max'], which='outlet', full_DOFs=False):
    """Calculate convergency tables from simulation files.

    Parameters
    ----------
    prefix: string
        Prefix of (all) the file names
    reference: np.ndarray or Cadet object or string
        Reference solution, Cadet object or h5 file name
    ax_methods : np.array
        Axial discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    ax_cells : np.array
        Axial discretization steps, i.e. cells.
    par_methods : np.array
        Particle discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    par_cells : np.array
        Particle discretization steps, i.e. cells.
    unit : string
        Unit name '000'-'999'
    error_types: np.array
        Contains error types for which convergence is computed.
    which : string
        Specifies which solution is considered, i.e. 'outlet', 'bulk', ...
    full_DOFs: boolean
        Default recommended! Specifies whether or not particle DOFs are considered for convergency.

    Returns
    -------
    np.array
        Header of convergency table
    np.array
        Convergency tables

    """
    tables = []
    ax_methods = np.array(ax_methods)

    if par_methods is None:  # i.e. LRMP or LRM
        par_methods = np.array(ax_methods.size * [None])
        par_cells = np.array(ax_methods.size * [None])

    for m in range(0, ax_methods.size):

        # Get simulation names
        simulation_names = generate_simulation_names(
            prefix=prefix,
            ax_methods=[ax_methods[m]],
            ax_cells=ax_cells[m],
            par_methods=[par_methods[m]],
            par_cells=par_cells[m])

        # Calculate errors
        errors = calculate_all_abs_errors(
            simulation_names, reference, unit=unit, which=which)

        # Calculate convergence tables
        header, table = convergency_table(
            method=[ax_methods[m], par_methods[m]],
            disc=[ax_cells[m], par_cells[m]],
            abs_errors=errors,
            error_types=error_types,
            full_DOFs=full_DOFs
        )

        tables.append(table)

    return header, tables


def get_compute_times_from_files(
        prefix,
        ax_methods, ax_cells,
        par_methods=None, par_cells=None):
    """returns compute times from simulation files.

    Parameters
    ----------
    prefix: string
        Prefix of (all) the file names
    ax_methods : np.array
        Axial discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    ax_cells : np.array
        Axial discretization steps, i.e. cells.
    par_methods : np.array
        Particle discretization methods, i.e. polynomial degree(s) of method with FV -> 0.
    par_cells : np.array
        Particle discretization steps, i.e. cells.

    Returns
    -------
    np.array
        Compute times per method

    """
    times = []
    ax_methods = np.array(ax_methods)

    if par_methods is None:  # i.e. LRMP or LRM
        par_methods = np.array(ax_methods.size * [None])
        par_cells = np.array(ax_methods.size * [None])

    for m in range(0, ax_methods.size):
        # Get simulation name
        simulation_names = generate_simulation_names(
            prefix=prefix,
            ax_methods=[ax_methods[m]],
            ax_cells=ax_cells[m],
            par_methods=[par_methods[m]],
            par_cells=par_cells[m])

        times.append(get_compute_times(simulation_names))

    return times


def std_plot(x_axis, y_axis, **kwargs):
    """adds a line to the current plot.

    Parameters
    ----------
    x_axis : array
        x values (one line per row)
    y_axis : array
        y values (one line per row)
    shape : array size 2
        shape of plot
    font_size : float
        Scale factor of font sozes, e.g. legend, title, ...
    x_scale : string
        Scale of x-axis, e.g. 'log'

    Returns
    -------
    np.array
        Compute times per method

    """
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)

    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    if 'markersize' not in kwargs:
        kwargs['markersize'] = 12

    if x_axis.ndim == 2:
        for i in range(0, x_axis.shape[0]):
            plt.plot(x_axis[i], y_axis[i], **kwargs)

    elif x_axis.ndim == 1:
        plt.plot(x_axis, y_axis, **kwargs)
    else:
        raise ValueError(
            "Unfeasible dimensionality of input, i.e. ndim = " +
            str(x_axis.ndim)
        )


def std_plot_prep(**kwargs):
    """Create standard plot setting.

    Parameters
    ----------
    x_axis : array
        x values (one line per row)
    y_axis : array
        y values (one line per row)
    shape : array size 2
        shape of plot
    font_size : float
        Scale factor of font sozes, e.g. legend, title, ...
    x_scale : string
        Scale of x-axis, e.g. 'log'

    Returns
    -------
    np.array
        Compute times per method

    """
    if 'title' in kwargs:
        title = kwargs.pop('title')
    else:
        title = None
    if 'x_label' in kwargs:
        x_label = kwargs.pop('x_label')
    else:
        x_label = None
    if 'y_label' in kwargs:
        y_label = kwargs.pop('y_label')
    else:
        y_label = None
    if 'shape' in kwargs:
        shape = kwargs.pop('shape')
    else:
        shape = [20, 10]
    if 'font_size_fac' in kwargs:
        font_size_factor = kwargs.pop('font_size_fac')
    else:
        font_size_factor = 1.5
    if 'x_scale' in kwargs:
        x_scale = kwargs.pop('x_scale')
    else:
        x_scale = 'log'
    if 'y_scale' in kwargs:
        y_scale = kwargs.pop('y_scale')
    else:
        y_scale = 'log'
    if 'x_lim' in kwargs:
        x_lim = kwargs['x_lim']
    else:
        x_lim = None
    if 'y_lim' in kwargs:
        y_lim = kwargs['y_lim']
    else:
        y_lim = None
    # TODO: ignores ticks right now
    # if 'x_ticks' in kwargs:
    #     x_ticks = kwargs['x_ticks']
    # elif x_lim is not None:
    #     x_ticks = np.logspace(x_lim[0], y_lim[1], num=4, endpoint=True)
    # if 'y_ticks' in kwargs:
    #     y_ticks = kwargs['y_ticks']
    # elif y_lim is not None:
    #     y_ticks = np.logspace(y_lim[0], y_lim[1], num=12, endpoint=True)

    if 'linewidth' not in kwargs:
        kwargs['linewidth'] = 3
    if 'marker' not in kwargs:
        kwargs['marker'] = 'o'
    if 'markersize' not in kwargs:
        kwargs['markersize'] = 12

    plt.rcParams["figure.figsize"] = (shape[0], shape[1])

    plt.legend(fontsize=15*font_size_factor)
    plt.grid()
    plt.yscale(y_scale)
    plt.xscale(x_scale)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=15*font_size_factor)
    if x_label is not None:
        plt.ylabel(y_label, fontsize=15*font_size_factor)
    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])
        plt.xticks(fontsize=15*font_size_factor)  # todo x_ticks
    else:
        plt.xticks(fontsize=15*font_size_factor)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
        plt.yticks(fontsize=15*font_size_factor)  # todo y_ticks
    else:
        plt.yticks(fontsize=15*font_size_factor)
    if title is not None:
        plt.title(title, fontsize=15*font_size_factor)


def std_latex_table(data_frame, latex_columns, math_mode=True, error_digits=3):
    """Create latex table from dataframe.

    Parameters
    ----------
    data_frame : pandas.dataframe
        table
    latex_columns : array
        column names to be included
    math_mode : bool
        print numbers in latex math mode
    error_digits : int
        Number of digits for errors.

    Returns
    -------
    string
        Latex table code

    """
    float_format = '%.' + str(error_digits) + 'e'
    eoc_format = '{:.2f}'
    if math_mode:
        float_format = '$' + float_format + '$'
        eoc_format = '$' + eoc_format + '$'

    # EOC as fixed point number
    eoc_columns = data_frame.filter(like='EOC').columns

    data_frame[eoc_columns] = data_frame[eoc_columns].applymap(
        eoc_format.format)

    data_frame['$N_d$'] = data_frame['$N_d$'].astype(int)
    data_frame['$N_e^z$'] = data_frame['$N_e^z$'].astype(int)
    data_frame['$N_e^r$'] = data_frame['$N_e^r$'].astype(int)

    if math_mode:
        return data_frame.to_latex(index=False,
                                   float_format=float_format,
                                   columns=latex_columns,
                                   escape=False,
                                   formatters={
                                       # needed for math mode $$
                                       '$N_d$': '${:d}$'.format,
                                       '$N_e^z$': '${:d}$'.format,
                                       '$N_e^r$': '${:d}$'.format,
                                   })
    else:
        return data_frame.to_latex(index=False,
                                   float_format=float_format,
                                   columns=latex_columns,
                                   escape=False,
                                   formatters={
                                       '$N_d$': '{:d}'.format,
                                       '$N_e^z$': '{:d}'.format,
                                       '$N_e^r$': '{:d}'.format,
                                   })


def recalculate_results(file_path, models,
                        ax_methods, ax_cells,
                        exact_names,
                        unit='001', which='outlet',
                        par_methods=[None], par_cells=None,
                        incl_min_val=True,
                        transport_model=None, ncomp=None, nbound=None,
                        save_path_=None):
    """Calculate results (EOC, Sim. time, errors) for saved simulations.

    Parameters
    ----------
    file_path : string
        Specifies path to where simulation files are stored.
    models : array
        Model names
    ax_methods : array
        Axial discretization methods, i.e. 0:FV;Z^+:cDG;Z^-:DG.
    ax_cells : array
        Axial number of cells
    par_methods : array
        Particle discretization methods, i.e. 0:FV;Z^+:cDG;Z^-:DG.
    par_cells : array
        Particle number of cells.
    exact_names : array
        Strings with reference solution names (full, without path)
            or np.arrays with reference solutions
    unit : string
        Unit of interest, i.e. '000'-'999'.
    which : string
        Concentration of interest, i.e. 'outlet', 'bulk'.
    transport_model : list or np.array
        Considered transport models, defaults to search string of models
        in variable 'models'.
    ncomp : np.array
        Number of components for each model, defaults to search string
        '_\d+comp_' in variable 'models'.
    nbound : np.array
        Number of total bound states for each model, defaults to ncomp
        (mult. parTypes not supported).
    save_path : string
        Specifies the directory to which the tables are saved, defaults to file path

    """

    for modelIdx in range(0, len(models)):

        # needed for DOF calculation
        if transport_model is None:
            try:
                transport_model = re.search(
                    'LRM(?!P)|LRMP|GRM',
                    models[modelIdx],
                    re.IGNORECASE).group(0)
            except:
                raise ValueError(
                    "Considered transport model neither explicitly specified \
                    nor given in model name: " + models[modelIdx])
        if ncomp is None:
            try:
                ncomp = int(re.search('(_)(\d+)(comp)',
                                      models[modelIdx],
                                      re.IGNORECASE).group(2))
            except:
                raise ValueError(
                    "Number of components neither explicitly specified \
                    nor given in model name: " + models[modelIdx])
        if nbound is None:
            nbound = ncomp

        # Data per discretization method
        simulation_names = []
        abs_errors = []
        DoFs = []
        bulk_DoFs = []
        convergence_tables = []
        if incl_min_val:
            min_val = []

        if type(exact_names[modelIdx]) is np.ndarray:
            reference = exact_names[modelIdx]
        else:
            reference = get_outlet(
                file_path+exact_names[modelIdx], unit)

        for m in range(0, len(ax_methods)):

            if len(ax_methods) > 1:
                ax_cells_ = ax_cells[m]
                par_cells_ = None if par_cells is None else par_cells[m]
            else:
                ax_cells_ = ax_cells
                par_cells_ = par_cells

            if par_methods[0] is None:
                simulation_names.append(
                    generate_simulation_names(
                        prefix=file_path+models[modelIdx],
                        ax_methods=[ax_methods[m]], ax_cells=ax_cells_
                    )
                )
            else:
                simulation_names.append(
                    generate_simulation_names(
                        prefix=file_path+models[modelIdx],
                        ax_methods=[ax_methods[m]], ax_cells=ax_cells_,
                        par_methods=[par_methods[m]], par_cells=par_cells_
                    )
                )

            abs_errors.append(
                calculate_all_abs_errors(
                    simulation_names[m],
                    reference,
                    unit,
                    which=which
                )
            )
            if incl_min_val:
                min_val.append(
                    calculate_all_min_vals(
                        simulation_names[m],
                        unit,
                        which=which
                    )
                )
            if par_methods[0] is None:
                DoFs.append(
                    calculate_DOFs(ax_cells_,
                                   ax_methods[m],
                                   full_DOFs=True,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                bulk_DoFs.append(
                    calculate_DOFs(ax_cells_,
                                   ax_methods[m],
                                   full_DOFs=False,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                # Convergence Table
                header, table = convergency_table(ax_methods[m],
                                                  ax_cells_,
                                                  abs_errors[m],
                                                  error_types=['max', 'L1'],
                                                  sim_names=simulation_names[m])

            else:
                DoFs.append(
                    calculate_DOFs([ax_cells_, par_cells_],
                                   [ax_methods[m], par_methods[m]],
                                   full_DOFs=True,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                bulk_DoFs.append(
                    calculate_DOFs([ax_cells_, par_cells_],
                                   [ax_methods[m], par_methods[m]],
                                   full_DOFs=False,
                                   model=transport_model,
                                   nComp=ncomp, nBound=nbound)
                )
                # Convergence Table
                header, table = convergency_table(
                    [ax_methods[m], par_methods[m]],
                    [ax_cells_, par_cells_],
                    abs_errors[m],
                    error_types=['max', 'L1'],
                    sim_names=simulation_names[m]
                )

            if incl_min_val:
                table = np.column_stack((np.array(table), min_val[m]))
                header.append('Min. value')

            convergence_tables.append(table)

            # Export Data
            if ax_methods[m] == 0:
                data_name = models[modelIdx] + "_FV"
            elif ax_methods[m] < 0:
                data_name = models[modelIdx] + \
                    "_DGexInt_P" + str(abs(ax_methods[m]))
            else:
                data_name = models[modelIdx] + \
                    "_DG_P" + str(abs(ax_methods[m]))
            if par_methods[0] is not None:
                if par_methods[m] < 0:
                    data_name += "_cDG_parP" + str(abs(par_methods[m]))
                elif abs(par_methods[m]) != abs(ax_methods[m]):
                    data_name += "parP" + str(abs(par_methods[m]))

            # add (bulk) DoFs to output table
            header = np.insert(header, len(header), 'DoF')
            result = np.hstack(
                (convergence_tables[m], np.atleast_2d(DoFs[m]).T))
            header = np.insert(header, len(header), 'Axial DoF')
            result = np.hstack(
                (result, np.atleast_2d(bulk_DoFs[m]).T))

            # add method to output table
            header = np.insert(header, 0, '$N_d$')
            arr = np.empty(len(ax_cells_), dtype=object)
            arr[:] = str(int(abs(ax_methods[m])))
            result = np.column_stack((arr, result))

            # export
            if save_path_ is None:
                save_path_ = file_path
            pd.DataFrame(result, columns=header).to_csv(
                save_path_ + data_name + ".csv", index=False)


def compare_h5_files(file1, file2):
    """Compares two h5 files, i.e. their structure, datatypes and datasets.

    Parameters
    ----------
    fil1 : string
        First file name.
    fil2 : string
        Second file name.
    """

    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        # Compare attributes of root group
        attrs1 = dict(f1.attrs.items())
        attrs2 = dict(f2.attrs.items())
        if attrs1 != attrs2:
            print("Attributes of root group are not equal")
            print("File 1:", attrs1)
            print("File 2:", attrs2)
            return False

        # Recursively compare file structure and datasets
        def compare_items(item1, item2):
            if isinstance(item1, h5py.Group):
                if not isinstance(item2, h5py.Group):
                    print(f"File 2 does not contain group '{item1.name}'")
                    return False
                # Compare group attributes
                attrs1 = dict(item1.attrs.items())
                attrs2 = dict(item2.attrs.items())
                if attrs1 != attrs2:
                    print(f"Attributes of group '{item1.name}' are not equal")
                    print("File 1:", attrs1)
                    print("File 2:", attrs2)
                    return False
                # Compare group keys
                keys1 = set(item1.keys())
                keys2 = set(item2.keys())
                if keys1 != keys2:
                    print(f"Keys in group '{item1.name}' are not equal")
                    print("File 1:", keys1)
                    print("File 2:", keys2)
                    return False
                # Recursively compare group items
                for key in keys1:
                    if not compare_items(item1[key], item2[key]):
                        return False
            elif isinstance(item1, h5py.Dataset):
                if not isinstance(item2, h5py.Dataset):
                    print(f"File 2 does not contain dataset '{item1.name}'")
                    return False
                # Compare dataset attributes
                attrs1 = dict(item1.attrs.items())
                attrs2 = dict(item2.attrs.items())
                if attrs1 != attrs2:
                    print(
                        f"Attributes of dataset '{item1.name}' are not equal")
                    print("File 1:", attrs1)
                    print("File 2:", attrs2)
                    return False
                # Compare dataset shape and data
                if item1.dtype != item2.dtype:
                    # if not (item1.dtype.kind == 'S' and item2.dtype.kind in ('S')):
                    print(
                        f"Dataset '{item1.name}' has different data types: {item1.dtype} and {item2.dtype}")
                    return False
                if item1.shape != item2.shape:
                    if not (  # scalar values might have different shapes
                            (item1.shape == () or all(dim == 1 for dim in item1.shape)) and
                            (item2.shape == () or all(
                                dim == 1 for dim in item2.shape))
                    ):
                        print(
                            f"Dataset '{item1.name}' has different shapes: {item1.shape} and {item2.shape}")
                        return False
                if (item1.shape == () or all(dim == 1 for dim in item1.shape)):
                    if item1[()] != item2[()]:
                        print(
                            f"Dataset '{item1.name}' has different scalar values: {item1[()]} and {item2[()]}")
                        return False
                # else:
                #     print(item1.shape == ())
                #     print(item2)
                #     return False
                elif not (item1[:] == item2[:]).all():
                    print(f"Dataset '{item1.name}' has different data values")
                    return False
            else:
                # Unsupported object type
                print(f"Unsupported object type '{type(item1)}'")
                return False
            return True

        # Compare root group items
        if not compare_items(f1, f2):
            return False

        return True


def mult_sim_rerun(file_path, cadet_path, n_wdh):
    """Rerun simulations and keep best simulation time (for benchmarking).

    Parameters
    ----------
    file_path : String
        Path to files to be rerun.
    cadet_path : String
        Path to cadet executable.
    n_wdh : int
        Number of reruns per simulation
    """
    Cadet.cadet_path = cadet_path
    model = Cadet()
    files = os.listdir(file_path)

    for file in files:  # file = files[0]

        if(re.search('.h5', file)):

            model.filename = file_path + "/" + file
            success = model.run_load()
            if not success.returncode == 0:
                print(success)
                break

            best_sim_time = model.root.meta.time_sim

            for i in range(0, n_wdh):

                success = model.run_load()
                if not success.returncode == 0:
                    print(success)
                    break

                # only keep faster simulation time
                best_sim_time = min(best_sim_time, model.root.meta.time_sim)

            model.load()
            model.root.meta.time_sim = best_sim_time
            model.save()
            print(file + " rerun")

        else:
            print(file + " kein h5")


if __name__ == '__main__':
    test_eoc()
    test_calculate_DOFs()
    test_convergency_table()
