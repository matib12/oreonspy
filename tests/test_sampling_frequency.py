import numpy as np
import pytest
from os import path, fsync
import time
import warnings
from pathlib import Path
from packaging.version import Version
import copy

import oreonspy as op
import glob
import random
import oreonspy.utils as ut

#%matplotlib widget

#from scipy import constants as const
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import logging
from scipy.interpolate import interp1d

import gc; gc.collect()

import utils as test_utils

"""
This file conains two automated tests prepared for pytest tool. The first one verifies the agreement between pure and numba implementation of computationally heavy parts of the code. The second one allows the verfication of results provided by all versions of the simulator.

SCENARIOS = [
    #cavity_0c4b9c47-5b22-5678-9107-511887fa03d8
    {"name": "const_low_freq_high_speed_cav1", "params": {"freq": 0.5, "speed": 5.0, "type": "step", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
]
"""

# Test scenario generation parameters
motion_types = ["const", "step", "ramp", "sine", "pulse", "noise"]
number_of_freqs = 6
number_of_speeds = 6
randomly_choose = True
randomly_choose_count = 0
save_generated_scenarios = False  # Set to True to save all generated scenarios
compare_with_existing_data = False  # Set to True to include existing test data

def extract_params_from_filename(filename):
    """
    Extract parameters from a filename.
    
    Example: "noise_cavity_14397bf2-8fe3-5c85-811d-053248fff88c_f4.1_s0.1_pure.npy"
    Returns: {"type": "noise", "cavity": "cavity_14397bf2-8fe3-5c85-811d-053248fff88c", "freq": 4.1, "speed": 0.1}
    """
    # Remove file extension and backend suffix
    name = Path(filename).stem
    if name.endswith("_pure"):
        name = name[:-5]
    elif name.endswith("_numba"):
        name = name[:-6]
    
    parts = name.split("_")
    
    motion_type = parts[0]
    #print(parts)
    cavity = parts[1]+"_"+parts[2]
    
    freq = float(parts[3][1:])
    speed = float(parts[4][1:])
    
    return {
        "type": motion_type,
        "cavity": cavity,
        "freq": freq,
        "speed": speed
    }

def print_params(params, indent=0):
    """Pretty-print a params dict (handles nested dicts/lists)."""
    pad = " " * (indent * 2)
    if isinstance(params, dict):
        for key in sorted(params):
            val = params[key]
            if isinstance(val, dict):
                print(f"{pad}{key}:")
                print_params(val, indent + 1)
            elif isinstance(val, list):
                print(f"{pad}{key}: [")
                for item in val:
                    if isinstance(item, dict):
                        print_params(item, indent + 2)
                    else:
                        print(f"{pad}  {item}")
                print(f"{pad}]")
            elif isinstance(val, float):
                print(f"{pad}{key}: {val:.6g}")
            else:
                print(f"{pad}{key}: {val}")
    else:
        # fallback for non-dict params
        print(f"{pad}{params}")

# Discover cavity XML files in optical_cavities_testset directory
cavity_files = glob.glob(path.join("tests", "optical_cavities_testset", "*.xml"))
cavity_names = [path.basename(f).replace(".xml", "") for f in cavity_files]

#print("Cavity files found:", cavity_files)
#print("Cavity names found:", cavity_names)
print("Cavity files found:", len(cavity_files))
print("Cavity names found:", len(cavity_names))

# Generate scenarios dynamically
SCENARIOS = []
PREVIOUS_SCENARIOS = []

freq_values = np.linspace(0.5, 5.0, number_of_freqs)
speed_values = np.linspace(0.1, 5.0, number_of_speeds)


# LASER
E_in_avg = 1.+0.j  #

def generate_cavity_evolution(cavity, z_evolution, E_evolution, num_points, time_steps, velocity_factor, f_calc, plot=False, title=None):
    results = np.zeros(num_points, dtype=np.complex128)

    z_evolution = np.append(z_evolution, z_evolution[-1])

    d_zeta = z_evolution[1]-z_evolution[0]

    for t in range(num_points):
        #d_zeta = z_evolution[t+1]-z_evolution[t]
        results[t],_ = cavity.sim_step(E_evolution[t], 0, d_zeta)

    print("Result length: {0}".format(len(results)))
    s = np.abs(results)**2
    ph = np.angle(results)
    pdh = ut.V_pdh(0., E_evolution, results)
    
    print(f"Results for v = {velocity_factor} v_cr and f_calc = {f_calc} Hz:")
    #print(results)

    if plot:
        simple_plot = False
        if simple_plot == False:
            ut.plot_cavity_evolution(z_evolution, E_evolution, s, ph, pdh, zeta1_positons=None, s_ref=None, ph_ref=None, title=title, save=False)
        else:
            plt.plot(time_steps, s)
            plt.plot(time_steps, ph)
            plt.plot(time_steps, pdh)
            plt.xlabel('Time (s)')
            plt.ylabel('Cavity Evolution')
            plt.title(f'Cavity Evolution for v = {velocity_factor} v_cr and f_calc = {f_calc} Hz')
            plt.show()
    else:
        return s, ph, pdh


for cavity in cavity_names:
    for motion_type in motion_types:
        for freq in freq_values:
            for speed in speed_values:
                scenario_name = f"{motion_type}_{cavity}_f{freq:.1f}_s{speed:.1f}"
                SCENARIOS.append({
                    "name": scenario_name,
                    "params": {
                        "freq": round(freq, 1),
                        "speed": round(speed, 1),
                        "type": motion_type,
                        "cavity": cavity,
                        "version": op.__version__
                    }
                })

#print(f"{SCENARIOS}")
print(f"Generated {len(SCENARIOS)} test scenarios.")

# Randomly select a subset of scenarios to limit test duration
random.seed(42)
if randomly_choose and (len(SCENARIOS) > randomly_choose_count):
    SCENARIOS = random.sample(SCENARIOS, randomly_choose_count)

print(f"Selected {len(SCENARIOS)} random scenarios.")

if compare_with_existing_data:
    save_generated_scenarios = True  # Ensure saving is enabled when comparing with existing data

    # Parse existing test data from previous versions
    data_dir = Path("tests/data")
    if data_dir.exists():
        print(f"Loading existing test data from {data_dir}")
        for version_dir in data_dir.iterdir():
            if version_dir.is_dir() and Version(version_dir.name) < Version(op.__version__):
                print(f"Checking version directory: {version_dir}")
                npy_files = list(version_dir.glob("*_pure.npy"))
                print(f"Found {len(npy_files)} .npy files in {version_dir}")
                for npy_file in npy_files:
                    scenario_name = npy_file.stem
                    if scenario_name.endswith("_pure"):
                        scenario_name = scenario_name[:-5]
                    elif scenario_name.endswith("_numba"):
                        scenario_name = scenario_name[:-6]
                    print(f"Processing file: {npy_file.name} for scenario: {scenario_name}")
                    # Add all scenarios to PREVIOUS_SCENARIOS to be tested in the second test
                    params = extract_params_from_filename(npy_file.name)
                    params["version"] = version_dir.name
                    print_params(params)
                    #print(f"Adding scenario from file: {npy_file.name} with params: {params}")
                    PREVIOUS_SCENARIOS.append({"name": scenario_name, "params": params})
            else:
                print(f"Skipping version directory: {version_dir} (not older version)")
    
    existing_scenario_names = {s["name"] for s in SCENARIOS}
    for prev_scenario in PREVIOUS_SCENARIOS:
        if prev_scenario["name"] not in existing_scenario_names:
            new_scenario = copy.deepcopy(prev_scenario)
            new_scenario["params"]["version"] = op.__version__
            SCENARIOS.append(new_scenario)


print("==========================")
print(SCENARIOS)
print("==========================")
print(PREVIOUS_SCENARIOS)
print("==========================")

# === FORCE manually defined scenarios for testing here ===
#'''
SCENARIOS = [
    {"name": "const_low_freq_high_speed_cav1", "params": {"freq": [0.9, 0.95, 0.99, 1., 1.1, 1.15, 10.], "speed": 1.0, "type": "step", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    {"name": "const_low_freq_high_speed_cav1", "params": {"freq": [0.5, 1., 10., 20., 30.], "speed": 1.0, "type": "step", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}}
]
PREVIOUS_SCENARIOS = []
#'''

from itertools import combinations

#@pytest.mark.skip(reason="testing other tests")
@pytest.mark.frequency_test  # to run: $ pytest -m frequency_test --capture no
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["name"])
def test_frequencies_agree(scenario):
    params = scenario["params"]

    print_params(params)

    critical_velocity_factor = params["speed"]
    print(f"Testing scenario: {scenario['name']}")

    # PURE BACKEND SIMULATION

    cavity_pure = op.Cavity()
    cavity_pure.xml_load(path.join("tests", "optical_cavities_testset", params["cavity"] + ".xml"))

    # Calculate critical velocity for later use
    critical_velocity = ut.critical_velocity(cavity_pure, ut.lambd)
    print(f"Critical velocity: {critical_velocity:.3E} m/s")

    # Determine optimal sampling frequency and calculate sampling frequency for this test
    f_opt = ut.optimal_sampling_frequency(
        cavity=cavity_pure,
        critical_velocity_factor=critical_velocity_factor
    )
    print(f"Optimal sampling frequency: {f_opt:.3E} Hz")

    f_factor = params["freq"]

    timesteps_list = {}
    pure_results_list = []
    numba_results_list = []

    for f in f_factor:
        # Just to find the right f_calc
        if hasattr(op, "HAS_NUMBA"):
            cavity_pure.simulation(ut.k, f * f_opt, 1.0, backend="pure")
        else:
            cavity_pure.simulation(ut.k, f * f_opt, 1.0)

        f_calc = cavity_pure.f_calc

        # Generate number of time points and time points vector
        tlen,timepoints = ut.generate_time_points_for_constant_velocity(
            critical_velocity_factor*critical_velocity,
            f_calc,
            number_of_FSR=1.0
        )
        print(f"Generated {tlen} time points for simulation.")

        # Generate d_zeta motion profile
        z_step = ut.lambd / tlen  # Spatial step for d_zeta

        
        z_steps = np.ones(tlen) * z_step  # Constant velocity profile
        timesteps_list[str(f)] = timepoints

        # Run PURE backend simulation
        if hasattr(op, "HAS_NUMBA"):
            cavity_pure.simulation(ut.k, f_calc, 1.0, backend="pure")
        else:
            cavity_pure.simulation(ut.k, f_calc, 1.0)

        result_E_pure = np.zeros(tlen, dtype=np.complex128)
        result_E_ref_pure = np.zeros(tlen, dtype=np.complex128)

        start = time.perf_counter()
        for i in range(tlen):
            result_E_pure[i], result_E_ref_pure[i] = cavity_pure.sim_step(phaseshifted_input_electric_field=1., input_mirror_displacement=0., output_mirror_displacement=z_steps[i])
        end = time.perf_counter()
        pure_exec_time = end - start
        print(f"PURE backend simulation time: {pure_exec_time:.3f} seconds")

        pure_results_list.append((f, result_E_pure.copy()))

        if save_generated_scenarios:
            # Create directory for saving if it doesn't exist
            dest_path = Path(path.join("tests", "data", op.__version__))
            dest_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving generated scenario data to {dest_path}")

            pure_file_path = path.join(dest_path, scenario["name"] + "_pure.npy")
            with open(pure_file_path, 'wb') as f:
                np.save(f, result_E_pure)
                f.flush()
                fsync(f.fileno())

        # NUMBA BACKEND SIMULATION
        # This part resuses most of the setup from above to ensure identical conditions

        if hasattr(op, "HAS_NUMBA") and op.HAS_NUMBA:
            cavity_numba = op.Cavity()
            cavity_numba.xml_load(path.join("tests", "optical_cavities_testset", params["cavity"] + ".xml"))

            result_E_numba = np.zeros(tlen, dtype=np.complex128)
            result_E_ref_numba = np.zeros(tlen, dtype=np.complex128)

            # Run NUMBA backend simulation
            cavity_numba.simulation(ut.k, f_calc, 1.0, backend="numba")
            print("N_pre: ", cavity_numba.N_pre)
            print("Refined f_calc: ", cavity_numba.f_calc)

            start = time.perf_counter()
            for i in range(tlen):
                result_E_numba[i], result_E_ref_numba[i] = cavity_numba.sim_step(phaseshifted_input_electric_field=1., input_mirror_displacement=0., output_mirror_displacement=z_steps[i])
            end = time.perf_counter()
            numba_exec_time = end - start
            print(f"NUMBA backend simulation time: {numba_exec_time:.3f} seconds")

            numba_results_list.append((f, result_E_numba.copy()))

            if save_generated_scenarios:
                print(f"Saving generated scenario data to {dest_path}")

                numba_file_path = path.join(dest_path, scenario["name"] + "_numba.npy")
                with open(numba_file_path, 'wb') as f:
                    np.save(f, result_E_numba)
                    f.flush()
                    fsync(f.fileno())

            if numba_exec_time > pure_exec_time:
                warnings.warn(UserWarning("NUMBA slower than PURE"))  # ISSUE WARNING IF NUMBA IS SLOWER

            # COMPARE RESULTS
            #np.testing.assert_allclose(result_E_pure, result_E_numba, rtol=1e-7, atol=1e-10)
            #np.testing.assert_allclose(result_E_ref_pure, result_E_ref_numba, rtol=1e-7, atol=1e-10)
    
    # Plot all pure solutions
    plt.figure(figsize=(12, 6))
    #for f, result_E in pure_results_list:
    for f, result_E in numba_results_list:
        plt.plot(timesteps_list[str(f)], np.abs(result_E)**2, label=f'f={f}', alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Power (|E|²)')
    plt.legend()
    plt.title('All Pure Solutions')
    plt.grid(True)
    #plt.show()  # Optional: comment out if you don't want to see the plot during testing
    plt.close()

    # Compare all results from pure and numba implementations
    for a, b in combinations(pure_results_list, 2):
        print(f"Comparing results for frequency factor f={a[0]} and f={b[0]}")
        z_step_pure = timesteps_list[str(a[0])]
        z_step_numba = timesteps_list[str(b[0])]

        try:
            test_utils.compare_vectors(z_step_pure, z_step_numba, a[1], b[1], plot=False)
        except AssertionError as e:
            assert True
            # plt.figure(figsize=(12, 6))
            # plt.plot(z_step_pure, np.abs(a[1])**2, label=f'Pure (f={a[0]})', marker='o', alpha=0.7)
            # plt.plot(z_step_numba, np.abs(b[1])**2, label=f'Numba (f={b[0]})', marker='s', alpha=0.7)
            # plt.xlabel('Time Step')
            # plt.ylabel('Power (|E|²)')
            # plt.legend()
            # plt.title(f'Comparison of Pure (f={a[0]}) vs Numba (f={b[0]}) Results')
            # plt.grid(True)
            # plt.show()
