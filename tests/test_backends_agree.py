import numpy as np
import pytest
from os import path
import time
import warnings
from pathlib import Path

import oreonspy as op
import glob
import random
import oreonspy.utils as ut

# Discover cavity XML files in optical_cavities_testset directory
cavity_files = glob.glob(path.join("tests", "optical_cavities_testset", "*.xml"))
cavity_names = [path.basename(f).replace(".xml", "") for f in cavity_files]

# Generate scenarios dynamically
SCENARIOS = []
motion_types = ["const", "step", "ramp", "sine", "pulse", "noise"]
number_of_freqs = 6
number_of_speeds = 6
randomly_choose = True
randomly_choose_count = 10
save_generated_scenarios = True  # Set to True to save all generated scenarios

freq_values = np.linspace(0.5, 5.0, number_of_freqs)
speed_values = np.linspace(0.1, 5.0, number_of_speeds)

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
                        "cavity": cavity
                    }
                })

#print(f"{SCENARIOS}")
print(f"Generated {len(SCENARIOS)} test scenarios.")

# Randomly select a subset of scenarios to limit test duration
random.seed(42)
if randomly_choose and (len(SCENARIOS) > randomly_choose_count):
    SCENARIOS = random.sample(SCENARIOS, randomly_choose_count)

print(f"Selected {len(SCENARIOS)} random scenarios.")

# Manually defined scenarios for testing
'''
SCENARIOS = [
    #cavity_0c4b9c47-5b22-5678-9107-511887fa03d8
    {"name": "const_low_freq_high_speed_cav1", "params": {"freq": 0.5, "speed": 5.0, "type": "step", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    {"name": "step_low_freq_high_speed_cav1", "params": {"freq": 1.0, "speed": 4.0, "type": "ramp", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    {"name": "ramp_low_freq_high_speed_cav1", "params": {"freq": 2.0, "speed": 3.0, "type": "sine", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    {"name": "sine_low_freq_high_speed_cav1", "params": {"freq": 3.0, "speed": 2.0, "type": "pulse", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    {"name": "pulse_low_freq_high_speed_cav1", "params": {"freq": 4.0, "speed": 1.0, "type": "noise", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    {"name": "noise_low_freq_high_speed_cav1", "params": {"freq": 5.0, "speed": 0.1, "type": "const", "cavity": "cavity_0c4b9c47-5b22-5678-9107-511887fa03d8"}},
    #cavity_99b865df-23c7-5d84-9c34-7d86b2ade673
    {"name": "const_low_freq_high_speed_cav1", "params": {"freq": 0.5, "speed": 5.0, "type": "const", "cavity": "cavity_99b865df-23c7-5d84-9c34-7d86b2ade673"}},
    {"name": "step_low_freq_high_speed_cav1", "params": {"freq": 1.0, "speed": 4.0, "type": "step", "cavity": "cavity_99b865df-23c7-5d84-9c34-7d86b2ade673"}},
    {"name": "ramp_low_freq_high_speed_cav1", "params": {"freq": 2.0, "speed": 3.0, "type": "ramp", "cavity": "cavity_99b865df-23c7-5d84-9c34-7d86b2ade673"}},
    {"name": "sine_low_freq_high_speed_cav1", "params": {"freq": 3.0, "speed": 2.0, "type": "sine", "cavity": "cavity_99b865df-23c7-5d84-9c34-7d86b2ade673"}},
    {"name": "pulse_low_freq_high_speed_cav1", "params": {"freq": 4.0, "speed": 1.0, "type": "pulse", "cavity": "cavity_99b865df-23c7-5d84-9c34-7d86b2ade673"}},
    {"name": "noise_low_freq_high_speed_cav1", "params": {"freq": 5.0, "speed": 0.1, "type": "noise", "cavity": "cavity_99b865df-23c7-5d84-9c34-7d86b2ade673"}},
    # add more parameter sets here
]
'''

@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s["name"])
def test_pure_vs_numba_agree(scenario):
    params = scenario["params"]

    critical_velocity_factor = params["speed"]
    print(f"Testing scenario: {scenario['name']} with speed factor {critical_velocity_factor}")

    # PURE BACKEND SIMULATION

    cavity_pure = op.Cavity()
    cavity_pure.xml_load(path.join("tests", "optical_cavities_testset", params["cavity"] + ".xml"))

    # Determine optimal sampling frequency and calculate sampling frequency for this test
    f_opt = ut.optimal_sampling_frequency(
        cavity=cavity_pure,
        critical_velocity_factor=critical_velocity_factor
    )
    print(f"Optimal sampling frequency: {f_opt:.3E} Hz")

    f_factor = params["freq"]
    f_calc = f_factor * f_opt

    # Calculate critical velocity for later use
    critical_velocity = ut.critical_velocity(cavity_pure, ut.lambd)
    print(f"Critical velocity: {critical_velocity:.3E} m/s")

    # Generate number of time points and time points vector
    tlen,_ = ut.generate_time_points_for_constant_velocity(
        critical_velocity_factor*critical_velocity,
        f_calc,
        number_of_FSR=1.0
    )
    print(f"Generated {tlen} time points for simulation.")

    # Generate d_zeta motion profile
    z_step = ut.lambd / tlen  # Spatial step for d_zeta

    if params["type"] == "const":
        z_steps = np.ones(tlen) * z_step
    elif params["type"] == "step":
        z_steps = np.zeros(tlen)
        z_steps[tlen // 2 :] = z_step * 2.
    elif params["type"] == "ramp":
        z_steps = np.linspace(0, z_step * 3., tlen)
    elif params["type"] == "sine":
        z_steps = (z_step * 2.0) * (1 + np.sin(2 * np.pi * np.arange(tlen) / tlen))
    elif params["type"] == "pulse":
        z_steps = np.zeros(tlen)
        pulse_width = tlen // 10
        start_idx = (tlen - pulse_width) // 2
        z_steps[start_idx:start_idx + pulse_width] = z_step * 3.
    elif params["type"] == "noise":
        np.random.seed(0)  # For reproducibility
        z_steps = np.random.uniform(0, z_step, tlen)
    else:
        raise ValueError(f"Unknown motion type: {params['type']}")

    # Run PURE backend simulation
    cavity_pure.simulation(ut.k, f_calc, 1.0, backend="pure")

    result_E_pure = np.zeros(tlen, dtype=np.complex128)
    result_E_ref_pure = np.zeros(tlen, dtype=np.complex128)

    start = time.perf_counter()
    for i in range(tlen):
        result_E_pure[i], result_E_ref_pure[i] = cavity_pure.sim_step(E_in_laser=1., d_zeta_in=0., d_zeta=z_steps[i])
    end = time.perf_counter()

    pure_exec_time = end - start
    print(f"PURE backend simulation time: {pure_exec_time:.3f} seconds")

    if save_generated_scenarios:
        # Create directory for saving if it doesn't exist
        dest_path = Path(path.join("tests", "data", op.__version__))
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving generated scenario data to {dest_path}")

        np.save(path.join(dest_path, scenario["name"] + "_pure.npy"), result_E_pure)

    # NUMBA BACKEND SIMULATION
    # This part resuses most of the setup from above to ensure identical conditions

    cavity_numba = op.Cavity()
    cavity_numba.xml_load(path.join("tests", "optical_cavities_testset", params["cavity"] + ".xml"))

    result_E_numba = np.zeros(tlen, dtype=np.complex128)
    result_E_ref_numba = np.zeros(tlen, dtype=np.complex128)

    # Run NUMBA backend simulation
    cavity_numba.simulation(ut.k, f_calc, 1.0, backend="numba")

    start = time.perf_counter()
    for i in range(tlen):
        result_E_numba[i], result_E_ref_numba[i] = cavity_numba.sim_step(E_in_laser=1., d_zeta_in=0., d_zeta=z_steps[i])
    end = time.perf_counter()
    numba_exec_time = end - start
    print(f"NUMBA backend simulation time: {numba_exec_time:.3f} seconds")

    if save_generated_scenarios:
        np.save(path.join(dest_path, scenario["name"] + "_numba.npy"), result_E_numba)

    if numba_exec_time > pure_exec_time:
        warnings.warn(UserWarning("NUMBA slower than PURE"))  # FAIL TEST IF NUMBA IS SLOWER

    # COMPARE RESULTS
    np.testing.assert_allclose(result_E_pure, result_E_numba, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result_E_ref_pure, result_E_ref_numba, rtol=1e-7, atol=1e-10)