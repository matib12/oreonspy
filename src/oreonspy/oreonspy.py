#!/usr/bin/env python3
# coding: utf-8

# ### Calculates electric field amplitude inside an optical resonator cavity in dynamic case.
#
# - Andrea Svizzeretto, <andrea.svizzeretto@studenti.unipg.it>
# - Mateusz Bawaj, <mateusz.bawaj@unipg.it>

import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import logging
from dataclasses import dataclass
from typing import Optional

from ._pure_impl import heavy as _pure_heavy

try:
    from ._numba_impl import heavy as _numba_heavy

    HAS_NUMBA = True
except Exception as e:
    HAS_NUMBA = False

c = 299792458.0  # Speed of light in vacuum [m/s]

logger = logging.getLogger(__name__.split(".")[-1])
logger.setLevel(logging.INFO)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


@dataclass
class CavityParams:
    t_a: float
    r_a: float
    r_b: float
    cavity_length: float

    @property
    def half_roundtrip_time(self):
        return self.cavity_length / c


@dataclass
class SimulationParams:
    wave_number: Optional[float] = None
    k2j: Optional[complex] = None
    requested_sampling_frequency: Optional[float] = None
    initial_input_electric_field: Optional[complex] = None
    f_calc: Optional[float] = None
    num_roundtrips: Optional[int] = None
    Theta: Optional[float] = None
    partial_Theta: Optional[bool] = None
    Theta_fraction: Optional[float] = None
    num_of_subhist: Optional[int] = None
    sampling_frequency_accuracy: Optional[float] = None


class Cavity:
    simulation_initialized = False

    def __init__(
        self,
        t_a=0.001,
        T_A=None,
        r_a=0.99,
        R_A=None,
        r_b=0.999,
        R_B=None,
        cavity_length=3000.0,
        debug=None,
        log_file=None,
    ):
        self.sim_params = SimulationParams()

        if T_A is not None:
            t_a = np.sqrt(T_A)
        else:
            t_a = t_a

        if R_A is not None:
            r_a = np.sqrt(R_A)
        else:
            r_a = r_a

        if R_B is not None:
            r_b = np.sqrt(R_B)
        else:
            r_b = r_b

        self.params = CavityParams(
            t_a=t_a, r_a=r_a, r_b=r_b, cavity_length=cavity_length
        )
        # self._sync_param_aliases()

        self.debug = debug
        if debug is None:
            logger.disabled = True
            pass
        else:
            if log_file is not None:
                logger.disabled = False
                logger.setLevel(debug)
                self.file_handler = logging.FileHandler(log_file)
                self.file_handler.setLevel(debug)
                formatter = logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(message)s"
                )
                self.file_handler.setFormatter(formatter)
                logger.addHandler(self.file_handler)
            else:
                logging.basicConfig(level=debug)
                logger.setLevel(debug)

            logger.debug("Cavity initialized with parameters:")
            logger.debug("t_a: {0}".format(self.params.t_a))
            logger.debug("r_a: {0}".format(self.params.r_a))
            logger.debug("r_b: {0}".format(self.params.r_b))
            logger.debug("L: {0}".format(self.params.cavity_length))
            logger.debug("T: {0}".format(self.params.half_roundtrip_time))
            logger.debug("Debug level: {0}".format(debug))

    def cavity_loss(self):
        loss = 1.0 - np.power(self.params.t_a, 2) - np.power(self.params.r_a, 2)

        if loss < 0.0:
            print("Attenti ai valori")
        else:
            print("loss: {0}".format(loss))

        return loss

    def N_eff(self):
        """
        Effective number of photon round trips in a FabryPerot cavity (Rakhmanov Eq. 1.57)
        """
        return int(np.round(1.0 / np.abs(np.log(self.params.r_a * self.params.r_b))))

    def finesse_coefficient(self):
        """
        Coefficient of finesse
        """
        return (
            4.0
            * self.params.r_a
            * self.params.r_b
            / np.power(1.0 - self.params.r_a * self.params.r_b, 2)
        )

    def tau_s(self):
        """
        Formula equivalent to Eq. 2.17 (Rakhmanov)

        return 2. * T * N_eff(r_a, r_b)
        """
        return self.finesse_coefficient() * self.params.cavity_length / (np.pi * c)

    def Finesse(self):
        """
        Finesse
        """
        return np.sqrt(self.finesse_coefficient()) * np.pi / 2.0

    def tau(self):
        """
        Cavity decay time: N_eff()*2.*__L__/c
        """
        return 2.0 * self.params.half_roundtrip_time * self.N_eff()

    def gain(self):
        """
        Amplitude gain of cavity Eq. 1.67 (Rakhmanov)
        """
        return self.params.t_a / (1.0 - self.params.r_a * self.params.r_b)

    def print_params(self):
        """
        Print cavity parameters
        """
        for field_name, field_value in self.params.__dict__.items():
            print(f"{field_name}: {field_value}")

    def number_of_subhistories(self, eta_2T):
        """
        Compute the number of phase-shifted subhistories required by the simulator.

        Parameters
        ----------
        eta_2T : float
            dimensionless round-trip normalized time-step expressing ratio
            between the sampling period and the cavity round-trip time.
            Must be strictly positive.

        Returns
        -------
        int
            Number of subhistories:
            - 1 when eta_2T < 1
            - round(eta_2T) otherwise
        """
        assert eta_2T > 0, "eta_2T must be positive"
        return int(np.round(1.0 / eta_2T) if eta_2T < 1.0 else 1)

    def round_for_inverse_curve(self, eta_2T):
        """
        Select the integer order used by the inverse-curve rounding rule.

        For each value `eta_2T`, let `k0 = floor(eta_2T)` and define the
        switching boundary

            b = 2*k0*(k0 + 1) / (2*k0 + 1)

        The returned integer is:
        - `k0` when `eta_2T < b`
        - `k0 + 1` otherwise

        Parameters
        ----------
        eta_2T : float or array-like of float
            Dimensionless normalized time-step value(s). Must be strictly positive.

        Returns
        -------
        tuple
            `(k, b)` where:
            - `k` is the selected integer order (int for scalar input, ndarray for array input)
            - `b` is the corresponding boundary value(s) used for the decision
        """
        assert np.all(np.asarray(eta_2T) > 0), "eta_2T must be positive"
        eta_2T_arr = np.asarray(eta_2T, dtype=float)

        k0 = np.floor(eta_2T_arr)
        b = 2 * k0 * (k0 + 1) / (2 * k0 + 1)
        k = np.where(eta_2T_arr < b, k0, k0 + 1).astype(int)

        # Preserve scalar return type for scalar input
        if np.isscalar(eta_2T):
            return int(k), float(b)
        return k, b

    def estimate_f_calc(self, f_calc_desired):
        """
        Simulator case algorithm from cavity.simulation() in oreonspy 3.2.3.

        Parameters
        ----------
        f_calc_desired -- the desired sampling frequency

        Returns
        -------
        f_calc -- the actual sampling frequency used by the simulator
        N -- the number of round trips in the cavity during the calculation time
        Theta -- the calculation time step
        partial_Theta -- a flag indicating whether the calculation time step is partially determined
        n_of_subhistories -- the number of phase-shifted subhistories required by the simulator

        """
        f_2T = 0.5 / self.params.half_roundtrip_time  # Half-round trip time

        N = 1  # Initlized to one in the constructor.
        n_of_subhistories = 1  # Initlized to one in the constructor.
        partial_Theta = False  # Initlized to False in the constructor.
        Theta_fraction = 1.0  # Initlized to 1.0 in the constructor.

        _N_eff_factor = 2

        eta_2T = f_2T / (f_calc_desired)

        if f_calc_desired > f_2T:
            n_of_subhistories = self.number_of_subhistories(eta_2T)
            f_calc = f_2T * n_of_subhistories
        else:
            N_max = _N_eff_factor * self.N_eff()
            if f_calc_desired < (f_2T / N_max):
                N = N_max
                f_calc = (
                    f_calc_desired  # In this case desired f_calc is exactly simulated.
                )
                partial_Theta = True
                Theta_fraction = 1.0 - (
                    N * f_calc * 2.0 * self.params.half_roundtrip_time
                )
            else:
                N = self.round_for_inverse_curve(eta_2T)[0]
                f_calc = f_2T / N  # in Hz

        Theta = 1.0 / f_calc  # in Seconds

        logger.debug("N: {0}".format(N))
        logger.debug("Number of subhistories: {0}".format(n_of_subhistories))
        logger.debug("Theta: {0}".format(Theta))
        logger.debug("Final f_calc: {0}".format(f_calc))

        return f_calc, N, Theta, partial_Theta, Theta_fraction, n_of_subhistories

    def simulation(
        self, wave_number, requested_sampling_frequency, initial_input_electric_field, backend="auto"
    ):
        """
        With respect to version 2.0.0, the simulation works with incident electric field instead of optical power.
        wave_number: wave number
        desired_f_calc: desired calculation frequency
        initial_input_electric_field: initial electric field amplitude
        backend: "pure" | "numba" | "auto"
        """
        logger.debug("Simulation started")
        logger.debug("wave_number: {0}".format(wave_number))
        logger.debug("Desired f_calc: {0}".format(requested_sampling_frequency))
        logger.debug(
            "Initial input electric field: {0}".format(initial_input_electric_field)
        )

        # Initial values
        k2j = -2.0j * wave_number  # Used frequently in step()

        # Estimate sampling frequency and related parameters
        sampling_frequency, N, Theta, partial_Theta, Theta_fraction, num_of_subhistories = (
            self.estimate_f_calc(requested_sampling_frequency)
        )

        sampling_frequency_accuracy = 1.0 - np.abs(sampling_frequency - requested_sampling_frequency) / requested_sampling_frequency

        self.sim_params = SimulationParams(
            wave_number=wave_number,
            k2j=k2j,
            requested_sampling_frequency=requested_sampling_frequency,
            initial_input_electric_field=initial_input_electric_field,
            f_calc=sampling_frequency,
            num_roundtrips=N,
            Theta=Theta,
            partial_Theta=partial_Theta,
            Theta_fraction=Theta_fraction,
            num_of_subhist=num_of_subhistories,
            sampling_frequency_accuracy=sampling_frequency_accuracy,
        )

        # self._sync_sim_param_aliases()

        # Arrays initialization
        self.n = np.arange(0, self.sim_params.num_roundtrips + 1, 1)
        self.rarbn = np.power(self.params.r_a * self.params.r_b, self.n)

        self.e2iknL = np.exp(
            -2.0j * self.sim_params.wave_number * (self.n) * self.params.cavity_length
        )  # Convert to the case when: L is multiple of lambd

        self.rarbne2iknL = self.rarbn * self.e2iknL

        logger.debug("n: {0}".format(self.n))
        logger.debug("rarbn: {0}".format(self.rarbn))
        logger.debug("e2iknL: {0}".format(self.e2iknL))
        logger.debug("rarbne2iknL: {0}".format(self.rarbne2iknL))

        plot_debug = False
        if plot_debug:
            # Plot self.rarbn, self.e2iknL, and self.rarbne2iknL in the same figure
            fig, ax1 = plt.subplots()

            # Plot self.rarbn
            ax1.plot(self.n, self.rarbn, label="$(r_a r_b)^n$", color="blue")
            ax1.set_xlabel("n")
            ax1.set_ylabel("$(r_a r_b)^n$", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            # Plot self.e2iknL magnitude
            # ax1.plot(self.n, np.abs(self.e2iknL), label="|$\exp(-2iknL)$|", color="green")  # Always 1
            # ax1.plot(self.n, np.abs(self.rarbne2iknL), label="|$(r_a r_b)^n \cdot \exp(-2iknL)$|", color="red")  # Equal to self.rarbn

            # Create a secondary y-axis for the phase
            ax2 = ax1.twinx()
            ax2.plot(
                self.n,
                np.unwrap(np.angle(self.e2iknL)),
                label=r"Phase of $\exp(-2iknL)$",
                color="orange",
                linestyle="--",
                marker="o",
            )
            ax2.set_ylabel("Phase [rad]", color="black")
            ax2.tick_params(axis="y", labelcolor="black")

            # Add legends
            fig.legend(
                loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
            )
            plt.title(
                r"Decay and Phase of $(r_a r_b)^n$, $\exp(-2iknL)$, and $(r_a r_b)^n \cdot \exp(-2iknL)$"
            )
            plt.show()

        self.frac, _ = np.modf(
            self.params.cavity_length * self.sim_params.wave_number / (2.0 * np.pi)
        )
        self.phi = 2.0 * np.pi * self.frac
        logger.debug("phi: {0}".format(self.phi))

        self.airy_phi = self.intracavity_electric_field_static_solution(
            self.sim_params.initial_input_electric_field, self.phi
        )
        logger.debug("E_adiabatic cplx: {0}".format(self.airy_phi))
        logger.debug("E_adiabatic abs : {0}".format(np.abs(self.airy_phi)))
        logger.debug("E_adiabatic angl: {0}".format(np.angle(self.airy_phi)))

        self.last_intracavity_electric_field_all_subhist = (
            self.airy_phi
            * np.ones(self.sim_params.num_of_subhist, dtype=np.complex128)
            * np.exp(1.0j * np.angle(self.sim_params.initial_input_electric_field))
        )

        # Define a list of deque buffers for the electric field
        self.input_electric_field_history_all_subhist = [
            self.sim_params.initial_input_electric_field
            * np.ones(self.sim_params.num_roundtrips, dtype=np.complex128)
            for _ in range(self.sim_params.num_of_subhist)
        ]
        logger.debug(
            "last_intracavity_electric_field_all_subhist cplx: {0}".format(
                self.last_intracavity_electric_field_all_subhist
            )
        )
        logger.debug(
            "last_intracavity_electric_field_all_subhist abs : {0}".format(
                np.abs(self.last_intracavity_electric_field_all_subhist)
            )
        )
        logger.debug(
            "last_intracavity_electric_field_all_subhist angl: {0}".format(
                np.angle(self.last_intracavity_electric_field_all_subhist)
            )
        )

        self.last_total_output_mirror_displacement_all_subhist = np.zeros(
            self.sim_params.num_of_subhist, dtype=np.float64
        )
        self.last_output_mirror_displacement_all_subhist = np.zeros(
            self.sim_params.num_of_subhist, dtype=np.float64
        )
        self.total_input_mirror_displacement = 0.0

        self.input_electric_field = np.zeros(
            self.sim_params.num_roundtrips, dtype=np.complex128
        )

        self.sim_step_counter = 0

        global heavy

        if backend == "pure":
            heavy = _pure_heavy

        elif backend == "numba":
            if not HAS_NUMBA:
                raise RuntimeError(
                    "Numba backend requested but numba is not available. "
                    "Install with `pip install mypkg[numba]`."
                )
            heavy = _numba_heavy

        elif backend == "auto":
            if HAS_NUMBA:
                heavy = _numba_heavy
            else:
                heavy = _pure_heavy

        else:
            raise ValueError(
                f"Unknown backend {backend!r}. " "Use 'pure', 'numba' or 'auto'."
            )

        self.simulation_initialized = True

    def __sim_step__(self, output_mirror_displacement=0.0, input_electric_field=1.0):
        """
        With respect to version 2.0.0, the simulation works with incident electric field instead of optical power.

        d_zeta: displacement of the output mirror
        E_in_curr: current electric input field
        """

        if self.simulation_initialized == False:
            print("Initialize first")
            return

        subhist_idx = self.sim_step_counter % self.sim_params.num_of_subhist
        # logger.debug("Chain idx: {0}".format(chain_idx))

        # Update the displacement of the output mirror
        self.last_output_mirror_displacement_all_subhist[subhist_idx] = (
            output_mirror_displacement
        )

        (
            self.input_electric_field_history_all_subhist[subhist_idx],
            intracavity_electric_field,
            self.last_total_output_mirror_displacement_all_subhist[subhist_idx],
        ) = heavy(
            output_mirror_displacement,
            input_electric_field,
            self.last_output_mirror_displacement_all_subhist,
            self.last_total_output_mirror_displacement_all_subhist[subhist_idx],
            self.sim_params.partial_Theta,
            self.sim_params.Theta_fraction,
            self.sim_params.num_roundtrips,
            self.input_electric_field_history_all_subhist[subhist_idx],
            self.rarbne2iknL,
            self.sim_params.k2j,
            self.params.t_a,
            self.last_intracavity_electric_field_all_subhist[subhist_idx],
        )

        # if not self.partial_Theta:
        self.last_intracavity_electric_field_all_subhist[subhist_idx] = (
            intracavity_electric_field
        )

        # logger.debug("E_last: {0}".format(self.last_intracavity_electric_field_all_subhist))

        self.sim_step_counter += 1  # Be carefull with the overflow!!!

        return intracavity_electric_field

    def sim_step(
        self,
        input_electric_field=1.0,
        input_mirror_displacement=0.0,
        output_mirror_displacement=0.0,
    ):
        """
        Simulate the electric field propagation through a two-mirror cavity.
        This method calculates the electric field after propagating through a
        two-mirror cavity with given initial electric field and mirror displacements.

        Parameters:
        -----------
        input_electric_field_amplitude : complex, optional
            The initial electric field amplitude emitted by the laser and referred to the external reference frame (default is 1.0).
        input_mirror_displacement : float, optional
            The displacement of the input mirror (default is 0.0).
        output_mirror_displacement : float, optional
            The displacement of the output mirror (default is 0.0).

        Returns:
        --------
        tuple:
            - electric_field_inside_cavity : complex
            The electric field inside the cavity after propagation.
            - reflected_electric_field : complex
            The reflected electric field from the cavity.

        Notes:
        ------
        - `self.k` is the wave number.
        - `self.total_input_mirror_displacement` is the sum of previous mirror displacements.
        """

        # Total cavity length deviation from the initial length
        cavity_length_variation = output_mirror_displacement - input_mirror_displacement

        # Displacement of the input mirror from the initial position
        self.total_input_mirror_displacement += input_mirror_displacement

        # Electric field on the input mirror
        phaseshifted_input_electric_field = input_electric_field * np.exp(
            self.sim_params.k2j * self.total_input_mirror_displacement
        )

        intracavity_electric_field = self.__sim_step__(
            output_mirror_displacement=cavity_length_variation,
            input_electric_field=phaseshifted_input_electric_field,
        )

        reflected_electric_field = self.compute_reflected_field(
            intracavity_electric_field=intracavity_electric_field,
            input_electric_field=phaseshifted_input_electric_field,
            total_input_mirror_displacement=self.total_input_mirror_displacement,
        )

        return intracavity_electric_field, reflected_electric_field

    def sim_reset(self):
        self.last_intracavity_electric_field_all_subhist = (
            self.airy_phi
            * np.ones(self.sim_params.num_of_subhist, dtype=np.complex128)
            * np.exp(1.0j * np.angle(self.sim_params.E_in_init))
        )
        self.input_electric_field_history_all_subhist = [
            self.sim_params.E_in_init
            * np.ones(self.sim_params.num_roundtrips, dtype=np.complex128)
            for _ in range(self.sim_params.num_of_subhist)
        ]
        self.last_total_output_mirror_displacement_all_subhist = np.zeros(
            self.sim_params.num_of_subhist
        )
        self.total_input_mirror_displacement = 0.0
        self.last_output_mirror_displacement_all_subhist = np.zeros(
            self.sim_params.num_of_subhist
        )
        self.sim_step_counter = 0

    def print_sim_params(self):
        for field_name, field_value in self.sim_params.__dict__.items():
            print(f"{field_name}: {field_value}")

    def plot_sim_factors(self):
        plt.plot(self.rarbn, label="$(r_a r_b)^n$")
        plt.title("Decaying $n$ powered $r_a r_b$ product")
        plt.xlabel("n")
        plt.legend()

        plt.plot(np.abs(self.e2iknL), label="Mag")
        plt.plot(np.angle(self.e2iknL), label="Phase")
        plt.title(r"$\exp(-2ik(n-1)L)$")
        plt.xlabel("n")
        plt.legend()

    def Airy(self, phi):
        """
        Calculate the Airy function for a given detuning phase phi.
        """
        return 1.0 / (1.0 + self.finesse_coefficient() * np.sin(phi) ** 2)

    def intracavity_electric_field_static_solution(self, input_electric_field, phi):
        r"""
        Calculate the adiabatic electric field inside the cavity based on the input electric field.

        This method uses the formula from Rakhmanov Eq. 1.72 to compute the adiabatic
        electric field.

        Parameters:
        -----------
        input_electric_field : complex
            The input electric field.
        phi : float
            The detuning phase is defined by both the length offset and the laser frequency offset: phi = k\Xi + \omega_s T.
            Where k is the wave number, \Xi is the length offset, \omega_s is the laser frequency offset, and T is the half cavity round-trip time.

        Returns:
        --------
        complex
            The adiabatic electric field.

        Notes:
        ------
        - `self.t_a` is a parameter related to the transmission coefficient.
        - `self.r_a` and `self.r_b` are parameters related to the reflection coefficients.
        - The formula involves the absolute value and angle of the input electric field.
        """
        """
         (Rakhmanov Eq. 1.72)
        """
        return (
            self.params.t_a
            * np.abs(input_electric_field)
            / (1.0 - self.params.r_a * self.params.r_b * np.exp(-2.0j * phi))
        )

    def xml_save(self, filename):
        """
        The method saves the following parameters of the Cavity object if they exist: r_a, r_b, t_a, __L__, T.

        Parameters:
        --------
        filename (str): The name of the file to save the XML data. If the filename does not end with ".xml", it will be appended automatically.

        Example:
        --------
        cavity = Cavity(r_a = 1.0, r_b = 2.0, t_a = 3.0, L = 4.0)
        cavity.xml_save("cavity_parameters.xml")

        This will create an XML file named "cavity_parameters.xml" with the parameters of the Cavity object.
        """
        root = ET.Element("Cavity")

        # Parameters to save from the CavityParams dataclass
        params_to_save = ["t_a", "r_a", "r_b", "cavity_length"]

        # Add parameters as sub-elements
        for param_name in params_to_save:
            if hasattr(self.params, param_name):
                param_value = getattr(self.params, param_name)
                param_element = ET.SubElement(root, param_name)
                param_element.text = str(param_value)

        # Also save the computed T property
        param_element = ET.SubElement(root, "half_roundtrip_time")
        param_element.text = str(self.params.half_roundtrip_time)

        # Create the tree and write to an XML file
        tree = ET.ElementTree(root)
        try:
            if not filename.endswith(".xml"):
                filename += ".xml"
            tree.write(filename)
        except Exception as e:
            logger.error(f"Error writing the XML file: {e}")

    def xml_load(self, filename=None):
        """
        Load cavity parameters from an XML file.

        This method supports two call styles:

        1) Instance style (updates the existing object):
           cavity = Cavity()
           cavity.xml_load('path/to/parameters.xml')

        2) Class style (creates and returns a loaded object):
           cavity = Cavity.xml_load('path/to/parameters.xml')
        """
        if isinstance(self, Cavity):
            cavity_obj = self
            if filename is None:
                raise TypeError(
                    "Cavity.xml_load() missing 1 required positional argument: 'filename'"
                )
            xml_filename = filename
        else:
            # Allow class-style call Cavity.xml_load('file.xml') by interpreting
            # the first argument as filename and creating a new object.
            cavity_obj = Cavity()
            xml_filename = self

        # Load the XML file
        try:
            tree = ET.parse(xml_filename)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Error parsing the XML file: {e}")
            return
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return

        # Extract parameters from the XML
        params = {}
        for param in root:
            params[param.tag] = float(param.text)

        # Reinitialize the Cavity object with the loaded parameters
        cavity_obj.__init__(
            t_a=params["t_a"],
            r_a=params["r_a"],
            r_b=params["r_b"],
            cavity_length=params["cavity_length"],
            debug=cavity_obj.debug,
        )
        return cavity_obj

    def compute_reflected_field(
        self,
        intracavity_electric_field,
        input_electric_field=1.0,
        total_input_mirror_displacement=0.0,
    ):
        """
        TODO: verify what is the phase parameter
        Calculate the reflected electric field (E_ref) based on the input electric field (E) and the input laser electric field (E_in_laser).

        Parameters:
        intracavity_electric_field (float): The electric field inside the cavity.
        input_electric_field (float, optional): The input electric field. Default is 1.

        Returns:
        float: The reflected electric field (E_ref).

        Notes:
        This function uses the formula from Eq 1.48:

        where:
        - self.r_a and self.t_a are predefined reflection and transmission coefficients, respectively.
        """
        return (
            np.exp(self.sim_params.k2j * total_input_mirror_displacement)
            * (
                (self.params.r_a**2 + self.params.t_a**2) * input_electric_field
                - self.params.t_a * intracavity_electric_field
            )
            / self.params.r_a
        )

    def close(self):
        """
        Close the file handler if it exists and remove it from the logger.
        This method assures correct writing to disctinct logs.
        """
        if hasattr(self, "file_handler") and self.file_handler:
            self.file_handler.close()
            logger.removeHandler(self.file_handler)


class TestCavity(Cavity):
    def __init__(self, debug=None):
        Cavity.__init__(
            self, T_A=0.19, R_A=0.81, R_B=0.81, cavity_length=3000.0, debug=debug
        )


class ArmCavity(Cavity):
    """
    TODO: Confirm parameters reproducing PDH signal of the arm cavity.

    Finesse 450-460  # "The advanced Virgo longitudinal control system for the O2 observing run" s2.0-S0927650519301835
    """

    def __init__(self, debug=None):
        MassThickness = 0.2  # m
        SubstrateAbsorption = 0.3e-4  # 1/m; bulk absorption coef
        MirrorSubstrateAbsorption = SubstrateAbsorption * MassThickness

        t_a = 0.014
        T_a = t_a**2
        R_a = 1.0 - MirrorSubstrateAbsorption**2 - T_a

        t_b = 5e-6
        T_b = t_b**2
        R_b = 0.99325

        lambd = 1064.0e-9
        # L = np.ceil(3000.0/lambd)*lambd - 0.05*lambd
        L = 3000.0

        # Values from Andrea's thesis
        Cavity.__init__(
            self, t_a=0.01377, r_a=0.986, r_b=0.99999, cavity_length=L, debug=debug
        )


class FilterCavity(Cavity):
    """
    Finesse 9582-10204  # "Thermal detuning of a bichromatic narrow linewidth optical cavity" L.D. BONAVENA
    t_a = 0.000562
    t_b = 0.00000316  # Thermal detuning of a bichromatic narrow linewidth optical cavity L.D. BONAVENA
    r_a = np.sqrt(1. - t_a**2)-0.00016
    r_b = np.sqrt(1. - t_b**2)-0.00016
    L = 284.9  # m
    """

    def __init__(self, debug=None):
        t_a = 0.000562
        T_a = t_a**2
        t_b = 0.00000316  # Thermal detuning of a bichromatic narrow linewidth optical cavity L.D. BONAVENA
        T_b = t_b**2
        R_a = 1.0 - T_a - 0.00016
        R_b = 1.0 - T_b - 0.00016
        L = 284.9  # m

        Cavity.__init__(self, T_A=T_a, R_A=R_a, R_B=R_b, cavity_length=L, debug=debug)
