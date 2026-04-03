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
    __L__: float

    @property
    def T(self):
        return self.__L__ / c


@dataclass
class SimulationParams:
    k: Optional[float] = None
    k2j: Optional[complex] = None
    desired_f_calc: Optional[float] = None
    E_in_init: Optional[complex] = None
    f_calc: Optional[float] = None
    N: Optional[int] = None
    Theta: Optional[float] = None
    partial_Theta: Optional[bool] = None
    Theta_fraction: Optional[float] = None
    number_of_2T_chains: Optional[int] = None
    f_calc_accuracy: Optional[float] = None


class Cavity:
    simulation_initialized = False

    def __init__(self, t_a=0.001, T_a=None, r_a=0.99, R_a=None, r_b=0.999, R_b=None, L=3000.0, debug=None, log_file=None):
        self.sim_params = SimulationParams()

        if T_a is not None:
            t_a = np.sqrt(T_a)
        else:
            t_a = t_a
        
        if R_a is not None:
            r_a = np.sqrt(R_a)
        else:
            r_a = r_a

        if R_b is not None:
            r_b = np.sqrt(R_b)
        else:
            r_b = r_b

        self.params = CavityParams(t_a=t_a, r_a=r_a, r_b=r_b, __L__=L)
        #self._sync_param_aliases()
        
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
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                self.file_handler.setFormatter(formatter)
                logger.addHandler(self.file_handler)
            else:
                logging.basicConfig(level=debug)
                logger.setLevel(debug)
            
            logger.debug("Cavity initialized with parameters:")
            logger.debug("t_a: {0}".format(self.params.t_a))
            logger.debug("r_a: {0}".format(self.params.r_a))
            logger.debug("r_b: {0}".format(self.params.r_b))
            logger.debug("L: {0}".format(self.params.__L__))
            logger.debug("T: {0}".format(self.params.T))
            logger.debug("Debug level: {0}".format(debug))
    '''
    def _sync_param_aliases(self):
        self.t_a = self.params.t_a
        self.r_a = self.params.r_a
        self.r_b = self.params.r_b
        self.__L__ = self.params.__L__  # [m]
        self.T = self.params.T  # [s] half cavity round-trip time
    
    def _sync_sim_param_aliases(self):
        self.k = self.sim_params.k
        self.k2j = self.sim_params.k2j
        self.desired_f_calc = self.sim_params.desired_f_calc
        self.E_in_init = self.sim_params.E_in_init
        self.f_calc = self.sim_params.f_calc
        self.N = self.sim_params.N
        self.N_pre = self.sim_params.N_pre
        self.Theta = self.sim_params.Theta
        self.partial_Theta = self.sim_params.partial_Theta
        self.sim_params.number_of_2T_chains = self.sim_params.number_of_2T_chains
        self.f_calc_accuracy = self.sim_params.f_calc_accuracy     
    '''
    def cavity_loss(self):
        Loss = 1.0 - np.power(self.params.t_a, 2) - np.power(self.params.r_a, 2)

        if Loss < 0.0:
            print("Attenti ai valori")
        else:
            print("Loss: {0}".format(Loss))

        return Loss

    def N_eff(self):
        """
        Effective number of photon round trips in a FabryPerot cavity (Rakhmanov Eq. 1.57)
        """
        return int(np.round(1.0 / np.abs(np.log(self.params.r_a * self.params.r_b))))

    def F(self):
        """
        Coefficient of finesse
        """
        return 4.0 * self.params.r_a * self.params.r_b / np.power(1.0 - self.params.r_a * self.params.r_b, 2)

    def tau_s(self):
        """
        Formula equivalent to Eq. 2.17 (Rakhmanov)

        return 2. * T * N_eff(r_a, r_b)
        """
        return self.F() * self.params.__L__ / (np.pi * c)

    def Finesse(self):
        """
        Finesse
        """
        return np.sqrt(self.F()) * np.pi / 2.0

    def tau(self):
        """
        Cavity decay time: N_eff()*2.*__L__/c
        """
        return 2.0 * self.params.T * self.N_eff()

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
        return int(np.round(1.0/eta_2T) if eta_2T < 1. else 1)

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
        f_2T = 0.5 / self.params.T  # Half-round trip time

        N = 1  # Initlized to one in the constructor.
        n_of_subhistories = 1  # Initlized to one in the constructor.
        partial_Theta = False  # Initlized to False in the constructor.
        Theta_fraction = 1.0  # Initlized to 1.0 in the constructor.

        _N_eff_factor = 2

        eta_2T = f_2T/(f_calc_desired)

        if f_calc_desired > f_2T:
            n_of_subhistories = self.number_of_subhistories(eta_2T)
            f_calc = f_2T * n_of_subhistories
        else:
            N_max = _N_eff_factor * self.N_eff()
            if f_calc_desired < (f_2T / N_max):
                N = N_max
                f_calc = f_calc_desired  # In this case desired f_calc is exactly simulated.
                partial_Theta = True
                Theta_fraction = 1.0 - (N * f_calc * 2.0 * self.params.T)
            else:
                N = self.round_for_inverse_curve(eta_2T)[0]
                f_calc = f_2T/N  # in Hz

        Theta = 1.0 / f_calc  # in Seconds

        logger.debug("N: {0}".format(N))
        logger.debug("Number of subhistories: {0}".format(n_of_subhistories))
        logger.debug("Theta: {0}".format(Theta))
        logger.debug("Final f_calc: {0}".format(f_calc))
                
        return f_calc, N, Theta, partial_Theta, Theta_fraction, n_of_subhistories

    def simulation(self, k, desired_f_calc, E_in_init, backend="auto"):
        '''
        With respect to version 2.0.0, the simulation works with incident electric field instead of optical power.
        k: wave number
        f_calc: calculation frequency
        E_in_init: initial electric field amplitude
        backend: "pure" | "numba" | "auto"
        '''
        logger.debug("Simulation started")
        logger.debug("k: {0}".format(k))
        logger.debug("Desired f_calc: {0}".format(desired_f_calc))
        logger.debug("E_in_init: {0}".format(E_in_init))


        # Initial values
        k2j = -2.0j * k  # Used frequently in step()

        # Estimate f_calc, N, N_pre, Theta, partial_Theta, number_of_2T_chains
        f_calc, N, Theta, partial_Theta, Theta_fraction, number_of_2T_chains = self.resolve_sampling_frequency(desired_f_calc)

        f_calc_accuracy = 1. - np.abs(f_calc - desired_f_calc) / desired_f_calc

        self.sim_params = SimulationParams(
            k=k,
            k2j=k2j,
            desired_f_calc=desired_f_calc,
            E_in_init=E_in_init,
            f_calc=f_calc,
            N=N,
            Theta=Theta,
            partial_Theta=partial_Theta,
            Theta_fraction=Theta_fraction,
            number_of_2T_chains=number_of_2T_chains,
            f_calc_accuracy=f_calc_accuracy,
        )

        #self._sync_sim_param_aliases()

        # Arrays initialization
        self.n = np.arange(0, self.sim_params.N + 1, 1)
        self.rarbn = np.power(self.params.r_a * self.params.r_b, self.n)

        self.e2iknL = np.exp(
            -2.0j * self.sim_params.k * (self.n) * self.params.__L__
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
            #ax1.plot(self.n, np.abs(self.e2iknL), label="|$\exp(-2iknL)$|", color="green")  # Always 1
            #ax1.plot(self.n, np.abs(self.rarbne2iknL), label="|$(r_a r_b)^n \cdot \exp(-2iknL)$|", color="red")  # Equal to self.rarbn

            # Create a secondary y-axis for the phase
            ax2 = ax1.twinx()
            ax2.plot(self.n, np.unwrap(np.angle(self.e2iknL)), label=r"Phase of $\exp(-2iknL)$", color="orange", linestyle="--", marker='o')
            ax2.set_ylabel("Phase [rad]", color="black")
            ax2.tick_params(axis="y", labelcolor="black")

            # Add legends
            fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
            plt.title(r"Decay and Phase of $(r_a r_b)^n$, $\exp(-2iknL)$, and $(r_a r_b)^n \cdot \exp(-2iknL)$")
            plt.show()

        self.frac,_ = np.modf(self.params.__L__*self.sim_params.k/(2.*np.pi))
        self.phi = 2.*np.pi*self.frac
        logger.debug("phi: {0}".format(self.phi))

        self.airy_phi = self.E_adiabatic(E_in_init, self.phi)
        logger.debug("E_adiabatic cplx: {0}".format(self.airy_phi))
        logger.debug("E_adiabatic abs : {0}".format(np.abs(self.airy_phi)))
        logger.debug("E_adiabatic angl: {0}".format(np.angle(self.airy_phi)))

        self.E_last = self.airy_phi*np.ones(self.sim_params.number_of_2T_chains, dtype=np.complex128)*np.exp(1.j*np.angle(self.sim_params.E_in_init))

        # Define a list of deque buffers for the electric field
        self.E_in_buffers = [self.sim_params.E_in_init*np.ones(self.sim_params.N, dtype=np.complex128) for _ in range(self.sim_params.number_of_2T_chains)]
        logger.debug("E_last cplx: {0}".format(self.E_last))
        logger.debug("E_last abs : {0}".format(np.abs(self.E_last)))
        logger.debug("E_last angl: {0}".format(np.angle(self.E_last)))

        self.Ze = np.zeros(self.sim_params.N + 2, dtype=np.float64)
        self.Z_last = np.zeros(self.sim_params.number_of_2T_chains, dtype=np.float64)
        self.d_zeta_last = np.zeros(self.sim_params.number_of_2T_chains, dtype=np.float64)
        self.Ze_in = 0.

        self.E_in = np.zeros(self.sim_params.N, dtype=np.complex128)

        self.__sim_step_counter__ = 0

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
            raise ValueError(f"Unknown backend {backend!r}. "
                                "Use 'pure', 'numba' or 'auto'.")

        self.simulation_initialized = True
    
    def __sim_step__(self, d_zeta=0., E_in_curr=1.):
        '''
        With respect to version 2.0.0, the simulation works with incident electric field instead of optical power.

        d_zeta: displacement of the output mirror
        E_in_curr: current electric input field
        '''
        
        if self.simulation_initialized == False:
            print("Initialize first")
            return

        chain_idx = self.__sim_step_counter__ % self.sim_params.number_of_2T_chains
        #logger.debug("Chain idx: {0}".format(chain_idx))

        # Update the displacement of the output mirror
        self.d_zeta_last[chain_idx] = d_zeta

        self.Ze, self.E_in_buffers[chain_idx], E, self.Z_last[chain_idx] = heavy(d_zeta, E_in_curr, self.d_zeta_last, self.Z_last[chain_idx], self.sim_params.partial_Theta, self.sim_params.Theta_fraction, self.sim_params.N, self.Ze, self.E_in_buffers[chain_idx], self.rarbne2iknL, self.sim_params.k2j, self.params.t_a, self.E_last[chain_idx])

        #if not self.partial_Theta:
        self.E_last[chain_idx] = E

        #logger.debug("E_last: {0}".format(self.E_last))

        self.__sim_step_counter__ += 1  # Be carefull with the overflow!!!

        return E
        
    def sim_step(self, E_in_laser=1., d_zeta_in=0., d_zeta=0.):
        '''
        Simulate the electric field propagation through a two-mirror cavity.
        This method calculates the electric field after propagating through a 
        two-mirror cavity with given initial electric field and mirror displacements.

        Parameters:
        -----------
        E_in_laser : float, optional
            The initial electric field amplitude emitted by the laser and referred to the external reference frame (default is 1.0).
        d_zeta_1 : float, optional
            The displacement of the input mirror (default is 0.0).
        d_zeta_2 : float, optional
            The displacement of the back mirror (default is 0.0).

        Returns:
        --------
        tuple:
            - E : complex
            The electric field inside the cavity after propagation.
            - E_ref_val : complex
            The reflected electric field from the cavity.

        Notes:
        ------
        - `self.k` is the wave number.
        - `self.Ze_in` is the sum of previous mirror displacements.
        '''

        # Total cavity length
        d_zeta_tot = d_zeta - d_zeta_in

        # Position of the input mirror
        self.Ze_in += d_zeta_in

        # Electric field on the input mirror
        E_in_laser = E_in_laser * np.exp(self.sim_params.k2j*self.Ze_in)

        E = self.__sim_step__(d_zeta=d_zeta_tot, E_in_curr=E_in_laser)

        E_ref_val = self.E_ref(E=E, E_in_laser=E_in_laser, Ze_in=self.Ze_in)

        return E, E_ref_val
    
    def sim_reset(self):
        self.E_last = self.airy_phi*np.ones(self.sim_params.number_of_2T_chains, dtype=np.complex128)*np.exp(1.j*np.angle(self.sim_params.E_in_init))
        self.E_in_buffers = [self.sim_params.E_in_init*np.ones(self.sim_params.N, dtype=np.complex128) for _ in range(self.sim_params.number_of_2T_chains)]
        self.Z_last = np.zeros(self.sim_params.number_of_2T_chains)
        self.Ze = np.zeros(self.sim_params.N + 2)
        self.Ze_in = 0.
        self.d_zeta_last = np.zeros(self.sim_params.number_of_2T_chains)
        self.__sim_step_counter__ = 0
        
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
        return 1. / (1. + self.F() * np.sin(phi)**2)
    
    def E_adiabatic(self, E_in, phi):
        r"""
        Calculate the adiabatic electric field inside the cavity based on the input electric field.

        This method uses the formula from Rakhmanov Eq. 1.72 to compute the adiabatic 
        electric field.

        Parameters:
        -----------
        E_in : complex
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
        '''
         (Rakhmanov Eq. 1.72)
        '''
        return self.params.t_a*np.abs(E_in)/(1.-self.params.r_a*self.params.r_b*np.exp(-2.j*phi))
    
    def xml_save(self, filename):
        '''
        The method saves the following parameters of the Cavity object if they exist: r_a, r_b, t_a, __L__, T.

        Parameters:
        --------
        filename (str): The name of the file to save the XML data. If the filename does not end with ".xml", it will be appended automatically.

        Example:
        --------
        cavity = Cavity(r_a = 1.0, r_b = 2.0, t_a = 3.0, L = 4.0)
        cavity.xml_save("cavity_parameters.xml")

        This will create an XML file named "cavity_parameters.xml" with the parameters of the Cavity object.
        '''
        root = ET.Element("Cavity")

        # Parameters to save from the CavityParams dataclass
        params_to_save = ["t_a", "r_a", "r_b", "__L__"]

        # Add parameters as sub-elements
        for param_name in params_to_save:
            if hasattr(self.params, param_name):
                param_value = getattr(self.params, param_name)
                param_element = ET.SubElement(root, param_name)
                param_element.text = str(param_value)

        # Also save the computed T property
        param_element = ET.SubElement(root, "T")
        param_element.text = str(self.params.T)

        # Create the tree and write to an XML file
        tree = ET.ElementTree(root)
        try:
            if not filename.endswith(".xml"):
                filename += ".xml"
            tree.write(filename)
        except Exception as e:
            logger.error(f"Error writing the XML file: {e}")

    def xml_load(self, filename=None):
        '''
        Load cavity parameters from an XML file.

        This method supports two call styles:

        1) Instance style (updates the existing object):
           cavity = Cavity()
           cavity.xml_load('path/to/parameters.xml')

        2) Class style (creates and returns a loaded object):
           cavity = Cavity.xml_load('path/to/parameters.xml')
        '''
        if isinstance(self, Cavity):
            cavity_obj = self
            if filename is None:
                raise TypeError("Cavity.xml_load() missing 1 required positional argument: 'filename'")
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
        cavity_obj.__init__(t_a=params['t_a'], r_a=params['r_a'], r_b=params['r_b'], L=params['__L__'], debug=cavity_obj.debug)
        return cavity_obj
    
    def E_ref(self, E, E_in_laser=1., Ze_in=0.):
        """
        TODO: verify what is the phase parameter
        Calculate the reflected electric field (E_ref) based on the input electric field (E) and the input laser electric field (E_in_laser).

        Parameters:
        E (float): The input electric field.
        E_in_laser (float, optional): The input laser electric field. Default is 1.

        Returns:
        float: The reflected electric field (E_ref).

        Notes:
        This function uses the formula from Eq 1.48:

        where:
        - self.r_a and self.t_a are predefined reflection and transmission coefficients, respectively.
        """
        return np.exp(self.sim_params.k2j*Ze_in) * ((self.params.r_a**2 + self.params.t_a**2) * E_in_laser - self.params.t_a * E) / self.params.r_a

    def close(self):
        '''
        Close the file handler if it exists and remove it from the logger.
        This method assures correct writing to disctinct logs.
        '''
        if hasattr(self, 'file_handler') and self.file_handler:
            self.file_handler.close()
            logger.removeHandler(self.file_handler)


class TestCavity(Cavity):
    def __init__(self, debug=None):
        Cavity.__init__(self, T_a=0.19, R_a=0.81, R_b=0.81, L=3000.0, debug=debug)


class ArmCavity(Cavity):
    '''
    TODO: Confirm parameters reproducing PDH signal of the arm cavity.

    Finesse 450-460  # "The advanced Virgo longitudinal control system for the O2 observing run" s2.0-S0927650519301835
    '''
    def __init__(self, debug=None):
        MassThickness = 0.2                   # m
        SubstrateAbsorption = 0.3e-4          # 1/m; bulk absorption coef
        MirrorSubstrateAbsorption = SubstrateAbsorption * MassThickness

        t_a = 0.014
        T_a = t_a**2
        R_a = 1. - MirrorSubstrateAbsorption**2 - T_a

        t_b = 5e-6
        T_b = t_b**2
        R_b = 0.99325

        lambd = 1064.e-9
        #L = np.ceil(3000.0/lambd)*lambd - 0.05*lambd
        L = 3000.0

        # Values from Andrea's thesis
        Cavity.__init__(self, t_a=0.01377, r_a=0.986, r_b=0.99999, L=L, debug=debug)


class FilterCavity(Cavity):
    '''
    Finesse 9582-10204  # "Thermal detuning of a bichromatic narrow linewidth optical cavity" L.D. BONAVENA
    t_a = 0.000562
    t_b = 0.00000316  # Thermal detuning of a bichromatic narrow linewidth optical cavity L.D. BONAVENA
    r_a = np.sqrt(1. - t_a**2)-0.00016
    r_b = np.sqrt(1. - t_b**2)-0.00016
    L = 284.9  # m
    '''
    def __init__(self, debug=None):
        t_a = 0.000562
        T_a = t_a**2
        t_b = 0.00000316  # Thermal detuning of a bichromatic narrow linewidth optical cavity L.D. BONAVENA
        T_b = t_b**2
        R_a = 1. - T_a - 0.00016
        R_b = 1. - T_b - 0.00016
        L = 284.9  # m

        Cavity.__init__(self, T_a=T_a, R_a=R_a, R_b=R_b, L=L, debug=debug)