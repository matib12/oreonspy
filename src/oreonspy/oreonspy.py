#!/usr/bin/env python3
# coding: utf-8

# ### Calculates electric field amplitude inside an optical resonator cavity in dynamic case.
#
# - Andrea Svizzeretto, <andrea.svizzeretto@studenti.unipg.it>
# - Mateusz Bawaj, <mateusz.bawaj@unipg.it>

from scipy import constants as const
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
import xml.etree.ElementTree as ET

import logging

logger = logging.getLogger(__name__.split(".")[-1])
logger.setLevel(logging.INFO)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

class Cavity:
    simulation_initialized = False

    def __init__(self, t_a=0.001, T_a=None, r_a=0.99, R_a=None, r_b=0.999, R_b=None, L=3000.0, debug=None, log_file=None):
        if T_a is not None:
            self.t_a = np.sqrt(T_a)
        else:
            self.t_a = t_a
        
        if R_a is not None:
            self.r_a = np.sqrt(R_a)
        else:
            self.r_a = r_a

        if R_b is not None:
            self.r_b = np.sqrt(R_b)
        else:
            self.r_b = r_b
        
        self.__L__ = L  # [m]
        self.T = L / const.c  # [s] half cavity round-trip time

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
            logger.debug("t_a: {0}".format(self.t_a))
            logger.debug("r_a: {0}".format(self.r_a))
            logger.debug("r_b: {0}".format(self.r_b))
            logger.debug("L: {0}".format(self.__L__))
            logger.debug("T: {0}".format(self.T))
            logger.debug("Debug level: {0}".format(debug))
            

    def cavity_loss(self):
        Loss = 1.0 - np.power(self.t_a, 2) - np.power(self.r_a, 2)

        if Loss < 0.0:
            print("Attenti ai valori")
        else:
            print("Loss: {0}".format(Loss))

        return Loss

    def N_eff(self):
        """
        Effective number of photon round trips in a FabryPerot cavity (Rakhmanov Eq. 1.57)
        """
        return int(np.round(1.0 / np.abs(np.log(self.r_a * self.r_b))))

    def F(self):
        """
        Coefficient of finesse
        """
        return 4.0 * self.r_a * self.r_b / np.power(1.0 - self.r_a * self.r_b, 2)

    def tau_s(self):
        """
        Formula equivalent to Eq. 2.17 (Rakhmanov)

        return 2. * T * N_eff(r_a, r_b)
        """
        return self.F() * self.__L__ / (np.pi * const.c)

    def Finesse(self):
        """
        Finesse
        """
        return np.sqrt(self.F()) * np.pi / 2.0

    def tau(self):
        return 2.0 * self.T * self.N_eff()

    def gain(self):
        """
        Amplitude gain of cavity Eq. 1.67 (Rakhmanov)
        """
        return self.t_a / (1.0 - self.r_a * self.r_b)

    def print_params(self):
        print("Coefficient of finesse: {0:.2f}".format(self.F()))
        print("Half round-trip time: {0:.2e} [s]".format(self.T))
        print("Effective number of photon round trip: {0:d}".format(self.N_eff()))
        print("Tau_s: {0:.2e} [s]".format(self.tau_s()))
        print("Finesse: {0:.2f}".format(self.Finesse()))
        print("Gain: {0:.2f}".format(self.gain()))

    def simulation(self, k, f_calc, E_in_init):
        '''
        With respect to version 2.0.0, the simulation works with incident electric field instead of optical power.
        k: wave number
        f_calc: calculation frequency
        E_in_init: initial electric field amplitude
        '''
        logger.debug("Simulation started")
        logger.debug("k: {0}".format(k))
        logger.debug("Required f_calc: {0}".format(f_calc))
        logger.debug("E_in_init: {0}".format(E_in_init))
        # Useful constants
        _2T = 2.0 * self.T  # Round trip time
        _N_eff_factor = 2   # Multiplier for the Effective number of photon round trips in a cavity

        # Algorithm parameters
        N_epsilon = 0.25     # Epsilon for the number of chains estimation

        # Initial values
        self.k = k
        self.k2j = -2.0j * k  # Used frequently in step()
        self.number_of_2T_chains = 1
        self.N = 1
        self.desired_f_calc = f_calc
        self.f_calc_accuracy = 1.

        self.E_in_init = E_in_init

        self.N_pre = 1.0 / (f_calc * _2T)  # N_pre is the number of round trips in the cavity during the calculation time
        logger.debug("N_pre: {0}".format(self.N_pre))

        self.partial_Theta = False

        # Number of chains must be integer
        if self.N_pre < 1.0 - N_epsilon:
            logger.info("2T x times bigger then Theta. (x is integer)")
            self.number_of_2T_chains = int(np.round(1.0 / self.N_pre))

            self.f_calc = self.number_of_2T_chains / _2T
            self.Theta = 1.0 / f_calc
            logger.warning(
                "Warning: approximated f_calc to: {0:.2f}".format(self.f_calc)
            )
            logger.warning("Number of chains: {0}".format(self.number_of_2T_chains))

        elif self.N_pre < 1.0 + N_epsilon:
            logger.info("2T comparable with Theta so N becomes 1")
            self.f_calc = 1.0 / _2T
            self.Theta = _2T
            logger.warning(
                "Warning: approximated f_calc to: {0:.2f}".format(self.f_calc)
            )

        else:
            N_max = _N_eff_factor * self.N_eff()
            logger.debug("N_max: {0}".format(N_max))
            if self.N_pre > N_max:
                logger.info("N times Cavity decay time shorter than the sampling period")
                self.N = N_max
                self.f_calc = f_calc
                self.Theta = 1.0 / f_calc

                self.partial_Theta = True
            else:
                logger.info("N times Cavity decay time longer than the sampling period")
                self.N = int(np.round(self.N_pre))
                self.Theta = _2T * self.N
                self.f_calc = 1.0 / self.Theta
                logger.warning(
                    "Warning: approximated f_calc to: {0:.2f}".format(self.f_calc)
                )

        self.f_calc_accuracy = 1. - np.abs(self.f_calc - self.desired_f_calc) / self.desired_f_calc
        
        logger.debug("N: {0}".format(self.N))
        logger.debug("Number of chains: {0}".format(self.number_of_2T_chains))
        logger.debug("Theta: {0}".format(self.Theta))
        logger.debug("Final f_calc: {0}".format(self.f_calc))
        logger.debug("f_calc accuracy: {0:.2f}%".format(100*self.f_calc_accuracy))

        # Arrays initialization
        self.n = np.arange(0, self.N + 1, 1)
        self.rarbn = np.power(self.r_a * self.r_b, self.n)

        self.e2iknL = np.exp(
            -2.0j * self.k * (self.n) * self.__L__
        )  # Convert to the case when: L is multiple of lambd

        self.rarbne2iknL = self.rarbn * self.e2iknL

        logger.debug("n: {0}".format(self.n))
        logger.debug("rarbn: {0}".format(self.rarbn))
        logger.debug("e2iknL: {0}".format(self.e2iknL))
        logger.debug("rarbne2iknL: {0}".format(self.rarbne2iknL))

        self.frac,_ = np.modf(self.__L__*k/(2.*np.pi))
        self.phi = 2.*np.pi*self.frac
        logger.debug("phi: {0}".format(self.phi))

        self.airy_phi = self.E_adiabatic(E_in_init, self.phi)
        logger.debug("E_adiabatic cplx: {0}".format(self.airy_phi))
        logger.debug("E_adiabatic abs : {0}".format(np.abs(self.airy_phi)))
        logger.debug("E_adiabatic angl: {0}".format(np.angle(self.airy_phi)))

        self.E_last = self.airy_phi*np.ones(self.number_of_2T_chains, dtype=np.complex128)*np.exp(1.j*np.angle(self.E_in_init))
        logger.debug("E_last cplx: {0}".format(self.E_last))
        logger.debug("E_last abs : {0}".format(np.abs(self.E_last)))
        logger.debug("E_last angl: {0}".format(np.angle(self.E_last)))

        # Define a list of deque buffers for the electric field
        self.E_in_buffers = [deque(E_in_init*np.ones(self.N, dtype=np.complex128), maxlen=self.N) for _ in range(self.number_of_2T_chains)]

        self.Ze = np.zeros(self.N + 1)
        self.Z_last = np.zeros(self.number_of_2T_chains)
        self.d_zeta_last = np.zeros(self.number_of_2T_chains)
        self.Ze_in = 0.

        self.E_in = np.zeros(self.N, dtype=np.complex128)

        self.__sim_step_counter__ = 0

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

        Sum = 0.0

        chain_idx = self.__sim_step_counter__ % self.number_of_2T_chains
        #logger.debug("Chain idx: {0}".format(chain_idx))

        # Update the displacement of the output mirror
        self.d_zeta_last[chain_idx] = d_zeta

        Z = np.sum(self.d_zeta_last) + self.Z_last[chain_idx]

        #logger.debug("Z_last: {0}".format(self.Z_last))

        Z_start = self.Z_last[chain_idx]
        if self.partial_Theta:
            Z_start += np.interp(self.N_pre-self.N , [0, self.N_pre], [0, d_zeta])
            #logger.debug("Z_start: {0}".format(Z_start))

        self.Ze[1:] = np.linspace(Z, Z_start, self.N, endpoint=True)
        self.Ze = np.add.accumulate(self.Ze)

        #self.Ze[1:] = np.linspace(Z_start, Z, self.N, endpoint=True)[::-1]
        #logger.debug(self.Ze)

        #self.Ze = np.add.accumulate(self.Ze)
        #logger.debug("Ze: {0}".format(self.Ze))

        # Update input electric field buffer
        self.E_in_buffers[chain_idx].appendleft(E_in_curr)

        # Calculate the sum
        for idx in range(self.N):
            # print("index: {0}".format(idx))
            Sum = Sum + self.rarbne2iknL[idx] * np.exp(
                self.k2j * self.Ze[idx]
            ) * self.E_in_buffers[chain_idx][idx]

        E = (
            self.t_a * Sum
            + self.rarbne2iknL[self.N]
            * np.exp(self.k2j * self.Ze[self.N])
            * self.E_last[chain_idx]
        )

        #if not self.partial_Theta:
        self.E_last[chain_idx] = E

        #logger.debug("E_last: {0}".format(self.E_last))
        
        self.Z_last[chain_idx] = Z

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
        E_in_laser = E_in_laser * np.exp(self.k2j*self.Ze_in)

        E = self.__sim_step__(d_zeta=d_zeta_tot, E_in_curr=E_in_laser)

        E_ref_val = self.E_ref(E=E, E_in_laser=E_in_laser, Ze_in=self.Ze_in)

        return E, E_ref_val
    
    def sim_reset(self):
        self.E_last = self.airy_phi*np.ones(self.number_of_2T_chains, dtype=np.complex128)*np.exp(1.j*np.angle(self.E_in_init))
        self.Z_last = np.zeros(self.number_of_2T_chains)
        self.Ze = np.zeros(self.N + 1)
        self.Ze_in = 0.
        self.d_zeta_last = np.zeros(self.number_of_2T_chains)
        self.__sim_step_counter__ = 0
        
    def print_sim_params(self):
        print("Theta: {0:.2e} [s]".format(self.Theta))
        print("Cavity RT: {0:.2e} [s]".format(2.0 * self.T))
        print("Calculation frequency: {0:.2e} [Hz]".format(self.f_calc))
        print("N_eff: {0:.2e}".format(self.N_eff()))

        print("N: {0}".format(self.N))

        print("Number of 2T chains: {0}".format(self.number_of_2T_chains))
        print("Partial Theta: {0}".format(self.partial_Theta))

    def plot_sim_factors(self):
        plt.plot(self.rarbn, label="$(r_a r_b)^n$")
        plt.title("Decaying $n$ powered $r_a r_b$ product")
        plt.xlabel("n")
        plt.legend()

        plt.plot(np.abs(self.e2iknL), label="Mag")
        plt.plot(np.angle(self.e2iknL), label="Phase")
        plt.title("$\exp(-2ik(n-1)L)$")
        plt.xlabel("n")
        plt.legend()

    def Airy(self, phi):
        return 1. / (1. + self.F() * np.sin(phi)**2)
    
    def E_adiabatic(self, E_in, phi):
        """
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
        return self.t_a*np.abs(E_in)/(1.-self.r_a*self.r_b*np.exp(-2.j*phi))
    
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

        # Parameters to save
        params = ["r_a", "r_b", "t_a", "__L__", "T"]

        # Add parameters as sub-elements
        cavity = {}
        for par in params:
            if hasattr(self, par):
                cavity[par] = getattr(self, par)

        for key, value in cavity.items():
            param = ET.SubElement(root, key)
            param.text = str(value)

        # Create the tree and write to an XML file
        tree = ET.ElementTree(root)
        try:
            if not filename.endswith(".xml"):
                filename += ".xml"
            tree.write(filename)
        except Exception as e:
            logger.error(f"Error writing the XML file: {e}")

    def xml_load(self, filename):
        '''
        Load the parameters of the Cavity object from an XML file.

        Parameters:
        --------
        filename (str): The path to the XML file containing the parameters.

        Example:
        --------
        cavity = Cavity()
        cavity.xml_load('path/to/parameters.xml')

        This will create a Cavity classobject named "cavity" with the parameters from the XML file "parameters.xml".
        '''
        # Load the XML file
        try:
            tree = ET.parse(filename)
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
        self.__init__(t_a=params['t_a'], r_a=params['r_a'], r_b=params['r_b'], L=params['__L__'], debug=self.debug)
    
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
        return np.exp(self.k2j*Ze_in) * ((self.r_a**2 + self.t_a**2) * E_in_laser - self.t_a * E) / self.r_a

    def close(self):
        '''
        Close the file handler if it exists and remove it from the logger.
        This method assures correct writing to disctinct logs.
        '''
        if hasattr(self, 'file_handler') and self.file_handler:
            self.file_handler.close()
            logger.removeHandler(self.file_handler)

class TestCavity(Cavity):
    def __init__(self, debug=""):
        Cavity.__init__(self, T_a=0.19, R_a=0.81, R_b=0.81, L=3000.0, debug=debug)


class ArmCavity(Cavity):
    '''
    TODO: Confirm parameters reproducing PDH signal of the arm cavity.

    Finesse 450-460  # "The advanced Virgo longitudinal control system for the O2 observing run" s2.0-S0927650519301835
    '''
    def __init__(self, debug=""):
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
    def __init__(self, debug=""):
        t_a = 0.000562
        T_a = t_a**2
        t_b = 0.00000316  # Thermal detuning of a bichromatic narrow linewidth optical cavity L.D. BONAVENA
        T_b = t_b**2
        R_a = 1. - T_a - 0.00016
        R_b = 1. - T_b - 0.00016
        L = 284.9  # m

        Cavity.__init__(self, T_a=T_a, R_a=R_a, R_b=R_b, L=L, debug=debug)