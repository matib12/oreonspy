#!/usr/bin/env python3
# coding: utf-8

# ### Calculates electric field amplitude inside an optical resonator cavity in dynamic case.
#
# - Andrea Svizzeretto, <andrea.svizzeretto@studenti.unipg.it>
# - Mateusz Bawaj, <mateusz.bawaj@unipg.it>

__authors__ = ["Andrea Svizzeretto", "Mateusz Bawaj"]
__contact__ = "mateusz.bawaj@unipg.it"
__credits__ = ["Andrea Svizzeretto", "Mateusz Bawaj"]
__date__ = "2024/04/19"
__deprecated__ = False
__email__ =  "mateusz.bawaj@unipg.it"
__license__ = "GPLv3"
__maintainer__ = "developer"
__status__ = "Production"
__version__ = '2.1.2'


from scipy import constants as const
import numpy as np
from matplotlib import pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class Cavity:
    simulation_initialized = False

    def __init__(self, t_a=0.001, T_a=None, r_a=0.99, R_a=None, r_b=0.999, R_b=None, L=3000.0, debug=""):
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
        if debug == "":
            logger.disabled = True
            pass
        else:
            logger.disabled = False
            logger.setLevel(debug)
            

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

    def simulation(self, k, f_calc, P_in_init=1.):
        E_in_init = np.sqrt(P_in_init)

        # Useful constants
        _2T = 2.0 * self.T
        _N_eff_factor = 2

        # Algorithm parameters
        N_epsilon = 0.1

        # Initial values
        self.k = k
        self.k2j = -2.0j * k  # Used frequently in step()
        self.number_of_2T_chains = 1
        self.N = 1

        self.N_pre = 1.0 / (f_calc * _2T)
        logger.debug("N_pre: {0}".format(self.N_pre))

        self.partial_Theta = False

        if self.N_pre < 1.0 - N_epsilon:
            logger.info("2T x times bigger then Theta. (x is integer)")
            self.number_of_2T_chains = int(np.ceil(1.0 / self.N_pre))

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
            if self.N_pre > N_max:
                logger.info("N times Cavity decay time shorter than the sampling period")
                self.N = N_max
                self.f_calc = f_calc
                self.Theta = 1.0 / f_calc

                self.partial_Theta = True
            else:
                logger.info("")
                self.N = int(np.round(self.N_pre))
                self.Theta = _2T * self.N
                self.f_calc = 1.0 / self.Theta
                logger.warning(
                    "Warning: approximated f_calc to: {0:.2f}".format(self.f_calc)
                )

        logger.debug("N: {0}".format(self.N))
        logger.debug("Number of chains: {0}".format(self.number_of_2T_chains))

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

        self.airy_phi = self.E_adiabatic(E_in_init, phi=self.phi)

        self.E_last = self.airy_phi*np.ones(self.number_of_2T_chains, dtype=np.complex128)

        self.Ze = np.zeros(self.N + 1)
        self.Z_last = np.zeros(self.number_of_2T_chains)
        self.d_zeta_last = np.zeros(self.number_of_2T_chains)

        self.__sim_step_counter__ = 0

        self.simulation_initialized = True

    def sim_step(self, d_zeta, P_in_curr):
        E_in_curr = np.sqrt(P_in_curr)

        if self.simulation_initialized == False:
            print("Initialize first")
            return

        Sum = 0.0

        chain_idx = self.__sim_step_counter__ % self.number_of_2T_chains
        logger.debug("Chain idx: {0}".format(chain_idx))

        self.d_zeta_last[chain_idx] = d_zeta

        Z = np.sum(self.d_zeta_last) + self.Z_last[chain_idx]

        logger.debug("Z_last: {0}".format(self.Z_last))

        Z_start = self.Z_last[chain_idx]
        if self.partial_Theta:
            Z_start += np.interp(self.N_pre-self.N , [0, self.N_pre], [0, d_zeta])
            logger.debug("Z_start: {0}".format(Z_start))

        self.Ze[1:] = np.linspace(Z_start, Z, self.N)
        # logger.debug(self.Ze)
        self.Ze = np.add.accumulate(self.Ze)
        logger.debug("Ze: {0}".format(self.Ze))

        for idx in np.arange(0, self.N, 1):
            # print("index: {0}".format(idx))
            Sum = Sum + self.rarbne2iknL[idx] * np.exp(
                self.k2j * self.Ze[idx]
            ) * E_in_curr

        res = (
            self.t_a * Sum
            + self.rarbne2iknL[self.N]
            * np.exp(self.k2j * self.Ze[self.N])
            * self.E_last[chain_idx]
        )

        #if not self.partial_Theta:
        self.E_last[chain_idx] = res

        logger.debug("E_last: {0}".format(self.E_last))
        
        self.Z_last[chain_idx] = Z

        self.__sim_step_counter__ += 1  # Be carefull with the overflow!!!

        return res
    
    def sim_reset(self):
        self.E_last = self.airy_phi*np.ones(self.number_of_2T_chains, dtype=np.complex128)
        self.Z_last = np.zeros(self.number_of_2T_chains)
        self.Ze = np.zeros(self.N + 1)
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
        '''
         (Rakhmanov Eq. 1.72)
        '''
        return self.t_a*E_in/(1.-self.r_a*self.r_b*np.exp(-2.j*phi))


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

        Cavity.__init__(self, T_a=T_a, R_a=R_a, R_b=R_b, L=L, debug=debug)


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