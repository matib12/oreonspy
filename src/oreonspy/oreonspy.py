#!/usr/bin/env python3
# coding: utf-8

# ### Calculates electric field amplitude inside an optical resonator cavity in dynamic case.
#
# - Andrea Svizzeretto, <andrea.svizzeretto@studenti.unipg.it>
# - Mateusz Bawaj, <mateusz.bawaj@unipg.it>

from scipy import constants as const
import numpy as np
from matplotlib import pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


class Cavity:
    simulation_initialized = False

    def __init__(self, t_a=0.001, r_a=0.99, r_b=0.999, L=3000.0, debug=""):
        self.t_a = t_a
        self.r_a = r_a
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

    def simulation(self, k, f_calc, E_last=0.0j):
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
        self.Z = 0.0
        # self.dt = 0.

        N_pre = 1.0 / (f_calc * _2T)
        logger.debug("N_pre: {0}".format(N_pre))

        if N_pre < 1.0 - N_epsilon:
            logger.info("2T x times bigger then Theta. (x is integer)")
            self.number_of_2T_chains = int(np.ceil(1.0 / N_pre))

            self.f_calc = self.number_of_2T_chains / _2T
            self.Theta = 1.0 / f_calc
            logger.warning(
                "Warning: approximated f_calc to: {0:.2f}".format(self.f_calc)
            )
            logger.warning("Number of chains: {0}".format(self.number_of_2T_chains))

        elif N_pre < 1.0 + N_epsilon:
            logger.info("2T comparable with Theta so N becomes 1")
            self.f_calc = 1.0 / _2T
            self.Theta = _2T
            logger.warning(
                "Warning: approximated f_calc to: {0:.2f}".format(self.f_calc)
            )

        else:
            N_max = _N_eff_factor * self.N_eff()
            if N_pre > N_max:
                logger.info("N times Cavity decay time shorter than sampling period")
                self.N = N_max
                self.f_calc = f_calc
                self.Theta = 1.0 / f_calc
            else:
                logger.info("")
                self.N = int(np.round(N_pre))
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

        logger.debug("n: {0}".format(self.n))
        logger.debug("rarbn: {0}".format(self.rarbn))
        logger.debug("e2iknL: {0}".format(self.e2iknL))

        #self.E_in = np.zeros(self.number_of_2T_chains, dtype=np.complex128)
        self.E_last = np.zeros(self.number_of_2T_chains, dtype=np.complex128)  # last term of Eq. 1.55 E(t - 2NT)
        # self.E_last[0] = E_last
        # self.E_last[1] = E_last * self.rarbn[-1]
        self.Ze = np.zeros(self.N + 1)

        self.Z_last = np.zeros(self.number_of_2T_chains)

        # Smooth raising of the input electric field.
        # Vector of factors in the range [0; 1] for the first 2T chains.
        E_in_init = np.linspace(-np.pi/2., np.pi/2., self.number_of_2T_chains)
        self.E_in_init = (np.sin(E_in_init)+1.)/2.

        self.__sim_step_counter__ = 0

        self.simulation_initialized = True

    def sim_step(self, d_zeta, E_in_curr):
        if self.simulation_initialized == False:
            print("Initialize first")
            return

        Sum = 0.0

        chain_idx = self.__sim_step_counter__ % self.number_of_2T_chains
        logger.debug("Chain idx: {0}".format(chain_idx))

        self.Z += d_zeta

        if self.__sim_step_counter__ < self.number_of_2T_chains:
            E_in_curr *= self.E_in_init[self.__sim_step_counter__]

        logger.debug("Z_last: {0}".format(self.Z_last))
        self.Ze[1:] = np.linspace(self.Z_last[chain_idx], self.Z, self.N)
        # logger.debug(self.Ze)
        self.Ze = np.add.accumulate(self.Ze)
        logger.debug("Ze: {0}".format(self.Ze))

        for idx in np.arange(0, self.N, 1):
            # print("index: {0}".format(idx))
            Sum = Sum + self.rarbn[idx] * self.e2iknL[idx] * np.exp(
                self.k2j * self.Ze[idx]
            ) * E_in_curr

        res = (
            self.t_a * Sum
            + self.rarbn[self.N]
            * self.e2iknL[self.N]
            * np.exp(self.k2j * self.Ze[self.N])
            * self.E_last[chain_idx]
        )

        self.E_last[chain_idx] = res
        logger.debug("E_last: {0}".format(self.E_last))
        
        self.Z_last[chain_idx] = self.Z

        self.__sim_step_counter__ += 1  # Be carefull with the overflow!!!

        return res
    
    def sim_reset(self):
        self.E_last = np.zeros(self.number_of_2T_chains, dtype=np.complex128)  # last term of Eq. 1.55 E(t - 2NT)
        self.Z_last = np.zeros(self.number_of_2T_chains)
        self.Ze = np.zeros(self.N + 1)
        self.Z = 0.
        self.__sim_step_counter__ = 0
        

    def print_sim_params(self):
        print("Theta: {0:.2e} [s]".format(self.Theta))
        print("Cavity RT: {0:.2e} [s]".format(2.0 * self.T))
        print("N_eff: {0:.2e} [s]".format(self.N_eff()))

        print("N: {0}".format(self.N))

        print("Number of 2T chains: {0}".format(self.number_of_2T_chains))

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
