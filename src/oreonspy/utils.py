import numpy as np
from scipy import constants as const

# Wavelength of the laser light
lambd = 1064e-9  # [m]

# Wave vector
k = 2.*np.pi / lambd  # [1/m]

def r_a_r_b(Finesse):
    """
    This function computes the product of reflectivities of two mirrors in an optical cavity
    based on the given Finesse value. It evaluates two possible branches of the calculation
    and returns the minimum value, ensuring the result is less than 1.0.
    Parameters:
        Finesse (float): The finesse of the optical cavity, a dimensionless parameter 
                         that characterizes the cavity's quality.
    Returns:
        float: The minimum product of reflectivities (r_a * r_b), constrained to be less than 1.0.
    """

    W = (2.0 * Finesse / np.pi)**2
    
    # “plus” branch:
    r_a_r_b_product_p = (W + 2.0 + 2.0 * np.sqrt(W + 1.0)) / W
    # “minus” branch:
    r_a_r_b_product_m = (W + 2.0 - 2.0 * np.sqrt(W + 1.0)) / W

    return np.min([r_a_r_b_product_p, r_a_r_b_product_m])  # must be less than 1.0


def critical_velocity(Finesse, L, wavelength):
    '''
    Calculate the critical velocity for a cavity.
    F = pi/2*sinh(T/tau)  # 2.18
    sinh(T/tau) = pi/(2*F)
    T/tau = asinh(pi/(2*F))
    tau = T/asinh(pi/(2*F))

    T = L/c  # 1.31
    '''
    
    return wavelength / (2. * Finesse * L/const.c/ np.arcsinh(np.pi/(2.*Finesse)))  # [m/s]!