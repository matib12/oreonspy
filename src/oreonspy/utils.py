import numpy as np
import matplotlib.pyplot as plt

c = 299792458.0  # Speed of light in vacuum [m/s]

# Wavelength of the laser light
lambd = 1064e-9  # [m]

# Wave vector
k = 2.*np.pi / lambd  # [1/m]

# Additional cavity parameters
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


def critical_velocity(cavity=None, wavelength=lambd, Finesse=None, L=None):
    '''
    Calculate the critical velocity for a cavity.
    If cavity parameter is provided, it will be used to calculate the critical velocity.
    If not, the function will use the provided wavelength, Finesse, and L parameters.
    The critical velocity is calculated using the formula:
    v_crit = lambda / (2 * Finesse * L / c * asinh(pi/(2*Finesse)))
    where:
    - lambda is the wavelength of the light
    - Finesse is the finesse of the cavity
    - L is the length of the cavity
    - c is the speed of light
    - asinh is the inverse hyperbolic sine function
    Parameters:
        cavity (object, optional): An object representing the cavity. If provided, it must have a Finesse() method.
        wavelength (float, optional): Wavelength of the light in meters. Default is 1064e-9 m.
        Finesse (float, optional): Finesse of the cavity. Default is None.
        L (float, optional): Length of the cavity in meters. Default is None.
    Returns:
        float: The critical velocity in meters per second.
    Raises:
        ValueError: If neither cavity nor wavelength, Finesse, and L are provided.
    
    Calculation:
    F = pi/2*sinh(T/tau)  # 2.18
    sinh(T/tau) = pi/(2*F)
    T/tau = asinh(pi/(2*F))
    tau = T/asinh(pi/(2*F))

    T = L/c  # 1.31
    '''
    if cavity is not None:
        Finesse = cavity.Finesse()
        L = cavity.params.cavity_length
    
    return wavelength / (2. * Finesse * L/c/ np.arcsinh(np.pi/(2.*Finesse)))  # [m/s]!


# Additional functions for the simulation
def V_pdh(gamma, E_in, E):
    """
    Calculate the Pound-Drever-Hall error signal. Based on the formula: 2.83 (Rakhmanov).

    Parameters:
        gamma (float): Demodulation phase of the coherent detection.
        E_in (complex): Input electric field.
        E (complex): Electric field inside the cavity.

    Returns:
        float: Pound-Drever-Hall error signal.
    """
    return -(np.exp(gamma*1.j) * np.conjugate(E_in) * E).imag


# Additional functions for the testing
def generate_time_points_for_constant_velocity(velocity, f_calc, number_of_FSR=2.):
    """
    Calculate the time window and the number of points for a given velocity and calculation frequency.

    Parameters:
        v (float): Velocity in meters per second.
        f_calc (float): Calculation frequency in Hertz.
        number_of_FSR (float, optional): Number of Free Spectral Ranges. Default is 2.

    Returns:
        tuple: A tuple containing the number of points (int) and a numpy array of time points (numpy.ndarray).
    """

    t_stop = number_of_FSR * lambd/(2.*velocity)
    #print("Time window [s]: {0}".format(t_stop))
    number_of_points = int(np.ceil(t_stop*f_calc))
    #print("Time window point number: {0}".format(number_of_points))

    return number_of_points, np.linspace(0., t_stop, number_of_points)

def Omega(t, v, cavity):
    # 2.70 
    return -k*v*t/cavity.params.half_roundtrip_time

def optimal_sampling_frequency(cavity, critical_velocity_factor):
    """
    Calculate the optimal sampling frequency for a given velocity and number of Free Spectral Ranges.

    Parameters:
        velocity (float): Velocity in meters per second.
        number_of_FSR (float, optional): Number of Free Spectral Ranges. Default is 2.

    Returns:
        float: Optimal sampling frequency in Hertz.
    """

    cav_tau = cavity.tau()
    oscillation_freq = Omega(3*cav_tau, -critical_velocity_factor*critical_velocity(cavity, lambd), cavity)  # rad/s or Hz?

    return oscillation_freq


def plot_cavity_evolution(zeta_positons, E_in_values, s, ph, pdh, zeta1_positons=None, s_ref=None, ph_ref=None, title=None, save=False):
    if save == True:
        plt.ioff()  # Disable interactive mode for correct memory management in iPython
    else:
        plt.ion()

    n_of_sublots = 5 if s_ref is not None or ph_ref is not None else 4
    
    fig, ax = plt.subplots(n_of_sublots, 1, figsize=(10, 10))
    fig.tight_layout()

    plot_idx = 0
    ax[0].plot(s, label="Cav mag")
    ax[0].grid()
    ax[0].set_ylabel("magn")
    ax[0].title.set_text("Electric field inside the cavity (m1 ref frame)")
    ax2 = ax[0].twinx()
    ax2.plot(np.unwrap(ph), label="Cav ph", color='tab:orange')
    ax2.set_ylabel("ph")
    ax2.tick_params(axis='y')
    plot_idx += 1

    ax[1].plot(pdh, label="PDH", c="darkred")
    ax[1].set_ylabel("PDH")
    ax[1].grid()
    ax[1].title.set_text("PDH signal")
    plot_idx += 1

    if s_ref is not None:
        ax[plot_idx].plot(s_ref, label="Refl mag", color='darkblue')
        ax[plot_idx].set_ylabel("magn")
        ax[plot_idx].grid()
        ax[plot_idx].title.set_text("Reflected electric field")
        ax3 = ax[plot_idx].twinx()
        if ph_ref is not None:
            ax3.plot(np.unwrap(ph_ref), label="Refl ph", color='orange')
            ax3.set_ylabel("ph")
            ax3.tick_params(axis='y')
        plot_idx += 1

    if zeta1_positons is not None:
        ax[plot_idx].plot(zeta1_positons, label="m2 pos", c='lightgreen')
    ax[plot_idx].plot(zeta_positons, label="m1 pos", ls='dashed', c='darkgreen')
    ax[plot_idx].set_ylabel("positions")
    ax[plot_idx].grid()
    ax[plot_idx].title.set_text("Mirror")
    plot_idx += 1

    ax[plot_idx].plot(np.abs(E_in_values)**2, label="Las mag", c='violet')
    ax[plot_idx].set_ylabel("magn")
    ax[plot_idx].grid()
    ax[plot_idx].title.set_text("Input electric field")
    ax4 = ax[plot_idx].twinx()
    ax4.plot(np.unwrap(np.angle(E_in_values)), label="Las ph", ls='dashed', c='magenta')
    ax4.set_ylabel("ph")
    ax4.tick_params(axis='y')

    # Prepare the legend
    lab = []
    han = []
    for a in ax:
        handles, labels = a.get_legend_handles_labels()
        han += handles
        lab += labels

    h, l = ax2.get_legend_handles_labels()
    han += h
    lab += l
    if s_ref is not None:
        h, l = ax3.get_legend_handles_labels()
        han += h
        lab += l
    h, l = ax4.get_legend_handles_labels()
    han += h
    lab += l

    # Reshufle the legend
    #order = [0,6,1,2,7,3,4,5,8]
    #fig.legend([han[idx] for idx in order],[lab[idx] for idx in order], ncol = 9, bbox_to_anchor=(0.5, .55, 0.5, 0.5))
    fig.legend()

    if save:
        # Save the figure
        if title is None:
            title = "cavity_evolution"
        else:
            title = title.replace(" ", "_")
        fig.savefig("../optical_cavities_testset/"+title+".png", dpi=300, bbox_inches='tight')
        
        plt.cla()
        plt.close(fig)
    else:
        plt.show()
        return fig