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
        L = cavity.params.__L__
    
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
def generate_time_points_for_constant_velocity(velocity, f_calc, lambd = lambd, number_of_FSR=2.):
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
    return -k*v*t/cavity.params.T

def optimal_sampling_frequency(cavity, critical_velocity_factor, lambd = lambd):
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


def plot_cavity_evolution(output_mirror_displacements, input_electric_field, intracav_electric_field, pdh, input_mirror_displacements=None, reflected_field=None, file_name=None, save=False):
    if save == True:
        plt.ioff()  # Disable interactive mode for correct memory management in iPython
    else:
        plt.ion()

    reflected_field_phase = np.angle(reflected_field) if reflected_field is not None else None
    intracav_electric_field_phase = np.angle(intracav_electric_field) if intracav_electric_field is not None else None
    input_electric_field_phase = np.angle(input_electric_field) if input_electric_field is not None else None
    intracav_power = np.abs(intracav_electric_field)**2

    n_of_sublots = 5 if reflected_field is not None or reflected_field_phase is not None else 4
    fig, ax = plt.subplots(n_of_sublots, 1, figsize=(10, 10))
    fig.tight_layout()
    
    plot_idx = 0
    ax[0].plot(intracav_power, label="Intracavity power")
    ax[0].grid()
    ax[0].set_ylabel("watt")
    ax[0].title.set_text("Optical power inside the cavity")
    ax2 = ax[0].twinx()
    ax2.plot(np.unwrap(intracav_electric_field_phase), label="Intracavity electric field phase", color='tab:orange')
    ax2.set_ylabel("phase")
    ax2.tick_params(axis='y')
    plot_idx += 1

    ax[1].plot(pdh, label="Pound-Drever-Hall", c="darkred")
    ax[1].set_ylabel("PDH")
    ax[1].grid()
    ax[1].title.set_text("PDH signal")
    plot_idx += 1

    if reflected_field is not None:
        ax[plot_idx].plot(np.abs(reflected_field)**2, label="Reflected power", color='darkblue')
        ax[plot_idx].set_ylabel("watt")
        ax[plot_idx].grid()
        ax[plot_idx].title.set_text("Reflected power")
        ax3 = ax[plot_idx].twinx()
        if reflected_field_phase is not None:
            ax3.plot(np.unwrap(reflected_field_phase), label="Reflected field phase", color='orange')
            ax3.set_ylabel("phase")
            ax3.tick_params(axis='y')
        plot_idx += 1

    if input_mirror_displacements is not None:
        ax[plot_idx].plot(input_mirror_displacements, label="m2 shift", c='lightgreen')
    ax[plot_idx].plot(output_mirror_displacements, label="m1 shift", ls='dashed', c='darkgreen')
    ax[plot_idx].set_ylabel("displacements")
    ax[plot_idx].grid()
    ax[plot_idx].title.set_text("Mirror")
    plot_idx += 1

    ax[plot_idx].plot(np.abs(input_electric_field), label="Input electric field magnitude", c='violet')
    ax[plot_idx].set_ylabel("magnitude")
    ax[plot_idx].grid()
    ax[plot_idx].title.set_text("Input electric field")
    ax4 = ax[plot_idx].twinx()
    ax4.plot(np.unwrap(input_electric_field_phase), label="Input electric field phase", ls='dashed', c='magenta')
    ax4.set_ylabel("phase")
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
    if reflected_field is not None:
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
        if file_name is None:
            file_name = "cavity_evolution"
        else:
            file_name = file_name.replace(" ", "_")
        fig.savefig("./"+file_name+".png", dpi=300, bbox_inches='tight')
        
        plt.cla()
        plt.close(fig)
    else:
        plt.show()
        return fig