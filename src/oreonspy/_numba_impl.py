import numpy as np
from numba import njit, types
from numba import int64, float64, complex128, boolean


@njit(float64[:](float64[:]))
def numba_add_accumulate(A):
    r = np.empty(len(A), dtype=np.float64)
    t = 0.0
    for i in range(len(A)):
        t += A[i]
        r[i] = t
    return r


@staticmethod
@njit(
    types.Tuple((complex128, float64))(
        float64,        # output_mirror_displacement
        complex128,     # input_electric_field
        float64[:],     # last_output_mirror_displacement_all_subhist
        float64,        # last_total_output_mirror_displacement
        boolean,        # partial_Theta
        float64,        # Theta_fraction
        int64,          # num_roundtrips
        complex128,     # last_input_electric_field
        complex128[:],  # rarbne2iknL
        complex128,     # k2j
        float64,        # t_a
        complex128      # last_intracavity_electric_field
    )
)
def heavy(
    output_mirror_displacement,
    input_electric_field,
    last_output_mirror_displacement_all_subhist,
    last_total_output_mirror_displacement,
    partial_Theta,
    Theta_fraction,
    num_roundtrips,
    last_input_electric_field,
    rarbne2iknL,
    k2j,
    t_a,
    last_intracavity_electric_field,
):
    total_output_mirror_displacement = (
        np.sum(last_output_mirror_displacement_all_subhist)
        + last_total_output_mirror_displacement
    )

    if partial_Theta:
        last_total_output_mirror_displacement += (
            Theta_fraction * output_mirror_displacement
        )
        last_input_electric_field += (
            Theta_fraction * (input_electric_field - last_input_electric_field)
        )

    output_mirror_position_grid = np.empty(num_roundtrips + 2, dtype=np.float64)
    output_mirror_position_grid[0] = 0.0
    output_mirror_position_grid[1:] = np.linspace(
        total_output_mirror_displacement,
        last_total_output_mirror_displacement,
        num=num_roundtrips + 1
    )
    output_mirror_position_grid = numba_add_accumulate(output_mirror_position_grid)

    input_electric_field_grid = np.linspace(
        input_electric_field,
        last_input_electric_field,
        num=num_roundtrips
    )

    # Calculate the sum
    Sum = 0.0
    for idx in range(num_roundtrips):
        Sum = (
            Sum
            + rarbne2iknL[idx]
            * np.exp(k2j * output_mirror_position_grid[idx])
            * input_electric_field_grid[idx]
        )

    intracavity_electric_field = (
        t_a * Sum
        + rarbne2iknL[num_roundtrips]
        * np.exp(k2j * output_mirror_position_grid[num_roundtrips])
        * last_intracavity_electric_field
    )

    return (
        intracavity_electric_field,
        total_output_mirror_displacement
    )
