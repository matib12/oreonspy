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
    types.Tuple((complex128[:], complex128, float64))(
        float64,
        complex128,
        float64[:],
        float64,
        boolean,
        float64,
        int64,
        complex128[:],
        complex128[:],
        complex128,
        float64,
        complex128
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
    input_electric_field_history,
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

    output_mirror_position_grid = np.empty(num_roundtrips + 2, dtype=np.float64)
    output_mirror_position_grid[0] = 0.0
    output_mirror_position_grid[1:] = np.linspace(
        total_output_mirror_displacement,
        last_total_output_mirror_displacement,
        num=num_roundtrips + 1
    )
    output_mirror_position_grid = numba_add_accumulate(output_mirror_position_grid)

    # Update input electric field buffer
    input_electric_field_history = np.roll(input_electric_field_history, 1)
    input_electric_field_history[0] = input_electric_field

    # Calculate the sum
    Sum = 0.0
    for idx in range(num_roundtrips):
        Sum = (
            Sum
            + rarbne2iknL[idx]
            * np.exp(k2j * output_mirror_position_grid[idx])
            * input_electric_field_history[idx]
        )

    intracavity_electric_field = (
        t_a * Sum
        + rarbne2iknL[num_roundtrips]
        * np.exp(k2j * output_mirror_position_grid[num_roundtrips])
        * last_intracavity_electric_field
    )

    return (
        input_electric_field_history,
        intracavity_electric_field,
        total_output_mirror_displacement
    )
