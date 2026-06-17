import numpy as np
#import logging
#logger = logging.getLogger(__name__)

def heavy(
    output_mirror_displacement,  # d_zeta
    input_electric_field,  # E_in_curr
    last_output_mirror_displacement_all_subhist,  # d_zeta_last
    last_total_output_mirror_displacement,  # Z_last_chain_idx
    partial_Theta,
    Theta_fraction,
    num_roundtrips,  # N
    last_input_electric_field,
    rarbne2iknL,
    k2j,
    t_a,
    last_intracavity_electric_field,  # E_last_chain_idx
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
    # logger.debug(output_mirror_position_grid)

    output_mirror_position_grid = np.add.accumulate(output_mirror_position_grid)
    # logger.debug("output_mirror_position_grid: {0}".format(output_mirror_position_grid))

    input_electric_field_grid = np.linspace(
        input_electric_field,
        last_input_electric_field,
        num=num_roundtrips
    )

    # Calculate the sum
    Sum = 0.0
    for idx in range(num_roundtrips):
        # print("index: {0}".format(idx))
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
        total_output_mirror_displacement,
    )
