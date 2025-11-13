import numpy as np
from numba import njit, types
from numba import int64, float64, complex128, boolean    # import the types

@njit(float64[:](float64[:]))
def numba_add_accumulate(A):
    r = np.empty(len(A), dtype=np.float64)
    t = 0.
    for i in range(len(A)):
        t += A[i]
        r[i] = t
    return r

@staticmethod
@njit(types.Tuple((float64[:], complex128[:], complex128, float64))(float64, complex128, float64[:], float64, boolean, float64, int64, float64[:], complex128[:], complex128[:], complex128, float64, complex128)) #,fastmath=True)  # Check if fastmath=True is correct
def heavy(d_zeta, E_in_curr, d_zeta_last, Z_last_chain_idx, partial_Theta, N_pre, N, Ze, E_in_buffers_chain_idx, rarbne2iknL, k2j, t_a, E_last_chain_idx):
    Z = np.sum(d_zeta_last) + Z_last_chain_idx

    #logger.debug("Z_last: {0}".format(self.Z_last))

    Z_start = Z_last_chain_idx
    if partial_Theta:
        Z_start += np.interp(N_pre-N , [0, N_pre], [0, d_zeta])
        #logger.debug("Z_start: {0}".format(Z_start))

    Ze[1:] = np.linspace(Z, Z_start, num=N + 1)
    # logger.debug(self.Ze)
    Ze = numba_add_accumulate(Ze)
    #logger.debug("Ze: {0}".format(Ze))

    # Update input electric field buffer
    E_in_buffers_chain_idx = np.roll(E_in_buffers_chain_idx, 1)
    E_in_buffers_chain_idx[0] = E_in_curr

    # Calculate the sum
    Sum = 0.0
    for idx in range(N):
        # print("index: {0}".format(idx))
        Sum = Sum + rarbne2iknL[idx] * np.exp(
            k2j * Ze[idx]
        ) * E_in_buffers_chain_idx[idx]

    E = (
        t_a * Sum
        + rarbne2iknL[N]
        * np.exp(k2j * Ze[N])
        * E_last_chain_idx
    )

    return Ze, E_in_buffers_chain_idx, E, Z