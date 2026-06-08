import numpy as np
from scipy.linalg import qr, svd, pinv

'''
===============================================================================
MOESP

References:
    
    Isermann, R., & Münchhof, M. (2011). 
    Identification of dynamic systems: An introduction with applications. 
    Springer. 
    https://doi.org/10.1007/978-3-540-78879-9

    Katayama, T. (2005). 
    Subspace methods for system identification. 
    Springer-Verlag London. 
    https://doi.org/10.1007/1-84628-158-X

===============================================================================
'''

def block_hankel(data, i):
    """
    Build block Hankel matrix.

    data: (N, d)
    returns: (i*d, N-i+1)
    """
    N, d = data.shape
    j = N - i + 1

    H = np.zeros((i * d, j))

    for row in range(i):
        H[row*d:(row+1)*d, :] = data[row:row+j, :].T

    return H


def MOESP(u, y, i, n):
    """
    MOESP subspace identification with internal Hankel construction.

    Parameters
    ----------
    u : ndarray (N, m)
    y : ndarray (N, p)
    i : int
        Number of block rows
    n : int
        System order

    Returns
    -------
    A, B, C, D
    """

    N, m = u.shape
    _, p = y.shape

    # --------------------------------------------------
    # Hankel matrices

    U = block_hankel(u, i)
    Y = block_hankel(y, i)

    km = U.shape[0]
    kp = Y.shape[0]

    # --------------------------------------------------
    # LQ decomposition

    W = np.vstack((U, Y))
    _, R = qr(W.T, mode='economic')
    L = np.triu(R).T

    # partition L
    L11 = L[:km, :km]
    L21 = L[km:km + kp, :km]

    # --------------------------------------------------
    # SVD

    UU, s, VVh = svd(L[km:km + kp, km:km + kp], full_matrices=False)

    U1 = UU[:, :n]
    Ok = U1 @ np.diag(np.sqrt(s[:n]))

    # --------------------------------------------------
    # C matrix is just the first block

    C = Ok[:p, :]

    # --------------------------------------------------
    # A matrix from performing ordinary least squares on shifted Ok

    A = pinv(Ok[:p * (i - 1), :]) @ Ok[p:p * i, :]

    # --------------------------------------------------
    # B and D matrices

    U2 = UU[:, n:]
    Z = U2.T @ L21 @ pinv(L11)

    XX = []
    RR = []

    for j in range(1, i + 1):

        XX.append(Z[:, m * (j - 1):m * j])

        Okj = Ok[:p * (i - j), :]

        upper1 = np.hstack([
            np.zeros((p * (j - 1), p)),
            np.zeros((p * (j - 1), n))
        ])

        middle = np.hstack([
            np.eye(p),
            np.zeros((p, n))
        ])

        lower = np.hstack([
            np.zeros((p * (i - j), p)),
            Okj
        ])

        Rj = np.vstack([upper1, middle, lower])

        RR.append(U2.T @ Rj)

    XX = np.vstack(XX)
    RR = np.vstack(RR)

    DB = pinv(RR) @ XX

    D = DB[:p, :]
    B = DB[p:, :]

    return A, B, C, D