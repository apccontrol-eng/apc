import numpy as np

def MOESP(u, y, i, n):
    """
    MOESP

    u: (N, m)
    y: (N, p)
    i: number of block rows
    n: system order
    """

    N = u.shape[0]
    m = u.shape[1]
    p = y.shape[1]

    j = N - 2 * i + 1

    # ----------------------------
    # Block Hankel
    # ----------------------------
    def block_hankel(data, rows, cols):
        d = data.shape[1]
        H = np.zeros((rows * d, cols))
        for r in range(rows):
            H[r*d:(r+1)*d, :] = data[r:r+cols].T
        return H

    U_p = block_hankel(u, i, j)
    Y_p = block_hankel(y, i, j)
    U_f = block_hankel(u[i:], i, j)
    Y_f = block_hankel(y[i:], i, j)

    # ----------------------------
    # QR decomposition
    # ----------------------------
    W = np.vstack((U_p, Y_p, U_f))
    _, R = np.linalg.qr(W.T)
    R = R.T

    # partition
    R32 = R[i*(m+p):, i*m:i*(m+p)]

    # ----------------------------
    # SVD
    # ----------------------------
    U_svd, S_svd, _ = np.linalg.svd(R32, full_matrices=False)

    U1 = U_svd[:, :n]
    S1 = np.diag(np.sqrt(S_svd[:n]))

    Gamma_i = U1 @ S1

    # ----------------------------
    # Extract C and A
    # ----------------------------
    C = Gamma_i[:p, :]

    Gamma_i_upper = Gamma_i[:-p, :]
    Gamma_i_lower = Gamma_i[p:, :]

    A = np.linalg.lstsq(Gamma_i_upper, Gamma_i_lower, rcond=None)[0]

    # ----------------------------
    # Estimate states (FIXED)
    # ----------------------------
    X = np.linalg.pinv(Gamma_i) @ Y_p   # (n, j)

    # ----------------------------
    # Estimate B and D (FIXED)
    # ----------------------------
    N_x = X.shape[1]

    X_curr = X[:, :-1]          # (n, j-1)
    X_next = X[:, 1:]           # (n, j-1)

    U_trim = u[i:i+N_x-1].T     # (m, j-1)
    Y_trim = y[i:i+N_x-1].T     # (p, j-1)

    Z = np.vstack((X_curr, U_trim))  # (n+m, j-1)

    # least squares (correct orientation!)
    AB = np.linalg.lstsq(Z.T, X_next.T, rcond=None)[0].T
    CD = np.linalg.lstsq(Z.T, Y_trim.T, rcond=None)[0].T

    A = AB[:, :n]
    B = AB[:, n:]

    C = CD[:, :n]
    D = CD[:, n:]

    return A, B, C, D


