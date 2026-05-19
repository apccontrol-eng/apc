import numpy as np

# =========================
# HILDRETH QP SOLVER
# =========================
def hildreth_qp(H, f, G, b, max_iter=100, tol=1e-10, lambda0=None):
    n = H.shape[0]
    m = G.shape[0]

    # Regularization of H
    H = H + 1e-8 * np.eye(n)

    H_inv = np.linalg.inv(H)
    P = G @ H_inv @ G.T
    d = G @ H_inv @ f + b

    lam = np.zeros(m) if lambda0 is None else lambda0.copy()

    for _ in range(max_iter):
        lam_old = lam.copy()

        for i in range(m):
            sum_term = np.dot(P[i, :], lam) - P[i, i] * lam[i]
            lam[i] = max(0.0, -(d[i] + sum_term) / P[i, i])

        if np.linalg.norm(lam - lam_old) < tol:
            break

    x = -H_inv @ (f + G.T @ lam)
    return x, lam