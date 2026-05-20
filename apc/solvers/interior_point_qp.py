import numpy as np

def primal_dual_interior_point_qp(H, f, G, b, max_iter=100, tol=1e-10):
    """
    Primal-dual interior point QP solver:
        min 0.5 x^T H x + f^T x
        s.t. Gx <= b
    """

    n = H.shape[0]
    m = G.shape[0]

    H = H + 1e-8 * np.eye(n)

    x = np.zeros(n)
    s = np.maximum(b - G @ x, 1e-3)         # slack variables
    lam = np.ones(m)       # dual variables

    mu = 1

    for _ in range(max_iter):

        # residuals
        rx = H @ x + f + G.T @ lam
        rs = G @ x + s - b
        rlam = lam * s

        # stopping condition
        if (
            np.linalg.norm(rx) < tol and
            np.linalg.norm(rs) < tol and
            np.linalg.norm(np.minimum(lam, 0)) < tol
        ):
            break

        # diagonal matrices
        S = np.diag(s)
        L = np.diag(lam)

        # KKT system
        KKT = np.block([
            [H,        G.T,        np.zeros((n, m))],
            [G,        np.zeros((m, m)), np.eye(m)],
            [np.zeros((m, n)), S, L]
        ])

        rhs = -np.hstack([
            rx,
            rs,
            rlam - (1e-2 * mu) * np.ones(m)
        ])

        dx = np.linalg.solve(KKT, rhs)

        dx_x = dx[:n]
        dx_l = dx[n:n+m]
        dx_s = dx[n+m:]

        # step size (ensure positivity)
        alpha = 1.0
        for d, v in [(dx_l, lam), (dx_s, s)]:
            idx = d < 0
            if np.any(idx):
                alpha = min(alpha, 0.999 * np.min(-v[idx] / d[idx]))

        # update
        x += alpha * dx_x
        lam += alpha * dx_l
        s += alpha * dx_s
        mu = max(1e-4, mu * 0.2)
        
    # final projection safety (numerical cleanup)
    viol = G @ x - b
    x = x - G.T @ np.maximum(viol, 0)
    #if np.max(viol) > 1e-10:
    #    print("Warning: small constraint violation:", np.max(viol))

    return x, lam