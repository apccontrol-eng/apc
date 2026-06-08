import numpy as np

def primal_dual_interior_point_qp(H, f, G, b, max_iter=100, tol=1e-10):
    """
    ===========================================================================
    Primal-Dual Interior-Point Quadratic Programming Solver
    ===========================================================================
    Parameters

    H : ndarray (n x n)
        Positive definite Hessian matrix of the quadratic cost.

    f : ndarray (n,)
        Linear cost vector.

    G : ndarray (m x n)
        Inequality constraint matrix.

    b : ndarray (m,)
        Inequality constraint vector.

    max_iter : int, optional
        Maximum number of interior-point iterations.

    tol : float, optional
        Numerical convergence tolerance.

    ===========================================================================
    Returns

    x : ndarray (n,)
        Optimal primal solution.

    lam : ndarray (m,)
        Optimal dual variables (Lagrange multipliers).

    ===========================================================================
    References

    Stephen J. Wright
    "Primal-Dual Interior-Point Methods"
    SIAM, 1997

    Jorge Nocedal and Stephen Wright
    "Numerical Optimization"
    Springer, Second Edition, 2006

    ===========================================================================
    """

    n = H.shape[0]
    m = G.shape[0]

    # =========================================================================
    # regularization for numerical stability
    H = H + 1e-8 * np.eye(n)

    # =========================================================================
    # initial primal variable
    x = np.zeros(n)

    # =========================================================================
    # slack variables, strictly positive
    s = np.maximum(b - G @ x, 1e-3)

    # =========================================================================
    # dual variables
    lam = np.ones(m)

    # =========================================================================
    # initial barrier parameter which is update in the loop
    mu = 1

    for _ in range(max_iter):

        # =====================================================================
        # KKT RESIDUALS:
        # stationarity residual
        rx = H @ x + f + G.T @ lam

        # primal feasibility residual
        rs = G @ x + s - b

        # complementarity residual
        rlam = lam * s

        # =====================================================================
        # checking convergence
        if (
            np.linalg.norm(rx) < tol and
            np.linalg.norm(rs) < tol and
            np.linalg.norm(np.minimum(lam, 0)) < tol
        ):
            break

        # =====================================================================
        # diagonal matrices
        S = np.diag(s)
        L = np.diag(lam)

        # =====================================================================
        # KKT MATRIX
        KKT = np.block([
            [H,        G.T,        np.zeros((n, m))],
            [G,        np.zeros((m, m)), np.eye(m)],
            [np.zeros((m, n)), S, L]
        ])

        # =====================================================================
        # RIGHT-HAND SIDE
        rhs = -np.hstack([
            rx,
            rs,
            rlam - (1e-2 * mu) * np.ones(m)
        ])

        # =====================================================================
        # NEWTON STEP
        dx = np.linalg.solve(KKT, rhs)

        dx_x = dx[:n]
        dx_l = dx[n:n+m]
        dx_s = dx[n+m:]

        # =====================================================================
        # STEP LENGTH COMPUTATION
        alpha = 1.0
        for d, v in [(dx_l, lam), (dx_s, s)]:
            idx = d < 0
            if np.any(idx):
                alpha = min(alpha, 0.999 * np.min(-v[idx] / d[idx]))

        # =====================================================================
        # VARIABLE UPDATE
        x += alpha * dx_x
        lam += alpha * dx_l
        s += alpha * dx_s

        # =====================================================================
        # barrier parameter reduction, mu update
        mu = max(1e-4, mu * 0.2)

    # =========================================================================
    # FINAL PROJECTION SAFETY STEP
    viol = G @ x - b
    x = x - G.T @ np.maximum(viol, 0)
    if np.max(viol) > 1e-10:
        print("Warning: small constraint violation:",
              np.max(viol))

    return x, lam

