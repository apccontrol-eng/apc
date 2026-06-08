import numpy as np

def hildreth_qp(H, f, G, b, max_iter=100, tol=1e-100, lambda0=None):
    """
    ===========================================================================
    Hildreth Quadratic Programming (QP) Solver
    ===========================================================================
    Parameters

    H : ndarray
        Hessian matrix of the quadratic cost function.
        Must be positive definite.

    f : ndarray
        Linear cost vector.

    G : ndarray
        Inequality constraint matrix.

    b : ndarray
        Inequality constraint bound vector.

    max_iter : int, optional
        Maximum number of solver iterations.
        Default is 100.

    tol : float, optional
        Convergence tolerance for Lagrange multiplier updates.
        Default is 1e-10.

    lambda0 : ndarray, optional
        Initial guess for Lagrange multipliers.
        Useful for warm-starting MPC optimization problems.

    ===========================================================================
    Returns

    x : ndarray
        Optimal solution vector.

    lam : ndarray
        Optimal Lagrange multipliers.
        
    ===========================================================================
    References
    
    Liuping Wang,
    Model Predictive Control System Design and Implementation Using MATLAB®,
    Springer, 2009.
    
    Hildreth, C.,
    A Quadratic Programming Procedure,
    Naval Research Logistics Quarterly, 1957.
    
    David G. Luenberger,
    Optimization by Vector Space Methods,
    John Wiley & Sons, 1969.

    ===========================================================================
    """
    
    n = H.shape[0]
    m = G.shape[0]

    # =========================================================================
    # adding regularization
    H = H + 1e-8 * np.eye(n)

    # =========================================================================
    # dual problem matrices
    H_inv = np.linalg.inv(H)

    P = G @ H_inv @ G.T
    d = G @ H_inv @ f + b

    # =========================================================================
    # initializing lagrange multipliers
    lam = np.zeros(m) if lambda0 is None else lambda0.copy()

    # =========================================================================
    # iteration of the Hildreth solver
    for _ in range(max_iter):
        lam_old = lam.copy()
        for i in range(m):
            # =================================================================
            # computing summation term excluding i-th multiplier
            sum_term = np.dot(P[i, :], lam) - P[i, i] * lam[i]

            # =================================================================
            # projecting onto feasible region (lambda_i >= 0)
            lam[i] = max(0.0, -(d[i] + sum_term) / P[i, i] )

        # =====================================================================
        # checking convergence
        if np.linalg.norm(lam - lam_old) < tol:
            break

    # =========================================================================
    # recovering primal solution
    x = -H_inv @ (f + G.T @ lam)

    return x, lam

