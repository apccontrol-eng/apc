'''
Hildreth Quadratic Programming (QP) Solver
==========================================

Description
-----------
This module implements the Hildreth iterative algorithm for solving
constrained quadratic programming (QP) problems of the form:

    minimize:
        0.5 * x^T H x + f^T x

    subject to:
        G x <= b

The solver is particularly useful in:
- Model Predictive Control (MPC)
- Constrained optimization
- Real-time embedded control systems
- Convex quadratic optimization problems

The Hildreth method solves the dual optimization problem iteratively
using Lagrange multipliers.

Optimization Problem
--------------------
Primal form:

    min_x  (1/2)x^T H x + f^T x

Subject to:

    Gx <= b

where:
    H : positive definite Hessian matrix
    f : linear cost vector
    G : inequality constraint matrix
    b : inequality bound vector

Method Overview
---------------
The algorithm:
1. Forms the dual QP problem
2. Iteratively updates Lagrange multipliers
3. Enforces non-negativity constraints
4. Computes the optimal primal solution

The method is computationally efficient for small-to-medium-sized MPC
applications.

Dependencies
------------
- numpy

Author
------
Emil

Notes
-----
- A small regularization term is added to H to improve numerical stability.
- The matrix H should be positive definite for guaranteed convergence.
- The solver handles inequality constraints only.
- Warm-starting is supported through the `lambda0` parameter.

Example
-------
    x_opt, lambda_opt = hildreth_qp(
        H,
        f,
        G,
        b
    )

References
----------
[1] Liuping Wang,
    Model Predictive Control System Design and Implementation Using MATLAB®,
    Springer, 2009.

[2] Hildreth, C.,
    A Quadratic Programming Procedure,
    Naval Research Logistics Quarterly, 1957.

[3] David G. Luenberger,
    Optimization by Vector Space Methods,
    John Wiley & Sons, 1969.
'''

# =========================
# HILDRETH QP SOLVER
# =========================

import numpy as np

def hildreth_qp(H, f, G, b, max_iter=100, tol=1e-10, lambda0=None):
    
    '''
    Solve a constrained quadratic programming (QP) problem using the
    Hildreth iterative algorithm.

    Parameters
    ----------
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

    Returns
    -------
    x : ndarray
        Optimal solution vector.

    lam : ndarray
        Optimal Lagrange multipliers.

    Optimization Form
    -----------------
    The solver minimizes:

        0.5 * x^T H x + f^T x

    Subject to:

        Gx <= b

    Algorithm
    ---------
    1. Regularize Hessian matrix:
           H = H + epsilon * I

    2. Compute dual problem matrices:
           P = G H^-1 G^T
           d = G H^-1 f + b

    3. Iteratively update Lagrange multipliers:
           lambda_i >= 0

    4. Recover primal solution:
           x = -H^-1 (f + G^T lambda)

    Convergence
    -----------
    The algorithm stops when:

        ||lambda_k - lambda_k-1|| < tol

    or when the maximum number of iterations is reached.

    Notes
    -----
    - Suitable for MPC applications with linear inequality constraints.
    - Warm-starting can significantly improve convergence speed.
    - Numerical regularization improves matrix inversion stability.
    '''

    n = H.shape[0]
    m = G.shape[0]

    # ==================================
    # HESSIAN REGULARIZATION
    # ==================================
    # Improves numerical stability and
    # prevents singular matrix inversion.
    H = H + 1e-8 * np.eye(n)

    # ==================================
    # DUAL PROBLEM MATRICES
    # ==================================
    H_inv = np.linalg.inv(H)

    P = G @ H_inv @ G.T
    d = G @ H_inv @ f + b

    # ==================================
    # INITIALIZE LAGRANGE MULTIPLIERS
    # ==================================
    lam = np.zeros(m) if lambda0 is None else lambda0.copy()

    # ==================================
    # HILDRETH ITERATIVE SOLVER
    # ==================================
    for _ in range(max_iter):

        lam_old = lam.copy()

        for i in range(m):

            # Compute summation term excluding i-th multiplier
            sum_term = np.dot(P[i, :], lam) - P[i, i] * lam[i]

            # Projection onto feasible region (lambda_i >= 0)
            lam[i] = max(
                0.0,
                -(d[i] + sum_term) / P[i, i]
            )

        # ==================================
        # CONVERGENCE CHECK
        # ==================================
        if np.linalg.norm(lam - lam_old) < tol:
            break

    # ==================================
    # RECOVER PRIMAL SOLUTION
    # ==================================
    x = -H_inv @ (f + G.T @ lam)

    return x, lam
