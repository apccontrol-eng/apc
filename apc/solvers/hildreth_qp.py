"""
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
