#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:31:00 2026

@author: emil
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:31:00 2026

@author: emil
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# -------------------------------------------------
# Discrete-time MIMO plant
# -------------------------------------------------

A = np.array([
    [1.05, 0.10, 0.00, 0.00],
    [0.00, 0.95, 0.20, 0.00],
    [0.00, 0.00, 0.90, 0.10],
    [0.00, 0.00, 0.00, 0.85]
])

Bw = np.array([
    [0.10, 0.00],
    [0.00, 0.15],
    [0.20, 0.00],
    [0.00, 0.10]
])

Bu = np.array([
    [0.10, 0.00],
    [0.00, 0.20],
    [0.30, 0.00],
    [0.00, 0.25]
])

Cz = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

Dzw = np.zeros((2, 2))
Dzu = np.zeros((2, 2))


# -------------------------------------------------
# Dimensions
# -------------------------------------------------

n = A.shape[0]
nw = Bw.shape[1]
nu = Bu.shape[1]
nz = Cz.shape[0]


# -------------------------------------------------
# Decision variables
# -------------------------------------------------

X = cp.Variable((n, n), symmetric=True)
Y = cp.Variable((nu, n))
W = cp.Variable((nz, nz), symmetric=True)


# -------------------------------------------------
# H2 LMIs
# -------------------------------------------------

# AX + Bu Y
Acl_X = A @ X + Bu @ Y

LMI1 = cp.bmat([
    [X,              Acl_X,                    Bw],
    [Acl_X.T,        X,                        np.zeros((n, nw))],
    [Bw.T,           np.zeros((nw, n)),        np.eye(nw)]
])


# Ccl X = Cz X + Dzu Y
CXY = Cz @ X + Dzu @ Y

LMI2 = cp.bmat([
    [W,              CXY],
    [CXY.T,          X]
])


# -------------------------------------------------
# Constraints
# -------------------------------------------------

constraints = [
    X >> 1e-6 * np.eye(n),
    W >> 0,
    LMI1 >> 1e-8 * np.eye(2*n + nw),
    LMI2 >> 1e-8 * np.eye(nz + n)
]


# -------------------------------------------------
# Optimization
# -------------------------------------------------

problem = cp.Problem(
    cp.Minimize(cp.trace(W)),
    constraints
)

problem.solve(solver=cp.SCS)


# -------------------------------------------------
# Recover K
# -------------------------------------------------

if problem.status in ["optimal", "optimal_inaccurate"]:

    K = Y.value @ np.linalg.inv(X.value)

    print("Status:", problem.status)

    print("Optimal H2 squared bound:")
    print(problem.value)

    print("H2 bound:")
    print(np.sqrt(problem.value))

    print("\nK =")
    print(K)


    # -------------------------------------------------
    # Closed-loop system
    # -------------------------------------------------

    Acl = A + Bu @ K

    # IMPORTANT:
    # z = (Cz + Dzu K)x + Dzw w
    Ccl = Cz + Dzu @ K

    print("\nClosed-loop eigenvalues:")
    print(np.linalg.eigvals(Acl))

    print("\nClosed-loop spectral radius:")
    print(np.max(np.abs(np.linalg.eigvals(Acl))))


    # =================================================
    # Simulation
    # =================================================

    N = 30

    # Arbitrary initial condition
    x0 = np.array([
        2.0,
        -1.0,
        1.5,
        -0.5
    ])

    # No external disturbance
    # Pure regulation experiment
    w = np.zeros((N, nw))


    # -------------------------------------------------
    # Allocate simulation variables
    # -------------------------------------------------

    x = np.zeros((N + 1, n))
    u = np.zeros((N, nu))
    z = np.zeros((N, nz))


    # Initial condition
    x[0, :] = x0


    # -------------------------------------------------
    # Simulate closed-loop system
    #
    # u[k] = K x[k]
    #
    # x[k+1] = Acl x[k] + Bw w[k]
    #
    # z[k] = Ccl x[k] + Dzw w[k]
    # -------------------------------------------------

    for k in range(N):

        # State feedback
        u[k, :] = K @ x[k, :]

        # Performance output
        z[k, :] = (
            Ccl @ x[k, :]
            + Dzw @ w[k, :]
        )

        # State update
        x[k + 1, :] = (
            Acl @ x[k, :]
            + Bw @ w[k, :]
        )


    # =================================================
    # Plot 1: State trajectories
    # =================================================

    time_x = np.arange(N + 1)

    plt.figure(figsize=(9, 5))

    for i in range(n):

        plt.plot(
            time_x,
            x[:, i],
            linewidth=2,
            label=fr"$x_{i+1}$"
        )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.xlabel("Discrete time $k$")
    plt.ylabel("State")

    plt.title(
        r"$H^2$ State Feedback: State Regulation"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


    # =================================================
    # Plot 2: Control inputs
    # =================================================

    time_u = np.arange(N)

    plt.figure(figsize=(9, 5))

    for i in range(nu):

        plt.step(
            time_u,
            u[:, i],
            where="post",
            linewidth=2,
            label=fr"$u_{i+1}$"
        )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.xlabel("Discrete time $k$")
    plt.ylabel("Control input")

    plt.title(
        r"$H^2$ State Feedback: Control Inputs"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


    # =================================================
    # Plot 3: Performance outputs
    # =================================================

    plt.figure(figsize=(9, 5))

    for i in range(nz):

        plt.plot(
            time_u,
            z[:, i],
            linewidth=2,
            label=fr"$z_{i+1}$"
        )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.xlabel("Discrete time $k$")
    plt.ylabel("Performance output")

    plt.title(
        r"$H^2$ State Feedback: Regulated Outputs"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


else:

    print("Optimization failed.")
    print("Status:", problem.status)