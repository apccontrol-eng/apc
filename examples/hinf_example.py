
"""
Discrete-time MIMO H-infinity state-feedback synthesis
using CVXPY.

System:

    x[k+1] = A x[k] + Bw w[k] + Bu u[k]

    z[k]   = Cz x[k] + Dzw w[k] + Dzu u[k]

State feedback:

    u[k] = K x[k]

Goal:

    minimize gamma

    subject to

        ||T_{w -> z}||_inf < gamma

and closed-loop stability.

The controller is recovered as

    K = Y X^{-1}
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# =================================================
# Discrete-time MIMO plant
# =================================================

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


# =================================================
# Dimensions
# =================================================

n = A.shape[0]          # number of states
nw = Bw.shape[1]        # number of disturbances
nu = Bu.shape[1]        # number of control inputs
nz = Cz.shape[0]        # number of performance outputs


# =================================================
# Decision variables
# =================================================

# X > 0
X = cp.Variable((n, n), symmetric=True)

# Y = K X
Y = cp.Variable((nu, n))

# H-infinity performance level
gamma = cp.Variable(nonneg=True)


# =================================================
# Closed-loop terms in convex variables
# =================================================

# (A + Bu K) X
#
# Since Y = K X:
#
# (A + Bu K) X = A X + Bu Y

Acl_X = A @ X + Bu @ Y


# (Cz + Dzu K) X
#
# Since Y = K X:
#
# (Cz + Dzu K) X = Cz X + Dzu Y

Ccl_X = Cz @ X + Dzu @ Y


# =================================================
# DT H-infinity bounded-real LMI
# =================================================

# The LMI is
#
# [ X       0       Acl_X       Bw          ]
# [ 0       gamma I Ccl_X       Dzw         ]
# [ Acl_X'  Ccl_X'  X           0           ]
# [ Bw'     Dzw'    0           gamma I     ]
#
# > 0
#
# This guarantees
#
#     ||T_(w -> z)||_inf < gamma
#
# together with closed-loop stability.

LMI_HINF = cp.bmat([

    [
        X,
        np.zeros((n, nz)),
        Acl_X,
        Bw
    ],

    [
        np.zeros((nz, n)),
        gamma * np.eye(nz),
        Ccl_X,
        Dzw
    ],

    [
        Acl_X.T,
        Ccl_X.T,
        X,
        np.zeros((n, nw))
    ],

    [
        Bw.T,
        Dzw.T,
        np.zeros((nw, n)),
        gamma * np.eye(nw)
    ]

])


# =================================================
# Constraints
# =================================================

eps = 1e-7

constraints = [

    # Positive definite Lyapunov matrix
    X >> eps * np.eye(n),

    # Strictly positive H-infinity bound
    gamma >= eps,

    # DT bounded-real LMI
    LMI_HINF >> eps * np.eye(2*n + nz + nw)

]


# =================================================
# Optimization
# =================================================

problem = cp.Problem(

    cp.Minimize(gamma),

    constraints

)


# =================================================
# Solve
# =================================================

problem.solve(
    solver=cp.SCS
)


# =================================================
# Recover controller
# =================================================

if problem.status in ["optimal", "optimal_inaccurate"]:

    # K = Y X^{-1}
    K = Y.value @ np.linalg.inv(X.value)

    print("=" * 60)
    print("DT H-INFINITY STATE-FEEDBACK SYNTHESIS")
    print("=" * 60)

    print("\nStatus:")
    print(problem.status)

    print("\nOptimal H-infinity bound:")
    print(gamma.value)

    print("\nK =")
    print(K)


    # =================================================
    # Closed-loop system
    # =================================================

    Acl = A + Bu @ K

    Ccl = Cz + Dzu @ K


    # =================================================
    # Closed-loop eigenvalues
    # =================================================

    eigvals = np.linalg.eigvals(Acl)

    spectral_radius = np.max(np.abs(eigvals))

    print("\nClosed-loop eigenvalues:")
    print(eigvals)

    print("\nClosed-loop spectral radius:")
    print(spectral_radius)

    if spectral_radius < 1.0:
        print("\nClosed loop is DISCRETE-TIME STABLE.")
    else:
        print("\nWARNING: Closed loop is NOT stable.")


    # =================================================
    # Verify H-infinity bound
    # =================================================

    print("\nLMI H-infinity bound:")
    print(gamma.value)

    print("\nTherefore:")
    print("||T_(w -> z)||_inf <=", gamma.value)


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

    # -------------------------------------------------
    # External disturbance
    # -------------------------------------------------
    #
    # For pure regulation:
    #
    #     w = 0
    #
    # For seeing disturbance rejection, you can replace
    # this with a nonzero disturbance later.

    w = np.zeros((N, nw))


    # =================================================
    # Allocate simulation variables
    # =================================================

    x = np.zeros((N + 1, n))
    u = np.zeros((N, nu))
    z = np.zeros((N, nz))


    # Initial condition
    x[0, :] = x0


    # =================================================
    # Simulate closed-loop system
    #
    # u[k] = K x[k]
    #
    # x[k+1] = Acl x[k] + Bw w[k]
    #
    # z[k] = Ccl x[k] + Dzw w[k]
    # =================================================

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
        r"$H^\infty$ State Feedback: State Regulation"
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
        r"$H^\infty$ State Feedback: Control Inputs"
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
        r"$H^\infty$ State Feedback: Regulated Outputs"
    )

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


else:

    print("=" * 60)
    print("OPTIMIZATION FAILED")
    print("=" * 60)

    print("Status:")
    print(problem.status)
