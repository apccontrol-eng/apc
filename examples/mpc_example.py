"""
Model Predictive Control (MPC) Example
===============================================================================

Description
===============================================================================
This script implements a constrained linear Model Predictive Control (MPC)
simulation for a discrete-time system using a quadratic programming (QP)
formulation.

The controller predicts future system behavior over a finite prediction
horizon and computes an optimal control sequence while satisfying input
constraints.

A Kalman filter is optionally used to estimate the system state in the
presence of Gaussian process noise.

Main Features
===============================================================================
- Linear MPC formulation
- Prediction matrix construction
- Quadratic cost function generation
- Input constraint handling
- QP optimization solver
- Optional Kalman filtering
- Closed-loop simulation
- State and control signal visualization

System Model
===============================================================================
The simulated system is a discrete-time linear state-space model:

    x[k+1] = A x[k] + B u[k] + w[k]

where:
    x : state vector
    u : control input
    w : Gaussian process noise
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emil/Documents/GitHub/apc')

# =============================================================================
# QP SOLVERS and KF

from apc.solvers.hildreth_qp import hildreth_qp
from apc.solvers.primal_dual_interior_point_qp import primal_dual_interior_point_qp
from apc.solvers.active_set_qp import active_set_qp
from apc.solvers.projected_gradient_descent_qp import projected_gradient_descent_qp
from apc.filters.kalman_filter import kalman_filter

# =============================================================================
# BUILD PREDICTION MATRICES

def build_prediction_matrices(A, B, N):
    n = A.shape[0]
    m = B.shape[1]

    Sx = np.zeros((N*n, n))
    Su = np.zeros((N*n, N*m))

    for i in range(N):
        A_power = np.linalg.matrix_power(A, i+1)
        Sx[i*n:(i+1)*n, :] = A_power

        for j in range(i+1):
            A_j = np.linalg.matrix_power(A, i-j)
            Su[i*n:(i+1)*n, j*m:(j+1)*m] = A_j @ B

    return Sx, Su

# =============================================================================
# BLOCK DIAGONAL MATRIX

def block_diag(Q, N):
    return np.kron(np.eye(N), Q)

# =============================================================================
# CONSTRUCTING QP MATRICES

def build_qp(Sx, Su, Q_bar, R_bar, x0):
    H = Su.T @ Q_bar @ Su + R_bar
    f = Su.T @ Q_bar @ (Sx @ x0)
    return H, f

# =============================================================================
# CONSTRUCTING INPUT CONSTRAINTS

def input_constraints(N, m, umin, umax):
    G = np.vstack((np.eye(N*m), -np.eye(N*m)))
    b = np.hstack((np.tile(umax, N), -np.tile(umin, N)))
    return G, b

# =============================================================================
# EXAMPLE MPC LOOP

# example system
A = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.0, 0.0, 0.7]
])

B = np.array([[0.1], [0.05], [0.02]])

n = A.shape[0]
m = B.shape[1]
N = 10  # horizon length

# weight matrices
Q = np.eye(n)
R = 0.1 * np.eye(m)

Q_bar = block_diag(Q, N)
R_bar = block_diag(R, N)

# constraints
umin = np.array([-1.0])
umax = np.array([1.0])
G, b = input_constraints(N, m, umin, umax)

# initial state
x = np.array([2.0, 0.0, 5.0])

x_history = [x.copy()]
u_history = []

sim_steps = 100
np.random.seed(42)

for k in range(sim_steps):

    Sx, Su = build_prediction_matrices(A, B, N)
    H, f = build_qp(Sx, Su, Q_bar, R_bar, x)

    U_opt, lam = hildreth_qp(H, f, G, b, lambda0=None)
    #U_opt, lam= primal_dual_interior_point_qp(H, f, G, b, max_iter=100, tol=1e-100)
    #U_opt, lam, W = active_set_qp(H, f, G, b, x0=None, tol=1e-10, max_iter=100)
    #U_opt = projected_gradient_descent_qp(H, f, G, b, x0=None, alpha=1e-1, max_iter=100, tol=1e-10)
    u = U_opt[:m]
    
    u_history.append(u.copy())

    if k == 40:
        bias = np.array([0.2, -0.3, -0.4])
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3) + bias
    else:
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3)
    
    # =========================================================================
    # no kalman filter:
    x = A @ x + B @ u + added_noise
    x_history.append(x.copy())

    # =========================================================================
    # kalman filter:
    '''
    y = A @ x + B @ u + added_noise
    C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    D = np.array([
        [0.0],
        [0.0]
    ])
    if k == 0:
        P = 0.01 * np.eye(3)
        Q = 0.0 * np.eye(3)                                                     # this demo only contains measurement noise, so we trust the process model to contain no noise
        R = 0.01 * np.eye(2)
        x_pred, P_pred = kalman_filter(A, B, C, D, u, x, P, y[0:2], Q, R)       # only the first two states are "observed" with Gaussian noise

    else:
        x_pred, P_pred = kalman_filter(A, B, C, D, u, x, P_pred, y[0:2], Q, R)
    x = x_pred
    x_history.append(x.copy())
    '''
    
    #print(f"Step {k}, state: {x}, control: {u}")


x_history = np.array(x_history)
u_history = np.array(u_history)


t = np.arange(len(x_history[:,0]))

plt.figure(figsize=(14, 8))

# =============================================================================
# OUTPUT COMPARISON

plt.subplot(2,1,1)

plt.plot(t, x_history[:,0], '--', label="state 1", linewidth=2)
plt.plot(t, x_history[:,1], '--', label="state 2", linewidth=2)
plt.plot(t, x_history[:,2], '--', label="state 3", linewidth=2)

plt.title("MPC controller input and states with added Gaussian noise", fontsize=14)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.grid()

# =============================================================================
# INPUT SIGNAL

plt.subplot(2,1,2)

plt.plot(t[0:-1], u_history[:,0], label="MPC Input (u)", linewidth=2)

plt.xlabel("Time step", fontsize=12)
plt.ylabel("MPC Input", fontsize=12)
plt.title("MPC Input Signal", fontsize=14)
plt.grid()

plt.tight_layout()
plt.show()

# =============================================================================

