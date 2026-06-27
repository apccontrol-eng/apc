"""
===============================================================================
Off-set free Model Predictive Control (MPC) Example
===============================================================================

Description

This script implements a constrained linear Model Predictive Control (MPC)
simulation for a discrete-time system using a quadratic programming (QP)
formulation.

The controller and Kalman Filter are based on an approximate process model
which are deviced for controlling true system model. The approximate model 
used for MPC and Kalman Filter will introduce a steady-state offset which
will be dealt with by augmenting the disturbance into the model and estimating
the disturbance with Kalman Filter.

===============================================================================

System and Measurement Models (### disturbance/bias d is identified ###)

The simulated system is a discrete-time linear state-space model:

    x[k+1] = A x_approx[k] + B u[k] + d[k]
    
    d[k+1] = d[k]
    
    y[k] = C x[k]
    
    where:
        x : state vector
        u : control input
        y : measurement of the system
        d : disturbance/bias lumped together

===============================================================================

References:

    Pannocchia, G. (2015)
    Offset-free tracking MPC: A tutorial review and comparison of different formulations
    In Proceedings of the 2015 European Control Conference (ECC) (pp. 527–532)
    IEEE 
    https://doi.org/10.1109/ECC.2015.7330597
    
    
    Pannocchia, G., Gabiccini, M., & Artoni, A. (2015)
    Offset-free MPC explained: Novelties, subtleties, and applications
    IFAC-PapersOnLine, 48 (23), 342–351
    https://doi.org/10.1016/j.ifacol.2015.11.304

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emil/Documents/GitHub/apc')
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
# CONSTRUCTING QP MATRICES (not tracking mpc version)
def build_qp(Sx, Su, Q_bar, R_bar, x0):
    H = Su.T @ Q_bar @ Su + R_bar
    f = Su.T @ Q_bar @ (Sx @ x0)
    return H, f

# =============================================================================
# CONSTRUCTING QP MATRICES (tracking mpc version which is needed in the offset-free mpc)
def build_qp_tracking(Sx, Su, Q_bar, R_bar, x0, r_bar, u_bar):
    H = Su.T @ Q_bar @ Su + R_bar
    f = ( Su.T @ Q_bar @ (Sx @ x0 - r_bar) - R_bar @ u_bar )
    return H, f

# =============================================================================
# CONSTRUCTING INPUT CONSTRAINTS
def input_constraints(N, m, umin, umax):
    G = np.vstack((np.eye(N*m), -np.eye(N*m)))
    b = np.hstack((np.tile(umax, N), -np.tile(umin, N)))
    return G, b

# =============================================================================
def is_controllable(A, B, tol=1e-100):
    #Test controllability of the pair (A, B)
    n = A.shape[0]
    C = B
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = Ak @ A
        C = np.hstack((C, Ak @ B))
    rank = np.linalg.matrix_rank(C, tol)
    return rank == n

print("\n")
print("==========================================================================================================================================================")
print("==========================================================================================================================================================")
print("==========================================================================================================================================================")
print("\n")

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ESTIMATING THE BIAS/INNOVATION/DISTURBANCE WITH KALMAN FILTER AND USING OFFSET FREE MPC TO FIX THE PLANT-MODEL MISMATCH
A = np.array([
    [0.72,  0.18,  0.02],
    [0.12,  0.63,  0.22],
    [0.01,  0.08,  0.53]
])

B = np.array([
    [0.12,  0.88,  0.22],
    [0.32,  0.23,  0.52],
    [0.21,  0.18,  0.83]
])

print("(A, B) pair is controllable : ",   is_controllable(A, B, tol=1e-1000))
print("Stable discrete-time system model should have all of its eigenvalues under unit circle (Schur stability):")
print("A max abs eigenvalue : ", abs(max(np.linalg.eigvals(A))))


C = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

Bd = np.eye(3)   # disturbance input matrix
Cd = np.eye(3)*0   # disturbance output matrix

# Identity for disturbance dynamics
I = np.eye(3)
Z = np.zeros((3, 3))

# ---- Augmented matrices ----

Aa = np.block([
    [A,  Bd],
    [Z,  I ]
])

Ba = np.vstack([
    B,
    np.zeros((3, 3))
])

Ca = np.hstack([
    C, Cd
])

print("Aa shape:", Aa.shape)  # (6, 6)
print("Ba shape:", Ba.shape)  # (6, 1)
print("Ca shape:", Ca.shape)  # (3, 6)
N = 10
na = Aa.shape[0]
ma = Ba.shape[1]

Qa = np.diag([
    1.0, 1.0, 1.0,
    1e-6, 1e-6, 1e-6
])

Ra = 0.1*np.eye(ma)

Qa_bar = block_diag(Qa, N)
Ra_bar = block_diag(Ra, N)

# constraints
umin = np.array([-1.0, -1.0, -1.0])
umax = np.array([ 1.0,  1.0,  1.0])
G, b = input_constraints(N, ma, umin, umax)

# initial state
x = np.array([2.0, 0.0, 5.0, 0.0, 0.0, 0.0]) # the last 3 are initial values for the bias

x_history = [x.copy()]
u_history = []

sim_steps = 100
np.random.seed(42)

for k in range(sim_steps):

    Sx, Su = build_prediction_matrices(Aa, Ba, N)
     
    if k > 0:
        
        d_hat = x[3:6].copy()
        x_hat = x[:3].copy()
        
        # =====================================================================
        # correct steady-state equation:
        # x = A x + B u + d
        # x_ss - A x_ss = B u + d
        # ⇒ (I - A)x_ss - B u = d
        
        #M = np.block([
        #    [np.eye(3) - A, -B]
        #])
        #sol = np.linalg.lstsq(M, d_hat, rcond=None)[0]
        #x_ss = sol[:3]
        #u_ss = sol[3:]
    
        # assume x_ss is zero and then perform u_ss = inverse(B) @ d so that the controller removes the identified offset that the nominal process model does not have
        x_ss = np.zeros(3)
        u_ss = np.linalg.inv(-B) @ d_hat
    
        # =====================================================================
        x_ref_stage = np.hstack([x_ss, d_hat])
        u_ref_stage = u_ss
    
        x_ref = np.tile(x_ref_stage, N)
        u_ref = np.tile(u_ref_stage, N)
    
        H, f = build_qp_tracking(
            Sx, Su,
            Qa_bar, Ra_bar,
            x,
            x_ref,
            u_ref
        )
    
    else:
        H, f = build_qp(Sx, Su, Qa_bar, Ra_bar, x)
    
    U_opt, lam = hildreth_qp(H, f, G, b, max_iter=100, tol=1e-100, lambda0=None)
    #U_opt, lam= primal_dual_interior_point_qp(H, f, G, b, max_iter=100, tol=1e-100)
    #U_opt, lam, W = active_set_qp(H, f, G, b, x0=None, tol=1e-10, max_iter=100)
    #U_opt = projected_gradient_descent_qp(H, f, G, b, x0=None, alpha=0.001, max_iter=100, tol=1e-10)
    u = U_opt[:ma]
    u_history.append(u.copy())
    
    #noise_std = 0.01
    #added_noise = noise_std * np.random.randn(3)
    
    bias = np.array([0.1, -0.23, -0.14]) #+ added_noise

    # =========================================================================
    # kalman filter:
    x_true_state = A @ x[0:3] + B @ u + bias
    y = C @ ( x_true_state ) # measured state (this example is not about rejecting Gaussian noise though)

    Da = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    if k == 0:
        P = 0.0001 * np.eye(6)
        Q = 0.0001 * np.eye(6)
        R = 0.0001 * np.eye(3)
        x_pred, P_pred, innovation = kalman_filter(Aa, Ba, Ca, Da, u, x, P, y[0:3], Q, R)

    else:
        x_pred, P_pred, innovation = kalman_filter(Aa, Ba, Ca, Da, u, x, P_pred, y[0:3], Q, R)
    x = x_pred
    x_history.append(x.copy())

    if k > 95:
        printable_x = np.round(x,4)
        printable_u = np.round(u,4)
        printable_innovation = np.round(innovation,4)
        
        print(f"Step {k}, state: {printable_x}, control: {printable_u}, innovation: {printable_innovation}")


x_history = np.array(x_history)
u_history = np.array(u_history)


t = np.arange(len(x_history[:,0]))

plt.figure(figsize=(14, 8))

# =============================================================================
# OUTPUT COMPARISON

plt.subplot(2,1,1)

plt.plot(t, x_history[:,0], '-', label="state 1", linewidth=2)
plt.plot(t, x_history[:,1], '-', label="state 2", linewidth=2)
plt.plot(t, x_history[:,2], '-', label="state 3", linewidth=2)

plt.title("Approximate model offset free MPC controller", fontsize=14)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.grid()

# =============================================================================
# INPUT SIGNAL

plt.subplot(2,1,2)

plt.plot(t[0:-1], u_history[:,0], 'x', label="MPC Input (u_1)", linewidth=2)
plt.plot(t[0:-1], u_history[:,1], 'x', label="MPC Input (u_2)", linewidth=2)
plt.plot(t[0:-1], u_history[:,2], 'x', label="MPC Input (u_3)", linewidth=2)

plt.xlabel("Time step", fontsize=12)
plt.ylabel("MPC Input", fontsize=12)
plt.title("MPC Input Signal", fontsize=14)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


