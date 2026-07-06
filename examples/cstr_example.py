"""
CSTR MPC Example (the dynamics are from the book by Rawlings)
===============================================================================

Reference: 

    James B. Rawlings, David Q. Mayne, Moritz M. Diehl (2017)
    Model Predictive Control: Theory, Computation, and Design (2nd ed.)
    Nob Hill Publishing


Description
===============================================================================
This script implements a constrained linear Model Predictive Control (MPC)
simulation for a discrete-time system using a quadratic programming (QP)
formulation.

The controller predicts future system behavior over a finite prediction
horizon and computes an optimal control sequence while satisfying input
constraints.

Main Features
===============================================================================
- Linear MPC formulation
- Prediction matrix construction
- Quadratic cost function generation
- Input constraint handling
- QP optimization
- Optional Kalman filtering
- Closed-loop simulation using non-linear dynamics
- State and control signal visualization

Non-linear system model of the CSTR
===============================================================================
The simulated system is a discrete-time non-linear state-space model:

    x[k+1] = f( x[k] , u[k] )

where:
    f : non-linear function
    x : state vector
    u : control input

Fixed point linearized system model
===============================================================================
The controller uses a discrete-time linear state-space model:

    dx[k+1] = A_d dx[k] + B_d du[k]

where:
    dx : state vector in deviation coordinates
    du : control input in deviation coordinates

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
# NON-LINEAR STATE-SPACE MODEL FOR CSTR    
def cstr_state(x,u):
    F_0 = 0.1
    T_0 = 350
    c_0 = 1
    r = 0.219
    k_0 = 7.2*(10**10)
    E_R = 8750
    U = 54.94
    roo = 1000
    C_p = 0.239
    deltaH = -5.0*(10**(4))
    pi = np.pi
    
    c = x[0]
    T = x[1]
    h = x[2]
    
    T_c = u[0]
    F = u[1]
    
    dc_dt = ( F_0*( c_0 - c ) ) / (pi*r*r*h) - k_0*np.exp( -E_R/T )*c
    dT_dt = ( F_0*( T_0 - T ) ) / (pi*r*r*h) + ( - deltaH / (roo*C_p) )*k_0*np.exp( -E_R/T )*c + ( (2*U) / (r*roo*C_p) )*(T_c - T)
    dh_dt = (F_0 - F) / (pi*r*r)
    
    dx_dt = np.array([ dc_dt , dT_dt , dh_dt ])
    
    return dx_dt

def cstr_discrete(x, u, Ts): # Runge-Kutta 4

    k1 = cstr_state(x, u)
    k2 = cstr_state(x + Ts/2 * k1, u)
    k3 = cstr_state(x + Ts/2 * k2, u)
    k4 = cstr_state(x + Ts * k3, u)

    x_next = x + Ts/6 * (k1 + 2*k2 + 2*k3 + k4)

    return x_next

Ts = 1  # sampling time

# =============================================================================
# EXAMPLE MPC LOOP

A = np.array([
    [0.2681, -0.00338, -0.00728],
    [9.703, 0.3279, -25.44],
    [0.0, 0.0, 1]
])

B = np.array([ 
    [-0.00537, 0.1655],
    [1.297, 97.91],
    [0.0, -6.637]
])

N = 20  # horizon length

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
    np.zeros((3, 2))
])

Ca = np.hstack([
    C, Cd
])

print("Aa shape:", Aa.shape)
print("Ba shape:", Ba.shape)
print("Ca shape:", Ca.shape)
N = 10
na = Aa.shape[0]
ma = Ba.shape[1]

Qa = np.diag([
    1.0, 1.0, 1.0,
    1e-6, 1e-6, 1e-6
])

Ra = 100*np.eye(ma)

Qa_bar = block_diag(Qa, N)
Ra_bar = block_diag(Ra, N)

# constraints
umin = np.array([-10, -0.1])
umax = np.array([40, 3.0])
G, b = input_constraints(N, ma, umin, umax)

# x_setpoint = (c = 0.8789 , (T = 324.5) , (h = 0.659)
x_setpoint = np.array([0.878, 324.5, 0.659, 0.0, 0.0, 0.0])
u_setpoint = np.array([300, 0.1])

# initial states (the nonlinear CSTR is not controllable too far from the setpoint values, the Arrhenius term explodes easily)
x_prop = np.array([0.50, 324.5, 0.29, 0.0, 0.0, 0.0]) # the last 3 are initial values for the bias
x_dev = x_prop - x_setpoint 
# steady-state around which MPC is defined
u_prop = np.array([310, 0.15])
u_dev = u_prop - u_setpoint

x_prop_history = [x_prop.copy()]
x_dev_history = [x_dev.copy()]
u_prop_history = [u_prop.copy()]
u_dev_history = [u_dev.copy()]

sim_steps = 50
np.random.seed(42)

for k in range(sim_steps):

    Sx, Su = build_prediction_matrices(Aa, Ba, N)
     
    if k > 1:
        
        d_hat = x_dev[3:6].copy()
        x_hat = x_dev[:3].copy()
        
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
        
        # =====================================================================
        # set x_ss (we want x_ss = 0) zero and then perform u_ss := inverse(B) @ d so that the controller removes the identified offset that the nominal process model does not have
        x_ss = np.zeros(3)
        #u_ss = np.array([0,0])
        u_ss = np.linalg.pinv(-B) @ d_hat
        
        #print("u_ss : ", u_ss)
        # =====================================================================
        x_ref_stage = np.hstack([x_ss, d_hat])
        u_ref_stage = u_ss
    
        x_ref = np.tile(x_ref_stage, N)
        u_ref = np.tile(u_ref_stage, N)
    
        H, f = build_qp_tracking(
            Sx, Su,
            Qa_bar, Ra_bar,
            x_dev,
            x_ref,
            u_ref
        )
    
    else:
        H, f = build_qp(Sx, Su, Qa_bar, Ra_bar, x_dev)
    
    U_opt, lam = hildreth_qp(H, f, G, b, max_iter=100, tol=1e-100, lambda0=None)
    #U_opt, lam= primal_dual_interior_point_qp(H, f, G, b, max_iter=100, tol=1e-100)
    #U_opt, lam, W = active_set_qp(H, f, G, b, x0=None, tol=1e-10, max_iter=100)
    #U_opt = projected_gradient_descent_qp(H, f, G, b, x0=None, alpha=0.001, max_iter=100, tol=1e-10)
    u_dev = U_opt[:ma]
    u_dev_history.append(u_dev.copy())
    u_prop = u_dev + u_setpoint
    u_prop_history.append(u_prop.copy())
    
    noise_std = 0.01
    added_noise = noise_std * np.random.randn(3)
    
    #bias = np.array([0.1, -0.23, -0.14]) #+ added_noise
    
    #x_prop = cstr_discrete(x_prop[0:3], u_prop, Ts)
        
    #print("x_prop : ", x_prop)
    
    #x_prop_history.append(x_prop.copy())

    y_dev = C @ ( cstr_discrete(x_prop[0:3], u_prop, Ts) - x_setpoint[0:3] ) #+ added_noise
    
    Da = np.zeros((3, 2))
    
    if k == 0:
        P = 0.00000001 * np.eye(6)
        Q = 0.00000001 * np.eye(6)
        R = 0.00000001 * np.eye(3)
        x_pred, P_pred, innovation = kalman_filter(Aa, Ba, Ca, Da, u_dev, x_dev, P, y_dev, Q, R)

    else:
        x_pred, P_pred, innovation = kalman_filter(Aa, Ba, Ca, Da, u_dev, x_dev, P_pred, y_dev, Q, R)
        
    x_dev = x_pred
    x_dev_history.append(x_dev.copy())
    
    x_prop = x_dev + x_setpoint
    x_prop_history.append(x_prop.copy())
    

    if k > 48:
        printable_x = np.round(x_prop,4)
        printable_u = np.round(u_prop,4)
        printable_innovation = np.round(innovation,4)
        
        #print(f"Step {k}, state: {printable_x}, control: {printable_u}, innovation: {printable_innovation}")
        
        #print("x_dev : ",  np.round(x_dev,3))
        #print("u_dev : ",  np.round(u_dev,3))
        print("x_prop : ", np.round(x_prop,3))
        print("u_prop : ", np.round(u_prop,3))
        

x_prop_history = np.array(x_prop_history)
u_prop_history = np.array(u_prop_history)


t = np.arange(len(x_prop_history[:,0]))

plt.figure(figsize=(14, 8))

# =============================================================================
# OUTPUT COMPARISON

plt.subplot(2,1,1)

plt.plot(t, x_prop_history[:,0], '--', label="x_1", linewidth=2)
plt.plot(t, x_prop_history[:,1]/100, '--', label="x_2/100", linewidth=2)
plt.plot(t, x_prop_history[:,2], '--', label="x_3", linewidth=2)

plt.title("Non-linear state-space model response to MPC controls", fontsize=14)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.ylim(-1, 4)
plt.grid()


# =============================================================================
# INPUT SIGNAL

plt.subplot(2,1,2)

plt.plot(t, u_prop_history[:,0]/100, label="MPC Input (u_1)/100", linewidth=2)
plt.plot(t, u_prop_history[:,1], label="MPC Input (u_2)", linewidth=2)

plt.xlabel("Time step", fontsize=12)
plt.ylabel("MPC Input", fontsize=12)
plt.title("MPC Input Signal", fontsize=14)
plt.ylim(-1, 4)
plt.grid()
plt.legend()
plt.show()
# =============================================================================

# open-loop stability sanity check (x should stay same at same u)
#x = np.array([0.878, 324.5, 0.659])    # also the set-point
#u = np.array([300, 0.1])               # also the set-point
#for k in range(sim_steps):
#    x = cstr_discrete(x[0:3], u, Ts)
#    print(x)

# add state constraints to MPC and safety layer because the nonlinear system has runaway dynamics

