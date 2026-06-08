import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emil/Documents/GitHub/apc')

"""
===============================================================================
Robust Infinite-Horizon MPC via Linear Matrix Inequalities Example
===============================================================================

Description
===============================================================================
This controller implements a robust infinite-horizon Model Predictive Control
(MPC) formulation based on Linear Matrix Inequalities (LMIs), 
following directly the approach proposed by:

Kothare, M. V., Balakrishnan, V., & Morari, M. (1996).
Robust constrained model predictive control using linear matrix inequalities.
Automatica, 32(10), 1361–1379.
DOI: 10.1016/0005-1098(96)00063-5

https://doi.org/10.1016/0005-1098(96)00063-5

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

    x[k+1] = A x[k] + B u[k] (+ w[k])

where:
    x : state vector
    u : control input
    w : Gaussian process noise

The controller seeks a state-feedback law

    u[k] = L x[k]
    

Optimization Problem
===============================================================================
Decision variables:

    Q_U     : Lyapunov matrix (positive semidefinite)
    Y       : controller parameter matrix
    X_U     : auxiliary input constraint variable
    γ       : upper bound on worst-case cost

The optimization solves

    minimize    γ

subject to the LMI constraints

1. State inclusion constraint

    [ 1      x_kᵀ ]
    [ x_k    Q_U  ]  ⪰ 0

2. Input constraint feasibility

    [ X_U   Y   ]
    [ Yᵀ   Q_U ] ⪰ 0

3. Robust performance / stability LMI

    [ Q_U              Q_UAᵀ + YᵀBᵀ    Q_UQ^(1/2)    YᵀR^(1/2) ]
    [ AQ_U + BY        Q_U             0            0          ]
    [ Q^(1/2)Q_U       0               γI           0          ]
    [ R^(1/2)Y         0               0            γI         ] ⪰ 0

4. Input magnitude constraint

    |u| ≤ u_max

implemented through

    X_U(0,0) ≤ u_max²

After solving the SDP, the feedback gain is recovered as

    L = Y Q_U⁻¹

and the control law becomes

    u = L x

===============================================================================
- CVXPY is used to formulate and solve the semidefinite program.
- CVXOPT solves the LMI optimization.
- The implementation assumes:
    - Linear time-invariant dynamics (for this example)
    - Full-state feedback
    - Quadratic cost
    - Input constraints
- Gaussian disturbances/noise added externally during simulation.
===============================================================================
"""



def robust_linear_mpc( state_x , A , B , Q , R ):

    Q_sqrt = np.linalg.cholesky(Q)
    R_sqrt = np.linalg.cholesky(R)

    # current state x(k|k)
    xk = state_x

    n = A.shape[0]
    m = B.shape[1]
    
    # decision variables
    XU = cp.Variable((m, m), symmetric=True)
    QU = cp.Variable((n, n), PSD=True)
    
    Y = cp.Variable((m, n))
    gamma = cp.Variable(nonneg=True)

    C1 = cp.bmat([
        [cp.Constant([[1.0]]), xk.reshape(1, -1)],
        [xk.reshape(-1, 1), QU]
    ])
    
    C2 = cp.bmat([
        [XU, Y],
        [Y.T, QU] 
    ])
    
    row1 = [QU, QU @ A.T + Y.T @ B.T, QU @ Q_sqrt, Y.T @ R_sqrt]
    
    row2 = [A @ QU + B @ Y, QU, cp.Constant(np.zeros((n, n))), cp.Constant(np.zeros((n, m)))]
    
    row3 = [Q_sqrt @ QU, cp.Constant(np.zeros((n, n))), gamma * np.eye(n), cp.Constant(np.zeros((n, m)))]
    
    row4 = [R_sqrt @ Y, cp.Constant(np.zeros((m, n))), cp.Constant(np.zeros((m, n))), gamma * np.eye(m)]
    
    LMI = cp.bmat([row1, row2, row3, row4])
    
    u_max = 1 # |u| <= u_max is a vector if we have many controls!
    
    # collecting constraints
    constraints = [
        C1 >> 0,
        C2 >> 0,
        LMI >> 0,
        -XU[0,0] >= - u_max**2 # here we have only one control so 0,0 means it is actually a scalar
    ]

    # objective
    objective = cp.Minimize(gamma)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT)

    # Output
    if prob.status == cp.OPTIMAL:
        print(f" Optimal γ: {gamma.value:.4f}")
        print("QU matrix:\n", QU.value)
        print("Y matrix:\n", Y.value)
        print("XU matrix:\n", XU.value)
    else:
        print(f" Optimization failed: {prob.status}")
        
    QU_inv = np.linalg.inv(QU.value)
    
    print("QU_inv : " , QU_inv)
    
    L = Y.value@QU_inv
    
    control_u = L@state_x
    
    new_state_x = A@state_x + B@control_u
    
    return new_state_x, control_u


###############################################################################
###############################################################################
###############################################################################

A = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.0, 0.0, 0.7]
])

A = np.array(A)
n = A.shape[0]

P = cp.Variable((n, n), symmetric=True)

# Define the LMI: A^T P + P A < 0
LMI = A.T @ P + P @ A

constraints = [
    P >> 1e-6 * np.eye(n),      # P > 0 (positive definite)
    LMI << -1e-6 * np.eye(n)    # Lyapunov inequality (strict negative definite)
]


objective = cp.Minimize(cp.trace(P))


prob = cp.Problem(objective, constraints)

prob.solve(solver=cp.CVXOPT)

if prob.status == cp.OPTIMAL:
    print("LMI is feasible. P:")
    print(P.value)
else:
    print("LMI problem is not feasible:", prob.status)

###############################################################################
###############################################################################
###############################################################################

# --------------------------------------------------
# EXAMPLE LMI RMPC LOOP


# system matrices
A = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.0, 0.0, 0.7]
])

B = np.array([[0.1], [0.05], [0.02]])

n = A.shape[0]
m = B.shape[1]

Q = np.eye(n)
R = 0.1 * np.eye(m)

# initial state
x = np.array([2.0, 0.0, 5.0])

x_history = [x.copy()]
u_history = []

sim_steps = 100
np.random.seed(42)

for k in range(sim_steps):

    x, u = robust_linear_mpc(x, A, B, Q, R)
    # saving the control
    u_history.append(u.copy())
    
    # system update
    if k == 40:
        bias = np.array([0.2, -0.3, -0.4])
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3) + bias
        x = x + added_noise
    else:
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3)
        x = x + added_noise
        
    x_history.append(x.copy())
    #print(f"Step {k}, state: {x}, control: {u}")

x_history = np.array(x_history)
u_history = np.array(u_history)

t = np.arange(len(x_history[:,0]))

plt.figure(figsize=(14, 8))

# --------------------------------------------------
# OUTPUTS

plt.subplot(2,1,1)
plt.plot(t, x_history[:,0], '--', label="state 1", linewidth=2)
plt.plot(t, x_history[:,1], '--', label="state 2", linewidth=2)
plt.plot(t, x_history[:,2], '--', label="state 3", linewidth=2)

plt.title("LMI-based infinite horizon MPC input and states with added Gaussian noise", fontsize=14)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.grid(False)

# --------------------------------------------------
# INPUT SIGNAL

plt.subplot(2,1,2)
plt.plot(t[0:-1], u_history[:,0], label="MPC Input (u)", linewidth=2)
plt.xlabel("Time step", fontsize=12)
plt.ylabel("MPC Input", fontsize=12)
plt.title("MPC Input Signal", fontsize=14)
plt.grid(False)

plt.tight_layout()
plt.show()

