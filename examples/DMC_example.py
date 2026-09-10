import numpy as np
import matplotlib.pyplot as plt
from apc.solvers.hildreth_qp import hildreth_qp

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




# ============================================================
# 1. Generate unit-step response data
# ============================================================

sim_steps = 100

x_1_unit_step_history = []
x_2_unit_step_history = []
x_3_unit_step_history = []
x_4_unit_step_history = []

for k in range(sim_steps):

    # u1 -> y1
    if 4 * k < 27:
        x_1_response_at_k = 0.0
    else:
        x_1_response_at_k = 4.05 * (
            1 - np.exp(-(4 * k - 27) / 50)
        )

    x_1_unit_step_history.append(
        x_1_response_at_k
    )

    # u2 -> y1
    if 4 * k < 28:
        x_2_response_at_k = 0.0
    else:
        x_2_response_at_k = 1.77 * (
            1 - np.exp(-(4 * k - 28) / 60)
        )

    x_2_unit_step_history.append(
        x_2_response_at_k
    )

    # u1 -> y2
    if 4 * k < 18:
        x_3_response_at_k = 0.0
    else:
        x_3_response_at_k = 5.39 * (
            1 - np.exp(-(4 * k - 18) / 50)
        )

    x_3_unit_step_history.append(
        x_3_response_at_k
    )

    # u2 -> y2
    if 4 * k < 14:
        x_4_response_at_k = 0.0
    else:
        x_4_response_at_k = 5.72 * (
            1 - np.exp(-(4 * k - 14) / 60)
        )

    x_4_unit_step_history.append(
        x_4_response_at_k
    )


# Convert to NumPy arrays
x_1_unit_step_history = np.asarray(
    x_1_unit_step_history,
    dtype=float
)

x_2_unit_step_history = np.asarray(
    x_2_unit_step_history,
    dtype=float
)

x_3_unit_step_history = np.asarray(
    x_3_unit_step_history,
    dtype=float
)

x_4_unit_step_history = np.asarray(
    x_4_unit_step_history,
    dtype=float
)


# ============================================================
# 2. Plot step responses
# ============================================================

t_model = np.arange(sim_steps)

plt.figure(figsize=(12, 6))

plt.plot(
    t_model,
    x_1_unit_step_history,
    '--',
    label="u1 -> y1",
    linewidth=2
)

plt.plot(
    t_model,
    x_2_unit_step_history,
    '--',
    label="u2 -> y1",
    linewidth=2
)

plt.plot(
    t_model,
    x_3_unit_step_history,
    '--',
    label="u1 -> y2",
    linewidth=2
)

plt.plot(
    t_model,
    x_4_unit_step_history,
    '--',
    label="u2 -> y2",
    linewidth=2
)

plt.xlabel("Sample")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# 3. DMC dynamic matrix
# ============================================================

def dynamic_matrix(g, p, m):

    g = np.asarray(g, dtype=float)

    if p < 1:
        raise ValueError("p must be at least 1.")

    if m < 1:
        raise ValueError("m must be at least 1.")

    if p > len(g) - 1:
        raise ValueError(
            "p must be <= len(g) - 1."
        )

    G = np.zeros((p, m))

    for i in range(p):

        for j in range(min(i + 1, m)):

            G[i, j] = g[i - j + 1]

    return G


# ============================================================
# 4. DMC horizons
# ============================================================

prediction_horizon = 60
control_horizon = 60

p = prediction_horizon
m = control_horizon


# ============================================================
# 5. Four SISO dynamic matrices
# ============================================================

# u1 -> y1
G11 = dynamic_matrix(
    x_1_unit_step_history,
    p,
    m
)

# u2 -> y1
G12 = dynamic_matrix(
    x_2_unit_step_history,
    p,
    m
)

# u1 -> y2
G21 = dynamic_matrix(
    x_3_unit_step_history,
    p,
    m
)

# u2 -> y2
G22 = dynamic_matrix(
    x_4_unit_step_history,
    p,
    m
)


# ============================================================
# 6. Complete MIMO dynamic matrix
# ============================================================
#
#       [ G11  G12 ]
# G  =  [ G21  G22 ]
#
# ============================================================

G = np.block([
    [G11, G12],
    [G21, G22]
])

print("G11 shape:", G11.shape)
print("G12 shape:", G12.shape)
print("G21 shape:", G21.shape)
print("G22 shape:", G22.shape)
print("Full G shape:", G.shape)


# ============================================================
# 7. Free response
# ============================================================

def calculate_free_response(
        y_current,
        du_history,
        g11,
        g12,
        g21,
        g22,
        p):

    g11 = np.asarray(g11, dtype=float)
    g12 = np.asarray(g12, dtype=float)
    g21 = np.asarray(g21, dtype=float)
    g22 = np.asarray(g22, dtype=float)

    y_current = np.asarray(
        y_current,
        dtype=float
    )

    Y01 = np.zeros(p)
    Y02 = np.zeros(p)

    N = len(du_history)

    for j in range(p):

        Y01[j] = y_current[0]
        Y02[j] = y_current[1]

        for i in range(1, N + 1):

            du1 = du_history[-i][0]
            du2 = du_history[-i][1]

            index_future = min(
                i + j,
                len(g11) - 1
            )

            index_now = min(
                i,
                len(g11) - 1
            )

            # u1 -> y1
            Y01[j] += (
                g11[index_future]
                - g11[index_now]
            ) * du1

            # u1 -> y2
            Y02[j] += (
                g21[index_future]
                - g21[index_now]
            ) * du1

            # u2 -> y1
            Y01[j] += (
                g12[index_future]
                - g12[index_now]
            ) * du2

            # u2 -> y2
            Y02[j] += (
                g22[index_future]
                - g22[index_now]
            ) * du2

    Y0 = np.concatenate(
        (Y01, Y02)
    )

    return Y0


# ============================================================
# 8. Hildreth QP solver
# ============================================================
#
# IMPORTANT:
#
# Put your existing hildreth_qp() function here.
#
# It must have the form:
#
# U_opt, lam = hildreth_qp(
#     H,
#     f,
#     G_constraints,
#     b,
#     max_iter=100,
#     tol=1e-100,
#     lambda0=None
# )
#
# ============================================================
# 9. Initial conditions
# ============================================================

y = np.array([
    4.0,
    -2.0
])

u = np.array([
    0.0,
    0.0
])

du_history = []


# ============================================================
# 10. Setpoints
# ============================================================

r1 = 0.0
r2 = 0.0

Yref = np.concatenate((
    np.full(p, r1),
    np.full(p, r2)
))


# ============================================================
# 11. DMC weighting
# ============================================================

lambda_ = 0.1


# ============================================================
# 12. QP matrices H and f
# ============================================================
#
# Objective:
#
# J =
# (Yref - Y0 - G*dU)^T
# (Yref - Y0 - G*dU)
#
# + lambda*dU^T*dU
#
# Therefore:
#
# H = G^T G + lambda I
#
# f = -G^T(Yref-Y0)
#
# ------------------------------------------------------------
# NOTE:
#
# This assumes your Hildreth solver solves:
#
# min 1/2 x^T H x + f^T x
#
# subject to:
#
# A x <= b
#
# ============================================================

H = (
    G.T @ G
    + lambda_ * np.eye(G.shape[1])
)


# ============================================================
# 13. Input constraints
# ============================================================
#
# These are EXAMPLE values.
#
# CHANGE THESE TO YOUR REAL CONSTRAINTS.
#
# ============================================================

# Absolute input constraints
u1_min = -10.0
u1_max =  10.0

u2_min = -5.0
u2_max =  5.0


# Input-move constraints
du1_min = -3.0
du1_max =  3.0

du2_min = -3.0
du2_max =  3.0


# ============================================================
# 14. Build cumulative-sum matrix
# ============================================================
#
# Future absolute inputs are:
#
# u(k+1) = u(k) + du(k)
#
# u(k+2) = u(k) + du(k) + du(k+1)
#
# etc.
#
# Therefore:
#
# U_future = U_current + S*dU
#
# where S is a lower-triangular matrix of ones.
#
# ============================================================

S_single = np.tril(
    np.ones((m, m))
)


# MIMO version:
#
# [S  0]
# [0  S]
#
S = np.block([
    [
        S_single,
        np.zeros((m, m))
    ],
    [
        np.zeros((m, m)),
        S_single
    ]
])


# ============================================================
# 15. Build constant constraint matrix
# ============================================================
#
# We want:
#
# du_min <= dU <= du_max
#
# and
#
# u_min <= u + S*dU <= u_max
#
# Everything must be converted to:
#
# A*dU <= b
#
# ============================================================

# ------------------------------------------------------------
# Move constraints
# ------------------------------------------------------------

A_du_upper = np.eye(
    2 * m
)

A_du_lower = -np.eye(
    2 * m
)


# ------------------------------------------------------------
# Absolute input constraints
# ------------------------------------------------------------

A_u_upper = S
A_u_lower = -S


# ------------------------------------------------------------
# Stack all constraints
# ------------------------------------------------------------

G_constraints = np.vstack([
    A_du_upper,
    A_du_lower,
    A_u_upper,
    A_u_lower
])


print(
    "Constraint matrix shape:",
    G_constraints.shape
)


# ============================================================
# 16. Closed-loop simulation
# ============================================================

closed_loop_steps = 80

y_history = [y.copy()]
u_history = [u.copy()]
du_applied_history = []


for k in range(closed_loop_steps):

    # ========================================================
    # Calculate free response
    # ========================================================

    Y0 = calculate_free_response(
        y_current=y,
        du_history=du_history,
        g11=x_1_unit_step_history,
        g12=x_2_unit_step_history,
        g21=x_3_unit_step_history,
        g22=x_4_unit_step_history,
        p=p
    )


    # ========================================================
    # QP linear term
    # ========================================================
    #
    # J =
    # (Yref-Y0-G*dU)^T
    # (Yref-Y0-G*dU)
    #
    # + lambda*dU^T*dU
    #
    # H = G^T G + lambda I
    #
    # f = -G^T(Yref-Y0)
    #
    # ========================================================

    f = -G.T @ (
        Yref - Y0
    )


    # ========================================================
    # Build constraint vector b
    # ========================================================
    #
    # Constraint ordering:
    #
    # 1.  dU <= du_max
    # 2. -dU <= -du_min
    # 3.  S*dU <= u_max-u_current
    # 4. -S*dU <= -(u_min-u_current)
    #
    # ========================================================

    du_max = np.concatenate((
        np.full(m, du1_max),
        np.full(m, du2_max)
    ))

    du_min = np.concatenate((
        np.full(m, du1_min),
        np.full(m, du2_min)
    ))

    u_max = np.concatenate((
        np.full(m, u1_max),
        np.full(m, u2_max)
    ))

    u_min = np.concatenate((
        np.full(m, u1_min),
        np.full(m, u2_min)
    ))

    u_current = np.concatenate((
        np.full(m, u[0]),
        np.full(m, u[1])
    ))


    # --------------------------------------------------------
    # Right-hand side
    # --------------------------------------------------------

    b_du_upper = du_max

    b_du_lower = -du_min

    b_u_upper = (
        u_max
        - u_current
    )

    b_u_lower = -(
        u_min
        - u_current
    )


    # --------------------------------------------------------
    # Complete b vector
    # --------------------------------------------------------

    b = np.concatenate([
        b_du_upper,
        b_du_lower,
        b_u_upper,
        b_u_lower
    ])


    # ========================================================
    # Solve constrained DMC QP using Hildreth
    # ========================================================

    U_opt, lam = hildreth_qp(
        H,
        f,
        G_constraints,
        b,
        max_iter=30,
        tol=1e-100,
        lambda0=None
    )


    # ========================================================
    # First optimal control move
    # ========================================================
    #
    # U_opt ordering:
    #
    # [du1(k)
    #  du1(k+1)
    #  ...
    #  du1(k+m-1)
    #  du2(k)
    #  du2(k+1)
    #  ...
    #  du2(k+m-1)]
    #
    # ========================================================

    du1 = U_opt[0]
    du2 = U_opt[m]

    new_du = np.array([
        du1,
        du2
    ])


    # ========================================================
    # Apply first control move
    # ========================================================

    u_previous = u.copy()

    u = u + new_du


    # ========================================================
    # Plant update
    # ========================================================

    def plant_step(
            u_current,
            u_previous,
            y_previous,
            g11,
            g12,
            g21,
            g22,
            du_history):

        du_current = (
            u_current
            - u_previous
        )

        all_du = du_history + [
            du_current.copy()
        ]

        dy1 = 0.0
        dy2 = 0.0

        for i, du in enumerate(
                reversed(all_du)
        ):

            index = min(
                i + 1,
                len(g11) - 1
            )

            previous_index = max(
                index - 1,
                0
            )

            # Impulse-response coefficients
            h11 = (
                g11[index]
                - g11[previous_index]
            )

            h12 = (
                g12[index]
                - g12[previous_index]
            )

            h21 = (
                g21[index]
                - g21[previous_index]
            )

            h22 = (
                g22[index]
                - g22[previous_index]
            )

            # y1
            dy1 += (
                h11 * du[0]
                + h12 * du[1]
            )

            # y2
            dy2 += (
                h21 * du[0]
                + h22 * du[1]
            )

        y_new = (
            y_previous
            + np.array([
                dy1,
                dy2
            ])
        )

        return y_new


    y = plant_step(
        u_current=u,
        u_previous=u_previous,
        y_previous=y,
        g11=x_1_unit_step_history,
        g12=x_2_unit_step_history,
        g21=x_3_unit_step_history,
        g22=x_4_unit_step_history,
        du_history=du_history
    )


    # ========================================================
    # Store histories
    # ========================================================

    du_history.append(
        new_du.copy()
    )

    du_applied_history.append(
        new_du.copy()
    )

    y_history.append(
        y.copy()
    )

    u_history.append(
        u.copy()
    )


# ============================================================
# 17. Convert histories to arrays
# ============================================================

y_history = np.asarray(
    y_history,
    dtype=float
)

u_history = np.asarray(
    u_history,
    dtype=float
)

du_applied_history = np.asarray(
    du_applied_history,
    dtype=float
)


# ============================================================
# 18. Plot outputs
# ============================================================

t = np.arange(
    len(y_history)
)

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)

plt.plot(
    t,
    y_history[:, 0],
    '--',
    label="DMC CV1 (y1)",
    linewidth=2
)

plt.plot(
    t,
    y_history[:, 1],
    '--',
    label="DMC CV2 (y2)",
    linewidth=2
)

plt.axhline(
    r1,
    linestyle=':',
    label="y1 reference"
)

plt.axhline(
    r2,
    linestyle=':',
    label="y2 reference"
)

plt.ylabel("Outputs")
plt.legend()
plt.grid()


# ============================================================
# 19. Plot manipulated variables
# ============================================================

plt.subplot(2, 1, 2)

plt.plot(
    t,
    u_history[:, 0],
    label="DMC MV1 (u1)",
    linewidth=2
)

plt.plot(
    t,
    u_history[:, 1],
    label="DMC MV2 (u2)",
    linewidth=2
)

plt.axhline(
    u1_min,
    linestyle=':'
)

plt.axhline(
    u1_max,
    linestyle=':'
)

plt.xlabel("Sample")
plt.ylabel("Inputs")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# 20. Print final values
# ============================================================

print()
print("Final output y1:", y_history[-1, 0])
print("Final output y2:", y_history[-1, 1])
print("Final input  u1:", u_history[-1, 0])
print("Final input  u2:", u_history[-1, 1])