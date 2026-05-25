import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emil/Documents/GitHub/apc')
from apc.calibration.PCA_eigen import *
from apc.solvers.hildreth_qp import hildreth_qp
from apc.filters.kalman_filter import kalman_filter


def column_stats(matrix):
    """
    Computes column-wise means and standard deviations of a matrix.

    Parameters:
        matrix (np.ndarray): A 2D NumPy array.

    Returns:
        col_means (np.ndarray): 1D array of column means.
        col_stds (np.ndarray): 1D array of column standard deviations.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    col_means = np.mean(matrix, axis=0)
    col_stds = np.std(matrix, axis=0, ddof=1)  # Use ddof=1 for sample std dev

    return col_means, col_stds


def standardize_matrix(matrix):
    """
    Centers and scales a matrix column-wise using mean and standard deviation.

    Parameters:
        matrix (np.ndarray): A 2D NumPy array.

    Returns:
        standardized_matrix (np.ndarray): The standardized matrix.
        means (np.ndarray): Column means used for centering.
        stds (np.ndarray): Column standard deviations used for scaling.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1)

    # Prevent division by zero for constant columns
    stds[stds == 0] = 1.0

    standardized_matrix = (matrix - means) / stds

    return standardized_matrix, means, stds



def plot_column_distributions_with_stats(matrix, bins=10):
    """
    Plots histograms of each column in the matrix to show value distributions,
    with mean and standard deviation lines and printed numeric labels.

    Parameters:
        matrix (np.ndarray): A 2D NumPy array.
        bins (int): Number of bins for histogram.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D matrix.")

    n_rows, n_cols = matrix.shape
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)

    for col in range(n_cols):
        col_data = matrix[:, col]
        mean = np.mean(col_data)
        std = np.std(col_data, ddof=1)

        ax = axes[0, col]
        ax.hist(col_data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(mean, color='green', linestyle='--', label=f'Mean = {mean:.2f}')
        ax.axvline(mean + std, color='red', linestyle=':', label=f'+1 Std = {mean + std:.2f}')
        ax.axvline(mean - std, color='red', linestyle=':', label=f'-1 Std = {mean - std:.2f}')

        ax.set_title(f'Column {col} Distribution')
        ax.set_xlabel('Value')
        if col == 0:
            ax.set_ylabel('Frequency')

        # Add mean and std as text inside the plot
        ax.text(0.95, 0.95, f"μ = {mean:.2f}\nσ = {std:.2f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.legend()

    plt.tight_layout()
    plt.show()
    
    
    
    


'''
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
'''


# =========================
# BUILD PREDICTION MATRICES
# =========================
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

# =========================
# BLOCK DIAGONAL MATRIX
# =========================
def block_diag(Q, N):
    return np.kron(np.eye(N), Q)

# =========================
# BUILD QP MATRICES
# =========================
def build_qp(Sx, Su, Q_bar, R_bar, x0):
    H = Su.T @ Q_bar @ Su + R_bar
    f = Su.T @ Q_bar @ (Sx @ x0)
    return H, f

# =========================
# INPUT CONSTRAINTS
# =========================
def input_constraints(N, m, umin, umax):
    G = np.vstack((np.eye(N*m), -np.eye(N*m)))
    b = np.hstack((np.tile(umax, N), -np.tile(umin, N)))
    return G, b

# =========================
# EXAMPLE MPC LOOP
# =========================

# System (example: double integrator)
A = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.0, 0.0, 0.7]
])

B = np.array([[0.1], [0.05], [0.02]])

n = A.shape[0]
m = B.shape[1]
N = 10  # horizon

# Cost
Q = np.eye(n)
R = 0.1 * np.eye(m)

Q_bar = block_diag(Q, N)
R_bar = block_diag(R, N)

# Constraints
umin = np.array([-1.0])
umax = np.array([1.0])
G, b = input_constraints(N, m, umin, umax)

# Initial state
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
    
    # Save control
    u_history.append(u.copy())

    # System update
    if k == 40:
        bias = np.array([0.2, -0.3, -0.4])
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3) + bias
    else:
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3)
    
    #no kalman filter

    x = A @ x + B @ u + added_noise
    x_history.append(x.copy())


    #kalman filter
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

# Convert to arrays
x_history = np.array(x_history)
u_history = np.array(u_history)


t = np.arange(len(x_history[:,0]))  # time index

plt.figure(figsize=(14, 8))  # bigger and clearer

# ----------------------------
# OUTPUT COMPARISON
# ----------------------------

plt.subplot(2,1,1)

plt.plot(t, x_history[:,0], '--', label="state 1", linewidth=2)
plt.plot(t, x_history[:,1], '--', label="state 2", linewidth=2)
plt.plot(t, x_history[:,2], '--', label="state 3", linewidth=2)

plt.title("MPC controller input and states with added Gaussian noise", fontsize=14)
plt.ylabel("Output", fontsize=12)
plt.legend()
plt.grid()

# ----------------------------
# INPUT SIGNAL
# ----------------------------
plt.subplot(2,1,2)

plt.plot(t[0:-1], u_history[:,0], label="MPC Input (u)", linewidth=2)

plt.xlabel("Time step", fontsize=12)
plt.ylabel("MPC Input", fontsize=12)
plt.title("MPC Input Signal", fontsize=14)
plt.grid()

plt.tight_layout()
plt.show()




'''
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
'''

# MPC RESULTS

mpc_states = x_history
mpc_controls = u_history

print("mpc_states : " , mpc_states)
print("mpc_controls : " , mpc_controls)
#print("mpc_states.shape : " , mpc_states.shape)
#print("mpc_controls.shape : " , mpc_controls.shape)


monitor_calibration_window = range(40,200)

calibration_data = mpc_states.T[monitor_calibration_window,:]

calibration_col_means_states, calibration_col_stds_states = column_stats(calibration_data)

#print("calibration_col_means_states : ", calibration_col_means_states)
#print("calibration_col_stds_states : ", calibration_col_stds_states)

calibration_standardized_matrix_states, calibration_means_states, calibration_stds_states = standardize_matrix(calibration_data)

#print("calibration_standardized_matrix_states : ", calibration_standardized_matrix_states)
#print("calibration_means_states : ", calibration_means_states)
#print("calibration_stds_states : ", calibration_stds_states)

n = calibration_data.shape[0]

calibration_state_covariance_matrix = ( 1/(n-1) ) * calibration_standardized_matrix_states.T @ calibration_standardized_matrix_states

print("calibration_state_covariance_matrix : ", calibration_state_covariance_matrix)



plot_column_distributions_with_stats(calibration_data, bins = 50)
plot_column_distributions_with_stats(calibration_standardized_matrix_states, bins = 50)


