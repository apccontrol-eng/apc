import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add local project path
sys.path.append('/Users/emil/Documents/GitHub/apc')

from apc.solvers.hildreth_qp import hildreth_qp

from apc.calibration.PCA_eigen import eigen_decomposition
from apc.calibration.PCA_eigen import pca_from_correlation
from apc.calibration.PCA_eigen import scale_with_reference
from apc.calibration.PCA_eigen import column_stats
from apc.calibration.PCA_eigen import standardize_matrix
from apc.calibration.PCA_eigen import plot_column_distributions_with_stats
from apc.calibration.PCA_eigen import pca_biplot_with_t2
from apc.calibration.PCA_eigen import compute_t2_q
from apc.calibration.PCA_eigen import plot_t2_q
from apc.calibration.PCA_eigen import hotelling_t2_threshold
from apc.calibration.PCA_eigen import q_residual_threshold

from apc.filters.kalman_filter import kalman_filter


os.makedirs("figures", exist_ok=True)
def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")


# ============================================================
# BUILD PREDICTION MATRICES
# ============================================================
def build_prediction_matrices(A, B, N):
    """
    Build MPC prediction matrices Sx and Su.

    Sx maps initial state evolution over the horizon.
    Su maps control inputs over the horizon.

    Returns:
        Sx: state prediction matrix
        Su: input prediction matrix
    """
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


# ============================================================
# BLOCK DIAGONAL MATRIX
# ============================================================
def block_diag(Q, N):
    """
    Create block diagonal matrix using Kronecker product.

    Equivalent to:
        diag(Q, Q, ..., Q) repeated N times
    """
    return np.kron(np.eye(N), Q)


# ============================================================
# BUILD QP MATRICES
# ============================================================
def build_qp(Sx, Su, Q_bar, R_bar, x0):
    """
    Construct quadratic programming matrices for MPC.

    Objective:
        min 1/2 U^T H U + f^T U

    Returns:
        H: Hessian matrix
        f: linear term
    """
    H = Su.T @ Q_bar @ Su + R_bar
    f = Su.T @ Q_bar @ (Sx @ x0)
    return H, f


# ============================================================
# INPUT CONSTRAINTS
# ============================================================
def input_constraints(N, m, umin, umax):
    """
    Build inequality constraints for input bounds:
        umin <= u <= umax
    """
    G = np.vstack((np.eye(N*m), -np.eye(N*m)))
    b = np.hstack((np.tile(umax, N), -np.tile(umin, N)))
    return G, b


# ============================================================
# SYSTEM DEFINITION
# ============================================================

# System matrices (3-state linear system)
A = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.0, 0.0, 0.7]
])

B = np.array([[0.1], [0.05], [0.02]])

n = A.shape[0]
m = B.shape[1]

N = 10  # MPC horizon


# Cost matrices
Q = np.eye(n)
R = 0.1 * np.eye(m)

Q_bar = block_diag(Q, N)
R_bar = block_diag(R, N)


# Input constraints
umin = np.array([-1.0])
umax = np.array([1.0])
G, b = input_constraints(N, m, umin, umax)


# Initial state
x = np.array([2.0, 0.0, 5.0])

x_history = [x.copy()]
u_history = []

sim_steps = 300
np.random.seed(42)


# ============================================================
# MPC SIMULATION LOOP
# ============================================================
for k in range(sim_steps):

    # Build prediction model
    Sx, Su = build_prediction_matrices(A, B, N)

    # Build QP problem
    H, f = build_qp(Sx, Su, Q_bar, R_bar, x)

    # Solve QP using Hildreth algorithm
    U_opt, lam = hildreth_qp(H, f, G, b, lambda0=None)
    u = U_opt[:m]

    u_history.append(u.copy())

    # Disturbance + fault injection
    if k == 150:
        bias = np.array([0.5, -0.5, -0.9])
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3) + bias
    else:
        noise_std = 0.01
        added_noise = noise_std * np.random.randn(3)

    # System update
    x = A @ x + B @ u + added_noise
    x_history.append(x.copy())


# Convert to arrays
x_history = np.array(x_history)
u_history = np.array(u_history)

t = np.arange(len(x_history[:,0]))


# ============================================================
# PLOT RESULTS
# ============================================================
plt.figure(figsize=(14, 8))

# States
plt.subplot(2,1,1)
plt.plot(t, x_history[:,0], '--', label="state 1")
plt.plot(t, x_history[:,1], '--', label="state 2")
plt.plot(t, x_history[:,2], '--', label="state 3")
plt.title("MPC states with noise and disturbance")
plt.legend()
plt.grid()

# Inputs
plt.subplot(2,1,2)
plt.plot(t[:-1], u_history[:,0], label="control input")
plt.title("MPC input signal")
plt.grid()

plt.tight_layout()
save_fig("MPC_states_controls")
plt.show()


# ============================================================
# PCA PROCESS MONITORING
# ============================================================

mpc_states = x_history
mpc_controls = u_history

# Combine states and inputs
process_data = np.hstack((mpc_states[:-1,:], mpc_controls))

# Calibration window selection
monitor_calibration_window = range(50,100)
calibration_data = process_data[monitor_calibration_window,:]

# Compute statistics
calibration_col_means_states, calibration_col_stds_states = column_stats(calibration_data)

# Standardize calibration data
calibration_standardized_matrix_states, calibration_means_states, calibration_stds_states = standardize_matrix(calibration_data)

# Covariance matrix
n = calibration_data.shape[0]
calibration_state_covariance_matrix = (1/(n-1)) * calibration_standardized_matrix_states.T @ calibration_standardized_matrix_states

print("calibration_state_covariance_matrix : ", calibration_state_covariance_matrix)


# Distribution plots
print("Calibration data before autoscaling:")
plot_column_distributions_with_stats(calibration_data, bins=10, name="Calibration data before autoscaling")
print("Calibration data after autoscaling:")
plot_column_distributions_with_stats(calibration_standardized_matrix_states, bins=10, name="Calibration data after autoscaling")


# ============================================================
# PCA MODEL
# ============================================================

calibration_standardized_matrix_states, calibration_means_states, calibration_stds_states = standardize_matrix(calibration_data)

n = calibration_data.shape[0]

calibration_state_covariance_matrix = (1/(n-1)) * calibration_standardized_matrix_states.T @ calibration_standardized_matrix_states

corr = calibration_state_covariance_matrix

eigvals, eigvecs, var_ratio = pca_from_correlation(corr)

# PCA model
calibration_P = eigvecs
calibration_T = calibration_standardized_matrix_states @ calibration_P

k = 2  # number of components

calibration_T2, calibration_Q = compute_t2_q(
    data=calibration_standardized_matrix_states,
    scores=calibration_T,
    loadings=calibration_P,
    num_components=k
)

# Thresholds
n = calibration_standardized_matrix_states.shape[0]
calibration_T2_thresh = hotelling_t2_threshold(n_samples=n, n_components=k)
calibration_Q_thresh = q_residual_threshold(eigenvalues=eigvals, n_components=k)

# Monitoring plots
plot_t2_q(calibration_T2, calibration_Q, threshold_T2=calibration_T2_thresh, threshold_Q=calibration_Q_thresh, name="Calibration T2 and SPE plot")


# PCA biplot
pca_biplot_with_t2(
    scores=calibration_T,
    loadings=calibration_P,
    explained_variance_ratio=var_ratio,
    T2_thresh=calibration_T2_thresh,
    feature_names=['X1', 'X2', 'X3', 'U1'], message="Calibration data PCA biplot"
)


# Loading plots
variables = ['X1', 'X2', 'X3', 'U1']
loadings = calibration_P

p1 = loadings[:, 0]
p2 = loadings[:, 1]

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].bar(variables, p1)
ax[0].set_title('Loading 1')
ax[0].grid(True, alpha=0.3)

ax[1].bar(variables, p2)
ax[1].set_title('Loading 2')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# ONLINE MONITORING
# ============================================================

print("THIS IS TESTING THE PCA MONITORING MODEL:")
print("new data is autoscaled with calibration means and standard deviations")

monitor_calibration_window = range(100,299)
monitored_data = process_data[monitor_calibration_window,:]

# scale using calibration statistics
calibration_autoscaled_monitored_data = scale_with_reference(
    X=monitored_data,
    mean_ref=calibration_means_states,
    std_ref=calibration_stds_states
)

print("Monitored new data before autoscaling with calibration means and stds:")
plot_column_distributions_with_stats(monitored_data, bins=10, name="Monitored new data before autoscaling")

print("Monitored new data after autoscaling with calibration means and stds:")
plot_column_distributions_with_stats(calibration_autoscaled_monitored_data, bins=10, name="Monitored new data after autoscaling")


# PCA projection
monitored_T = calibration_autoscaled_monitored_data @ calibration_P
monitored_X_hat = monitored_T @ calibration_P

k = 2

monitored_T2, monitored_Q = compute_t2_q(
    data=calibration_autoscaled_monitored_data,
    scores=monitored_T,
    loadings=calibration_P,
    num_components=k
)

plot_t2_q(monitored_T2, monitored_Q, threshold_T2=calibration_T2_thresh, threshold_Q=calibration_Q_thresh, name="Monitoring data T2 and SPE plot")

pca_biplot_with_t2(
    scores=monitored_T,
    loadings=calibration_P,
    explained_variance_ratio=var_ratio,
    T2_thresh=calibration_T2_thresh,
    feature_names=['X1', 'X2', 'X3', 'U1'], message="Monitoring data PCA biplot"
)


# ============================================================
# FAULT DIAGNOSIS (CONTRIBUTION PLOTS)
# ============================================================

print("Selecting one of the samples that is out of spec:")
i_fault = 150

variables = ['X1', 'X2', 'X3', 'U1']
contrib = calibration_autoscaled_monitored_data[i_fault, :] * calibration_P[:, 0] * 1/np.sqrt(eigvals[0])

plt.figure(figsize=(10,4))
plt.bar(range(len(contrib)), contrib)
plt.axhline(0, color='k')
plt.xticks(range(len(contrib)), variables)
plt.title(f'Sample {i_fault} contribution to PC1')
save_fig(f'Sample {i_fault} contribution to PC1')
plt.show()

contrib = calibration_autoscaled_monitored_data[i_fault, :] * calibration_P[:, 1] * 1/np.sqrt(eigvals[1])

plt.figure(figsize=(10,4))
plt.bar(range(len(contrib)), contrib)
plt.axhline(0, color='k')
plt.xticks(range(len(contrib)), variables)
plt.title(f'Sample {i_fault} contribution to PC2')
save_fig(f'Sample {i_fault} contribution to PC2')
plt.show()






