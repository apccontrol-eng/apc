import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/emil/Documents/GitHub/apc')
from apc.solvers.hildreth_qp import hildreth_qp
from apc.calibration.PCA_eigen import eigen_decomposition
from apc.calibration.PCA_eigen import pca_from_correlation
from apc.filters.kalman_filter import kalman_filter
from matplotlib.patches import Ellipse
from scipy.stats import f as f_
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score


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

sim_steps = 300
np.random.seed(42)

for k in range(sim_steps):

    Sx, Su = build_prediction_matrices(A, B, N)
    H, f = build_qp(Sx, Su, Q_bar, R_bar, x)

    U_opt, lam = hildreth_qp(H, f, G, b, lambda0=None)
    u = U_opt[:m]
    
    # Save control
    u_history.append(u.copy())

    # System update
    if k == 150:
        bias = np.array([0.5, -0.5, -0.9])
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

#print("mpc_states : " , mpc_states)
#print("mpc_controls : " , mpc_controls)
print("mpc_states.shape : " , mpc_states.shape)
print("mpc_controls.shape : " , mpc_controls.shape)

process_data = np.hstack((mpc_states[:-1,:], mpc_controls))

monitor_calibration_window = range(50,100)

calibration_data = process_data[monitor_calibration_window,:]

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



plot_column_distributions_with_stats(calibration_data, bins = 10)
plot_column_distributions_with_stats(calibration_standardized_matrix_states, bins = 10)



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







def pca_biplot_with_t2(scores, loadings, explained_variance_ratio, T2_thresh,
                       labels=None, feature_names=None, scale_scores=1.0, scale_loadings=1.0):
    """
    Plots a PCA biplot including Hotelling's T² threshold as an ellipse.
    """
    plt.figure(figsize=(8, 6))

    # Plot scores
    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = labels == label
            plt.scatter(scores[idx, 0] * scale_scores, scores[idx, 1] * scale_scores, label=f"Class {label}", alpha=0.7)
        plt.legend()
    else:
        plt.scatter(scores[:, 0] * scale_scores, scores[:, 1] * scale_scores, alpha=0.7, color='gray')

    # Plot loadings as red arrows
    for i in range(loadings.shape[0]):
        plt.arrow(0, 0,
                  loadings[i, 0] * scale_loadings,
                  loadings[i, 1] * scale_loadings,
                  color='red', alpha=0.8, head_width=0.03)
        name = feature_names[i] if feature_names is not None else f"Var{i+1}"
        plt.text(loadings[i, 0] * scale_loadings * 1.1,
                 loadings[i, 1] * scale_loadings * 1.1,
                 name, color='red', ha='center', va='center', fontsize=9)

    # Variance explained in axis labels
    pc1_var = explained_variance_ratio[0] * 100
    pc2_var = explained_variance_ratio[1] * 100
    plt.xlabel(f"PC1 ({pc1_var:.1f}% in calibration)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}% in calibration)")

    # Add T² confidence ellipse (assuming PC scores are uncorrelated & standardized)
    lambda1 = np.var(scores[:, 0])  # ≈ eigenvalue 1
    lambda2 = np.var(scores[:, 1])  # ≈ eigenvalue 2
    width = 2 * np.sqrt(T2_thresh * lambda1)
    height = 2 * np.sqrt(T2_thresh * lambda2)
    ellipse = Ellipse(xy=(0, 0), width=width, height=height,
                      edgecolor='blue', fc='None', lw=2, linestyle='--', label="T² calibration threshold")
    plt.gca().add_patch(ellipse)

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title("PCA Biplot with Hotelling's T² Ellipse")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    


def compute_t2_q(data, scores, loadings, num_components):
    """
    Computes Hotelling's T² and Q residual statistics for PCA.

    Parameters:
        data (np.ndarray): Standardized data (Z-scores).
        scores (np.ndarray): PCA scores (samples x PCs).
        loadings (np.ndarray): PCA loadings (features x PCs).
        num_components (int): Number of PCs to include in model.

    Returns:
        T2 (np.ndarray): Hotelling’s T² statistic per sample.
        Q (np.ndarray): Q residual statistic per sample.
    """
    # Truncate to selected PCs
    T = scores[:, :num_components]
    P = loadings[:, :num_components]

    # Hotelling's T² = sum of squared standardized scores
    T2 = np.sum((T / np.std(T, axis=0))**2, axis=1)

    # Reconstruct data from selected PCs
    X_hat = T @ P.T
    residuals = data - X_hat
    Q = np.sum(residuals**2, axis=1)

    return T2, Q

def plot_t2_q(T2, Q, threshold_T2=None, threshold_Q=None):
    """
    Plots Hotelling's T² and Q residuals with optional thresholds.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].scatter(range(len(T2)), T2, color='steelblue')
    axs[0].axhline(threshold_T2, color='red', linestyle='--', label='Threshold' if threshold_T2 else None)
    axs[0].set_title("Hotelling's T²")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("T²")
    axs[0].grid(True)

    axs[1].scatter(range(len(Q)), Q, color='darkgreen')
    axs[1].axhline(threshold_Q, color='red', linestyle='--', label='Threshold' if threshold_Q else None)
    axs[1].set_title("Q Residuals (SPE)")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Q")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def hotelling_t2_threshold(n_samples, n_components, alpha=0.95):
    """Computes Hotelling's T² critical value using F-distribution."""
    F_crit = f_.ppf(alpha, dfn=n_components, dfd=n_samples - n_components)
    T2_crit = (n_components * (n_samples**2 - 1)) / (n_samples * (n_samples - n_components)) * F_crit
    return T2_crit



def q_residual_threshold(eigenvalues, n_components, alpha=0.95):
    """Computes Q (SPE) critical value using Jackson’s method (chi-square approx)."""
    residual_eigvals = eigenvalues[n_components:]
    
    #print(" eigenvalues: " , eigenvalues)
    #print(" residual_eigvals: " , residual_eigvals)
    
    theta1 = np.sum(residual_eigvals)
    theta2 = np.sum(residual_eigvals**2)
    theta3 = np.sum(residual_eigvals**3)

    #print(" theta1: " , theta1)
    #print(" theta2: " , theta2)
    #print(" theta3: " , theta3)

    h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
    z_alpha = norm.ppf(alpha)

    Q_crit = theta1 * ((z_alpha * np.sqrt(2 * theta2) / theta1) + (theta2 * h0 * (h0 - 1)) / (theta1**2) + 1)**(1 / h0)
    
    #print(" h0: " , h0)
    #print(" z_alpha: " , z_alpha)
    #print(" Q_crit: " , Q_crit)
        
    return Q_crit


calibration_standardized_matrix_states, calibration_means_states, calibration_stds_states = standardize_matrix(calibration_data)

#print("calibration_standardized_matrix_states : ", calibration_standardized_matrix_states)
#print("calibration_means_states : ", calibration_means_states)
#print("calibration_stds_states : ", calibration_stds_states)

n = calibration_data.shape[0]

calibration_state_covariance_matrix = ( 1/(n-1) ) * calibration_standardized_matrix_states.T @ calibration_standardized_matrix_states

#print("calibration_state_covariance_matrix : ", calibration_state_covariance_matrix)

corr = calibration_state_covariance_matrix

eigvals, eigvecs, var_ratio = pca_from_correlation(corr)

#print("Eigenvalues (explained variance):", eigvals)
#print("Explained variance ratio:", var_ratio)
#print("Principal components (eigenvectors):\n", eigvecs)

# eigenvector = loading = P

calibration_P = eigvecs
calibration_T = calibration_standardized_matrix_states @ calibration_P


# k principal components
k = 2
calibration_T2, calibration_Q = compute_t2_q(data = calibration_standardized_matrix_states, scores = calibration_T, loadings = calibration_P, num_components = k)

n = calibration_standardized_matrix_states.shape[0]

calibration_T2_thresh = hotelling_t2_threshold(n_samples=n, n_components=k)
calibration_Q_thresh = q_residual_threshold(eigenvalues=eigvals, n_components=k)

#plot_t2_q(calibration_T2, Q, threshold_T2=np.percentile(calibration_T2, 95), threshold_Q=np.percentile(calibration_Q, 95))

plot_t2_q(calibration_T2, calibration_Q, threshold_T2 = calibration_T2_thresh, threshold_Q = calibration_Q_thresh)


pca_biplot_with_t2(
    scores = calibration_T,
    loadings = calibration_P,
    explained_variance_ratio = var_ratio,
    T2_thresh = calibration_T2_thresh,
    feature_names = ['X1', 'X2', 'X3', 'X4']
)





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

print("THIS IS TESTING THE PCA MONITORING MODEL:")
print("new data is autoscaled with calibration means and standard deviations")
print("the new autoscaled data is projected to the calibration data principal component space")

# testing PCA model
def scale_with_reference(X, mean_ref, std_ref):
    """
    Scales input matrix X using precomputed mean and std (from training data).

    Parameters:
        X: np.ndarray, shape (n_samples, n_features) — new data
        mean_ref: np.ndarray, shape (n_features,) — training set means
        std_ref: np.ndarray, shape (n_features,) — training set std deviations

    Returns:
        X_scaled: scaled version of X
    """
    mean_ref = np.asarray(mean_ref).ravel()
    std_ref = np.asarray(std_ref).ravel()

    if X.shape[1] != mean_ref.shape[0]:
        raise ValueError("Feature dimension mismatch between X and reference mean/std.")

    # Avoid divide by zero
    std_ref_safe = np.where(std_ref == 0, 1.0, std_ref)

    X_scaled = (X - mean_ref) / std_ref_safe
    return X_scaled


monitor_calibration_window = range(100,299)
monitored_data = process_data[monitor_calibration_window,:]

calibration_autoscaled_monitored_data = scale_with_reference(X = monitored_data, 
                                                             mean_ref = calibration_means_states, 
                                                             std_ref = calibration_stds_states)

plot_column_distributions_with_stats(monitored_data, bins = 10)
plot_column_distributions_with_stats(calibration_autoscaled_monitored_data, bins = 10)


monitored_T = calibration_autoscaled_monitored_data @ calibration_P
monitored_X_hat = monitored_T @ calibration_P

# k components
k = 2
monitored_T2, monitored_Q = compute_t2_q(data = calibration_autoscaled_monitored_data, scores = monitored_T, loadings = calibration_P, num_components = k)


plot_t2_q(monitored_T2, monitored_Q, threshold_T2 = calibration_T2_thresh, threshold_Q = calibration_Q_thresh)

# tarkista Q_thresh kaava

pca_biplot_with_t2(
    scores = monitored_T,
    loadings = calibration_P,
    explained_variance_ratio = var_ratio,
    T2_thresh = calibration_T2_thresh,
    feature_names = ['X1', 'X2', 'X3', 'X4']
)









