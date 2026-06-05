import os
import numpy as np
import matplotlib.pyplot as plt
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
    Centers and scales a NumPy array.

    - If input is 2D: standardizes column-wise.
    - If input is 1D: standardizes the whole vector.

    Returns:
        standardized (np.ndarray)
        means (np.ndarray)
        stds (np.ndarray)
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Handle vector input
    if matrix.ndim == 1:
        mean = np.mean(matrix)
        std = np.std(matrix, ddof=1)

        if std == 0:
            std = 1.0

        standardized = (matrix - mean) / std

        return standardized, mean, std

    # Handle matrix input
    if matrix.ndim != 2:
        raise ValueError("Input must be a 1D vector or 2D matrix.")

    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0, ddof=1)

    # Prevent division by zero for constant columns
    stds = np.where(stds == 0, 1.0, stds)

    standardized = (matrix - means) / stds

    return standardized, means, stds


def plot_column_distributions_with_stats(matrix, bins=10, name=""):
    """
    Plots histograms of each column in the matrix to show value distributions,
    with mean and standard deviation lines and printed numeric labels.

    Parameters:
        matrix (np.ndarray): A 2D NumPy array.
        bins (int): Number of bins for histogram.
    """
    
    os.makedirs("figures", exist_ok=True)
    def save_fig(name):
        plt.tight_layout()
        plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    
    
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
    plt.grid(False)
    save_fig(name)
    plt.show()
    

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


def eigen_decomposition(matrix):
    """
    Performs eigen decomposition of a symmetric matrix.
    Returns sorted eigenvalues and eigenvectors (descending order).
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    return sorted_eigenvalues, sorted_eigenvectors

def pca_from_correlation(corr_matrix):
    """
    Performs PCA given a correlation matrix using eigen decomposition.

    Parameters:
        corr_matrix (np.ndarray): A square correlation matrix.

    Returns:
        explained_variance (np.ndarray): Eigenvalues (variances along PCs).
        principal_components (np.ndarray): Eigenvectors (PC directions).
    """
    if not isinstance(corr_matrix, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("Input matrix must be square (correlation matrix).")

    eigenvalues, eigenvectors = eigen_decomposition(corr_matrix)
    
    # Normalize eigenvalues to get variance explained (optional)
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

    return eigenvalues, eigenvectors, explained_variance_ratio


def pca_biplot_with_t2(scores, loadings, explained_variance_ratio, T2_thresh,
                       labels=None, feature_names=None, scale_scores=1.0, scale_loadings=1.0, message=""):
    """
    Plots a PCA biplot including Hotelling's T² threshold as an ellipse.
    """
    
    os.makedirs("figures", exist_ok=True)
    def save_fig(name):
        plt.tight_layout()
        plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    
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
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    save_fig(message)
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

def plot_t2_q(T2, Q, threshold_T2=None, threshold_Q=None, name=""):
    """
    Plots Hotelling's T² and Q residuals with optional thresholds.
    """
    
    os.makedirs("figures", exist_ok=True)
    def save_fig(name):
        plt.tight_layout()
        plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].scatter(range(len(T2)), T2, color='steelblue')
    axs[0].axhline(threshold_T2, color='red', linestyle='--', label='Threshold' if threshold_T2 else None)
    axs[0].set_title("Hotelling's T²")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("T²")
    axs[0].grid(False)

    axs[1].scatter(range(len(Q)), Q, color='darkgreen')
    axs[1].axhline(threshold_Q, color='red', linestyle='--', label='Threshold' if threshold_Q else None)
    axs[1].set_title("Q Residuals (SPE)")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Q")
    axs[1].grid(False)

    plt.tight_layout()
    save_fig(name)
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

    Q_crit = theta1 * (1 + (z_alpha * np.sqrt(2 * theta2 * h0**2) / theta1) + (theta2 * h0 * (h0 - 1)) / (theta1**2))**(1 / h0)
    
    #print(" h0: " , h0)
    #print(" z_alpha: " , z_alpha)
    #print(" Q_crit: " , Q_crit)
        
    return Q_crit





