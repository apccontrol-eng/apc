import numpy as np

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

