import numpy as np

def pls_nipals(X, Y, n_components):
    """
    Batch-wise PLS (NIPALS) algorithm.
    Assumes X and Y are zero-mean and unit-variance.

    Parameters:
        X: np.ndarray of shape (n_samples, n_features)
        Y: np.ndarray of shape (n_samples,) or (n_samples, n_targets)
        n_components: number of PLS components to extract

    Returns:
        T: X scores
        U: Y scores
        P: X loadings
        Q: Y loadings
        W: X weights
        B: Inner model coefficients (uᵗt / tᵗt)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n, px = X.shape
    _, py = Y.shape

    # Initialize
    E = X.copy()
    F = Y.copy()

    T = np.zeros((n, n_components))
    U = np.zeros((n, n_components))
    P = np.zeros((px, n_components))
    Q = np.zeros((py, n_components))
    W = np.zeros((px, n_components))
    B = np.zeros((n_components,))

    for h in range(n_components):
        u = F[:, 0].copy().reshape(-1, 1)

        for _ in range(100):
            w = E.T @ u / (u.T @ u)
            w /= np.linalg.norm(w)
            t = E @ w
            q = F.T @ t / (t.T @ t)
            q /= np.linalg.norm(q)
            u_new = F @ q

            if np.allclose(u, u_new, atol=1e-10):
                break
            u = u_new

        p = E.T @ t / (t.T @ t)
        b = float((u.T @ t) / (t.T @ t))

        E = E - t @ p.T
        F = F - b * t @ q.T

        T[:, h] = t.ravel()
        U[:, h] = u.ravel()
        P[:, h] = p.ravel()
        Q[:, h] = q.ravel()
        W[:, h] = w.ravel()
        B[h] = b
    
    B = np.diag(B)
    W_star = W@np.linalg.inv(P.T@W)
    
    return T, U, P, Q, W, B, W_star


