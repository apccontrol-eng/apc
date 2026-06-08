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
        
    References:
    
        Kadlec, P., Gabrys, B., & Strandt, S. (2009). 
        Data-driven soft sensors in the process industry. 
        Computers & Chemical Engineering, 33 (4), 795–814. 
        https://doi.org/10.1016/j.compchemeng.2008.12.012
        
        Kadlec, P., Grbić, R., & Gabrys, B. (2011). 
        Review of adaptation mechanisms for data-driven soft sensors. 
        Computers & Chemical Engineering, 35 (1), 1–24. 
        https://doi.org/10.1016/j.compchemeng.2010.07.034
        
        Qin, S. J. (1998). 
        Recursive PLS algorithms for adaptive data modeling. 
        Computers & Chemical Engineering, 22 (4), 503–514. 
        https://doi.org/10.1016/S0098-1354(97)00262-7
        
        Geladi, P., & Kowalski, B. R. (1986). 
        Partial least-squares regression: A tutorial. 
        Analytica Chimica Acta, 185, 1–17. 
        https://doi.org/10.1016/0003-2670(86)80028-9    
        
        Wise, B. M., & Gallagher, N. B. (1996). 
        The process chemometrics approach to process monitoring and fault detection. 
        Journal of Process Control, 6 (6), 329–348. 
        https://doi.org/10.1016/0959-1524(96)00009-1
        
    """
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n, px = X.shape
    _, py = Y.shape

    if n_components > px:
        raise("Latent variables cannot exceed the number of columns in X")

    # initialize
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


