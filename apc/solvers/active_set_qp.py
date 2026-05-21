import numpy as np

def active_set_qp(H, f, G, b, x0=None, tol=1e-10, max_iter=100):
    """
    Active-Set Quadratic Programming Solver
    =======================================

    Solves the quadratic programming problem:

        minimize:
            (1/2) x^T H x + f^T x

        subject to:
            Gx <= b

    using a classical active-set method.

    -------------------------------------------------------------------
    Idea
    -------------------------------------------------------------------

    The algorithm maintains a working set of constraints assumed to be
    active (binding) at the solution.

    At every iteration:

        1. Solve the equality-constrained QP using the active set.
        2. Compute a search direction.
        3. If direction is small:
               - compute Lagrange multipliers
               - remove constraints with negative multipliers
        4. Otherwise:
               - move along the search direction
               - add blocking constraints

    The process repeats until the KKT conditions are satisfied.

    -------------------------------------------------------------------
    Parameters
    -------------------------------------------------------------------

    H : ndarray (n x n)
        Positive definite Hessian matrix.

    f : ndarray (n,)
        Linear cost vector.

    G : ndarray (m x n)
        Inequality constraint matrix.

    b : ndarray (m,)
        Inequality constraint vector.

    x0 : ndarray (n,), optional
        Initial feasible point.
        If None, uses zero vector.

    tol : float
        Numerical tolerance.

    max_iter : int
        Maximum number of iterations.

    -------------------------------------------------------------------
    Returns
    -------------------------------------------------------------------

    x : ndarray
        Optimal primal solution.

    lambda_full : ndarray
        Lagrange multipliers for all constraints.

    active_set : list
        Indices of active constraints at termination.

    -------------------------------------------------------------------
    References
    -------------------------------------------------------------------

    Nocedal & Wright
    Numerical Optimization

    Gill, Murray & Wright
    Practical Optimization
    """

    n = H.shape[0]
    m = G.shape[0]

    # ==============================================================
    # INITIAL FEASIBLE POINT
    # ==============================================================

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Feasibility check
    if np.any(G @ x - b > tol):
        raise ValueError("Initial point is not feasible")

    # Initial active set
    W = [i for i in range(m) if abs(G[i] @ x - b[i]) < tol]

    for iteration in range(max_iter):

        # ==========================================================
        # GRADIENT OF OBJECTIVE
        # ==========================================================

        g = H @ x + f

        # ==========================================================
        # BUILD ACTIVE CONSTRAINT MATRIX
        # ==========================================================

        if len(W) > 0:
            A = G[W]

            # KKT system
            KKT = np.block([
                [H, A.T],
                [A, np.zeros((len(W), len(W)))]
            ])

            rhs = -np.hstack([g, np.zeros(len(W))])

            sol = np.linalg.solve(KKT, rhs)

            p = sol[:n]
            lambda_w = sol[n:]

        else:
            # Unconstrained Newton step
            p = -np.linalg.solve(H, g)
            lambda_w = np.array([])

        # ==========================================================
        # TEST SEARCH DIRECTION
        # ==========================================================

        if np.linalg.norm(p) < tol:

            # ------------------------------------------------------
            # Check multipliers
            # ------------------------------------------------------

            if len(W) == 0:
                break

            min_lambda = np.min(lambda_w)

            # Optimality satisfied
            if min_lambda >= -tol:
                break

            # Remove most negative multiplier constraint
            idx_remove = np.argmin(lambda_w)
            del W[idx_remove]

        else:

            # ======================================================
            # COMPUTE STEP LENGTH
            # ======================================================

            alpha = 1.0
            blocking = -1

            for i in range(m):

                if i in W:
                    continue

                Gi = G[i]
                denom = Gi @ p

                # Constraint may become active
                if denom > tol:
                    alpha_i = (b[i] - Gi @ x) / denom

                    if alpha_i < alpha:
                        alpha = alpha_i
                        blocking = i

            # ======================================================
            # UPDATE PRIMAL VARIABLE
            # ======================================================

            x = x + alpha * p

            # Add blocking constraint
            if blocking >= 0:
                W.append(blocking)

    # ==============================================================
    # BUILD FULL MULTIPLIER VECTOR
    # ==============================================================

    lambda_full = np.zeros(m)

    if len(W) > 0:

        A = G[W]

        KKT = np.block([
            [H, A.T],
            [A, np.zeros((len(W), len(W)))]
        ])

        rhs = -np.hstack([
            H @ x + f,
            np.zeros(len(W))
        ])

        sol = np.linalg.solve(KKT, rhs)

        lambda_full[np.array(W)] = sol[n:]

    return x, lambda_full, W
