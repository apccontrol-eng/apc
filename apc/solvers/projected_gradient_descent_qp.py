import numpy as np

def project_onto_polyhedron(x, G, b, max_proj_iter=50):
    """
    ===========================================================================
    sequential projection onto halfspaces:
        Gx <= b

    uses cyclic orthogonal projections.
    ===========================================================================
    """

    x_proj = x.copy()

    for _ in range(max_proj_iter):

        violated = False

        for i in range(G.shape[0]):

            g = G[i]
            violation = g @ x_proj - b[i]

            if violation > 0:
                violated = True

                x_proj = x_proj - (
                    violation / (np.dot(g, g) + 1e-12)
                ) * g

        if not violated:
            break

    return x_proj


def projected_gradient_descent_qp(
    H,
    f,
    G,
    b,
    x0=None,
    alpha=1e-1,
    max_iter=1000,
    tol=1e-8
):
    """
    ===========================================================================
    Projected Gradient Descent QP Solver
    
    ===========================================================================
    Reference

    Stephen J. Wright
    "Primal-Dual Interior-Point Methods"
    SIAM, 1997
    
    ===========================================================================
    """
    n = H.shape[0]

    if x0 is None:
        x = np.zeros(n)
    else:
       x = x0.copy()

    # =========================================================================
    # ensure initial feasibility
    x = project_onto_polyhedron(x, G, b)

    for _ in range(max_iter):

        # =========================================================================        
        # gradient of quadratic objective
        grad = H @ x + f

        # =========================================================================
        # gradient step
        y = x - alpha * grad

        # =========================================================================
        # projection step
        x_new = project_onto_polyhedron(y, G, b)

        # =========================================================================
        # checking convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return x

