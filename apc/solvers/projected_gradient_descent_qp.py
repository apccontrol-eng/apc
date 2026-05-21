import numpy as np

def project_onto_polyhedron(x, G, b, max_proj_iter=50):
    """
    Sequential projection onto halfspaces:

        Gx <= b

    Uses cyclic orthogonal projections.
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
    Projected Gradient Descent QP Solver
    """

    n = H.shape[0]

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # Ensure initial feasibility
    x = project_onto_polyhedron(x, G, b)

    for _ in range(max_iter):

        # Gradient of quadratic objective
        grad = H @ x + f

        # Gradient step
        y = x - alpha * grad

        # Projection step
        x_new = project_onto_polyhedron(y, G, b)

        # Convergence test
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new

    return x
