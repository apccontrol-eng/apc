import numpy as np


def kkt_qp(H, f, G, b, max_iter=100, tol=1e-10, active_set0=None):
    """
    Active-set KKT QP solver:

        min 0.5 x^T H x + f^T x
        s.t. Gx + b >= 0
    """

    n = H.shape[0]
    m = G.shape[0]

    H = H# + 1e-10 * np.eye(n)

    active = np.zeros(m, dtype=bool)
    if active_set0 is not None:
        active = active_set0.copy()

    x = np.zeros(n)

    for _ in range(max_iter):

        # --- build active constraint system ---
        G_A = G[active]
        b_A = b[active]

        k = G_A.shape[0]

        # --- solve KKT system ---
        if k == 0:
            x = -np.linalg.solve(H, f)
        else:
            KKT = np.block([
                [H, G_A.T],
                [G_A, np.zeros((k, k))]
            ])

            rhs = -np.concatenate([f, b_A])

            sol = np.linalg.solve(KKT, rhs)

            x = sol[:n]
            lam_A = sol[n:]

        # --- check feasibility ---
        c = G @ x + b

        violated = np.where(c < -tol)[0]

        if len(violated) > 0:
            # add most violated constraint
            i = violated[np.argmin(c[violated])]
            active[i] = True
            continue

        # --- check multipliers ---
        if k > 0:
            remove = False
            for idx, i in enumerate(np.where(active)[0]):
                if lam_A[idx] < -tol:
                    active[i] = False
                    remove = True
                    break

            if remove:
                continue

        break

    # final solve for consistency
    G_A = G[active]
    k = G_A.shape[0]

    if k == 0:
        x = -np.linalg.solve(H, f)
        lam = np.zeros(m)
    else:
        KKT = np.block([
            [H, G_A.T],
            [G_A, np.zeros((k, k))]
        ])

        rhs = -np.concatenate([f, b[active]])

        sol = np.linalg.solve(KKT, rhs)

        x = sol[:n]

        lam = np.zeros(m)
        lam[active] = sol[n:]

    return x, lam