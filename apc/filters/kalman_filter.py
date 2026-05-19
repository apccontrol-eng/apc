import numpy as np

def kalman_filter(A, B, C, D, u, x, P, y, Q=None, R=None):
    """
    Linear Kalman Filter

    System:
        x_k+1 = A x_k + B u_k + w
        y_k   = C x_k + D u_k + v

    Inputs
    ------
    A : State transition matrix
    B : Control input matrix
    C : Measurement matrix
    D : Feedthrough matrix
    u : Control input vector
    x : Previous state estimate (mean)
    P : Previous covariance matrix
    y : Measurement vector
    Q : Process noise covariance
    R : Measurement noise covariance

    Returns
    -------
    x_new : Updated state estimate
    P_new : Updated covariance
    """

    x = np.asarray(x).reshape(-1)
    u = np.asarray(u).reshape(-1)
    y = np.asarray(y).reshape(-1)

    n = A.shape[0]
    m = C.shape[0]

    # Default noise covariances
    if Q is None:
        Q = 0.5 * np.eye(n)
    if R is None:
        R = 0.25 * np.eye(m)

    # ==================================
    # PREDICTION STEP
    # ==================================

    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Q

    # ==================================
    # MEASUREMENT PREDICTION
    # ==================================

    y_pred = C @ x_pred + D @ u
    innovation = y - y_pred

    # Innovation covariance
    S = C @ P_pred @ C.T + R

    # Kalman gain
    K = P_pred @ C.T @ np.linalg.inv(S)

    # ==================================
    # UPDATE STEP
    # ==================================

    x_new = x_pred + K @ innovation
    P_new = (np.eye(n) - K @ C) @ P_pred

    return x_new, P_new