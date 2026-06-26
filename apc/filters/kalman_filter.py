import numpy as np

"""
===============================================================================
Kalman Filter
===============================================================================

Description

Standard discrete-time linear Kalman Filter for state estimation of dynamic 
systems affected by process and measurement noise.

===============================================================================

Process Model

    x[k+1] = A x[k] + B u[k] + w[k]

Measurement Model

    y[k] = C x[k] + D u[k] + v[k]

where:
    x : system state vector
    u : control input vector
    y : measurement/output vector
    w : process noise
    v : measurement noise

===============================================================================

Assumptions

The filter assumes:
- Process noise w follows a zero-mean Gaussian distribution
- Measurement noise v follows a zero-mean Gaussian distribution
- Process and measurement noise are uncorrelated

===============================================================================

Reference
    
    Simo Särkkä and Lennart Svensson,
    Bayesian Filtering and Smoothing,
    Second Edition,
    Cambridge University Press, 2023.

===============================================================================
"""

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

    x = np.asarray(x)
    u = np.asarray(u)
    y = np.asarray(y)

    n = A.shape[0]
    m = C.shape[0]

    # =========================================================================
    # initial process model covariance matrix
    if Q is None:
        Q = 5 * np.eye(n)
        
    # =========================================================================
    # initial measurement model covariance matrix
    if R is None:
        R = 5 * np.eye(m)

    # =========================================================================
    # PREDICTION STEP
    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Q

    # =========================================================================
    # MEASUREMENT PREDICTION
    y_pred = C @ x_pred + D @ u
    innovation = y - y_pred

    # =========================================================================
    # Innovation covariance
    S = C @ P_pred @ C.T + R

    # =========================================================================
    # Kalman gain
    K = P_pred @ C.T @ np.linalg.inv(S)

    # =========================================================================
    # UPDATE STEP
    x_new = x_pred + K @ innovation
    P_new = (np.eye(n) - K @ C) @ P_pred

    return x_new, P_new, innovation

