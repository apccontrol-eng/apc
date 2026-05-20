"""
Kalman Filter Implementation
============================

Description
-----------
This module implements a standard discrete-time linear Kalman Filter for
state estimation of dynamic systems affected by process and measurement noise.

The Kalman filter recursively estimates the internal system state using:
1. A prediction model based on system dynamics
2. Measurement updates from observed outputs

The algorithm minimizes estimation uncertainty by combining model predictions
with noisy sensor measurements.

System Model
------------
State equation:

    x[k+1] = A x[k] + B u[k] + w[k]

Measurement equation:

    y[k] = C x[k] + D u[k] + v[k]

where:
    x : system state vector
    u : control input vector
    y : measurement/output vector
    w : process noise
    v : measurement noise

Noise Assumptions
-----------------
The filter assumes:
- Process noise w follows a zero-mean Gaussian distribution
- Measurement noise v follows a zero-mean Gaussian distribution
- Process and measurement noise are uncorrelated

Functions
---------
kalman_filter(...)
    Executes one iteration of the linear Kalman filter:
    - prediction step
    - innovation computation
    - state correction/update

Dependencies
------------
- numpy

Author
------
Emil

Notes
-----
- If Q or R are not provided, default covariance matrices are used.
- The implementation supports generic linear systems.
- This implementation performs:
    1. State prediction
    2. Covariance prediction
    3. Innovation calculation
    4. Kalman gain computation
    5. State update
    6. Covariance update

Example
-------
    x_new, P_new = kalman_filter(
        A, B, C, D,
        u,
        x_prev,
        P_prev,
        y_measured
    )

References
----------
[1] Simo Särkkä and Lennart Svensson,
    Bayesian Filtering and Smoothing,
    Second Edition,
    Cambridge University Press, 2023.

"""
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

    x = np.asarray(x)
    u = np.asarray(u)
    y = np.asarray(y)

    n = A.shape[0]
    m = C.shape[0]

    # initial process model covariance matrix
    if Q is None:
        Q = 0.05 * np.eye(n)
    # initial measurement model covariance matrix
    if R is None:
        R = 0.05 * np.eye(m)

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
