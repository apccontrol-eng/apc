# MPC simulation framework with process monitoring
A simulation framework combining **Model Predictive Control (MPC)** with **Principal Component Analysis (PCA)-based statistical process monitoring** for fault detection and diagnosis in a multivariable dynamic system. Additional examples of **Infinite-Horizon Robust Model Predictive Control** and **Partial Least Squares Regression (PLSR) Soft Sensor**. The controller and monitoring combination is best suited for fixed point operations where MPC steers the system to one steady-state which is treated as 'normal operating conditions' for which variance and covariance based multivariable calibration/monitoring/fault-detection methods namely PCA and PLS can be constructed.

---

## Overview
This project demonstrates:  
- Three different formulations of constrained MPC:
  - Finite-Horizon constrained MPC solved with different QP solvers (Hildreth, Projected gradient descent, Primal-dual interior point and Active set method)
  - Offset-free MPC w/ Kalman Filter
  - Infinite-Horizon constrained MPC formulated as a Linear Matrix Inequality (LMI) and solved as a Semidefinite Program (SDP)
- Kalman Filter and partial state observations
- PCA-based multivariate process monitoring
  - Fault detection using T² and Q statistics and biplot
  - Fault diagnosis via contribution plots  
- PLSR-based Soft Sensor
  - T² and Q statistics and biplot
- Open-loop subspace identification:
  - MOESP
  - N4SID

---