# MPC + PCA-Based Fault Detection and Process Monitoring
A simulation framework combining **Model Predictive Control (MPC)** with **Principal Component Analysis (PCA)-based statistical process monitoring** for fault detection and diagnosis in a multivariable dynamic system.

---

## Overview
This project demonstrates a closed-loop system:
- Constrained Model Predictive Control (MPC)
- Kalman Filter and partial state observations
- Quadratic Programming solved via several QP algorithms
- Linear Matrix Inequality Infinite Horizon MPC formulation
- Process disturbance + fault injection
- PCA-based multivariate monitoring
  - Fault detection using T² and Q statistics
  - Fault diagnosis via contribution plots

---

## Model Predictive Control (MPC)
- Linear state-space model
- Quadratic cost on states and inputs
- Box constraints on control inputs
- Optimization solved using QP algorithms
  
---

## Robust Model Predictive Control (RMPC) (Kothare 1996)
- Linear state-space model
- Quadratic cost on states and inputs
- Box constraints on control inputs
- Optimization solved using LMI

$$
x_{k+1}=Ax_k+Bu_k
$$

$$
u_k=Lx_k
$$

$$
J =
\sum_{k=0}^{\infty}
\left(
x_k^\top Q x_k
+
u_k^\top R u_k
\right)
$$

$$
Q \succ 0,
\qquad
R \succ 0
$$

$$
Y = LQ_U
$$

$$
\min_{Q_U,Y,X_U,\gamma}
\gamma
$$

subject to

$$
\begin{bmatrix}
1 & x_k^\top \\
x_k & Q_U
\end{bmatrix}
\succeq 0
$$

$$
\begin{bmatrix}
X_U & Y \\
Y^\top & Q_U
\end{bmatrix}
\succeq 0
$$

$$
\begin{bmatrix}
Q_U
&
Q_UA^\top + Y^\top B^\top
&
Q_UQ^{1/2}
&
Y^\top R^{1/2}
\\
AQ_U + BY
&
Q_U
&
0
&
0
\\
Q^{1/2}Q_U
&
0
&
\gamma I
&
0
\\
R^{1/2}Y
&
0
&
0
&
\gamma I
\end{bmatrix}
\succeq
0
$$

$$
|u_k| \le u_{\max}
$$

implemented as

$$
X_U \le u_{\max}^2
$$

Recover the feedback gain via

$$
L = YQ_U^{-1}
$$

and compute control input

$$
u_k=Lx_k
$$


### Reference
Kothare, M. V., Balakrishnan, V., & Morari, M. (1996).  
**Robust constrained model predictive control using linear matrix inequalities**.  
*Automatica*, 32(10), 1361–1379.  
DOI: 10.1016/0005-1098(96)00063-5

---
---
---


# Plots of the MPC + PCA Monitoring example:
### States and control inputs over time

![MPC States and Controls](examples/figures/MPC_states_controls.png)
---
# Fault Injection
A fault is introduced into the system at time step **k = 150**:
- Gaussian noise at all times
- Additional bias simulates abnormal process behavior

---

# PCA-Based Process Monitoring
### Steps:
- Autoscaling (mean / variance normalization)
- Covariance matrix estimation
- Eigen decomposition
- Projection into principal component space

---

### Calibration Phase (Normal Operation)
### Before autoscaling
![Calibration Before Scaling](examples/figures/Calibration_data_before_autoscaling.png)

### After autoscaling
![Calibration After Scaling](examples/figures/Calibration_data_after_autoscaling.png)

---

## PCA Model (Calibration Data)
### PCA Biplot (Calibration)
![Calibration PCA Biplot](examples/figures/Calibration_data_PCA_biplot.png)

---

## Fault Detection Statistics (Calibration)

### Hotelling’s T² and Q (SPE)
![Calibration T2 SPE](examples/figures/Calibration_T2_and_SPE_plot.png)

---

## Online Monitoring (New Data)
New process data is projected into the PCA model built from calibration data.

---

## Monitoring Data Distribution

### Before autoscaling
![Monitoring Before Scaling](examples/figures/Monitored_new_data_before_autoscaling.png)

### After autoscaling
![Monitoring After Scaling](examples/figures/Monitored_new_data_after_autoscaling.png)

---

## PCA Monitoring Results
### PCA Biplot (Monitoring data)
![Monitoring PCA Biplot](examples/figures/Monitoring_data_PCA_biplot.png)

---

### Hotelling’s T² and Q (Monitoring)
![Monitoring T2 SPE](examples/figures/Monitoring_data_T2_and_SPE_plot.png)

---

## Fault Diagnosis (Contribution Analysis)

When a fault is detected, contribution plots identify responsible variables.

---

## Variable Contributions to Principal Components

### Sample 150 — PC1 Contribution
![PC1 Contribution](examples/figures/Sample_150_contribution_to_PC1.png)

---

### Sample 150 — PC2 Contribution
![PC2 Contribution](examples/figures/Sample_150_contribution_to_PC2.png)


---
