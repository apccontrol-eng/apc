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
- Optimization solved using QP algorithms or LMI/SDP

---


@article{kothare1996robust,
  title={Robust constrained model predictive control using linear matrix inequalities},
  author={Kothare, Mayuresh V and Balakrishnan, Venkataramanan and Morari, Manfred},
  journal={Automatica},
  volume={32},
  number={10},
  pages={1361--1379},
  year={1996},
  publisher={Elsevier},
  doi={10.1016/0005-1098(96)00063-5}
}
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
