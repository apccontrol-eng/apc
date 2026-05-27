# MPC simulation framework with process monitoring
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

## Model Predictive Control (MPC) (e.g. Fink 2021)
- Linear state-space model
- Quadratic cost on states and inputs
- Box constraints on control inputs
- Optimization solved using QP algorithms
- Lifted matrices dictated by prediction horizon

Lifted system matrices form when prediction horizon is set to N:

```math
X_k = A_{lifted} x_k + B_{lifted} U_k
```

```math
X_k =
\begin{bmatrix}
\hat{x}_{k|k} \\
\hat{x}_{k+1|k} \\
\vdots \\
\hat{x}_{k+N|k}
\end{bmatrix}
\in \mathcal{X}^{N+1} \subseteq \mathbb{R}^{n(N+1)},
\quad
```

```math
U_k =
\begin{bmatrix}
\hat{u}_{k|k} \\
\hat{u}_{k+1|k} \\
\vdots \\
\hat{u}_{k+N-1|k}
\end{bmatrix}
\in \mathcal{U}^{N} \subseteq \mathbb{R}^{mN}
```

```math
A_{lifted} =
\begin{bmatrix}
I \\
A \\
A^2 \\
\vdots \\
A^N
\end{bmatrix}
\in \mathbb{R}^{n(N+1)\times n}
```

```math
B_{lifted} =
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
B & 0 & \cdots & 0 \\
AB & B & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
A^{N-1}B & A^{N-2}B & \cdots & B
\end{bmatrix}
\in \mathbb{R}^{n(N+1)\times m(N+1)}
```

The Quadratic Programming problem is of form:

```math
U_k^* = \arg\min_{U_k}\; X_k^\top \tilde{Q} X_k + U_k^\top \tilde{R} U_k
```
subject to
```math
X_k = A_{lifted} x_k + B_{lifted} U_k
```

```math
U_k \in \mathcal{U}_{ad}(x_k)
```
with
```math
\tilde{Q} =
\begin{bmatrix}
Q &  &  \\
 & \ddots &  \\
 &  & Q_f
\end{bmatrix}
\in \mathbb{R}^{n(N+1)\times n(N+1)},
\qquad
\tilde{R} =
\begin{bmatrix}
R &  &  \\
 & \ddots &  \\
 &  & R
\end{bmatrix}
\in \mathbb{R}^{mN\times mN}  
```  

The QP problem can be reduced to the following form if $X_k$ is substituted to the cost function.  

```math
U_k^* =
\arg\min_{U_k}\;
U_k^\top(B_{lifted}^\top \tilde{Q}B_{lifted} + \tilde{R})U_k
+ 2x_k^\top A_{lifted}^\top B_{lifted} U_k
+ x_k^\top A_{lifted}^\top \tilde{Q}A_{lifted} x_k
```
subject to
```math
U_N \in \mathcal{U}_{ad}(x_k)
```

### Reference
Michael Fink (2021).  
Implementation of Linear Model Predictive Control — Tutorial.  
arXiv preprint arXiv:2109.11986.  
https://arxiv.org/abs/2109.11986  

---
---
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

## Kalman Filter
Process model:
```math
x_k = A x_{k-1} + B u_{k-1} + w_{k-1}
```
Measurement model:
```math
y_k = C x_k + v_k
```

Where:
- $x_k$: state vector  
- $u_k$: control input  
- $y_k$: measurement  
- $A$: state transition matrix  
- $B$: control input matrix  
- $C$: observation matrix  
- $w_k$ ~ N(0, $Q$): process noise  
- $v_k$ ~ N(0, $R$): measurement noise  

### 2. Initialization

- $\hat{x}_{0}$: initial state estimate  
- $P_0$: initial covariance  

### 3. Prediction Step (Time Update)

#### State prediction
```math
\hat{x}_{k}^{-} = A \hat{x}_{k-1} + B u_{k-1}
```
#### Covariance prediction
```math
P_{k}^{-} = A P_{k-1} A^{T} + Q
```
### 4. Update Step (Measurement Update)

#### Innovation (residual)
```math
y_{k,res} = y_{k} - C \hat{x}_{k}^{-}
```
#### Innovation covariance
```math
S_{k} = C P_{k}^{-} C^{T} + R
```
#### Kalman Gain
```math
K_{k} = P_{k}^{-} C^{T} S_{k}^{-1}
```
#### State update
```math
\hat{x}_{k} = \hat{x}_{k}^{-} + K_{k} y_{k,res}
```
#### Covariance update
```math
P_{k} = (I - K_{k} C) P_{k}^{-}
```
---

### 5. Summary (Compact Form)

Predict:
```math
\hat{x}_{k}^{-} = A \hat{x}_{k-1} + B u_{k-1}
```
```math
P_{k}^{-} = A P_{k-1} A^{T} + Q  
```
Update:
```math
K_{k} = P_{k}^{-} C^{T} S_{k}^{-1}
```
```math
\hat{x}_{k} = \hat{x}_{k}^{-} + K_{k} y_{k,res}
```
```math
P_{k} = (I - K_{k} C) P_{k}^{-}  
```

### Reference
Särkkä, S., & Svensson, L. (2023).  
*Bayesian Filtering and Smoothing* (2nd ed.).  
Cambridge: Cambridge University Press.  

---
---
---


# Plots of the MPC + PCA Monitoring example:
### States and control inputs over time
<img src="examples/figures/MPC_states_controls.png" alt="drawing" width="650"/>

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
<img src="examples/figures/Calibration_data_before_autoscaling.png" alt="drawing" width="800"/>

### After autoscaling
<img src="examples/figures/Calibration_data_after_autoscaling.png" alt="drawing" width="800"/>

---

## PCA Model (Calibration Data)
### PCA Biplot (Calibration)
<img src="examples/figures/Calibration_data_PCA_biplot.png" alt="drawing" width="650"/>

---

## Fault Detection Statistics (Calibration)

### Hotelling’s T² and Q (SPE)
<img src="examples/figures/Calibration_T2_and_SPE_plot.png" alt="drawing" width="800"/>

---

## Online Monitoring (New Data)
New process data is projected into the PCA model built from calibration data.

---

## Monitoring Data Distribution

### Before autoscaling
<img src="examples/figures/Monitored_new_data_before_autoscaling.png" alt="drawing" width="800"/>

### After autoscaling
<img src="examples/figures/Monitored_new_data_after_autoscaling.png" alt="drawing" width="800"/>

---

## PCA Monitoring Results
### PCA Biplot (Monitoring data)
<img src="examples/figures/Monitoring_data_PCA_biplot.png" alt="drawing" width="650"/>

---

### Hotelling’s T² and Q (Monitoring)
<img src="examples/figures/Monitoring_data_T2_and_SPE_plot.png" alt="drawing" width="800"/>

---

## Fault Diagnosis (Contribution Analysis)

When a fault is detected, contribution plots identify responsible variables.

---

## Variable Contributions to Principal Components

### Sample 150 — PC1 Contribution
<img src="examples/figures/Sample_150_contribution_to_PC1.png" alt="drawing" width="500"/>

---

### Sample 150 — PC2 Contribution
<img src="examples/figures/Sample_150_contribution_to_PC2.png" alt="drawing" width="500"/>

---
