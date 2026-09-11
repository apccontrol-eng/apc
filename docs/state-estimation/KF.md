# Kalman Filter

The goal of the Kalman Filter is to obtain estimates for the given process states when only measurements or partial state observations are available. The standard Kalman Filter assumes linear process and measurement models, and the noise is assumed to be Gaussian, which is only an approximation of real-world process uncertainty and measurement noise.

## 1. Process and Measurement Models

### Process model

$$
x_k = A x_{k-1} + B u_{k-1} + w_{k-1}
$$

### Measurement model

$$
y_k = C x_k + v_k
$$

Where:

* $x_k$: state vector
* $u_k$: control input
* $y_k$: measurement
* $A$: state transition matrix
* $B$: control input matrix
* $C$: observation matrix
* $w_k \sim \mathcal{N}(0,Q)$: process noise
* $v_k \sim \mathcal{N}(0,R)$: measurement noise

## 2. Initialization

* $\hat{x}_0$: initial state estimate
* $P_0$: initial covariance

## 3. Prediction Step (Time Update)

### State prediction

$$
\hat{x}_{k}^{-}
=
A\hat{x}_{k-1}
+
Bu_{k-1}
$$

### Covariance prediction

$$
P_{k}^{-}
=
AP_{k-1}A^{T}
+
Q
$$

## 4. Update Step (Measurement Update)

### Innovation (residual)

$$
y_{k,\mathrm{res}}
=
y_k
-
C\hat{x}_{k}^{-}
$$

### Innovation covariance

$$
S_k
=
CP_k^{-}C^{T}
+
R
$$

### Kalman Gain

$$
K_k
=
P_k^{-}C^{T}S_k^{-1}
$$

### State update

$$
\hat{x}_k
=
\hat{x}_k^{-}
+
K_k y_{k,\mathrm{res}}
$$

### Covariance update

$$
P_k
=
(I-K_kC)P_k^{-}
$$

---

## 5. Summary (Compact Form)

### Predict

$$
\hat{x}_{k}^{-}
=
A\hat{x}_{k-1}
+
Bu_{k-1}
$$

$$
P_k^{-}
=
AP_{k-1}A^{T}
+
Q
$$

### Update

$$
K_k
=
P_k^{-}C^{T}S_k^{-1}
$$

$$
\hat{x}_k
=
\hat{x}_k^{-}
+
K_k y_{k,\mathrm{res}}
$$

$$
P_k
=
(I-K_kC)P_k^{-}
$$

## Reference

Särkkä, S., & Svensson, L. (2023).

*Särkkä, S., & Svensson, L. (2023). Bayesian Filtering and Smoothing* (2nd ed.).

Cambridge: Cambridge University Press.
