## Kalman Filter

The goal of the Kalman Filter is to obtain estimates for the given process states when only measurements or partial state observations are available. The standard Kalman Filter assumes linear process and measurement models, and the noise is assumed to be Gaussian, which is only an approximation of real-world process uncertainty and measurement noise.

Process model:

$$
x_k = A x_{k-1} + B u_{k-1} + w_{k-1}
$$
Measurement model:

$$
y_k = C x_k + v_k
$$

Where:
$x_k$: state vector  
$u_k$: control input  
$y_k$: measurement  
$A$: state transition matrix  
$B$: control input matrix  
$C$: observation matrix  
$w_k$ ~ N(0, $Q$): process noise  
$v_k$ ~ N(0, $R$): measurement noise  

### 2. Initialization

- $\hat{x}_{0}$: initial state estimate  
- $P_0$: initial covariance  

### 3. Prediction Step (Time Update)

#### State prediction

$$
\hat{x}_{k}^{-} = A \hat{x}_{k-1} + B u_{k-1}
$$
#### Covariance prediction

$$
P_{k}^{-} = A P_{k-1} A^{T} + Q
$$

### 4. Update Step (Measurement Update)

#### Innovation (residual)

$$
y_{k,res} = y_{k} - C \hat{x}_{k}^{-}
$$

#### Innovation covariance

$$
S_{k} = C P_{k}^{-} C^{T} + R
$$
#### Kalman Gain

$$
K_{k} = P_{k}^{-} C^{T} S_{k}^{-1}
$$

#### State update

$$
\hat{x}_{k} = \hat{x}_{k}^{-} + K_{k} y_{k,res}
$$

#### Covariance update

$$
P_{k} = (I - K_{k} C) P_{k}^{-}
$$
---

### 5. Summary (Compact Form)

Predict:

$$
\hat{x}_{k}^{-} = A \hat{x}_{k-1} + B u_{k-1}
$$

$$
P_{k}^{-} = A P_{k-1} A^{T} + Q  
$$
Update:

$$
K_{k} = P_{k}^{-} C^{T} S_{k}^{-1}
$$

$$
\hat{x}_{k} = \hat{x}_{k}^{-} + K_{k} y_{k,res}
$$

$$
P_{k} = (I - K_{k} C) P_{k}^{-}  
$$

### Reference

Särkkä, S., & Svensson, L. (2023).  
*Bayesian Filtering and Smoothing* (2nd ed.).  
Cambridge: Cambridge University Press.  
