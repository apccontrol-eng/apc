Dynamic Matrix Control is one of the earliest commercial Model Predictive Control schemes. DMC utilizes step-response models instead of state-space models and states. One of the advantages is in that MIMO models do not have to be updated completely and MIMO models can be constructed step-testing channels i.e. input-output pairs individually. The other advantage is that the DMC handles process dead-times naturally. 

## Example system for MIMO Dynamic Matrix Control (DMC)

A 2x2 FOPDT MIMO process model is the following:

$$
G(s)=
\begin{bmatrix}
\dfrac{K_{11}e^{-\theta_{11}s}}{\tau_{11}s+1} &
\dfrac{K_{12}e^{-\theta_{12}s}}{\tau_{12}s+1} \\
\dfrac{K_{21}e^{-\theta_{21}s}}{\tau_{21}s+1} &
\dfrac{K_{22}e^{-\theta_{22}s}}{\tau_{22}s+1}
\end{bmatrix}
$$

As an example it contains four First Order Plus Dead Time models. The identified models are all open-loop stable and also $y_1(0)=0$ and $y_2(0)=0$.

As an example the first input-output channel $u_1$ to $y_1$:

$$
Y_1 = \dfrac{K_{11}e^{-\theta_{11}s}}{\tau_{11}s+1}U_1
$$

$$
sY_1 = -\frac{1}{\tau_{11}}Y_1 + \frac{K_{11}}{\tau_{11}}e^{-\theta_{11}s}U_1
$$

$$
\frac{d}{dt}y_1(t) =
-\frac{1}{\tau_{11}}y_1(t)
+\frac{K_{11}}{\tau_{11}}u_1(t-\theta_{11})
$$


## Unit-step response model

The unit-step response model can be analytically obtained from the system model by using the unit-step Laplace-domain model, which is $U(s)=1/s$.

$$
Y_1 = \dfrac{K_{11}e^{-\theta_{11}s}}{\tau_{11}s+1}U_1
$$

$$
Y_1 = \dfrac{K_{11}e^{-\theta_{11}s}}{\tau_{11}s+1}\frac{1}{s}
$$

The unit-step response of the model is then obtained by inverse Laplace transform:

$$
y_1(t) =
K_{11}\left(1-e^{-(t-\theta_{11})/\tau_{11}}\right)H(t-\theta_{11})
$$

$$
y_1(t)=
\begin{cases}
0, & t < \theta_{11} \\[2mm]
K_{11}\left(1-e^{-(t-\theta_{11})/\tau_{11}}\right), & t \geq \theta_{11}
\end{cases}
$$

The original model is described in minutes and the sampling period is 4 minutes, i.e. the discretization is carried using $4k=t$:

$$
y_1(t)=
\begin{cases}
0, & 4k < \theta_{11} \\[2mm]
K_{11}\left(1-e^{-(4k-\theta_{11})/\tau_{11}}\right), & 4k \geq \theta_{11}
\end{cases}
$$

The unit-step response models are then simulated from $k=0$ up to the point where transient responses turn to the model gains ("simulate up to steady-states"). The unit-step response model is the recorded value of $y_{11}$ at times $k$ and is marked with $g_i$ in the DMC prediction model.

## Prediction model in DMC

The utilized step response model:

$$
y_k=\sum_{i=1}^{\infty}g_i\,\Delta u(k-i)
$$

The $p$-step ahead prediction model given $y$ is measured at $k$:

$$
\hat{y}(k+p\mid k)
=
\sum_{i=1}^{p}g_i\,\Delta u(k+p-i)
+
f(k+p)
$$

The free response term, $f(k+p)$, is constructed every loop instance with $N$ previous controls and the current measurement $y_m(k)$. The usual DMC assumption is that the step response has reached steady state after $N$ samples.

$$
f(k+p)
=
y_m(k)
+
\sum_{i=1}^{N}
\left(g_{p+i}-g_i\right)
\Delta u(k-i)
$$

With the assumption that for an open-loop stable model the step coefficients converge,

$$
g_i=g_N,\qquad i\geq N.
$$

Finally, the $p$-step ahead prediction model consists of the free response term and the forced response term:

$$
\hat{y}(k+p\mid k)
=
\sum_{i=1}^{p}
g_i\,\Delta u(k+p-i)
+
y_m(k)
+
\sum_{i=1}^{N}
\left(g_{p+i}-g_i\right)
\Delta u(k-i)
$$

For easier notation and computation, the same prediction model is written in matrix-vector notation:

$$
\hat{\mathbf{y}}=\mathbf{G}\Delta\mathbf{u}+\mathbf{f}
$$

The $\mathbf{G}$ is known as the dynamic matrix, which consists of the unit-step response coefficients.

$$
\mathbf{G}=
\begin{bmatrix}
g_1 & 0 & \cdots & 0\\
g_2 & g_1 & \cdots & 0\\
g_3 & g_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
g_p & g_{p-1} & \cdots & g_{p-m+1}
\end{bmatrix}
$$

Only the free response term needs to be updated before computing the next control.

## MIMO extension via principle of superposition

For the example 2x2 MIMO, we have four dynamic matrices $\mathbf{G}_{11}$, $\mathbf{G}_{12}$, $\mathbf{G}_{21}$, and $\mathbf{G}_{22}$, which have corresponding matrix entries from the unit-step response experiment up to the $p$ prediction horizon length. For the sake of simplicity, we choose $m$, i.e. the length of the control horizon, to be the same as $p$.

The MIMO model takes the following structure because of the principle of superposition:

$$
\hat{\mathbf{y}}_{1}
=
\mathbf{G}_{11}\Delta\mathbf{u}_1
+
\mathbf{f}_{11}
+
\mathbf{G}_{12}\Delta\mathbf{u}_2
+
\mathbf{f}_{12}
$$

$$
\hat{\mathbf{y}}_{2}
=
\mathbf{G}_{21}\Delta\mathbf{u}_2
+
\mathbf{f}_{21}
+
\mathbf{G}_{22}\Delta\mathbf{u}_2
+
\mathbf{f}_{22}
$$

These can be concatenated into one vector-matrix equation:

$$
\begin{bmatrix}
\hat{\mathbf{y}}_1\\[1mm]
\hat{\mathbf{y}}_2
\end{bmatrix}
=
\left[
\begin{array}{cc}
\mathbf{G}_{11} & \mathbf{G}_{12}\\
\mathbf{G}_{21} & \mathbf{G}_{22}
\end{array}
\right]
\begin{bmatrix}
\Delta\mathbf{u}_1\\
\Delta\mathbf{u}_2
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{f}_1\\
\mathbf{f}_2
\end{bmatrix}
$$

where

$$
\mathbf{f}_1 = \mathbf{f}_{11} + \mathbf{f}_{12}
$$

$$
\mathbf{f}_2 = \mathbf{f}_{21} + \mathbf{f}_{22}
$$

This model can be further condensed to the following form:

$$
\hat{\mathbf{Y}} = \mathscr{B}\Delta\mathbf{U} + \mathscr{F}
$$

## DMC optimization formulation

The next control action is calculated by optimizing the squared error cost between the chosen set-point reference $\mathbf{Y}_{ref}$ and the model predictions $\hat{\mathbf{Y}}$. The future control changes are also penalized by adding the penalty term to the cost function $J$. The cost function also has scaling matrices $\mathbf{Q}$ and $\mathbf{R}$.

$$
\boxed{
\begin{aligned}
\min_{\Delta\mathbf{U}}\quad
J
&=
\left(
\mathbf{Y}_{ref}
-\hat{\mathbf{Y}}
\right)^{T}
\mathbf{Q}
\left(
\mathbf{Y}_{ref}
-\hat{\mathbf{Y}}
\right)
+
\Delta\mathbf{U}^{T}
\mathbf{R}
\Delta\mathbf{U}
\\[2mm]
\text{s.t.}\quad
&\hat{\mathbf{Y}}
=
\mathscr{B}\Delta\mathbf{U}
+
\mathscr{F},
\\[1mm]
&\mathbf{Y}_{\min}
\leq
\hat{\mathbf{Y}}
\leq
\mathbf{Y}_{\max},
\\[1mm]
&\Delta\mathbf{U}_{\min}
\leq
\Delta\mathbf{U}
\leq
\Delta\mathbf{U}_{\max},
\\[1mm]
&\mathbf{U}_{\min}
\leq
\mathbf{U}
\leq
\mathbf{U}_{\max}.
\end{aligned}
}
$$

### References

Camacho, E. F., & Bordons, C. (2007).
*Model Predictive Control* (2nd ed.).
Springer London.
[**https://doi.org/10.1007/978-0-85729-398-5**](https://doi.org/10.1007/978-0-85729-398-5)

Ogunnaike, B. A., & Ray, W. H. (1994).
*Process Dynamics, Modeling, and Control*.
Oxford University Press.
ISBN: 978-0-19-509119-9.

Corriou, J.-P. (2018).
*Process Control: Theory and Applications* (2nd ed.).
Springer Cham.
[**https://doi.org/10.1007/978-3-319-61143-3**](https://doi.org/10.1007/978-3-319-61143-3)

García, C. E., & Morshedi, A. M. (1986).
*Quadratic Programming Solution of Dynamic Matrix Control (QDMC).*
*Chemical Engineering Communications*, 46, 73–87.
[**https://doi.org/10.1080/00986448608911397**](https://doi.org/10.1080/00986448608911397)


