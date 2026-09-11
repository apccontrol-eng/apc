# Shell Heavy Oil Fractionator DMC Example

## System description

The subsystem of the Shell Heavy Oil Fractionator model is the following.

$$
G(s)=
\begin{bmatrix}
\dfrac{4.05e^{-27s}}{50s+1} &
\dfrac{1.77e^{-28s}}{60s+1} \\[8pt]
\dfrac{5.39e^{-18s}}{50s+1} &
\dfrac{5.72e^{-14s}}{60s+1}
\end{bmatrix}
$$

It consists of four First Order Plus Dead Time models. The identified models are all open-loop stable and also $y_1(0)=0$ and $y_2(0)=0$.

The first input-output channel $u_1$ to $y_1$:

$$
Y_1 = \dfrac{4.05e^{-27s}}{50s+1}U_1
$$

$$
sY_1 = -\frac{1}{50}Y_1 + \frac{4.05}{50}e^{-27s}U_1
$$

$$
\frac{d}{dt}y_1(t) =
-\frac{1}{50}y_1(t)
+\frac{4.05}{50}u_1(t-27)
$$

Input-output channel $u_2$ to $y_1$:

$$
\frac{d}{dt}y_1(t) =
-\frac{1}{60}y_1(t)
+\frac{1.77}{60}u_2(t-28)
$$

Input-output channel $u_1$ to $y_2$:

$$
\frac{d}{dt}y_2(t) =
-\frac{1}{50}y_2(t)
+\frac{5.39}{50}u_1(t-18)
$$

Input-output channel $u_2$ to $y_2$:

$$
\frac{d}{dt}y_2(t) =
-\frac{1}{60}y_2(t)
+\frac{5.72}{60}u_2(t-14)
$$

## Unit-step response model

The unit-step response model can be analytically obtained from the system model by using the unit-step Laplace-domain model, which is $U(s)=1/s$.

$$
Y_1 = \dfrac{4.05e^{-27s}}{50s+1}U_1
$$

$$
Y_1 = \dfrac{4.05e^{-27s}}{50s+1}\frac{1}{s}
$$

The unit-step response of the model is then obtained by inverse Laplace transform:

$$
y_1(t) =
4.05\left(1-e^{-(t-27)/50}\right)H(t-27)
$$

$$
y_1(t)=
\begin{cases}
0, & t < 27 \\[2mm]
4.05\left(1-e^{-(t-27)/50}\right), & t \geq 27
\end{cases}
$$

The original model is described in minutes and the sampling period is 4 minutes, i.e. the discretization is carried using $4k=t$:

$$
y_1(t)=
\begin{cases}
0, & 4k < 27 \\[2mm]
4.05\left(1-e^{-(4k-27)/50}\right), & 4k \geq 27
\end{cases}
$$

The remaining unit-step response models:

$$
y_1(t)=
\begin{cases}
0, & 4k < 28 \\[2mm]
1.77\left(1-e^{-(4k-28)/60}\right), & 4k \geq 28
\end{cases}
$$

$$
y_2(t)=
\begin{cases}
0, & 4k < 18 \\[2mm]
5.39\left(1-e^{-(4k-18)/50}\right), & 4k \geq 18
\end{cases}
$$

$$
y_2(t)=
\begin{cases}
0, & 4k < 14 \\[2mm]
5.72\left(1-e^{-(4k-14)/60}\right), & 4k \geq 14
\end{cases}
$$

The unit-step response models are then simulated from $k=0$ up to the point where steady-states are reached.

### Table 1. Unit-step responses for the four input-output channels

| Sample | $u_1 \rightarrow y_1$ | $u_2 \rightarrow y_1$ | $u_1 \rightarrow y_2$ | $u_2 \rightarrow y_2$ | Sample | $u_1 \rightarrow y_1$ | $u_2 \rightarrow y_1$ | $u_1 \rightarrow y_2$ | $u_2 \rightarrow y_2$ |
| -----: | --------------------: | --------------------: | --------------------: | --------------------: | -----: | --------------------: | --------------------: | --------------------: | --------------------: |
|      0 |                  0.00 |                  0.00 |                  0.00 |                  0.00 |     50 |                  3.92 |                  1.67 |                  5.25 |                  5.46 |
|      1 |                  0.00 |                  0.00 |                  0.00 |                  0.00 |     51 |                  3.93 |                  1.68 |                  5.26 |                  5.48 |
|      2 |                  0.00 |                  0.00 |                  0.00 |                  0.00 |     52 |                  3.94 |                  1.68 |                  5.27 |                  5.49 |
|      3 |                  0.00 |                  0.00 |                  0.00 |                  0.00 |     53 |                  3.95 |                  1.69 |                  5.28 |                  5.51 |
|      4 |                  0.00 |                  0.00 |                  0.00 |                  0.19 |     54 |                  3.96 |                  1.69 |                  5.29 |                  5.52 |
|      5 |                  0.00 |                  0.00 |                  0.21 |                  0.54 |     55 |                  3.97 |                  1.70 |                  5.30 |                  5.54 |
|      6 |                  0.00 |                  0.00 |                  0.61 |                  0.88 |     56 |                  3.97 |                  1.70 |                  5.30 |                  5.55 |
|      7 |                  0.08 |                  0.00 |                  0.98 |                  1.19 |     57 |                  3.98 |                  1.71 |                  5.31 |                  5.56 |
|      8 |                  0.39 |                  0.11 |                  1.32 |                  1.48 |     58 |                  3.98 |                  1.71 |                  5.32 |                  5.57 |
|      9 |                  0.67 |                  0.22 |                  1.63 |                  1.76 |     59 |                  3.99 |                  1.72 |                  5.32 |                  5.58 |
|     10 |                  0.93 |                  0.32 |                  1.92 |                  2.01 |     60 |                  3.99 |                  1.72 |                  5.33 |                  5.59 |
|     11 |                  1.17 |                  0.41 |                  2.19 |                  2.25 |     61 |                  4.00 |                  1.72 |                  5.33 |                  5.60 |
|     12 |                  1.39 |                  0.50 |                  2.43 |                  2.47 |     62 |                  4.00 |                  1.73 |                  5.34 |                  5.60 |
|     13 |                  1.59 |                  0.58 |                  2.66 |                  2.68 |     63 |                  4.01 |                  1.73 |                  5.34 |                  5.61 |
|     14 |                  1.78 |                  0.66 |                  2.87 |                  2.88 |     64 |                  4.01 |                  1.73 |                  5.34 |                  5.62 |
|     15 |                  1.96 |                  0.73 |                  3.06 |                  3.06 |     65 |                  4.01 |                  1.73 |                  5.35 |                  5.63 |
|     16 |                  2.12 |                  0.80 |                  3.24 |                  3.23 |     66 |                  4.02 |                  1.74 |                  5.35 |                  5.63 |
|     17 |                  2.27 |                  0.86 |                  3.41 |                  3.39 |     67 |                  4.02 |                  1.74 |                  5.35 |                  5.64 |
|     18 |                  2.40 |                  0.92 |                  3.56 |                  3.54 |     68 |                  4.02 |                  1.74 |                  5.36 |                  5.64 |
|     19 |                  2.53 |                  0.98 |                  3.70 |                  3.69 |     69 |                  4.02 |                  1.74 |                  5.36 |                  5.65 |
|     20 |                  2.65 |                  1.03 |                  3.83 |                  3.82 |     70 |                  4.02 |                  1.74 |                  5.36 |                  5.65 |
|     21 |                  2.76 |                  1.07 |                  3.95 |                  3.94 |     71 |                  4.03 |                  1.75 |                  5.36 |                  5.66 |
|     22 |                  2.85 |                  1.12 |                  4.06 |                  4.05 |     72 |                  4.03 |                  1.75 |                  5.37 |                  5.66 |
|     23 |                  2.95 |                  1.16 |                  4.16 |                  4.16 |     73 |                  4.03 |                  1.75 |                  5.37 |                  5.66 |
|     24 |                  3.03 |                  1.20 |                  4.26 |                  4.26 |     74 |                  4.03 |                  1.75 |                  5.37 |                  5.67 |
|     25 |                  3.11 |                  1.24 |                  4.34 |                  4.36 |     75 |                  4.03 |                  1.75 |                  5.37 |                  5.67 |
|     26 |                  3.18 |                  1.27 |                  4.43 |                  4.44 |     76 |                  4.03 |                  1.75 |                  5.37 |                  5.67 |
|     27 |                  3.25 |                  1.30 |                  4.50 |                  4.53 |     77 |                  4.04 |                  1.75 |                  5.37 |                  5.68 |
|     28 |                  3.31 |                  1.33 |                  4.57 |                  4.60 |     78 |                  4.04 |                  1.76 |                  5.38 |                  5.68 |
|     29 |                  3.37 |                  1.36 |                  4.63 |                  4.68 |     79 |                  4.04 |                  1.76 |                  5.38 |                  5.68 |
|     30 |                  3.42 |                  1.39 |                  4.69 |                  4.74 |     80 |                  4.04 |                  1.76 |                  5.38 |                  5.69 |
|     31 |                  3.47 |                  1.41 |                  4.74 |                  4.81 |     81 |                  4.04 |                  1.76 |                  5.38 |                  5.69 |
|     32 |                  3.51 |                  1.44 |                  4.79 |                  4.86 |     82 |                  4.04 |                  1.76 |                  5.38 |                  5.69 |
|     33 |                  3.55 |                  1.46 |                  4.84 |                  4.92 |     83 |                  4.04 |                  1.76 |                  5.38 |                  5.69 |
|     34 |                  3.59 |                  1.48 |                  4.88 |                  4.97 |     84 |                  4.04 |                  1.76 |                  5.38 |                  5.69 |
|     35 |                  3.63 |                  1.50 |                  4.92 |                  5.02 |     85 |                  4.04 |                  1.76 |                  5.38 |                  5.70 |
|     36 |                  3.66 |                  1.51 |                  4.96 |                  5.07 |     86 |                  4.04 |                  1.76 |                  5.38 |                  5.70 |
|     37 |                  3.69 |                  1.53 |                  4.99 |                  5.11 |     87 |                  4.04 |                  1.76 |                  5.38 |                  5.70 |
|     38 |                  3.72 |                  1.55 |                  5.02 |                  5.15 |     88 |                  4.04 |                  1.76 |                  5.38 |                  5.70 |
|     39 |                  3.74 |                  1.56 |                  5.05 |                  5.18 |     89 |                  4.05 |                  1.76 |                  5.38 |                  5.70 |
|     40 |                  3.77 |                  1.57 |                  5.08 |                  5.22 |     90 |                  4.05 |                  1.76 |                  5.38 |                  5.70 |
|     41 |                  3.79 |                  1.59 |                  5.10 |                  5.25 |     91 |                  4.05 |                  1.76 |                  5.39 |                  5.70 |
|     42 |                  3.81 |                  1.60 |                  5.12 |                  5.28 |     92 |                  4.05 |                  1.76 |                  5.39 |                  5.70 |
|     43 |                  3.83 |                  1.61 |                  5.14 |                  5.31 |     93 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |
|     44 |                  3.84 |                  1.62 |                  5.16 |                  5.34 |     94 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |
|     45 |                  3.86 |                  1.63 |                  5.18 |                  5.36 |     95 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |
|     46 |                  3.88 |                  1.64 |                  5.20 |                  5.38 |     96 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |
|     47 |                  3.89 |                  1.65 |                  5.21 |                  5.41 |     97 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |
|     48 |                  3.90 |                  1.66 |                  5.22 |                  5.43 |     98 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |
|     49 |                  3.91 |                  1.66 |                  5.24 |                  5.45 |     99 |                  4.05 |                  1.77 |                  5.39 |                  5.71 |

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

For the Shell Heavy Oil Fractionator, we have four dynamic matrices $\mathbf{G}*{11}$, $\mathbf{G}*{12}$, $\mathbf{G}*{21}$, and $\mathbf{G}*{22}$, which have corresponding matrix entries from the unit-step response experiment up to the $p$ prediction horizon length. For the sake of simplicity, we choose $m$, i.e. the length of the control horizon, to be the same as $p$.

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
