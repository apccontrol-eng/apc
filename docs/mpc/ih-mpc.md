## Infinite-horizon Model Predictive Control

The structural difference between finite-horizon MPC and infinite-horizon MPC is in that finite-horizon problem is a scheduling problem from the viewpoint of optimization. The infinite-horizon MPC problem requires a different approach and the optimized control action is a constant control gain matrix $L$ and not a vector of future control actions as in the finite-horizon MPC. The following presents the optimization form for infinite-horizon MPC which is later cleverly solved with Linear Matrix Inequalities.

Discrete-time state-space model:  
$$
x_{k+1}=Ax_k+Bu_k
$$

Cost function:  

$$
J =
\sum_{k=0}^{\infty}
\left(
x_k^\top Q x_k
+
u_k^\top R u_k
\right)
$$  

Positive definite and positive semi-definite weight matrices for states and control effort:  

$$
Q \succeq 0,
\qquad
R \succ 0
$$

The general infinite-horizon MPC problem:

$$
\boxed{
\begin{aligned}
\min
\quad &
J(x_0)
=
\sum_{k=0}^{\infty}
\left(
x_k^T Q x_k
+
u_k^T R u_k
\right)
\\[4pt]
\text{subject to}
\quad &
x_{k+1}
=
Ax_k+Bu_k,
\qquad k=0,1,\ldots
\\
&
x_0
=
\hat{x}_0
\\
&
x_k \in \mathcal{X},
\qquad k=0,1,\ldots
\\
&
u_k \in \mathcal{U},
\qquad k=0,1,\ldots
\end{aligned}
}
$$


For the infinite-horizon MPC problem we seek for a full-state feedback control law that minimizes the upper bound of the cost function:
  
$$
u_k=Lx_k
$$

The LMI variable trick/substitution:  

$$
Y = LQ_U
$$  

The Semi-Definite Programming (SDP) optimization problem with LMIs (considering only input box constraints):

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

Where the box constraint:  

$$
|u_{k,j}| \le u_{j,\max}
$$

is implemented as  

$$
(X_U)_{jj} \le u_{j,\max}^2
$$  

The feedback gain is recovered from the variable trick:  

$$
L = YQ_U^{-1}
$$

The calculated control input at time k:  

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
