## Model Predictive Control (MPC)
- Linear state-space model:  


$$
x_{k+1} = A x_k + B u_k
$$  
where  
$$
 A \in \mathbb{R}^{nxn}
$$  
$$
 B \in \mathbb{R}^{nxm}
$$   
$$
 x_{k} \in \mathbb{R}^{nx1}
$$  
$$
 u_{k} \in \mathbb{R}^{mx1}
$$  

- Quadratic cost on states and inputs:  
$$
J = \sum_{k=0}^{N}\left(x_k^\top Q x_k+u_k^\top R u_k\right)
$$
where  
$$
Q \succeq 0,
\qquad
$$
$$
R \succ 0,
\qquad
$$  
are weighting matrices for states and control effort.  

- Box constraints (lower and upper bounds) on control inputs:  
$$
u_{lb} <= u_{k} <= u_{ub}  
$$  
- Optimization solved using QP algorithms
- Lifted matrices dictated by prediction horizon

Lifted system matrices form when prediction horizon is set to N:

$$
X_k = A_{lifted} x_k + B_{lifted} U_k
$$

$$
X_k =
\begin{bmatrix}
\hat{x}_{k|k} \\
\hat{x}_{k+1|k} \\
\vdots \\
\hat{x}_{k+N|k}
\end{bmatrix}
\in \mathcal{X}^{N+1} \subseteq \mathbb{R}^{n(N+1)},
\quad
$$

$$
U_k =
\begin{bmatrix}
\hat{u}_{k|k} \\
\hat{u}_{k+1|k} \\
\vdots \\
\hat{u}_{k+N-1|k}
\end{bmatrix}
\in \mathcal{U}^{N} \subseteq \mathbb{R}^{mN}
$$

$$
A_{lifted} =
\begin{bmatrix}
I \\
A \\
A^2 \\
\vdots \\
A^N
\end{bmatrix}
\in \mathbb{R}^{n(N+1)\times n}
$$

$$
B_{lifted} =
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
B & 0 & \cdots & 0 \\
AB & B & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
A^{N-1}B & A^{N-2}B & \cdots & B
\end{bmatrix}
\in \mathbb{R}^{n(N+1)\times m(N+1)}
$$

The Quadratic Programming problem is of form:

$$
U_k^* = \arg\min_{U_k}\; X_k^\top \tilde{Q} X_k + U_k^\top \tilde{R} U_k
$$
subject to
$$
X_k = A_{lifted} x_k + B_{lifted} U_k
$$

$$
U_k \in \mathcal{U}_{ad}(x_k)
$$
with
$$
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
$$  

The QP problem can be reduced to the following form when $X_k$ is substituted to the cost function.  

$$
U_k^* =
\arg\min_{U_k}\;
U_k^\top(B_{lifted}^\top \tilde{Q}B_{lifted} + \tilde{R})U_k
+ 2x_k^\top A_{lifted}^\top B_{lifted} U_k
+ x_k^\top A_{lifted}^\top \tilde{Q}A_{lifted} x_k
$$
subject to
$$
U_k \in \mathcal{U}_{ad}(x_k)
$$

### Reference
Michael Fink (2021).  
Implementation of Linear Model Predictive Control — Tutorial.   
https://arxiv.org/abs/2109.11986  

---
---
---

## Robust Model Predictive Control (RMPC)
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
Q \succeq 0,
\qquad
R \succ 0
$$

$$
Y = LQ_U
$$  

The LMI formulation of the problem:  

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

The feedback gain is recovered from  

$$
L = YQ_U^{-1}
$$

RMPC control input:  

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
