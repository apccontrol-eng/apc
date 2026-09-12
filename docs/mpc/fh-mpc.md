
## Finite-horizon Model Predictive Control (MPC)

Finite-horizon MPC is a control scheme where the linear nominal model of the plant is at every loop instance used for predicting a future horizon with the knowledge or measured state information. The prediction horizon is of finite length and therefore does not guarantee stability. However the common practise is to have a reasonably long prediction horizon (depending on system and model needs) as longer prediction horizon gets closer to the infinite-horizon results. The finite-horizon formulation can therefore be thought as a scheduling problem that obeys linear dynamics. Linear model, box constraints, quadratic cost function and state enforcing equality constraints guarantee convexity of the optimization problem. The LQR type analysis about controllability is required for the model matrices A and B. The connection of finite-horizon LQR and linear finite-horizon MPC is in that MPC allows constraints and uses new state measurements. MPC optimizes control action in every control loop instance even though the optimization solution has control actions for future control actions. This is achieved by considering only the first control action from the optimization solution. For short prediction lengths the optimization problem is dense and for long horizons the problem becomes sparse if the resulting QP optimization problem is carried out with lifted system matrices.

The nominal plant linear state-space model:  

$$
x_{k+1} = A x_k + B u_k
$$  
where  
$$
A \in \mathbb{R}^{n \times n}
$$  
$$
B \in \mathbb{R}^{n \times m}
$$   
$$
x_{k} \in \mathbb{R}^{n \times 1}
$$  
$$
u_{k} \in \mathbb{R}^{m \times 1}
$$  

The quadratic cost on states and inputs for the optimization problem:  
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

The controls are always bounded by real-life constraints which can be formulated as box constraints (lower and upper bounds) on control inputs:  
$$
u_{lb} \leq u_{k} \leq u_{ub}  
$$  

Lifted system matrices form when prediction horizon is set to $N$:

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

### References
Michael Fink (2021).  
Implementation of Linear Model Predictive Control — Tutorial.   
https://arxiv.org/abs/2109.11986  

Maciejowski, J. M. (2002).
Predictive Control with Constraints.
Prentice Hall.


---
---
---
