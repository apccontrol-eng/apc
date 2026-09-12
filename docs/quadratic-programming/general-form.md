## Quadratic Programming (QP)

Linear MPC and DMC rely on optimization. The mentioned control schemes usually use quadratic cost functions for states and penalized control actions. With linear process models, box constraints, quadratic costs and finite horizons, the underlying optimization problem is typically categorized as a quadratic optimization problem, hence the name Quadratic Programming (QP). 


### 1. Standard QP Formulation

The standard form of a quadratic programming problem is

$$
\begin{aligned}
\min_{\mathbf{x}} \quad &
J(\mathbf{x})
=
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}
\\[4pt]
\text{subject to} \quad &
A_{\mathrm{eq}}\mathbf{x}
=
\mathbf{b}_{\mathrm{eq}}
\\
&
A_{\mathrm{ineq}}\mathbf{x}
\leq
\mathbf{b}_{\mathrm{ineq}}
\\
&
\mathbf{x}_{\mathrm{lb}}
\leq
\mathbf{x}
\leq
\mathbf{x}_{\mathrm{ub}}.
\end{aligned}
$$

where:

* $\mathbf{x} \in \mathbb{R}^{n}$ is the **decision-variable vector**.
* $H \in \mathbb{R}^{n\times n}$ is the **Hessian matrix** of the quadratic objective.
* $\mathbf{f} \in \mathbb{R}^{n}$ is the **linear objective coefficient vector**.
* $A_{\mathrm{eq}} \in \mathbb{R}^{m_e\times n}$ is the **equality-constraint matrix**.
* $\mathbf{b}_{\mathrm{eq}} \in \mathbb{R}^{m_e}$ is the **equality-constraint vector**.
* $A_{\mathrm{ineq}} \in \mathbb{R}^{m_i\times n}$ is the **inequality-constraint matrix**.
* $\mathbf{b}_{\mathrm{ineq}} \in \mathbb{R}^{m_i}$ is the **inequality-constraint vector**.
* $\mathbf{x}*{\mathrm{lb}}$ and $\mathbf{x}*{\mathrm{ub}}$ are the **lower and upper bounds** on the decision variables.

The objective function is therefore

$$
J(\mathbf{x})
=
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}.
$$

The objective is to determine the optimal decision vector $\mathbf{x}^*$ that minimizes $J(\mathbf{x})$ while satisfying all constraints.

---

### 2. Convexity Requirements

For the QP to be a **convex optimization problem**, the Hessian matrix must be positive semidefinite:

$$
H = H^T
$$

and

$$
H \succeq 0.
$$

Positive semidefiniteness means that

$$
\mathbf{z}^{T}H\mathbf{z} \geq 0
$$

for every vector $\mathbf{z}$.

If the Hessian is positive definite,

$$
H \succ 0,
$$

then the objective function is **strictly convex**.

For a feasible strictly convex QP, the optimal decision vector $\mathbf{x}^*$ is unique.

Thus:

$$
H \succeq 0
\quad\Rightarrow\quad
\text{convex QP}
$$

while

$$
H \succ 0
\quad\Rightarrow\quad
\text{strictly convex QP and unique minimizer}.
$$

---

### 3. Equality and Inequality Constraints

For deriving the optimality conditions, the inequality constraints are written in the standard form

$$
\mathbf{g}(\mathbf{x}) \leq 0.
$$

Define

$$
\mathbf{g}(\mathbf{x})
=
A_{\mathrm{ineq}}\mathbf{x}
-
\mathbf{b}_{\mathrm{ineq}}
$$

and

$$
\mathbf{h}(\mathbf{x})
=
A_{\mathrm{eq}}\mathbf{x}
-
\mathbf{b}_{\mathrm{eq}}.
$$

The QP can therefore be expressed as

$$
\begin{aligned}
\min_{\mathbf{x}} \quad &
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}
\\
\text{subject to} \quad &
\mathbf{h}(\mathbf{x}) = 0
\\
&
\mathbf{g}(\mathbf{x}) \leq 0.
\end{aligned}
$$

---

### 4. Lagrangian

Introduce the Lagrange multiplier vectors

$$
\boldsymbol{\lambda}
\in
\mathbb{R}^{m_e}
$$

for the equality constraints and

$$
\boldsymbol{\mu}
\in
\mathbb{R}^{m_i}
$$

for the inequality constraints.

The Lagrangian is

$$
\mathcal{L}
=
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}
+
\boldsymbol{\lambda}^{T}
\left(
A_{\mathrm{eq}}\mathbf{x}
-
\mathbf{b}_{\mathrm{eq}}
\right)
+
\boldsymbol{\mu}^{T}
\left(
A_{\mathrm{ineq}}\mathbf{x}
-
\mathbf{b}_{\mathrm{ineq}}
\right).
$$

For the inequality constraints, the Lagrange multipliers must satisfy

$$
\boldsymbol{\mu} \geq 0.
$$

---

### 5. Karush-Kuhn-Tucker (KKT) Conditions

For a convex QP, the KKT conditions characterize the optimal solution, provided an appropriate constraint qualification is satisfied.

The optimal solution is denoted by

$$
\mathbf{x} = \mathbf{x}^*,
$$

with corresponding optimal Lagrange multipliers

$$
\boldsymbol{\lambda}
=
\boldsymbol{\lambda}^*
$$

and

$$
\boldsymbol{\mu}
=
\boldsymbol{\mu}^*.
$$

The four KKT conditions are **stationarity, primal feasibility, dual feasibility, and complementary slackness**.

### 5.1 Stationarity

The gradient of the Lagrangian with respect to $\mathbf{x}$ must be zero:

$$
\nabla_{\mathbf{x}}\mathcal{L}
=
0.
$$

Therefore,

$$
H\mathbf{x}
+
\mathbf{f}
+
A_{\mathrm{eq}}^T\boldsymbol{\lambda}
+
A_{\mathrm{ineq}}^T\boldsymbol{\mu}
=
0.
$$

At the optimum,

$$
H\mathbf{x}^*
+
\mathbf{f}
+
A_{\mathrm{eq}}^T\boldsymbol{\lambda}^*
+
A_{\mathrm{ineq}}^T\boldsymbol{\mu}^*
=
0.
$$

---

#### 5.2 Primal Feasibility

The optimal decision vector must satisfy all equality and inequality constraints:

$$
A_{\mathrm{eq}}\mathbf{x}^*
=
\mathbf{b}_{\mathrm{eq}}
$$

and

$$
A_{\mathrm{ineq}}\mathbf{x}^*
\leq
\mathbf{b}_{\mathrm{ineq}}.
$$

The variable bounds must also be satisfied:

$$
\mathbf{x}_{\mathrm{lb}}
\leq
\mathbf{x}^*
\leq
\mathbf{x}_{\mathrm{ub}}.
$$

---

#### 5.3 Dual Feasibility

The multipliers associated with inequality constraints must be nonnegative:

$$
\boldsymbol{\mu}^*
\geq
0.
$$

There is no corresponding sign restriction on the equality multipliers:

$$
\boldsymbol{\lambda}^*
\in
\mathbb{R}^{m_e}.
$$

---

#### 5.4 Complementary Slackness

For each inequality constraint,

$$
\mu_i^*
\left(
A_{\mathrm{ineq}}\mathbf{x}^*
-
\mathbf{b}_{\mathrm{ineq}}
\right)_i
=
0.
$$

In vector form,

$$
\boldsymbol{\mu}^{*T}
\left(
A_{\mathrm{ineq}}\mathbf{x}^*
-
\mathbf{b}_{\mathrm{ineq}}
\right)
=
0.
$$

Complementary slackness means that each inequality constraint satisfies one of two conditions:

$$
\mu_i^* = 0
$$

or

$$
\left(
A_{\mathrm{ineq}}\mathbf{x}^*
-
\mathbf{b}_{\mathrm{ineq}}
\right)_i
=
0.
$$

Thus, an inequality constraint is either **inactive**, with zero multiplier, or **active**, with zero constraint slack.

---

### 6. Complete KKT Conditions

The complete KKT conditions for the QP are therefore

$$
\boxed{
H\mathbf{x}^*
+
\mathbf{f}
+
A_{\mathrm{eq}}^T\boldsymbol{\lambda}^*
+
A_{\mathrm{ineq}}^T\boldsymbol{\mu}^*
=
0
}
$$

$$
\boxed{
A_{\mathrm{eq}}\mathbf{x}^*
-
\mathbf{b}_{\mathrm{eq}}
=
0
}
$$

$$
\boxed{
A_{\mathrm{ineq}}\mathbf{x}^*
-
\mathbf{b}_{\mathrm{ineq}}
\leq
0
}
$$

$$
\boxed{
\boldsymbol{\mu}^*
\geq
0
}
$$

and

$$
\boxed{
\boldsymbol{\mu}^{*T}
\left(
A_{\mathrm{ineq}}\mathbf{x}^*
-
\mathbf{b}_{\mathrm{ineq}}
\right)
=
0.
}
$$

These conditions collectively characterize the optimal solution.

---

### 7. Unconstrained QP

If there are no constraints, the optimization problem reduces to

$$
\min_{\mathbf{x}}
\quad
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}.
$$

The first-order optimality condition is

$$
\nabla J(\mathbf{x})
=
H\mathbf{x}
+
\mathbf{f}
=
0.
$$

Therefore, if $H$ is nonsingular,

$$
\boxed{
\mathbf{x}^*
=
-H^{-1}\mathbf{f}.
}
$$

For a strictly convex QP, where $H\succ0$, this solution is the unique global minimizer.

---

### 8. Equality-Constrained QP

If only equality constraints are present,

$$
\begin{aligned}
\min_{\mathbf{x}}\quad&
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}
\\
\text{subject to}\quad&
A_{\mathrm{eq}}\mathbf{x}
=
\mathbf{b}_{\mathrm{eq}},
\end{aligned}
$$

the KKT equations can be written as the linear system

$$
\begin{bmatrix}
H & A_{\mathrm{eq}}^T \\
A_{\mathrm{eq}} & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{x}^* \\
\boldsymbol{\lambda}^*
\end{bmatrix}
=
\begin{bmatrix}
-\mathbf{f} \\
\mathbf{b}_{\mathrm{eq}}
\end{bmatrix}.
$$

Solving this system gives the optimal decision vector $\mathbf{x}^*$ and its corresponding equality multipliers $\boldsymbol{\lambda}^*$.

---

### 9. Inequality-Constrained QP and Active Constraints

For inequality-constrained problems, the main difficulty is determining which constraints are **active** at the optimum.

An inequality constraint

$$
a_i^T\mathbf{x}
\leq
b_i
$$

is active if

$$
a_i^T\mathbf{x}^*
=
b_i.
$$

It is inactive if

$$
a_i^T\mathbf{x}^*
<
b_i.
$$

From complementary slackness:

$$
\mu_i^*
\left(
a_i^T\mathbf{x}^*
-
b_i
\right)
=
0.
$$

Therefore:

$$
a_i^T\mathbf{x}^*
<
b_i
\quad\Rightarrow\quad
\mu_i^*=0,
$$

whereas

$$
\mu_i^*>0
\quad\Rightarrow\quad
a_i^T\mathbf{x}^*
=
b_i.
$$

QP algorithms exploit this structure to determine the active set or otherwise solve the constrained optimization problem directly.

---

### 10. Conditions for an Optimal Solution

A QP has a well-defined global solution when the following requirements are appropriately satisfied:

1. **Well-defined objective**

   The matrices $H$ and $\mathbf{f}$ must be specified.

2. **Linear constraints**

   The equality and inequality constraints must have the form

   $$
   A_{\mathrm{eq}}\mathbf{x}
   =
   \mathbf{b}_{\mathrm{eq}}
   $$

   and

   $$
   A_{\mathrm{ineq}}\mathbf{x}
   \leq
   \mathbf{b}_{\mathrm{ineq}}.
   $$

3. **Convexity**

   For a convex QP,

   $$
   H\succeq0.
   $$

4. **Feasibility**

   There must exist at least one $\mathbf{x}$ satisfying all constraints.

5. **Constraint qualification**

   An appropriate constraint qualification, such as LICQ or Slater's condition where applicable, should hold so that the KKT conditions properly characterize optimality.

6. **KKT optimality**

   The candidate solution must satisfy stationarity, primal feasibility, dual feasibility, and complementary slackness.

For a convex QP, satisfaction of the KKT conditions is sufficient for **global optimality**.

If, in addition,

$$
H\succ0,
$$

the objective is strictly convex and the optimal decision vector is unique whenever the feasible set is nonempty.

---

### 11. Summary

The general QP can therefore be summarized as

$$
\boxed{
\begin{aligned}
\min_{\mathbf{x}}\quad&
\frac{1}{2}\mathbf{x}^{T}H\mathbf{x}
+
\mathbf{f}^{T}\mathbf{x}
\\
\text{subject to}\quad&
A_{\mathrm{eq}}\mathbf{x}
=
\mathbf{b}_{\mathrm{eq}}
\\
&
A_{\mathrm{ineq}}\mathbf{x}
\leq
\mathbf{b}_{\mathrm{ineq}}.
\end{aligned}
}
$$

The optimal solution $\mathbf{x}^*$ is characterized by

$$
\boxed{
\begin{aligned}
H\mathbf{x}^*
+\mathbf{f}
+A_{\mathrm{eq}}^T\boldsymbol{\lambda}^*
+A_{\mathrm{ineq}}^T\boldsymbol{\mu}^*
&=0,
\\
A_{\mathrm{eq}}\mathbf{x}^*
-\mathbf{b}_{\mathrm{eq}}
&=0,
\\
A_{\mathrm{ineq}}\mathbf{x}^*
-\mathbf{b}_{\mathrm{ineq}}
&\leq0,
\\
\boldsymbol{\mu}^*
&\geq0,
\\
\boldsymbol{\mu}^{*T}
\left(
A_{\mathrm{ineq}}\mathbf{x}^*
-\mathbf{b}_{\mathrm{ineq}}
\right)
&=0.
\end{aligned}
}
$$

For a convex QP, these KKT conditions provide the fundamental optimality conditions used by QP solvers to determine the globally optimal decision vector.


### References

Boyd, S., & Vandenberghe, L. (2004).
*Convex Optimization*.
Cambridge University Press.
[**https://doi.org/10.1017/CBO9780511804441**](https://doi.org/10.1017/CBO9780511804441)

Nocedal, J., & Wright, S. J. (2006).
*Numerical Optimization* (2nd ed.).
Springer New York.
[**https://doi.org/10.1007/978-0-387-40065-5**](https://doi.org/10.1007/978-0-387-40065-5)

