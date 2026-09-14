## Quadratic Programming (QP)

Linear MPC and DMC rely on optimization. The mentioned control schemes usually use quadratic cost functions for states and penalized control actions. With linear process models, box constraints, quadratic costs and finite horizons, the underlying optimization problem is typically categorized as a quadratic optimization problem, hence the name Quadratic Programming (QP). The following shows the standard QP structure and required conditions for optimality. Constructing the Lagrangian and satisfying Karush-Kuhn-Tucker conditions are the essential steps for reaching optimality when equality and inequality constraints are present. If no constraints are present, the problem structure often collapses to a least squares solution.


### 1. Standard QP Formulation

The standard form of a quadratic programming problem is

$$
\begin{aligned}
\min_{x} \quad &
J(x)
=
\frac{1}{2}x^{T}Hx
+
f^{T}x
\\[4pt]
\text{subject to} \quad &
A_{\mathrm{eq}}x
=
b_{\mathrm{eq}}
\\
&
A_{\mathrm{ineq}}x
\leq
b_{\mathrm{ineq}}
\\
&
x_{\mathrm{lb}}
\leq
x
\leq
x_{\mathrm{ub}}
\end{aligned}
$$

where:

* $x \in \mathbb{R}^{n}$   (the decision variables)
* $H \in \mathbb{R}^{n\times n}$
* $f \in \mathbb{R}^{n}$
* $A_{\mathrm{eq}} \in \mathbb{R}^{m_e\times n}$
* $b_{\mathrm{eq}} \in \mathbb{R}^{m_e}$
* $A_{\mathrm{ineq}} \in \mathbb{R}^{m_i\times n}$
* $b_{\mathrm{ineq}} \in \mathbb{R}^{m_i}$
* $x_{\mathrm{lb}} \in \mathbb{R}^{n}$  (lower bound for decision variables $x$)
* $x_{\mathrm{ub}} \in \mathbb{R}^{n}$ (upper bound for decision variables $x$)

The objective is to determine the optimal decision vector $x^*$ that minimizes $J(x)$ while satisfying all constraints.

---

### 2. Convexity Requirements

For the QP to be a convex optimization problem, the Hessian matrix must be positive semidefinite:

$$
H = H^T
$$

and

$$
H \succeq 0
$$

Positive semidefiniteness means that

$$
z^{T}Hz \geq 0
$$

for every vector $z$.

If the Hessian is positive definite,

$$
H \succ 0
$$

then the objective function is strictly convex.

For a feasible strictly convex QP, the optimal decision vector $x^*$ is unique.

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
g(x) \leq 0
$$

Define

$$
g(x)
=
A_{\mathrm{ineq}}x
-
b_{\mathrm{ineq}}
$$

and

$$
h(x)
=
A_{\mathrm{eq}}x
-
b_{\mathrm{eq}}
$$

The QP can therefore be expressed as

$$
\begin{aligned}
\min_{x} \quad &
\frac{1}{2}x^{T}Hx
+
f^{T}x
\\
\text{subject to} \quad &
h(x) = 0
\\
&
g(x) \leq 0
\end{aligned}
$$

---

### 4. Lagrangian

Introduce the Lagrange multiplier vectors

$$
\lambda
\in
\mathbb{R}^{m_e}
$$

for the equality constraints and

$$
\mu
\in
\mathbb{R}^{m_i}
$$

for the inequality constraints.

The Lagrangian is

$$
\mathcal{L}
=
\frac{1}{2}x^{T}Hx
+
f^{T}x
+
\lambda^{T}
\left(
A_{\mathrm{eq}}x
-
b_{\mathrm{eq}}
\right)
+
\mu^{T}
\left(
A_{\mathrm{ineq}}x
-
b_{\mathrm{ineq}}
\right)
$$

For the inequality constraints, the Lagrange multipliers must satisfy

$$
\mu \geq 0
$$

---

### 5. Karush-Kuhn-Tucker (KKT) Conditions

For a convex QP, the KKT conditions characterize the optimal solution, provided an appropriate constraint qualification is satisfied.

The optimal solution is denoted by

$$
x = x^*
$$

with corresponding optimal Lagrange multipliers

$$
\lambda
=
\lambda^*
$$

and

$$
\mu
=
\mu^*
$$

The four KKT conditions are stationarity, primal feasibility, dual feasibility, and complementary slackness.

### 5.1 Stationarity

The gradient of the Lagrangian with respect to $x$ must be zero:

$$
\nabla_{x}\mathcal{L}
=
0.
$$

Therefore,

$$
Hx
+
f
+
A_{\mathrm{eq}}^T\lambda
+
A_{\mathrm{ineq}}^T\mu
=
0
$$

At the optimum,

$$
Hx^*
+
f
+
A_{\mathrm{eq}}^T\lambda^*
+
A_{\mathrm{ineq}}^T\mu^*
=
0
$$

---

#### 5.2 Primal Feasibility

The optimal decision vector must satisfy all equality and inequality constraints:

$$
A_{\mathrm{eq}}x^*
=
b_{\mathrm{eq}}
$$

and

$$
A_{\mathrm{ineq}}x^*
\leq
b_{\mathrm{ineq}}
$$

The variable bounds must also be satisfied:

$$
x_{\mathrm{lb}}
\leq
x^*
\leq
x_{\mathrm{ub}}
$$

---

#### 5.3 Dual Feasibility

The multipliers associated with inequality constraints must be nonnegative:

$$
\mu^*
\geq
0
$$

There is no corresponding sign restriction on the equality multipliers:

$$
\lambda^*
\in
\mathbb{R}^{m_e}
$$

---

#### 5.4 Complementary Slackness

For each inequality constraint,

$$
\mu_i^*
\left(
A_{\mathrm{ineq}}x^*
-
b_{\mathrm{ineq}}
\right)_i
=
0
$$

In vector form,

$$
\mu^{*T}
\left(
A_{\mathrm{ineq}}x^*
-
b_{\mathrm{ineq}}
\right)
=
0
$$

Complementary slackness means that each inequality constraint satisfies one of two conditions:

$$
\mu_i^* = 0
$$

or

$$
\left(
A_{\mathrm{ineq}}x^*
-
b_{\mathrm{ineq}}
\right)_i
=
0
$$

Thus, an inequality constraint is either inactive, with zero multiplier, or active, with zero constraint slack.

---

### 6. Complete KKT Conditions

The complete KKT conditions for the QP are therefore

$$
\boxed{
Hx^*
+
f
+
A_{\mathrm{eq}}^T\lambda^*
+
A_{\mathrm{ineq}}^T\mu^*
=
0
}
$$

$$
\boxed{
A_{\mathrm{eq}}x^*
-
b_{\mathrm{eq}}
=
0
}
$$

$$
\boxed{
A_{\mathrm{ineq}}x^*
-
b_{\mathrm{ineq}}
\leq
0
}
$$

$$
\boxed{
\mu^*
\geq
0
}
$$

and

$$
\boxed{
\mu^{*T}
\left(
A_{\mathrm{ineq}}x^*
-
b_{\mathrm{ineq}}
\right)
=
0
}
$$

These conditions collectively characterize the optimal solution.

---

### 7. Unconstrained QP

If there are no constraints, the optimization problem reduces to

$$
\min_{x}
\quad
\frac{1}{2}x^{T}Hx
+
f^{T}x
$$

The first-order optimality condition is

$$
\nabla J(x)
=
Hx
+
f
=
0
$$

Therefore, if $H$ is nonsingular,

$$
\boxed{
x^*
=
-H^{-1}f
}
$$

For a strictly convex QP, where $H\succ0$, this solution is the unique global minimizer.

---

### 8. Equality-Constrained QP

If only equality constraints are present,

$$
\begin{aligned}
\min_{x}\quad&
\frac{1}{2}x^{T}Hx
+
f^{T}x
\\
\text{subject to}\quad&
A_{\mathrm{eq}}x
=
b_{\mathrm{eq}}
\end{aligned}
$$

the KKT equations can be written as the linear system

$$
\begin{bmatrix}
H & A_{\mathrm{eq}}^T \\
A_{\mathrm{eq}} & 0
\end{bmatrix}
\begin{bmatrix}
x^* \\
\lambda^*
\end{bmatrix}
=
\begin{bmatrix}
-f \\
b_{\mathrm{eq}}
\end{bmatrix}
$$

Solving this system gives the optimal decision vector $x^*$ and its corresponding equality multipliers $\lambda^*$.

---

### 9. Inequality-Constrained QP and Active Constraints

For inequality-constrained problems, the main difficulty is determining which constraints are active at the optimum.

An inequality constraint

$$
a_i^Tx
\leq
b_i
$$

is active if

$$
a_i^Tx^*
=
b_i
$$

It is inactive if

$$
a_i^Tx^*
<
b_i
$$

From complementary slackness:

$$
\mu_i^*
\left(
a_i^Tx^*
-
b_i
\right)
=
0
$$

Therefore:

$$
a_i^Tx^*
<
b_i
\quad\Rightarrow\quad
\mu_i^*=0
$$

whereas

$$
\mu_i^*>0
\quad\Rightarrow\quad
a_i^Tx^*
=
b_i
$$

QP algorithms exploit this structure to determine the active set or otherwise solve the constrained optimization problem directly.

---

### 10. Conditions for an Optimal Solution

A QP has a well-defined global solution when the following requirements are appropriately satisfied:

1. Well-defined objective

   The matrices $H$ and $f$ must be specified.

2. Linear constraints

   The equality and inequality constraints must have the form

   $$
   A_{\mathrm{eq}}x
   =
   b_{\mathrm{eq}}
   $$

   and

   $$
   A_{\mathrm{ineq}}x
   \leq
   b_{\mathrm{ineq}}
   $$

3. Convexity

   For a convex QP,

   $$
   H\succeq0
   $$

4. Feasibility

   There must exist at least one $x$ satisfying all constraints.

5. Constraint qualification

   An appropriate constraint qualification, such as LICQ or Slater's condition where applicable, should hold so that the KKT conditions properly characterize optimality.

6. KKT optimality

   The candidate solution must satisfy stationarity, primal feasibility, dual feasibility, and complementary slackness.

For a convex QP, satisfaction of the KKT conditions is sufficient for global optimality.

If, in addition,

$$
H\succ0
$$

the objective is strictly convex and the optimal decision vector is unique whenever the feasible set is nonempty.

---

### 11. Summary

The general QP can therefore be summarized as

$$
\boxed{
\begin{aligned}
\min_{x}\quad&
\frac{1}{2}\mathbf{x}^{T}Hx
+
f^{T}x
\\
\text{subject to}\quad&
A_{\mathrm{eq}}x
=
b_{\mathrm{eq}}
\\
&
A_{\mathrm{ineq}}x
\leq
b_{\mathrm{ineq}}
\end{aligned}
}
$$

The optimal solution $x^*$ is characterized by

$$
\boxed{
\begin{aligned}
Hx^*
+f
+A_{\mathrm{eq}}^T\lambda^*
+A_{\mathrm{ineq}}^T\mu^*
&=0
\\
A_{\mathrm{eq}}x^*
-b_{\mathrm{eq}}
&=0
\\
A_{\mathrm{ineq}}x^*
-b_{\mathrm{ineq}}
&\leq0
\\
\mu^*
&\geq0
\\
\mu^{*T}
\left(
A_{\mathrm{ineq}}x^*
-b_{\mathrm{ineq}}
\right)
&=0
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

