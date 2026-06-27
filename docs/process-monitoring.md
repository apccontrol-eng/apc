## PCA

PCA relation (X auto-scaled):  
```math
X = TP^{T} +E
```
In rank-1 terms we solve the scores T and loadings P separately and deflate X either to the full column space or just to get e.g. the first two principal components.

```math
X=tp^{T}
```  
```math
t=Xp
```  
```math
var(t) = \frac{1}{n-1} t^{T}t \propto t^{T}t
```  
$\max_{p} var(t)$
subject to 
$p^{T}p=1$

$\max_{p} (Xp)^{T}(Xp)$
subject to
$p^{T}p=1$  

$\max_{p} p^{T}X^{T}Xp - \lambda(p^{T}p-1)$  

$X^{T}Xp - \lambda(p)=0$  
$(p^{T}p-1)=0$  

$X^{T}Xp = \lambda p$  
$p^{T}p = 1$  

This is an eigenvalue problem (EVP) for which there are multiple ways of solving it. After p (the loading) is solved, t (the scores) are available by definition t = Xp.

The remaining scores t and loadings p are calculated by repeating the optimization problem for deflated X.

The deflation step:
```math
X:=X-tp^{T}
```  

Suppose $k$ principal components are chosen, then the X is approximated as:  

```math
\hat{X} = T_{1:k}P_{1:k}^{T}
```  
The common ways of directly solving for all scores T and loadings P directly are SVD and Eigenvalue decomposition.  
SVD: $X = TP^{T}$ where $T = U\Sigma$ and $P = V$  


### Reference
Dunn, K. G. (2026).  
*Process Improvement using Data* (v2026.05.19).  
Zenodo.  
https://doi.org/10.5281/zenodo.20284935  

---
---
---
## PLSR

PLSR outer relations (X and Y auto-scaled):  
```math
X = TP^{T} +E
```
```math
Y = UQ^{T} +F
```
PLSR inner relation (we are interested in regression/prediction of Y variables):  
```math
U = TB +R
```  

In rank-1 terms, PLS is solved iteratively.  

```math
t=Xw
```
```math
u=Yc
```  
```math
cov(t,u) = \frac{1}{n-1} t^{T}u \propto t^{T}u
```  
$\max_{w,c} cov(t,u)$  
subject to  
$w^{T}w=1$  
$c^{T}c=1$  

$\max_{w,c} (w^{T}X^{T}Yc)$  
subject to  
$w^{T}w=1$  
$c^{T}c=1$  

$\max_{w,c} w^{T}X^{T}Yc - \lambda_{1}(w^{T}w-1) - \lambda_{2}(c^{T}c-1)$  

Differentiation w.r.t Lagrange multipliers:  
$w^{T}w-1=0$  
$c^{T}c-1=0$  

Differentiation w.r.t $w$:  
$X^{T}Yc-2\lambda_1w=0$  

Differentiation w.r.t $c$:  
$Y^{T}Xw-2\lambda_2c=0$  

By substituting $c=\frac{1}{2\lambda_2}Y^{T}Xw$ to $X^{T}Yc=2\lambda_1w$:  
$X^{T}YY^{T}Xw=2\lambda_1 2\lambda_2 w$  
and setting $\lambda = (2\lambda_1) (2\lambda_2)$ the equation becomes a regular EVP:  
$X^{T}YY^{T}Xw=\lambda w$  

For solving c, $w=\frac{1}{2\lambda_1}X^{T}Yc$ to $Y^{T}Xw=2\lambda_2c$:  
$Y^{T}XX^{T}Yc=(2\lambda_1) (2\lambda_2) c$  
$Y^{T}XX^{T}Yc=\lambda c$  

The solution to maximizing covariance is a matter of finding the solution to this set of equations (notice similarities with SVD on $X^{T}Y$):  
```math
X^{T}YY^{T}Xw=\lambda w  
```
```math
Y^{T}XX^{T}Yc=\lambda c  
```
```math
w^{T}w=1  
```
```math
c^{T}c=1  
```  

The scores $t$ and $u$ are readily solved by calculating $t = Xw$ and $u = Yc$.
The loadings for the outer PLSR relation are solved by ordinary least squares:  

```math
p = (X^{T}w)/(w^{T}w)
```  
```math
q = (Y^{T}c)/(c^{T}c)
```  

The inner relation is then calculated by ordinary least squares on the scores:  

```math
b = (u^{T}t)/(t^{T}t)
```  
After these steps we have $w$, $c$, $p$, $q$ and $b$. The next latent variables are calculated for the deflated matrices:  

```math
X:=X-tp^{T}
```  
```math
Y:=Y-uq^{T}
```  
or
```math
Y:=Y-tbq^{T}
```  

The PLSR model is usually calculated with NIPALS algorithm.  

The following matrix is useful for predicting purposes:  
```math
W^{*} = W(P^{T}W)^{-1}
```  
which is used for new measurement data (autoscaled with calibration means and standard deviations) followingly:  
```math
T = XW^{*}
```  
```math
U = TB
```  
```math
\hat{Y} = TBQ^{T} = XW^{*}BQ^{T}
```  
Which is still in autoscaled form and is brought back to original scale by:  
```math
\hat{Y}_{unscaled} = \hat{Y}diag(s_{Y}) + 1\mu_{Y}
```  

The autoscaling that is assumed for both PCA and PLSR can be expressed in the following way:

```math
Y = (Y_{unscaled} - 1\mu_{Y})diag(s_{Y})^{-1}
```  
```math
X = (X_{unscaled} - 1\mu_{X})diag(s_{X})^{-1}
```  
$\mu_{X}$: mean vector of calibration block $X$.  
$\mu_{Y}$: mean vector of calibration block $Y$.  
$s_{X}$: sample standard deviation vector of calibration block $X$.  
$s_{Y}$: sample standard deviation vector of calibration block $Y$.  

The PLSR predicting power is in first checking whether the scores T of new X data fall under threshold which gives confidence for predicting estimates.  

### Reference(s)
Rosipal, R., & Krämer, N. (2006)   
*Overview and Recent Advances in Partial Least Squares*.  
In C. Saunders et al. (Eds.), *Subspace, Latent Structure and Feature Selection* (LNCS 3940, pp. 34–51).  
Springer.  
https://www.ofai.at/~roman.rosipal/Papers/pls_book06.pdf

Geladi, P., & Kowalski, B. R. (1986)  
*Partial least-squares regression: A tutorial*  
*Analytica Chimica Acta*, *185*, 1–17.  
https://doi.org/10.1016/0003-2670(86)80028-9  

Wegelin, J. A. (2000)  
*A survey of partial least squares (PLS) methods, with emphasis on the two-block case* (Technical Report No. 371)  
Department of Statistics, University of Washington, Seattle  
Available at: https://stat.uw.edu/research/tech-reports/survey-partial-least-squares-pls-methods-emphasis-two-block-case  

Qin, S. J. (1993)  
*Partial least squares regression for recursive system identification*  
In *Proceedings of the 32nd IEEE Conference on Decision and Control* (pp. 2617–2621).  
IEEE.  
https://doi.org/10.1109/CDC.1993.325671  
