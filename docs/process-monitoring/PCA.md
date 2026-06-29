
## PCA

PCA relation (X auto-scaled):  

$$
X = TP^{T} +E
$$

In rank-1 terms we solve the scores T and loadings P separately and deflate X either to the full column space or just to get e.g. the first two principal components.

$$
X=tp^{T}
$$  

$$
t=Xp
$$  

$$
var(t) = \frac{1}{n-1} t^{T}t \propto t^{T}t
$$  

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

$$
X:=X-tp^{T}
$$  

Suppose $k$ principal components are chosen, then the X is approximated as:  

$$
\hat{X} = T_{1:k}P_{1:k}^{T}
$$  

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
