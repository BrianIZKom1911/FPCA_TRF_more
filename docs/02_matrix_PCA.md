Reference: Jolliffe, I.T. 2002. *Principal Component Analysis*, 2nd Ed. Springer, New York, NY.

# Background
## Derivation of Principal Components

Let $X=(X_1, X_2, \ldots, X_m)^{\top}$ be the m-dimensional *random vector*. To do standard PCA, we are seeking the linear combinations called **principal components** (PC) satisfying: 

(1) $w_k^{\top} X=(w_{k1}, \ldots, w_{km})(X_1, \ldots, X_m)^{\top}$ has maximum variance; 

(2) $w_k^{\top} X$ is uncorrelated with all preceding PCs, for $k=1, 2, \ldots, m$. 

- Up to $m$ PCs could be found, but it is hoped, in general, that most of the variation in $X$ will be accounted for by $p$ PCs, $p<<m$.

- For now, presume $\Sigma=\text{var}(X)$ is known, then the k-th PC turns out to be $Z_k=a_k^{\top} X$ where $a_k$ is an eigenvector of $\Sigma$ corresponding to its k-th largest eigenvalue $\lambda_k$. Furthermore, if we choose $a_k$ to have unit length, $a_k^{\top} a_k=1$, then $\text{var}(Z_k)=\lambda_k$.

**Derivation.**
(The maximization is subject to a constant norm of the coefficient vector; otherwise, the inner product can be as large as one wishes. For simplicity, just set it to unit length.)

> $$\underset{w_1\in \mathbb{R}}{\max} \text{var}(w_1^{\top} X)=w_1^{\top}\Sigma w_1 \text{ subject to } w_1^{\top} w_1=1$$
>
> Using Lagrange multipliers, 
>
> $$\underset{w_1\in \mathbb{R}}{\max} \text{var}(w_1^{\top} X)=w_1^{\top}\Sigma w_1 - \lambda (w_1^{\top}w_1-1)$$
> 
> Differentiation w.r.t. $w_1$ gives $\Sigma w_1-\lambda w_1=0$ or $(\Sigma -\lambda I_m) w_1=0$. Thus, $\lambda$ is an eigenvalue of $\Sigma$ and $w_1$ is the corresponding eigenvector. Then, the quantity to be maximized is $w_1^{\top}\Sigma w_1=w_1^{\top}\lambda w_1=\lambda$, so $\lambda$ must be as large as possible, which is the largest eigenvalue of $\Sigma$, i.e., $\lambda =\lambda_1$. From now on, let $a_j$ denote the unit-length eigenvector associated with the j-th largest eigenvalue $\lambda_j$.
>
> Now, for $k≥2$, suppose the first $k$ PCs are given by $Z_j=Xa_j$; $a_j^{\top}a_j=1$; $\text{cov}(a_j^{\top} X, a_i^{\top} X)=0$ for any $i<j$, $j=k-1, \ldots, 1$. (For $k=1$ we do not need to assume any conditions beyond the first step and prove directly the nature of $w_2$.) We want to prove the coefficient defining the (k+1)th PC is $a_{k+1}$. That is, $a_{k+1}$ maximizes $w_{k+1}^{\top}\Sigma w_{k+1}$ subject to $w_{k+1}^{\top} w_{k+1}=1$ and $\text{cov}(w_{k+1}^{\top} X, a_j^{\top} X)=0$. Note that
>
> $$\text{cov}(w_{k+1}^{\top} X,a_j^{\top} X)=w_{k+1}^{\top}\Sigma a_j=w_{k+1}^{\top}\lambda_j a_j=\lambda_j w_{k+1}^{\top} a_j$$
>
> Thus the zero correlation constraints are equivalent to $w_{k+1}^{\top} a_j=0$ for any $j\leq k$.
>
> $$\underset{w_{k+1}}{\max} w_{k+1}^{\top}\Sigma w_{k+1}-\lambda (w_{k+1}^{\top} w_{k+1}-1)-\sum_{j\leq k}\delta_j (w_{k+1}^{\top} a_j)$$
>
> Differentiation w.r.t. $w_{k+1}$ gives $\Sigma w_{k+1}-\lambda w_{k+1}-\sum \delta_j a_j=0$. Left multiplying $a_1^{\top}$ on both sides gives
>
> $$\begin{align}
> a_1^{\top} \Sigma w_{k+1}-\lambda a_1^{\top} w_{k+1}-\sum \delta_j a_1^{\top} a_j &=0\\
> \sum \delta_j a_1^{\top} a_j &=0
> \end{align}$$
>
> But $a_1^{\top} a_j=0$ for all $j\neq 1$ and $a_1^{\top} a_1=1$. So $\delta_1=0$.
> Doing so with $a_2^{\top},\ldots,a_k^{\top}$, it must be that $\delta_j=0$ for any $j\leq k$. Therefore, $\Sigma w_{k+1}-\lambda w_{k+1}=0$, or $(\Sigma -\lambda I_m) w_{k+1}=0$. $\lambda$ is an eigenvalue of $\Sigma$. Again, $w_{k+1}^{\top} \Sigma w_{k+1}=\lambda$ must be as large as possible. Assuming $\Sigma$ does not have repeated eigenvalues*, $\lambda$ cannot equal any of $\lambda_1, \ldots, \lambda_k$. If it did, it would follow that $w_{k+1}=a_k$ violating $w_{k+1}^{\top} a_k=0$. So, $\lambda =\lambda_{k+1}, w_{k+1}=a_{k+1}$.
> Hence, the proof follows from mathematical induction. ||

- $\Sigma$ might have repeated eigenvalues. That is, some eigenvalue's (algebraic or geometric) multiplicity $=q>1$. From [Spectral Theorem II](01_matrix_factorization.md/#spectral-theorem), since $\Sigma$ is symmetric (real and Hermitian), all its eigenvalues are real and the algebraic and geometric multiplicities agree.
In this case, the $q$ eigenvectors corresponding to the $q$ equal eigenvalues span a certain unique q-dimensional space, but, within this space, they are arbitrary.
Therefore, these $q$ PCs are not uniquely determined---Any set of orthogonal vectors that span this eigenspace can be chosen as the direction of each of the PCs. But the total variance explained by the components in this subspace is fixed. What we lose is the interpretability.
More discussions can be found in Jolliffe (2002), Sections 2.4, 3.7. 

## Properties (based on a known population variance)
Let $Z^q=(Z_1, Z_2, \ldots, Z_q)^{\top} (q\leq m)$. $Z_k$ is the k-th PC with k-th largest variance $\text{var}(Z_k)=\lambda_k$ which equals the k-th (largest) eigenvalues. Then $Z^q={A^q}^{\top} X$, where $A^q$ is the first $q$ columns of $A$, the $m\times m$ orthogonal matrix $(a_1 \ldots a_m)$ whose k-th column is the k-th eigenvectors of $\Sigma$. That is, the PCs are defined by an orthonormal linear transformation of $X$. We have

$$\Sigma A^q=A^q \Lambda_q$$

where $\Lambda_q=diag(\lambda_1,\ldots,\lambda_q)$. But only when $q=m$ do we have these equations:

$$\begin{align}
A^{\top}\Sigma A &=\Lambda \\
\Sigma &=A\Lambda A^{\top}
\end{align}$$

(Trivially, $A^1\equiv a_1; A^m\equiv A$)

> The derivation can be found in [Spectral Decomposition](01_matrix_factorization.md/#spectral-theorem).

**Property A1.** For any integer $q$, $1\leq q\leq m$, consider the orthonormal linear transformation $\xi=B^{\top} X$ of $X$, where $\xi$ is a q-dimensional vector and $B$ is an $(m\times q)$ matrix. Let $\Sigma_{\xi}:=\text{var}(\xi)=B^{\top} \Sigma B$. Then $tr(\Sigma_{\xi})$ is maximized by taking $B=A^q$.

**Property A2.**
Let $\xi=B^{\top} X$, then $tr(\Sigma_{\xi})$ is minimized by taking $B=A_{\*}^q$ where $A_{\*}^q$ consists of the last $q$ columns of $A$.

**Property A3.** (spectral decomposition)

$$\Sigma =\lambda_1 a_1 a_1^{\top}+\lambda_2 a_2 a_2^{\top}+\ldots+\lambda_m a_m a_m^{\top}$$

> This is a direct implication from $\Sigma =A\Lambda A^{\top}$ by expanding the RHS matrix product.

- Looking at diagonal elements, we see $\text{var}(X_j)=\sum_{k=1}^m \lambda_k a_kj^2$. Not only can we decompose the variances of all the elements of $X$ into decreasing contributions $(\lambda_1,\ldots,\lambda_m)$ due to each PC, we can also decompose the whole covariance matrix $\Sigma$ into contributions $\lambda_k a_k a_k^{\top}$ from each PC.
- Property A1 emphasizes that the PCs explain, successively, as much as possible of $tr(\Sigma)$. The current property shows, intuitively, that they also do a good job of explaining the off-diagonal elements of $\Sigma$.

**Corollary 1.** The linear combination of $X$ that has maximum variance conditional on the first $q$ PCs, $Z^q$, is precisely the (q+1)th PC. That is,

$$a_{q+1}= \underset{w}{\text{argmax}}\text{ var}(Xw|Z^q)$$

Notice: This proposition is true ONLY IF $E(X|Z^q)$ is linear in $Z^q$ and $\text{var}(X|Z^q)$ is constant, which is a necessary condition but not pointed out in the textbook.

> *Sketched proof.* Under the above condition, we have a useful result that $\text{var}(X|Z^q)$ can be decomposed as $\Sigma-\Sigma_{xz}\Sigma_{zz}^{-1}\Sigma_{zx}$, where $\Sigma_{zz}=\text{var}(Z^q)$, $\Sigma_{xz}=\text{cov}(X,Z^q)$, $\Sigma_{zx}=\Sigma_{xz}^{\top}$. (Without the above conditions, we only have $\Sigma-\Sigma_{xz}\Sigma_{zz}^{-1}\Sigma_{zx}=\text{var}(X-BZ^q)$, the variance of the prediction error from the best linear predictor of $X$ by $Z^q$, $B:=\Sigma_{xz}\Sigma_{zz}^{-1}$.)
> 
> The k-th column of $\Sigma_{xz}$ is $\lambda_k a_k$. The matrix $\Sigma_{zz}^{-1}=diag(\lambda_1^{-1},\ldots,\lambda_q^{-1}$), so
> 
> $$\Sigma_{xz}\Sigma_{zz}^{-1} \Sigma_{zx}=\sum_{k=1}^q\lambda_k a_k a_k^{\top}$$
>
> and
>
> $$\Sigma -\Sigma_{xz}\Sigma_{zz}^{-1} \Sigma_{zx}=\sum_{k=q+1}^m\lambda_k a_k a_k^{\top}$$
>
> Finding a linear function of $X$ having maximum conditional variance reduces to finding the eigenvalues and eigenvectors of the conditional covariance matrix. It's easy to verify that these are simply $(\lambda_{q+1},a_{q+1}),(\lambda_{q+2},a_{q+2}),\ldots,(\lambda_m,a_m)$. The eigenvector associated with the largest
of these eigenvalues is $a_{q+1}$. || 

**Property A4.**
Let $\xi=B^{\top} X$, then $\det(\Sigma_\xi)$ is maximized when $B=A^q$, in which case $\det(\Sigma_\xi)=\det⁡({A^q}^{\top}\Sigma A^q)=\prod_{k=1}^q \lambda_k$. 

**Prerequisite to Cor.1, Property A3**

> *Partition of variance-covariance matrix.* Let $Y=(Y_1,Y_2)^{\top}, Y\sim (\mu,\Sigma)$. Partitioning 
>
> $$
> \Sigma=\begin{pmatrix}
> \Sigma_{11} & \Sigma_{12}\\
> \Sigma_{21} & \Sigma_{22} \end{pmatrix}, \mu=\begin{pmatrix}
> \mu_1\\
> \mu_2\end{pmatrix}
> $$
>
> Define $Z=Y_1-AY_2, A:=\Sigma_{12} \Sigma_{22}^{-1}$. Its mean and variance are given as follows
>
> $$
> \begin{align}
> E(Z) &=\mu_1-A\mu_2\\
> \text{var}(Z) = \text{var}(Y_1-AY_2) &=\text{var}(Y_1)+A\text{var}(Y_2) A^{\top}-\text{cov}(Y_1,Y_2) A^{\top}-A\text{cov}(Y_2,Y_1)\\
> &=\Sigma_{11}+\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}\\
> &=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21} \quad \text{(Schur complement)}
> \end{align}
> $$
> 
> $Z$ is uncorrelated to $Y_2$ (not requiring normality):
>
> $$\text{cov}(Z,Y_2)=\text{cov}(Y_1,Y_2)-\text{cov}(AY_2,Y_2)=\Sigma_{12}-A\text{var}(Y_2)=\Sigma_{12}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{22}=0$$
>
> Thus, $E(Y_1|Y_2)=E(Z+AY_2|Y_2)=E(Z|Y_2)+AY_2$. 
>
> Now, suppose $Y$ is **normally distributed**. Then $Z$ and $Y_2$ are jointly normal, so they are independent. 
>
> $$E(Y_1|Y_2)=E(Z)+AY_2=\mu_1+A(Y_2-\mu_2 )=\mu_1-\Sigma_{12} \Sigma_{22}^{-1} \mu_2$$
>
> $$
> \begin{align}
> \text{var}(Y_1|Y_2) &=\text{var}(Z+AY_2|Y_2)\\
> &=\text{var}(Z|Y_2)+\text{var}(AY_2|Y_2)+\text{cov}(Z,AY_2|Y_2)+\text{cov}(AY_2,Z|Y_2)\\
> &=\text{var}(Z|Y_2)+0+0+0
> \end{align}
> $$
>
> It is because $\text{cov}(Z,f(X)|X)=0$ for any $Z\in R^k$ with $E\Vert Z\Vert^2<\infty$ and any function $f(.)$. So,
>
> $$
> \begin{align}
> \text{var}(Y_1|Y_2)=\text{var}(Z)&=\text{var}(Y_1-AY_2)\\
> &=\text{var}(Y_1)+A\text{var}(Y_2) A^{\top}-\text{cov}(Y_1,Y_2) A^{\top}-A\text{cov}(Y_2,Y_1)\\
> &=\Sigma_{11}+\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}\\
> &=\Sigma_{11}-\Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}
> \end{align}
> $$
>
> Q.E.D.

Also see [a direct derivation](https://statproofbook.github.io/P/mvn-cond.html) of the conditional density under normality. 

## Principal Components from the Correlation Matrix
- The derivation and properties of PCs considered above are based on the eigenvectors and eigenvalues of the **covariance matrix**. In practice, it is more common to deﬁne principal components as $Z=A^{\top} X^{\*}$ where $A$ now has columns consisting of the eigenvectors of the **correlation matrix** (of $X$), and $X^{\*}$ consists of standardized variables (of $X$). $X_j^{\*}:=\sigma_{jj}^{-1/2}X_j, j=1,\ldots,m$. Thus we can adopt the same approach to find the PCs from the covariance matrix of $X^{\*}$ (which is the correlation matrix of $X$).
- A third possibility, instead of using covariance or correlation matrices, is to use covariances of $X_j/\omega_j$, where the weights $\omega_j$ are chosen to reﬂect some a priori idea of the relative importance of the variables. --> Section 14.2
- All the properties of the previous two sections are still **valid** for correlation matrices, or indeed for covariances based on other sets of weights, except that we are now considering PCs of $X^{\*}$ (or some other transformation of $X$), instead of $X$.
- The eigenvalues and eigenvectors of the correlation matrix have no simple relationship with those of the corresponding covariance matrix. The transformation from $X$ to $X^{\*}$ is not orthogonal.

Merits of correlation-based PCs:
(1) More comparable (2) Insensitivity to the units of measurement used for each elements of $X$

Advantages of covariance-based PCs:
(1) Statistical inference based on sample PCs is easier --> Section 3.7 (2) However, in practice, it is more common to use PCA as a descriptive, rather than an inferential, tool.

<!-- *Special Notes:* When applying PCA to functional data, we must center the data before computing eigenvectors, which, in the population sense, utilizes $E(XX^{\top})$ rather than $\text{var}(X)$. What's more, we shall not standardize the data that represent functions by coefficient vectors. -->

## Sample Principal Components
Suppose each $x_i$ is an observed value of random vector $X$, and observations $\{x_1,\ldots,x_n\}$ are independently distributed. 
Let 

$$𝐗=\begin{bmatrix}
x_1^{\top}\\ 
\vdots\\ 
x_n^{\top}
\end{bmatrix}=\[𝐱_1 𝐱_2 \ldots 𝐱_m\]$$

denote the $n\times m$ observation matrix. If $\Sigma$ were still known, we would define $z_{ik}=a_k^{\top} x_i$ as the score for the i-th observation on the k-th (population) PC. However, it is usually *unknown*.

Therefore, let $\hat{z}_{i1}=w_1^{\top} x_i,i=1,\ldots,n$. Analogously,

$$\underset{w_1}{\max}(n-1)^{-1}\sum_i(\hat{z}_{i1}-\bar{z}_1)^2 \text{ subject to } w_1^{\top} w_1=1$$

Writing $\hat{z}_1=𝐗w_1$, the sample variance to be maximized may be written as 

$$(n-1)^{-1}(\hat{z}_1^{\top}\hat{z}_1-n\bar{z}_1^2)=(n-1)^{-1}(w_1^{\top}𝐗^{\top}𝐗w_1-n\hat{z}_1^2)$$ 

in which $\bar z_1=n^{-1}\hat z_1^{\top} 𝟏_n$. Next let $\hat z_{2i}=w_2^{\top} x_i$ and choose $w_2$ to maximize the sample variance subject to $w_2^{\top} w_2=1$ and $cov(\hat z_{2i}, \hat z_{1i})=0$. Continuing this process, we have a sample version of the definition of PCs.

To have a clear analogy, define $\tilde{𝐗}=𝐗-𝟏_n\bar{x}^{\top}=𝐗-n^{-1}𝟏_n 𝟏_n^{\top}𝐗=(I_n-n^{-1} 𝟏_n𝟏_n^{\top})𝐗$ as the matrix of observed values centered about the mean of each variable. 
Then, the sample variance of $\hat{z}_k$ can be written as $(n-1)^{-1}w_k^{\top}\tilde{𝐗}^{\top}\tilde{𝐗}w_k=w_k^{\top}Sw_k$. The derivation is straightforward by replacing $\Sigma$ with $S$. 
Hence, the k-th sample PC would be determined by the unit-length eigenvector $\hat{a}_k$ corresponding to the k-th largest eigenvalue of $S$.

- Same as Sec 1.1, define $\hat Z_k:=\hat a_k^{\top} X$ as the k-th **sample PC** \[The only difference is between $\hat a_k$ and $a_k$\] and call $\hat z_{ik}$ the **score** for the i-th observation on the k-th PC, thus $\hat z_k$ the k-th PC scores for the whole sample. $\hat z_k$, as the realization of $Z_k$, uses the sample vector to transform the sample data. Write $\hat{𝐙}=\[\hat z_1 \ldots \hat z_m\]=\[\hat z_{ik}\]_{n\times m}$. Then $\hat{𝐙}=𝐗\hat{A}$, where $\hat{A}=\[\hat a_1 \ldots \hat a_m\]$.
- The eigenspaces of $S≡(n-1)^{-1}\tilde{𝐗}^{\top}\tilde{𝐗}$ and $\tilde{𝐗}^{\top}\tilde{𝐗}$ are identical, and the eigenvalues of the former are simply $(n-1)^{-1}$ times the latter. Because of this, it will be convenient in some places below to work in terms of $\tilde{𝐗}^{\top}\tilde{𝐗}$ rather than $S$.
- Finally, it will be convenient to deﬁne the matrix of PC scores as $\tilde{𝐙}:=\tilde{𝐗}\hat{A}$. These PC scores have exactly the same variances and covariances as those in $\hat{𝐙}$ but have zero means.


## Properties

All conclusions hold by replacing $\Sigma$ with $S$. Formally, deﬁne $\xi_i=B^{\top} x_i$, where $B$ is an $(m\times q)$ matrix whose columns are orthonormal. Properties A1, A2, A4, A5, still hold, but with the sample covariance matrix $S_{\xi}$ replacing $\Sigma_{\xi}$ and $\hat{A}$ replacing $A$. 

**Property A7.** (PC's use in regression)
Consider $\tilde{𝐗}$, deﬁned as above, as $n$ observations on $m$ predictors $X$ measured about their sample means, and that the corresponding regression equation is

$$\tilde{𝐲}=\tilde{𝐗}\beta+𝛆$$

where $\tilde{𝐲}$ is the vector of $n$ observations on the dependent variable, again measured about the sample mean. Suppose $\tilde{𝐙}=\tilde{𝐗}B$, where $B$ is an $(m\times m)$ orthogonal matrix. The regression equation is then rewritten as

$$\tilde{𝐲}=\tilde{𝐙}\gamma+𝛆$$

where $\gamma=B^{-1}\beta$. The OLS estimator for $\gamma$ is $\hat{\gamma}=(\tilde{𝐙}^{\top}\tilde{𝐙})^{-1}\tilde{𝐙}^{\top}\tilde{𝐲}$.
Then the elements of $\hat{\gamma}$ have, successively, the smallest possible *conditional* variances if $B=\hat{A}$, the matrix whose k-th column is the k-th eigenvector of $\tilde{𝐗}^{\top}\tilde{𝐗}$, and hence that of $S$.

> [Proof] 
> From standard results in regression \[*with homoscedastic error\], the covariance matrix of the least squares estimator $\hat{\gamma}$ is proportional to
>
> $$( \tilde{𝐙}^{\top} \tilde{𝐙} )^{-1}=(B^{\top}\tilde{𝐗}^{\top}\tilde{𝐗}B)^{-1}=B^{-1}(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}(B^{\top})^{-1}=B^{\top}(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}B \quad\text{(orthogonality)}$$
>
> Conditional variance of $\hat\gamma_j$ is proportional to $\[(\tilde{𝐙}^{\top}\tilde{𝐙} )^{-1}\]_{jj}$. Minimizing such variances is equivalent to minimizing each $tr\left( B^{qT}(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}B^q \right), q=1,2,\ldots,m$. $B^q$ consists of the first $q$ columns of $B$.
> 
> But using **Property A2** with $\Sigma$ replaced by $(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}$, so that $tr\left(B^{qT}(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}B^q \right)$ is minimized by taking $B^q=\hat{A}_{\*}^q|(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}$, i.e., the last $q$ columns of the matrix whose k-th column is the k-th eigenvector of $( \tilde{𝐗}^{\top}\tilde{𝐗} )^{-1}$.
> Furthermore, $(\tilde{𝐗}^{\top}\tilde{𝐗})^{-1}$ has the same eigenvectors as $\tilde{𝐗}^{\top}\tilde{𝐗}$ except that their (eigenvalues') order is reversed, so that $B^q=\hat{A}^q |\tilde{𝐗}^{\top}\tilde{𝐗}$. And this holds for $q=1,2,\ldots,m$ ||

*Notice:* Without conditional homoskedasticity, $var(\hat{\gamma}|\tilde{𝐙})=(\tilde{𝐙}^{\top}\tilde{𝐙})^{-1}\tilde{𝐙}^{\top} \Sigma_{\varepsilon} \tilde{𝐙}(\tilde{𝐙}^{\top} \tilde{𝐙})^{-1}$ where $\Sigma_{\varepsilon}:=E(𝛆𝛆^{\top}|\tilde{𝐙})$. This condition is also omitted in the textbook.

## Singular Value Decomposition

Suppose $X$ is a matrix of $n$ observations on $m$ variables measured about their means. It can be written as

$$X=UDV^{\top}$$

where (i) $U, V$ are $(n\times r), (m\times r)$ matrices, each of which has orthonormal columns so that $U^{\top} U=I_r=V^{\top} V$. (ii) $D$ is $(r\times r)$ diagonal matrix, $r=rank(X)$. 

It directly results from applying the alternative [SVD Theorem](01_matrix_factorization.md#singular-value-decomposition-svd), 

*Notice: The proof in the textbook is flawed.*
