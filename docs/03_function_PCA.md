Reference: 

[1] Ramsay, J.O. and B.W. Silverman. 2005. *Functional Data Analysis*, 2nd Ed. Springer, New York, NY.

[2] Bosq, Denis. 2000. *Linear Processes in Function Spaces*, Springer, New York, NY.

# Intuition
## Revisit of the Matrix Problem

Consider the random sample $\{x_1,\ldots,x_n\}$ whose elements are $\mathbb{R}^m$-valued random vectors. $\mathbb{R}^m$ is an *inner product space* over real numbers with **inner product** $\langle x, y \rangle =x^{\top}y\in\mathbb{R}$. The **outer product** can be viewed as a "tensor product" defined for $𝐮\in\mathbb{R}^m, 𝐯\in\mathbb{R}^n$ as

$$
𝐮\otimes 𝐯=𝐮𝐯^{\top}=\begin{pmatrix}
u_1v_1 & \cdots & u_1v_n\\
\vdots & \ddots & \vdots\\
u_mv_1 & \cdots & u_mv_n
\end{pmatrix}
$$

It is easy to see $(𝐮\otimes 𝐯_1)𝐯_2=\langle 𝐯_1, 𝐯_2\rangle 𝐮$ and $𝐮_2^{\top}(𝐮_1\otimes 𝐯)=\langle 𝐮_1, 𝐮_2\rangle 𝐯^{\top}$. 

For the remainder of this section, suppose the observed values result from subtracting the sample mean of each variable, so that $\sum_{i=1}^n x_{ij}=0$ for all $j=1,\ldots,m$. We now define the **sample variance operator** 

$$Q_n:=(n-1)^{-1}\sum_{i=1}^n x_i\otimes x_i=(n-1)^{-1}\sum_{i=1}^n x_ix_i^{\top}=(n-1)^{-1}𝐗^{\top}𝐗$$ 

(Comparing the notations of the prior chapter, $𝐗=\tilde{𝐗}$, $Q_n=S$.) Also, any matrix represents a linear transformation between two vector spaces. Seen this way, $Q_n 𝐯$ is the linear transformation of $𝐯\in\mathbb{R}^m$ into the same space $\mathbb{R}^m$, satisfying

$$
\begin{align}
(n-1)Q_n 𝐯 &=\sum_{i=1}^n(x_i\otimes x_i)𝐯=\sum_{i=1}^n\langle 𝐯, x_i\rangle x_i \text{ and}\\
\langle 𝐯, Q_n 𝐯\rangle &= 𝐯^{\top}Q_n 𝐯 =(n-1)^{-1}\sum_{i=1}^n\langle 𝐯, x_i\rangle 𝐯^{\top}x_i=(n-1)^{-1}\sum_{i=1}^n \langle 𝐯, x_i\rangle^2
\end{align}
$$

Therefore, we may express the criterion for finding the first PC weight vector as

$$
\max_{w_1} w_1^{\top}Q_n w_1\equiv\langle w_1, Q_n w_1\rangle \text{ subject to } w_1^{\top}w_1=1
$$

As shown in that chapter, this maximization problem is solved by finding the solution with largest eigenvalue $\lambda$ of the *eigenequation*

$$
\langle w_1, Q_n w_1\rangle = \lambda
$$

