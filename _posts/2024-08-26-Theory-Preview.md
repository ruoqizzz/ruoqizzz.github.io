---
layout: post
title: Preview of Asymptotic Properties of Recursive Identification Methods
date: 2024-08-26 11:52:00
description: 
tags: recursive_sysid, linear_model
categories: control
---

## Preview of Asymptotic Properties of Recursive Identification Methods

The general family of recursive algorithms can be written as 

$$
\begin{align}
\label{eq:quad_criterion_with_general_form:start}
&\varepsilon(t) = y(t)-\hat{y}(t) \\
&\hat{\Lambda}(t) = \hat{\Lambda}(t-1) + \alpha(t)[\varepsilon(t)\varepsilon^T(t) - \hat{\Lambda}(t-1)]\\
&\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}}\\
&\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t)\\
&\begin{pmatrix}
   \hat{y}(t+1) \\
   \text{col } \psi(t+1)
   \end{pmatrix} = C(\hat{\theta}(t)) \xi(t+1).
   \label{eq:quad_criterion_with_general_form:end}
\end{align}
$$

where $$\psi(t)$$ which is the gradient of the prediction w.r.t $$\theta$$ plays a crucial role.

**Assumption** The generation of the input sequence $$\{u(t)\}$$ is not based on the current estimates $$\theta$$. 

### Recursive Prediction Error Methods

Consider the general criterion function

$$
\bar{V}(\theta)=\bar{\mathbb{E}}l(t,\theta,\varepsilon(t,\theta))
$$

where $$\bar{\mathbb{E}}$$ is defined as 

$$
\bar{\mathbb{E}} f(t) = \lim_{N\rightarrow \infty}\frac{1}{N}\mathbb{E}f(t)
$$

This algorithm does exactly what it is required to:

> The estimate $$\hat{\theta}(t)$$ will converge w.p.1to a local minimum of $$\bar{V}$ as t approaches infinity. (This will be shown later in theorems x).

There are two issues to be discussed

1. "***Local***": If $$\bar{V}(\theta)$$ has several local minima, we can not expect global convergence to a gloabal minimum.
2. The quoted convergence results holds *whether or not the **true system** belongs to the model set*: Even if the true system is more complex then the models, the identification procedure will picl the best approximation of the system w.r.t the criterion. 

#### Asymptotic Distribution

Suppose we use a Gauss-Newton algorithm with a gain sequence such that $$t \cdot \alpha(t)\rightarrow 1$$ as $$t\rightarrow \infty$$, then we have

$$
\sqrt{t}(\theta(t) - \theta^*) \xrightarrow{d} \mathcal{N}(0, P)
$$

where $$\theta^*$ denotes the convergence value. This result implies that as more data becomes available, the estimation error becomes smaller, but the distribution of the scaled error approaches a normal distribution. The matrix $P$ , which describes the precision of the parameter estimates, is the same as the offline case. 

If the true system is in the model set and $$\theta^*=\theta_\circ$$ then the prediction errors $$\{\varepsilon(t,\theta_\circ)\}$$ is a sequence of indenpendent random vectors each of zero mean and covariance matrix $$\Lambda_\circ$ and with recursive algorithms used, we have

$$
\begin{align}
R^{-1}(t)\rightarrow P = \left[
\bar{\mathbb{E}}\psi(t,\theta_\circ)\Lambda_\circ\psi(t,\theta_\circ)^T
\right]^{-1}~~~~w.p.1.\text{ as }t\rightarrow\infty,
\label{eq:RPEM_convergence_R}
\end{align}
$$

where $$R(t)$$ is the matrix in the recursive Gauss-Newton algorithm and $$P$$ is the asymptotic covariance matrix.

### Pseudolinear Regressions

According to the definition,

$$
\psi(t) = -\frac{\partial \hat{y}(t\vert \theta)}{\partial \theta}
$$

If the implicit $$\theta$$-dependencen in $$\phi(t,\theta)$$ is neglected, we could obtain an appeoximate expression

$$
\frac{\partial \hat{y}(t\vert \theta)}{\partial \theta} \approx \phi(t,\theta).
$$

With this approximation, the recursive algorithm can be rewritten as

$$
\begin{align}
&\varepsilon(t) = y(t)-\hat{\theta}^T(t-1)\phi(t) \\
&R(t)=R(t-1) +\alpha (t)\left[\phi(t)\phi^T(t) - R(t-1)\right]\\
&\hat{\theta} = \hat{\theta}(t-1) + \alpha(t) R^{-1}\phi (t)\varepsilon (t)
\end{align}
$$

Due to the system is modeled as linear in the parameters, even if the relationship between the input and output is nonlinear, we call this algorithm "psuedolinear regression".

Usually, we use filter $$\phi(t)$$ as the approximated gradient. The filters is associated with the estimate $$\hat{\theta}(t)$$ and depends on the particular model set. There are two examples

- ARMAX model
  $$
  A(q^{-1})y(t)=B(q^{-1})u(t)+C(q^{-1})e(t)
  $$
  
  then 
  
  $$
  \psi(t)=\frac{1}{\hat{C}_t(q^{-1})}\phi(t)
  $$
  
  where $$\hat{C}_t(q^{-1})$$ is the current estimate of the $C$-polynomial. 

- Output error model

  $$
  y(t)=\frac{B(q^{-1})}{F(q^{-1})}u(t)+e(t)
  $$
  
  then 
  
  $$
  \psi(t)=\frac{1}{\hat{F}_t(q^{-1})}\phi(t)
  $$


PLR can be seen as neglecting the filtering. Let's give a closer look the influence of its approximation in ARMAX case.

The true system is given by

$$
A_\circ(q^{-1})y(t)=B_\circ(q^{-1})u(t)+C_\circ(q^{-1})e(t),
$$

then a sufficient condition for convergence of $$\hat{\theta}(t)$$ to the true parameters $$\theta_\circ$$ is that 

$$
\vert C_\circ(e^{i\omega})-1\vert < 1~\forall \omega
$$

provided the input is general enough. <span style="color:red"> WHY?</span>

This condition can be interpreted as a measure of how good an approximation it is to replace $$\psi(t)$$ by $$\phi(t)$$ close to $$\hat{\theta}(t) = \theta_\circ$.  It can also be rewritten as 

$$
\text{Re}\left[\frac{1}{C_\circ(e^{i\omega})}-\frac{1}{2}\right]>0~~\forall \omega,
$$

which is often expressed as "the filter $$\frac{1}{C_\circ(q^{-1})}-\frac{1}{2}$ is strictly positive real."

**Proof:**

$$
\begin{align}\text{Re}\left[\frac{1}{z}-\frac{1}{2}\right]>0 \rightleftharpoons \frac{2\text{Re }(z) -\vert z\vert^2}{\vert z\vert^2}>0  \rightleftharpoons 2\text{Re }(z) >\vert z\vert^2,\\
\vert z - 1\vert < 1 \rightleftharpoons (z-1)(\bar{z}-1)<1 \rightleftharpoons \vert z\vert^2-2\text{Re}(z) + 1< 1 \rightleftharpoons \vert Z(e^{i\omega})-1 \vert < 1.
\end{align}
$$

While the aforementioned conditions are *sufficient* for convergence, it is also known that PLR algorithms may not converge. In fact, if

$$
\begin{align}
\text{Re} C_\circ(e^{i\omega})>0~~\forall \omega
\label{eq:reC}
\end{align}
$$

does not hold, we can always an A-polynomial, B-polynomial and a input signal such that the probability that $$\hat{\theta}(t)$$ converges to the desired value $$\theta_\circ$$ is zero. This means that as a condition on C-polynomial alone equation $$\eqref{eq:reC}$$ is necessary to assure convergence. 

Note that these convergence results apply only if the true system indeedn belongs to the model set!

No explicit expression for the asympototic convariance matrix for the estimates obtained by PLR is knwon in general. It is not ture for a PLR algorithm that $$R(t)$$ converges to the asymptotic covariance matrix in contrast to the case in $$\eqref{eq:RPEM_convergence_R}$$.



### Instrumental Variable 

In linear regression, we assume that 

$$
\mathbb{E}\phi(t)v(t)=0,
$$

such that $$\theta_\circ$$ is a solution to 

$$
\mathbb{E}\phi(t)[y(t) - \phi^T(t)\theta]=0.
$$

This means that $$v(t)$$ must be zero mean and uncorrelated with $$\phi(t)$$. 

For linear different equations

$$
\begin{align}
y(t)+a_1y(t-1) + \dots + a_n y(t-n) = b_1u(t-1)+\dots+b_mu(t-m)+v(t)
\label{eq:linear_diff}
\end{align} 
$$

where $$\{v(t)\}$$ is s sequence of zero-mean disturbances of unspecified character, not necessarily white noise. The aforementioned assumption leads to $n=0$ and $$\{v(t)\}$$ and $$\{u(t)\}$$ are independent or $$\{v(t)\}$$ is white noise.

Let 

$$
\phi^T(t)=\begin{bmatrix}
-y(t-1) &\dots -y(t-n)&u(t-1)&\dots 
u(t-m)
\end{bmatrix}
$$

Then we can rewrite $$~\eqref{eq:linear_diff}$$ as

$$
\begin{align}
\begin{aligned}
y(t)=\theta^T\phi(t)+v(t)
\end{aligned}
\label{eq:linear_phi}
\end{align}
$$

Therefore, in order to obtain a sequence of estimates that converges to $$\theta_\circ$$ as $$t$$ approahces infinity, we should instead solve 

$$
\mathbb{E} \zeta (t)[y(t)-\phi^T(t)\theta]=0
$$

where $$\zeta (t)$$ should be uncorrelated with $$v(t)$$ but correlated with the gradient

$$
\mathbb{E}\zeta (t)v(t)=0\\
\mathbb{E}\zeta (t)\phi^T(t) \text{ positive definite or at least nonsigular.}
$$

The vector $$\zeta (t)$$ is called the instrumental variable (IV) and the algorithm 

$$
\hat{\theta}(t) = \hat{\theta}(t-1) +\alpha(t) R^{-1}(t)\zeta (t)[y(t)-\phi^T(t)\hat{\theta}(t-1)]
$$

is a recursive instrumental varibale (RIV) algorithm. 

There are many ways of choosing $$R(t)$$ and $$\zeta(t)$$. Here we first focus on the refined IV

$$
\begin{align}
&\hat{\theta}(t) = \hat{\theta}(t-1) +\alpha(t) R^{-1}(t)\zeta (t)[y_F(t)-\phi_F^T(t)\hat{\theta}(t-1)],\\
&R(t)=R+\alpha(t)[\zeta(t)phi^T_F(t) - R(t-1)]\\
&y_F(t)=T(q^{-1})y(t)\\
&\phi_F=T(q^{-1})\phi(t)\\
&t\cdot\alpha(t)\rightarrow 1 \text{ as }t\rightarrow \infty
\end{align}
$$

Convergence and asymptotic properties of the nonsymmetric method can be established by examination of the offline expression

$$
\hat{\theta}(t)=\left[
\sum_{1}^t\zeta(k)\phi_F^T(k)
\right]^{-1}
\sum_{1}^t\zeta(k)y_F(k)
$$

when the filters are invovled in the generation of $$\{\zeta(t)\}$$ are time-invariant.

If $$\eqref{eq:linear_phi}$$ describes the true system, then $$\hat{t}$$ will tend to $$\theta_\circ$$ as $$t \rightarrow \infty$$ if 

$$
\bar{\mathbb{E}}\zeta(t)\phi^T(t)
$$

is nonsingular and 

$$
\bar{\mathbb{E}}\zeta(t)v(t)=0.
$$


In general, these two conditions depend on the properties of true system and the choice of $$\{\zeta(t)\}$$ and $$T(q^{-1})$$. Also, it can be shown that

$$
\sqrt{t}(\theta(t) - \theta_\circ) \xrightarrow{d} \mathcal{N}(0, \bar{P})
$$

where an explicit expression of $$\bar{P}$$ can be given in terms of the true system, the properties of $$\{v(t)\}$ and the chosen instrumental variables and filters. 
