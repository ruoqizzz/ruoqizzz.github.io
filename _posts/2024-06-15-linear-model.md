---
layout: post
title: Recursive Least Squares Derived from Offline Estimation
date: 2024-06-15 16:05:00
description: 
tags: linear_model, recursive_sysid
categories: control
---

An obvious approach from offline estimation to recursive (online) estimation is to take any offline method and modify it. Here we show a method of how to modify the offline least squares to recursive least squares.

## Linear Difference Equation


$$
\begin{align}
y(t)+a_1y(t-1) + \dots + a_n y(t-n) = b_1u(t-1)+\dots+b_mu(t-m)+v(t)
\label{eq:linear}
\end{align}
$$

where $$\{u(t)\},\{y(t)\}$$  are input and output signals and $$v(t)$$ is some disturbance of unspecified character. 

Let $$q^{-1}$$ be backward shift operator, then 

$$
A(q^{-1})y(t) = B(q^{-1})u(t) + v(t),
$$

where $$A(q^{-1})$$ and $$B(q^{-1})$$ are polynomials in the delay operator:

$$
A(q^{-1}) = 1+a_1q^{-1}+\dots + a_nq^{-n}\\
B(q^{-1}) = 1+b_1q^{-1}+\dots + b_m q^{-m}
$$

Introduce the vector of lagged input-output data (regressor),

$$
\phi^T(t)=\begin{bmatrix}
-y(t-1) &\dots -y(t-n)&u(t-1)&\dots 
u(t-m)
\end{bmatrix}
$$

Then, $$~\eqref{eq:linear}$$ can be rewritten as 

$$
y(t)=\theta^T\phi(t)+v(t)
$$

where $$\theta^T=\begin{bmatrix}a_1 & \dots &a_n &b_1 &\dots b_m \end{bmatrix}^T$$ is the paramter vector.

If the character of the disturbance term $$v(t)$$ is not specified, it is natural to use 

$$
\hat{y}(t\vert \theta)\triangleq\theta^T\phi(t)
$$

as the prediction of $$y(t)$$ having observed previous inputs and outputs. 

## Offline Identification: The least squares

The parameter vector can be estimated from the measurements of $$y(t)$$ and $$\phi(t)$$ with $$t=1,2,\dots,N$. A common way to choose the estimation is to minimize

$$
V_N(\theta)=\frac{1}{N}\sum_1^N \alpha_t[y(t)-\theta^T\phi(t)]^2
$$


with respect to $$\theta$$ and $$\{\alpha_t\}$$ is a sequence of positive numbers allowing to give different weights to different observations. This criterion $$V_N(\theta)$$ is quadratic in $$\theta$$ and thus it can be minimized analytically, 

$$
\begin{align}
\hat{\theta}(N) = \left[
\sum_{1}^N \alpha_t\phi(t)\phi(t)^T
\right]^{-1}
\sum_{1}^N \alpha_t\phi(t)y(t),
\label{eq:offline_ls}
\end{align}
$$

where we assume that the inverse exists. It can be written in a recursive fashion. Let

$$
\bar{R}(t) = \sum_{1}^N \alpha_t\phi(t)\phi(t)^T.
$$

Then, from $$\eqref{eq:offline_ls}$$, we can get 

$$
\sum_{1}^N \alpha_t\phi(t)y(t) = \bar{R}(t-1)\hat{\theta}(t-1).
$$

From the definition of $$\bar{R}(t)$$,

$$
\bar{R}(t-1)=\bar{R}(t)-\alpha_t\phi(t)\phi^T(t)
$$

Thus
$$
\begin{align}
\hat{\theta}(t) &=\bar{R}^{-1}(t)\left[
\sum_{k=1}^{t-1} \alpha_k\phi(k)y(k) + \alpha_t \phi(t)y(t)
\right]\\
&=\bar{R}^{-1}(t)\left[
\bar{R}(t-1)\hat{\theta}(t-1)+ \alpha_t \phi(t)y(t)
\right]\\
&=\bar{R}^{-1}(t)\left[
\bar{R}(t)\hat{\theta}(t-1)+ \alpha_t \phi(t)[-\phi^T(t)\hat{\theta}(t-1) + y(t)]
\right]\\
&=\hat{\theta}(t-1) + \bar{R}^{-1}(t)\phi(t)\alpha_t[y(t) - \hat{\theta}^T(t-1)\phi(t)]
\end{align}
$$

and 

$$
\bar{R}(t)=\bar{R}(t-1) + \alpha_t\phi(t)\phi^T(t)
$$

Sometimes we may prefer to work with

$$
R(t) \triangleq \frac{1}{t}\bar{R}(t)
$$

Then

$$
R(t)=\frac{1}{t} \left [\bar{R}(t-1) + \alpha_t\phi(t)\phi^T(t)\right]=\frac{t-1}{t}
R(t-1)+\frac{1}{t}\alpha_t\phi(t)\phi^T(t)\\
=R(t-1) + \frac{1}{t}[
\alpha_t\phi(t)\phi^T(t)-R(t-1)
]
$$


In summary, we can write 
$$
\begin{align}
\hat{\theta}(t)&=\hat{\theta}(t-1) + \frac{1}{t}R^{-1}(t)\phi(t)\alpha_t [y(t)-\theta^T(t-1)\phi(t)],\\
R(t)&=R(t-1) + \frac{1}{t}[
\alpha_t\phi(t)\phi^T(t)-R(t-1)].
\label{eq:offline_rls1}
\end{align}
$$

## An Equivalent Form: Recursive Least Squares

Equation $$\eqref{eq:offline_rls1}$$ is not that suited for computation since a matrix inverse has to be calculated in each time step. It's more natural to introduce

$$
P(t)=\bar{R}^{-1}=\frac{1}{t}R^{-1}(t)
$$

and update $$P(t)$$ directly instead. This can be done by matrix inverse lemma

$$
[A+BCD]^{-1}=A^{-1}-A^{-1}B[DA^{-1}B+C^{-1}]^{-1}DA^{-1}.
$$

The proof can be done by multiplying the RHS by $$(A+BCD)$.

Let $$A=P(t-1)$, $$B=\phi(t)$, $$C=\alpha_t$$ and $$D=\phi^T(t)$$,

$$
\begin{align}
P(t)&=\left[P^{-1}(t-1)+\phi(t)\alpha_t\phi^T(t) \right]^{-1}\\
&=P(t-1)-P(t-1)\phi(t)
\left[
\phi^T(t)P(t-1)\phi(t)+\frac{1}{\alpha_t}
\right]^{-1}\phi^T(t)P(t-1)\\
&=P(t-1)-\frac{P(t-1)\phi(t)\phi^T(t)P(t-1)}{1/\alpha_t + \phi^T(t)P(t-1)\phi(t)}
\end{align}
$$

Now the inversion of a square matrix of dim $$\theta$$ is replaced by inversion of a scalar $$\alpha_t$$.

Thus the recursive least squares can be written as

$$
\begin{align}
\hat{\theta}(t)&=\hat{\theta}(t-1) + L(t)[y(t) -\hat{\theta}^T(t-1)\phi(t)],\\
L(t)&=\frac{P(t-1)\phi(t)}{1/\alpha_t + \phi^T(t)P(t-1)\phi(t)},\\
P(t)&=P(t-1)-\frac{P(t-1)\phi(t)\phi^T(t)P(t-1)}{1/\alpha_t + \phi^T(t)P(t-1)\phi(t)}.
\label{eq:rls}
\end{align}
$$


### Initial Conditions

The only assumption we made is the $$\bar{R}(t)$$ is invertible and typically it becomes invertible at time $$t_0=dim~ \phi(t) = dim~\theta$$. Thus strictly speaking, the proper initials values for $$\eqref{eq:rls}$$ are obtained if starting at the time $$t_0$$ for which 

$$
P(t_0)=
\left [
\sum_{k}^{t_0} \alpha_k\phi(k)\phi(k)^T \right ]^{-1} \\
\hat{\theta}(t_0) = P(t_0)\sum_{k}^{t_0}\alpha_k \phi(k)y(k).
$$


It is more common to start at $$t=0$$ with some invertible matrix $$P(0)$$ and a vector $$\hat{\theta}(0)$. Then, the resulting estimates are

$$
\hat{\theta}(t) = \left [ 
P^{-1}(0)+\sum_{k=1}^t \alpha_k\phi(k)\phi^T(k)
\right]^{-1}\left [ 
P^{-1}(0)\hat{\theta}(0)+\sum_{k=1}^t \alpha_k\phi(k)y(k)
\right].
$$

We can see that the relative importance of the initial values decays over time as the magnitudes of the sums increase. Also, as $$P^{-1}(0)\rightarrow0$$, the recursive estimate goes to the offline one. A common choice of initial values is to take $$P(0)=C \cdot I$$ and $$\hat{\theta}(0)=0$$, where $$C$$ is some large constant.

### Asymptotic Properties

We assume that the data are generated by 

$$
y(t) = \theta_\circ \phi(t) + v(t)
$$

Inserting this equation to $$\eqref{eq:offline_ls}$$,

$$
\begin{align}
\hat{\theta}(N) &= \left[
\sum_{t=1}^N \alpha_t\phi(t)\phi(t)^T
\right]^{-1}
\left \{
\sum_{t=1}^N \alpha_t\left[\phi(t)\phi^T(t)\theta_\circ+\phi(t)v(t)\right]
\right\}\\
&= \theta_\circ +  \left[ \frac{1}{N}
\sum_{t=1}^N \alpha_t\phi(t)\phi(t)^T
\right]^{-1}
\frac{1}{N}\sum_{t=1}^N \alpha_t\phi(t)v(t)
\end{align}
$$

According to the law of large numbers, the sum $$\frac{1}{N}\sum_{t=1}^N \alpha_t\phi(t)v(t)$$ will converge to its expected values as $$N$$ goes to infinity. The expected values depend on the correlation between the disturbance term $$v(t)$$ and the data vector $$\phi(t)$$. It's zero only when $$v(t)$$ and $$\phi(t)$$ are uncorrelated. This is true when 

- $$\{v(t)\}$$ is i.i.d with zero means
- $$n=0$$ and $$\{u(t)\}$$ is independent of the zero-mean noise sequence $$\{v(t)\}$$

In both cases, the $$\hat{\theta}(N)$$  approaches to $$\theta_\circ$$ as $$N$$ goes to infinity. 

### Interpretations of RLS

$$
\begin{align}
\hat{\theta}(N) &= \left[
\sum_{t=1}^N \alpha_t\phi(t)\phi(t)^T
\right]^{-1}
\left \{
\sum_{t=1}^N \alpha_t\left[\phi(t)\phi^T(t)\theta_\circ+\phi(t)v(t)\right]
\right\}\\
&= \theta_\circ +  \left[ \frac{1}{N}
\sum_{t=1}^N \alpha_t\phi(t)\phi(t)^T
\right]^{-1}
\frac{1}{N}\sum_{t=1}^N \alpha_t\phi(t)v(t)
\end{align}
$$



1. Beforehand, we have shown that how RLS derived from the offline LS version and 
   $$
   \hat{\theta}(t) = \left [ 
   P^{-1}(0)+\sum_{k=1}^t \alpha_k\phi(k)\phi^T(k)
   \right]^{-1}\left [ 
   P^{-1}(0)\hat{\theta}(0)+\sum_{k=1}^t \alpha_k\phi(k)y(k)
   \right]
   $$
   
   With $$P^{-1}(0)=0$$, it minimizes the least sqaures criterion 
   
   $$
   V_t(\theta)=\frac{1}{t}\sum_{k=1}^t \alpha_k[y(k)-\theta^T\phi(k)]^2
   $$

2. The estimate $$\hat{\theta}$$ can be seen as the Kalman filter state estimate for state-space model
   $$
   \theta(t+1) =\theta(t)\\
   y(t)=\phi^T(t)\theta(t)+v(t)
   $$

3. The RLS is a recursive minimization of
   $$
   \bar{V}(\theta)=\mathbb{E}\frac{1}{2}[y(t)-\theta^T\phi(t)]^2.
   $$
   
   The factor $$\phi(t)[y(t) -\hat{\theta}^T(t-1)\phi(t)]$$ is then an estimate of the gradient while
   
   $$
   t\cdot P(t) =\left [ \frac{1}{t}
   P^{-1}(0)+\sum_{k=1}^t \alpha_k\phi(k)\phi^T(k)
   \right]^{-1}
   $$
   
   is the inverse of an estimate of the second derivate of the criterion. The updating $$\hat{\theta}(t)$$ is thus updated with the direction of "Newton" and a decaying step size $$\frac{1}{t}$$. More can be found in another blog post "Recursive Least Squares Derived from Stochastic Approximation Approach ".

