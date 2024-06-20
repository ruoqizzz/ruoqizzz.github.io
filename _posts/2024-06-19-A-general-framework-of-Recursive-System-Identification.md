---
layout: post
title: A general framework of Recursive System Identification (Part I)
date: 2024-06-19 12:05:00
description: 
tags: recursive_sysid
categories: control
---

This blog post aims to summarise Chapter 3 of [Theory and Practice of Recursive Identificatio](,https://mitpress.mit.edu/9780262620581/theory-and-practice-of-recursive-identification/) which derives a general recursive identification method that can be applied to any set of (linear) models.

To develop a unified approach to recursive identification consists 3 phases：

1. Define the framework
2. Derive the Algorithm: In the book, the authors mainly focus on **minimizing the prediction error variance recursively**, using the idea from stochastic approximation (See blog post Recursive Least Squares Derived from Stochastic Approximation Approach for details).
3. Apply the Algorithm: We will show how the general algorithm can be applied to a particular model set, **a linear regression model** later.



## Systems and Models

### Systems
The system is the physical object that generates the obverations, the output signals $$\{y_t\}$$ where $$t$$ is a discrete time index and $$y_t$$ is a  $$p$$-dimensional column vector. Many systems also have a measure input signals $$\{u_t\}$$ with dim $$r$$,  which can be chosen by the user. To get a reasonable identification results of the system, some properties of the inputs are required. In loose term, the input should excite all modes of system, called "persistenly exciting". Let $$\{u_t\}$$ be such that the limits 

$$
  \lim_{N\rightarrow\infty}\frac{1}{N}\sum_{t=1}^Nu(t)u^T(t-j)\triangleq r(j)
$$

exist for all $$0\leq j\leq n$$. The block matrix $$R_m$$ whose $$i,k$$ entry is $$r(i-k)$$.  The inputs $$\{u_t\}$$ is said to be *persistently exciting of order $$n$$*, if $$R_n$$ is nonsigular.

### Models

A model is a link between the past oservations and the unknown future. At time $$t-1$$, with the observations, the input-output data $$z^{t-1}=[y(t-1)^T, u(t-1)^T]^T$$, it is possible to predict the output at time $$t$$ with a model,

$$
\hat{y}(t\vert \theta)=g_\mathscr{M}(\theta;t,z^{t-1})
$$

Here, for a fixed $$\theta$$,  $$g_\mathscr{M}(\theta;\cdot,\cdot)$$ is a determistic function from $$\mathbb{R}\times \mathbb{R}^{t(r+p)}$$ to $$\mathbb{R}^p$$  and its parameter $$\theta$$ is a vector of dim $$d$$.  The model is denoted by $$\mathscr{M}(\theta)$$. The set of models condered will be obtained as $$\theta$$ ranges over a subset $$\mathscr{D}_\mathscr{M}$$ of $$\mathbb{R}^d$$,

$$
\mathscr{M}=\{\mathscr{M} \vert \theta \in \mathscr{D}_\mathscr{M}\}.
$$



**[Example: linear predictor models]**
$$
\begin{align}
\phi(t+1,\theta) &= \mathscr{F}(\theta)\phi(t,\theta)+\mathscr{G}(\theta)z(t)\\
\hat{y}(t\vert \theta)&=\mathscr{H}(\theta)\phi(t,\theta)
\label{eq:linear_predictor}
\end{align}
$$

where $$\mathscr{F},\mathscr{G},\mathscr{H}$$ are matrix functions of $$\theta$$.



**[Example: nonlinear predictor models]**
$$
\phi(t+1,\theta) = f(\theta;t,\phi(t,\theta),z(t))\\
\hat{y}(t\vert \theta)=h(\theta;t,\phi(t,\theta))
$$

For the linear predictor models $$\eqref{eq:linear_predictor}$$, we assume that the model is differentiable with respect to $$\theta$,

$$
\left [
\frac{d}{d\theta}\hat{y}(t\vert \theta)
\right ]^T=\psi(t,\theta)
$$

where $$\psi(t,\theta)$$ is a $$d\times p $$-matrix and can be formed from $$z^{t}$$ by a finite-dimension filter by introducing

$$
\zeta(t,\theta)=(\phi^{(1)}(t,\theta)~\cdots~\phi^{(d)}(t,\theta)~~~\text{(a }d\times n~\text{matrix)}
$$

where the superscript $$(i)$$ denotes diferentiation with respect to $$\theta_i$$. Since $$\mathscr{F}(\theta)$$ is a matrix and $$\theta$$ is a vector, the derivative of  $$\mathscr{F}(\theta)$$ is a tensor (a quantity with three indices). To simplify the notaiton, we introduce

$$
\left [
\frac{d}{d\theta}\mathscr{F}(\theta)\phi+\mathscr{G}(\theta)z
\right ]^T= M(\theta,\psi,z)\\
\left [
\frac{d}{d\theta}\mathscr{H}(\theta)\phi
\right ]^T= D(\theta,\psi,z)\\
$$

For $$M(\theta,\psi,z)$$, its $$i$$-th column is 

$$
\left [
\frac{d}{\partial \theta_i}\mathscr{F}(\theta)
\right ]\phi+
\left [
\frac{d}{\partial \theta_i}\mathscr{G}(\theta)
\right ]z
$$

Then,

$$
\begin{align}
\zeta(t+1,\theta)&=\mathscr{F}\zeta(t,\theta) + M(\theta,\phi(t,\theta),z(t)) \\
\psi^T(t,\theta)&=\mathscr{H}\zeta(t,\theta)+D(\theta,\phi(t,\theta)).
\label{eq:linear_predictor_d}
\end{align}
$$

z theoretical considerations, it is more convenient to collect $$\eqref{eq:linear_predictor}$$ and $$\eqref{eq:linear_predictor_d}$$ into one filter with $$z(t)$$ as input and $$\hat{y}(t\vert \theta)$$ and $$\psi(t,\theta)$$ as output. Let's introduce a state vector

$$
\xi (t,\theta) = \begin{pmatrix}\phi(t,\theta)\\
\text{col}~\zeta(t,\theta)
\end{pmatrix}
$$

where $$\text{col} A$$ here means a column vector constructed from the matrix $$A$$ by stacking its columns under each other. Similarly, we can introduce the output vector

$$
\begin{pmatrix}\hat{y}(t\vert \theta)\\
\text{col}~\psi(t,\theta)
\end{pmatrix}.
$$

We can now rewrite $$\eqref{eq:linear_predictor}$$ and $$\eqref{eq:linear_predictor_d}$$ for some matrices $$A(\theta), B(\theta), \text{and}~C(\theta), $$ as

$$
\begin{align}
\xi (t+1, \theta) = A(\theta) \xi (t, \theta) + B(\theta)z(t),\\
\begin{pmatrix}\hat{y}(t\vert \theta)\\
\text{col}~\psi(t,\theta)
\end{pmatrix}=C(\theta)\xi(t,\theta)
\end{align}
$$

We can see that $$A(\theta)$$ is a matrix with dimension $$(d+1)n\times (d+1)n$$ matrix and it containt the matrix $$\mathscr{F}(\theta)$$ in each of its $$d+1$$ block-diagnoal entries and has all zeros above the block diagonal. 

$$
A(\theta) = \begin{pmatrix}
\mathscr{F}(\theta) & 0 & \cdots & 0 \\
0 & \mathscr{F}(\theta) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \mathscr{F}(\theta)
\end{pmatrix}?????
$$


Hence $$A(\theta)$$ has the same eigenvalues as $$\mathscr{F}(\theta)$$ but with higher multiplicities. The stability of $$A(\theta)$$ there coincide with those of $$\mathscr{F}(\theta)$$ . Let us introduce the set 

$$
\begin{align}
D_{s}=\{
\theta \vert \mathscr{F}(\theta) \text{ has all eigenvalues strictly inside the unit circle}
\}.
\label{eq:Ds}
\end{align}
$$





**[Example: A difference Equation Model]**

For the difference equation 

$$
\begin{align}
y(t)+a_1y(t-1) + \dots + a_n y(t-n) = b_1u(t-1)+\dots+b_mu(t-m)+v(t)
\label{eq:linear}
\end{align}
$$

we may take

$$
\phi^T(t, \theta)=\begin{bmatrix}
-y(t-1) &\dots -y(t-n)&u(t-1)&\dots 
u(t-m)
\end{bmatrix}
$$

which is actually independent of $$\theta$$. The matrix in 

$$
\begin{aligned}
&
\mathscr{F}(\theta) =
\left(
\begin{array}{cccc|cccc}
0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
1 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & 0 & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1 & 0 & 0 & \cdots & 0 \\
\hline
0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & 1 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 1 \\
\end{array}
\right)
 \leftarrow 
\text{row } n + 1,\\
& \mathscr{G}(\theta)=\left(\begin{array}{rr}
-1 & 0 \\
0 & 0 \\
\vdots & \vdots \\
0 & 0 \\
0 & 1 \\
0 & 0 \\
\vdots & \vdots \\
0 & 0
\end{array}\right) \leftarrow \text { row } n+1 \text {, } \\
& z(t)=\binom{y(t)}{u(t)} \text {, } \\
& \mathscr{H}(\theta)=\theta^{\mathrm{T}}, \\
&
\end{aligned}
$$

For the gradient, 

$$
\phi^{(i)}(t,\theta)=0,~~\psi(t,\theta)=\phi(t,\theta)
$$

The matrix $$\mathscr{F}(\theta)$$ is a lower triangular with zeros along its diagonal and thus its eigenvalues are all zeros. Thus the set $$\eqref{eq:Ds}$$ is equal to $$\mathbb{R}^d$$.
