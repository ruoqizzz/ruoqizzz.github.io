---
layout: post
title: Recursive Least Squares Derived from Stochastic Approximation Approach 
date: 2024-06-18 16:05:00
description: 
tags: linear_model, recursive_sysid
categories: control
---

## Problem Statement: 

Let's consider the simple difference equation model

$$
y(t)=\theta^T\phi(t)+v(t)
$$


where $$y(t)$$ and $$\phi(t)$$ are measured quantities, $$\theta$$ is a determined model parameter and $$v(t)$$ is the equation error. It is natural to select $$\theta$$ by minimizing the variance of $$v(t)$$,

$$
\begin{align}
\min_\theta V(\theta)
\label{eq:minimizingV}
\end{align}
$$

where 

$$
V(\theta)=\frac{1}{2}\mathbb{E}\left [y(t)-\phi^T(t)\theta
\right]^2
$$

Since $$V(\theta)$$ is quadratic in $$\theta$$, $$\eqref{eq:minimizingV}$$ can be found by solving

$$
\left [
-\frac{d}{d\theta}V(\theta)
\right ]^T = \mathbb{E}\phi(t)\left [y(t)-\phi^T(t)\theta
\right]=0.
$$

Since the probability distribution of $$y(t), \phi(t)$$ is not known in general, one way to replace expectations with sample means which goes to least squares (check Recursive Least Square blog).

## Stochastic Approximation 

Stochastic approximation is a mathematical technique used to find the roots of functions or to optimize functions when there is noise in measurements. Let $$\{e(t)\}$$ be a sequence of random variables from the same distribution and $$t$$ is the discrete-time variable. A typical problem for stochastic approximation can be to find the solution to 

$$
\begin{align}
\mathbb{E}_e Q(x, e(t))=f(x)=0
\label{eq:stochastic_approx_problem}
\end{align}
$$

where in general, the distribution of $$e(t)$$ is unknown, the exact form of the function $$Q$$ is not unknown but the values can be accessed for any $$x$$. In our case,

$$
\begin{align}
x&=\theta\\
e(t)&=\begin{bmatrix}y(t) \\ \phi(t)\end{bmatrix}\\
Q(x, e(t))&=\phi(t)[y(t)-\phi^T(t)\theta],
\end{align}
$$

where $$e(t)$$ can be observed and $$Q$$ is a known function.

One trivial way would be first to fix $$x$$, get a huge number of observations $Q(x, e(t))$ for this $$x$$ which gives a good estimate of $$f(x)$$ and change the value of $$x$$ and repeat this procedure. It is more efficient to change the value of $$x$$ for each observation to not waste lots of effort on estimating $$f(x)$$ where $$x$$ values are far away from the solution. It is suggested to use the Robbins-Monro scheme,

$$
\begin{align}
\hat{x}(t)=\hat{x}(t-1) + \alpha(t)Q(\hat{x}(t-1), e(t)),
\label{eq:robbins-monro}
\end{align}
$$

where $$\{\alpha(t)\}$$ is a sequence of positive scalars tending to zeros (step sizes). 

## Apply to Linear Regression

With this, we can get

$$
\hat{\theta}(t)=\hat{\theta}(t-1)+\alpha(t)\phi(t)\left [y(t)-\phi^T(t)\theta
\right].
$$

The sequence $$\{\alpha(t)\}$$ in control literature is called "gain sequence". Some common choices are

1. Constant: $$\alpha(t)=\alpha_0$$ 
2. Normalized: $$\alpha(t)=\alpha_0 / \vert \phi(t)\vert^2$$
3. Normalized and decreasing: $$\alpha(t)=\left[ \sum_{k=1}^t  \vert \phi(t)\vert^2\right]^{-1}$$

The normalized choices give the invariance under the scaling of the signals $$y(t) $$ and $$\phi(t)$$.

### The Robbins-Monro as Stochastic Gradient Method

In terms of the general formulation $$\eqref{eq:stochastic_approx_problem}$$, we could think the original problem $$\eqref{eq:minimizingV}$$  as

$$
\begin{align}
\min_x V(x) \\
V(x)=\mathbb{E}_e J(x, e(t)).
\end{align}
$$

Let

$$
-\frac{d}{dx}V(x)=f^T(x),
$$

and suppose that the gradient 

$$
-\frac{d}{dx}J(x, e(t)) = Q^T(x, e(t))
$$

can be obtained for any chosen $$x$$. Then the solution can be obtained by

$$
0=\left [
-\frac{d}{dx}V(x)
\right ]^T
=f(x)=\mathbb{E} Q(x, e(t)).
$$

Now we interchanged expectations and differentiation  and back to  $$\eqref{eq:stochastic_approx_problem}$$. Thus, the Robbins-Monro $$\eqref{eq:robbins-monro}$$ can be seen as an algorithm to minimize $$V(x)$$ and it adjusts $$x$$ in the direction of the negative gradient of the observed $J(x, e(t))$. On average, the adjustments are in the negative gradient direction of $$V(x)$$ which is the stochastic gradient method (SGD).

### Newton Direction

It is well-known that the SGD is fairly inefficient especially when its iterates are getting close to the minimum. The Newtown method can give better results in which the search direction is modified from negative gradient to 

$$
\left [
-\frac{d^2}{dx^2}V(x)
\right ]^{-1}\left [
-\frac{d}{dx}V(x)
\right ]^T
$$

where $$\left [
-\frac{d^2}{dx^2}V(x)
\right ]$$ is the Hessian. The iteration can be written as

$$
x^{(t+1)} = x^{(t)} - \left [
-\frac{d^2}{dx^2}V(x)
\right ]^{-1}\left [
-\frac{d}{dx}V(x)
\right ]^T \bigg \vert _{x=x^{(t)}}
$$

where the iteration will converge in one step to the minimum of $$V(x)$$ if its function is quadratic in $$x$$. Therefore, it is very efficient when close to the minimum where the approximation of $$V(x)$$ well describes the function while otherwise, it is inefficient or even diverges. Thus, the Hessian is usally replaced by a guaranteed positive-definite approximation to secure a search direction that points downhill.

### A General Stochastic Newton Method

Since the hessian gives a clear improvement in effiency, it is reasonable to consider the Newton variant of Robbins-Monro scheme,

$$
\begin{align}
\hat{x}(t)=\hat{x}(t-1) + \alpha(t)
\left [
\bar{V}^{''}(\hat{x}(t-1), e(t))
\right]^{-1}
Q(\hat{x}(t-1), e(t)),
\label{eq:stochastic_newton}
\end{align}
$$

which is called "stochastic Newton algorithm".

### Apply to Linear Regression

For the quadratic criterion $$V(\theta)$$, we got

$$
\frac{d^2}{d\theta^2}V(\theta)=\mathbb{E}_e \phi(t)\phi^T(t),
$$

which is independent of $$\theta$$. The Hessian can be determined as the solution $$R$$ of the equation,

$$
\mathbb{E}[ \phi(t)\phi^T(t) - R] =0.
$$

Applying the Robbins-Monro scheme, 

$$
R(t) = R(t-1) + \alpha(t)[\phi(t)\phi^T(t)-R(t-1)].
$$

With this Hession estimate, we can obtain

$$
\hat{x}(t)=\hat{x}(t-1) + \alpha(t)
R^{-1}
\phi(t)\left [y(t)-\phi^T(t)\theta(t-1)
\right].
$$

When $$\alpha(t)=\frac{1}{t}$$, this coincides with Recursive Least Squares.  
