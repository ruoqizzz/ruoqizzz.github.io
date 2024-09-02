---
layout: post
title: The Recipe for convergence analysis of recursive identification - a simple example
date: 2024-08-29 17:36:00
description: 
tags: recursive_sysid, linear_model
categories: control
---

## The Recipe for Convergence Analysis 

Here we consider the basic structure for algorithms related to quadratic criteria 
$$
\begin{align}
V(\theta) = \mathbb{E} l (t,\theta,\varepsilon)= \mathbb{E} 
\frac{1}{2}\varepsilon^T(t,\theta)\Lambda^{-1}\varepsilon^T(t,\theta).
\end{align}
$$

$$
\begin{aligned}
&\varepsilon(t) = y(t)-\hat{y}(t) \\
&R(t) = R(t-1) + \alpha(t)[\eta(t)\Lambda^{-1}(t)\eta^T(t) - R(t-1)]\\
&\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \eta(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}}\\
&\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t)\\
&\begin{pmatrix}
   \hat{y}(t+1) \\
   \text{col } \eta(t+1)
\end{pmatrix} = C(\hat{\theta}(t)) \xi(t+1).
\end{aligned}
\label{eq:general_form}
$$
where $\eta(t)$ is a vector related to the gradient of the predition $\hat{y}(t)$ w.r.t $\hat{\theta}$, for exawmple $\psi=\frac{d}{d\theta}\hat{y}$, $\phi, \zeta$ discussed before and $z(t) = \begin{pmatrix} y(t) \\ u(t) \end{pmatrix}$.  $R(t)$ is an approximation of the Hessian of the criterion,

1. Compute the prediction errors $\varepsilon(t, \theta)$ and gradient approximations $\eta(t,\theta)$ that would be obtained for a fixed and constant model $\theta$
2. Evaluate the average updating direction for the algorithm, based on these variables
3. Define a differential equation that has this direction as the right hand side
	$$
	\dot{\theta_D} = \bar{R}^{-1}f(\theta_D)\\
	\dot{R_D} =G(\bar{\theta}) - R_D
	$$
4. Study the stability properties of this differential equation.

## A Simple Example $y(t) = e(t) + c e(t-1)$

$$
\begin{aligned}
&\varepsilon(t) = y(t)-\hat{y}(t) \\
&R(t) = R(t-1) + \alpha(t)[\eta(t)\Lambda^{-1}(t)\eta^T(t) - R(t-1)]\\
&\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \eta(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}}\\
&\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t)\\
&\begin{pmatrix}
   \hat{y}(t+1) \\
   \text{col } \eta(t+1)
\end{pmatrix} = C(\hat{\theta}(t)) \xi(t+1).
\end{aligned}
$$





Step 1. Compute $\varepsilon(t, c)$ and $\eta(t,c)$