---
layout: post
title: A general framework of Recursive System Identification (Part III)
date: 2024-07-02 18:11:00
description: 
tags: recursive_sysid
categories: control
---


## A Recursive Prediction Error Identification Algorithm for General Critera

### General Criteria

Now let's consider the general criterion,

$$
\mathbb{E}_{z^t}l(t, \theta, \varepsilon(t,\theta))
$$


The recursive minimization of this criterion is entirely analogous to the quadratic criterion.

$$
\left[ \frac{d}{d\theta}l (t,\theta,\varepsilon(t,\theta))\right]^T = 
l_\theta^T(t,\theta,\varepsilon(t,\theta))-\psi^T(t,\theta)l_\varepsilon^T(t,\theta,\varepsilon(t,\theta)),
$$

where 

$$
\varepsilon(t,\theta) = y(t)-\hat{y}(t,\theta)\\
\frac{d}{d\theta}\varepsilon(t,\theta)=\frac{d}{d\theta}[y(t)-\hat{y}(t\vert \theta)=-\psi^T(t,\theta).
$$

$$l_\theta$$ and $$l_\varepsilon$$ denotes the partial derivatives with respect to $$\theta$$ and $$\varepsilon$$. The approach of [Part II](https://ruoqizzz.github.io/blog/2024/A-general-framework-of-Recursive-System-Identification-2/) now leads to the algorithm

$$
\begin{align}
\hat{\theta}(t)=\hat{\theta}(t-1) + \alpha(t)
R^{-1}(t)
\left
[-l_\theta^T(t,\hat{\theta}(t-1),\varepsilon(t))+\psi^T(t,\theta)l_\varepsilon^T(t,\hat{\theta}(t-1),\varepsilon(t))
\right ]
\label{eq:general_stochastic_newton_theta}
\end{align}
$$

If $$l$$ only depends on $$\varepsilon$$ then $$l_\theta=0$$ and 

$$
\hat{\theta}(t)=\hat{\theta}(t-1) + \alpha(t)
R^{-1}(t)
\psi^T(\theta)l_\varepsilon^T(\varepsilon(t))
$$

the only difference from qudratic criterion is $$\hat{\Lambda}(t)\varepsilon(t)$$ is replaced by $$l_\varepsilon^T(\varepsilon(t))$$.


### General Search Direction

In $$\eqref{eq:general_stochastic_newton_theta}$$  $$R(t)$$ is the approximation of Hession of $$V(\theta) = \mathbb{E} l (t,\theta,\varepsilon)$$. Then we can get

$$
\begin{align}
R(t)=R(t-1)+\alpha(t)
\left[
l_{\theta\theta}(t,\hat{\theta}(t-1),\varepsilon(t))+
\psi(t)l_{\varepsilon\varepsilon}(t,\hat{\theta}(t-1),\varepsilon(t))\psi^T(t)
-R(t-1)
\right ]
\label{eq:general_Rt_update}
\end{align}
$$

where $$l_{\theta\theta}$$ and $$l_{\varepsilon\varepsilon}$$ are the second derivatives w.r.t $$\theta$$ and $$\varepsilon$$. This gives a Newton-type updating direction which is known to be efficient in nonlinear programming problems. Of course any other updateing direction forms a sharp angle with the gradient will move $$\hat{\theta}$$ downhill on average. This means $$R(t)$$ can be any positive deifnite matrix and still ensure the criterion is minimized. Then we can replace $$\eqref{eq:general_Rt_update}$$ with

$$
R(t)=R(t-1)+\alpha(t)H\left(
R(t-1),\hat{\theta}(t-1),\varepsilon(t),\psi(t)
\right),
$$

where $$H$$ is a function such that the positive definiteness of $$R(t)$$ is guaranteed.

### Stochastic Gradient Algorithms

A common choice of $$R(t)$$ is a multiple of the identity matirx, which makes the updating direction coincide with the negative gradient of the criterion and written as

$$
\begin{align}
R(t)&=r(t)I,\\
r(t)&=r(t-1) + \alpha(t)\{
\text{tr}[l_{\theta\theta}(t,\hat{\theta}(t-1),\varepsilon(t))+
\psi(t)l_{\varepsilon\varepsilon}(t,\hat{\theta}(t-1),\varepsilon(t))\psi^T(t)]
-r(t-1).
\}
\end{align}
$$
