---
layout: post
title: The users' summary of a general recursive identification method
date: 2024-08-26 10:37:00
description: 
tags: recursive_sysid, linear_model
categories: control
---

## A general framework

The model set is defined in general terms as a one-step-ahead predictor $$\hat{y}(t \vert \theta)$$ that depends on the model parameter vector $$\theta$$. This prediction can be formed using a linear finite-dimensional filter acting on the observed input-output data $$\{z(t)\}$$
$$
\phi(t+1, \theta) = \mathscr{F}(\theta)\phi(t,\theta) + \mathscr{G}(\theta)z(t)\\
\hat{y}(t\vert \theta) = \mathscr{H}(\theta)\phi(t,\theta).
$$
There are some particular explaples of model sets

- Linear Regression Models
  $$
  \hat{y}(t\vert \theta) = \phi^T(t)\theta
  $$
  
- A General SISO Model:
  $$
  A(q^{-1})y(t)=\frac{B(q^{-1})}{F(q^{-1})}u(t)+\frac{C(q^{-1})}{D(q^{-1})}e(t)
  $$
  
- State-space Models
  $$
  x(t+1) = F(\theta)x(t)+G(\theta)u(t)+w(t)\\
  y=H(\theta)x(t)+e(t)
  $$



For the general form, the gradient of $$\hat{y}(t \vert \theta)$$ w.r.t $$\theta$$, denoted by $$\psi(t,\theta)$$ can be computed by means of 
$$
\begin{align}
\label{eq:quad_criterion_with_general_form:start}
&\varepsilon(t) = y(t)-\hat{y}(t) \\
&\hat{\Lambda}(t) = \hat{\Lambda}(t-1) + \alpha(t)[\varepsilon(t)\varepsilon^T(t) - \hat{\Lambda}(t-1)]\\
&\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}}\\
&\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t)\\
&\begin{pmatrix}
   \hat{y}(t) \\
   \text{col } \psi(t)
   \end{pmatrix} = C(\hat{\theta}(t-1)) \xi(t).
   \label{eq:quad_criterion_with_general_form:end}
\end{align}
$$
Here $$\{\alpha(t)\}$$ is a sequence of positive scalars, $$D_\mathscr{M}$$ is the projection into the stable model region. $$R(t)$$ is a positive definite matrix that modefies the seasrch direction. For example, there are two common choices,

- Gauss-Newton direction
  $$
  R(t)=R+\gamma\left[
  \psi(t)\hat(t){\Lambda(t)}\psi^T(t)-R(t-1).
  \right]
  $$
  
- Gradient direction
  $$
  \begin{align}
  R(t) &= r(t)\cdot I\\
  r(t) &= r(t-1)+\gamma(t) \left[
  \text{tr}\psi(t)\hat{\Lambda(t)}\psi^T(t)-r(t-1)
  \right]
  \end{align}
  $$

The idea between the general form and its update rule is simple

> Derive an expression to show how the prediction $$ \hat{y}(t \mid \theta)$$ depends on the model parameters. Then, derive an expression for the gradient $$ \psi(t, \theta)$$ of $$ \hat{y}(t \mid \theta)$$ with respect to $$ \theta$$. These expressions will result in filters that depend on $$ \theta$$ and utilize observed data as inputs. Subsequently, $$ \hat{y}(t)$$ and $$ \psi(t)$$ are obtained from these expressions by replacing past values $$ \hat{y}(t-k \mid \theta)$$ and $$ \psi(t-k, \theta)$$ with $$ \hat{y}(t-k)$$ and $$ \psi(t-k)$$, respectively, and by substituting $$ \theta$$ with its most recent estimate.

The algorithm $$\ref{eq:quad_criterion_with_general_form:start}\sim\ref{eq:quad_criterion_with_general_form:end}$$  aims to minizing the quadratic criterion
$$
\mathbb{E}\frac{1}{2}\varepsilon(t,\theta)\Lambda_0 \varepsilon(t,\theta)
$$
where $$\Lambda_0$$ is the covariance matrix of the prediction errors. If we instead aim at minimizing the general criterion
$$
\mathbb{E} l(\varepsilon(t,\theta)),
$$
the only difference is that $$\hat{\Lambda}^{-1}(t)\varepsilon(t)$$ must be repaced by $$l_\varepsilon^T(\varepsilon(t))$$, then the updated version is 
$$
\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) l_\varepsilon^T(\varepsilon(t))\right ]_{D_\mathscr{M}}
$$


## Users choice

1. Model set $$\mathscr{F},\mathscr{G},\mathscr{H}$$
2. Input signal $$\{ u(t)\}$$
3. Criterion function $$l$$
4. Gain sequence $$\{ \alpha(t)\}$$
5. Search direction $$R(t)$$
6. Intial conditions $$\hat{\theta}(0), R(0),\dots$$
7. $$\dots$$
