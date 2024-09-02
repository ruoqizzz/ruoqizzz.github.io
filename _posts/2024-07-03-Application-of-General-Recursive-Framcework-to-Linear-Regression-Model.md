---
layout: post
title: Application of General Recursive Framcework to Linear Regression Model
date: 2024-07-03 18:05:00
description: 
tags: recursive_sysid, linear_model
categories: control
---

## Application to Linear Regression Models

Here, we will discuss how a  general framework of Recursive System Identification ( [(Part I](https://ruoqizzz.github.io/blog/2024/A-general-framework-of-Recursive-System-Identification/), [Part II](https://ruoqizzz.github.io/blog/2024/A-general-framework-of-Recursive-System-Identification-2/), [Part III](https://ruoqizzz.github.io/blog/2024/A-general-framework-of-Recursive-System-Identification-3/)) can be appliced to a very special case: when the predcition is linear in parameters $$\theta$$. 

### The Model Set

The linear prediction can be written as

$$
\begin{align}
\hat{y}(t\vert \theta)=g_\mathscr{M}(\theta;t,z^{t-1})=\phi^T(t)\theta+\mu(t),
\label{eq:linear_model}
\end{align}
$$

where $$\phi(t)$$ is a $$d\times p$$-dimensional matrix function of $$t$$ and $$z^{t-1}$$, $$\theta$$ is a $$d\times 1$$ column vector and $$\mu(t)$$ is a knwon $$p\times 1$$ column vector function  of $$t$$ and $$z^{t-1}$$. 

Sometimes we use the model
$$
\begin{align}
\hat{y}(t\vert \theta)=\theta^T\phi(t)+\mu(t),
\label{eq:linear_model_mp}
\end{align}
$$
which is different to $$\eqref{eq:linear_model}$$. Here $$\theta$$ is an $$n'\times p$$-matrix and $$\phi(t)$$ is an $$n' \times 1$$-column vector function of $$t$$ and $$z^{t-1}$$.  The $$k$$-th row of $$\eqref{eq:linear_model_mp}$$,
$$
\hat{y}_k(t\vert \theta)=\theta_k^T\phi(t)+\mu_k(t),
$$
where $$\hat{y}_k(t\vert \theta)$$ is the $$k$$-th component of $$\hat{y}$$ and $$\theta_k$$ is the $$k$$-th column of $$\theta$$. This is a linear regression model with paramter vecotor $$\theta_k$$ and it is independent to other rows. Thus we can treat this model as a collection of $$p$$ indenpendent linear regressions with same regression vector $$\phi(t)$$.



**[Example 1: Linear Difference Equations]**

Consider a linear difference equations with $$p$$-dimensional output and $$r$$-dimensional input $$u(t)$$,
$$
\begin{align}
y(t)+A_1y(t-1) + \dots + A_n y(t-n) = B_1u(t-1)+\dots+B_m u(t-m)+v(t)
\label{eq:linear_difference}
\end{align}
$$
Where $$A_i$$ are $$p\times p$$  unkown matrices, $$B_k$$ are $$p\times r$$ unkown matrices and $$v(t)$$ is a $$p$$-dimensional disturbance term which is usually either of unspecified character or supposed to be a sequence of indenpendent random vectors each of zero mean. A reasonable predictor is given by
$$
\hat{y}(t\vert \theta)=
\theta^T \phi(t),
$$
with 
$$
\theta^T=
\begin{pmatrix}
A_1 & \dots  & A_n & B_1 & \dots & B_m 
\end{pmatrix}\\
\phi^T(t)=\begin{pmatrix}
-y^T(t-1) & \dots  & -y^T(t-n) & u^T(t-1) & \dots & u^T(t-m)
\end{pmatrix}.
$$
where $$\theta^T$$ is a $$p\times (np+mr)$$-matrix.

**[Example 1b: Comcrete Example]**

$$p=2,n=2,r=1,m=1$$:
$$
\hat{y}(t\vert \theta)=
\begin{bmatrix}
-y_1(t-1) & -y_2(t-1) & 0 & 0 & -y_1(t-2) & -y_2(t-2) & 0 &  0 & u_1(t-1) & 0\\
0 & 0 & -y_1(t-1) & -y_2(t-1) & 0 & 0 & -y_1(t-2) & -y_2(t-2) & 0 & u_1(t-1) 
\end{bmatrix}
\begin{bmatrix}
a_{11}^{(1)} \\
a_{12}^{(1)} \\
a_{21}^{(1)} \\
a_{22}^{(1)} \\
a_{11}^{(2)} \\
a_{11}^{(2)} \\
a_{21}^{(2)} \\
a_{22}^{(2)} \\
b_{11}^{(2)} \\
b_{21}^{(2)} \\
\end{bmatrix}
$$
where $$A_k=(a_{ij}^{(k)})$$, $$B_k=(b_{ij}^{(k)})$$.



THe gradient of the model in $$\eqref{eq:linear_model}$$ is given by
$$
\left [
\frac{d}{d\theta} \hat{y}(t\vert \theta)
\right ]=
\psi(t).
$$
For linear differnece euations, the stability region is easy to demtermine because only have finite past $$y(t)$$ and $$u(t)$$,
$$
D_s = \mathbb{R}^d.
$$

### The Recursive Prediction Error Algorithm

[The general framework of recursive identification](https://mitpress.mit.edu/9780262620581/theory-and-practice-of-recursive-identification-3)  can be directlyo applied to model in $$\eqref{eq:linear_model}$$ 
$$
\begin{align}
&\varepsilon(t) = y(t)-\phi^T(t)\hat{\theta}(t-1)-\mu(t) \\
&\hat{\Lambda}(t) = \hat{\Lambda}(t-1) + \alpha(t)[\varepsilon(t)\varepsilon^T(t) - \hat{\Lambda}(t-1)]\\
&R(t)=R(t-1)+\alpha(t)[\phi(t)\Lambda^{-1}\phi^T(t) - R(t-1)]\\
&\hat{\theta}(t) = 
\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t)
\end{align}
$$

### Approximate Gradient: The instrumental Variable Method

