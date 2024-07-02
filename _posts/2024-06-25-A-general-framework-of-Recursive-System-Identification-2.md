---
layout: post
title: A general framework of Recursive System Identification (Part II)
date: 2024-06-25 14:58:00
description: 
tags: recursive_sysid
categories: control
---

This blog post aims to summarise Chapter 3 of [Theory and Practice of Recursive Identification](,https://mitpress.mit.edu/9780262620581/theory-and-practice-of-recursive-identification/) which derives a general recursive identification method that can be applied to any set of (linear) models.

To develop a unified approach to recursive identification consists 3 phases：

1. Define the framework
2. Derive the Algorithm: In the book, the authors mainly focus on **minimizing the prediction error variance recursively**, using the idea from stochastic approximation (See blog post Recursive Least Squares Derived from Stochastic Approximation Approach for details).
3. Apply the Algorithm: We will show how the general algorithm can be applied to a particular model set, **a linear regression model** later.

This is the second part of this blog series, please first see *[A general framework of Recursive System Identification (Part I): System and Models](https://ruoqizzz.github.io/blog/2024/A-general-framework-of-Recursive-System-Identification/).*

## Recursive Gauss-Newton Algorithms for Quadratic Criteria

Here we will talk about how to derive a recursive algorithm for the estimation of model parameters minimizing a prediction error criterion. The development will lead to basic general algorithm that is the main object of recursive identification. 

In the spirit of the offline criterion (see blog *[Offline Identification]()*), we would like to select $$\theta$$ such that the criterion 

$$
\mathbb{E}_{z^t}l(t, \theta, \varepsilon(t,\theta))
$$

where $$z^t = (y(t), u(t))$$ gives the current outputs and inputs. This type of criteria can be minimized recursively using the stochastic approximation approach (see blog *[Recursive Least Squares Derived from Stochastic Approximation Approach](https://ruoqizzz.github.io/blog/2024/Recursive-Least-Squares-Derived-from-Stochastic-Approximation-Approach/)* for details.)  For example, for the problem

$$
\begin{align}
\min_x V(x) \\
V(x)=\mathbb{E}_e J(x, e(t)),
\end{align}
$$

it can be recursively minimized by the stochastic Newton method,

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

where $$-Q(x,e)$$ is the gradient of $$J(x,e)$$ with respect to $$x$$ and $$\bar{V}^{''}(x,e)$$ is some approximation of the second derivative of $$V(x)$$ based on the observations up to time $$t$$.

To be more general, here we assume that the limit

$$
\bar{E} l(t, \theta, \varepsilon(t,\theta)) \triangleq \lim_{N \rightarrow \infty}  l(t, \theta, \varepsilon(t,\theta))  \triangleq \bar{V}(\theta)
$$

exists.



### A General Minimization Algorithm for Quadratic Criteria

Here we talk about the special case when $$l$$ is quadratic in $$\varepsilon$$,

$$
\begin{align}
V(\theta) = \mathbb{E} l (t,\theta,\varepsilon)= \mathbb{E} 
\frac{1}{2}\varepsilon^T(t,\theta)\Lambda^{-1}\varepsilon^T(t,\theta).
\label{eq:V}
\end{align}
$$


Here we have

$$
\left[ \frac{d}{d\theta}l (t,\theta,\varepsilon)\right]^T=-\psi(t,\theta)\Lambda^{-1}\varepsilon(t,\theta)
$$

where 

$$
\frac{d}{d\theta}\varepsilon(t,\theta)=\frac{d}{d\theta}[y(t)-\hat{y}(t\vert \theta)=-\psi^T(t,\theta).
$$

Let's denote the second-derivative approximation $$\bar{V}^{''}(x,e)$$ by  $$R(t)$$, the algorithm $$\eqref{eq:stochastic_newton}$$ becomes,

$$
\begin{align}
\hat{\theta}(t)=\hat{\theta}(t-1) + \alpha(t)
R^{-1}
\psi(t,\hat{\theta}(t-1)\Lambda^{-1}\varepsilon(t,\hat{\theta}(t-1)).\
\label{eq:stochastic_newton_theta}
\end{align}
$$

The quantities $$\psi(t,\hat{\theta}(t-1)$$ and $$\varepsilon(t,\hat{\theta}(t-1))$$ can be computed using 

$$
\begin{align}
\xi (k+1, \hat{\theta}(t-1)) &= A(\hat{\theta}(t-1) \xi (k, \hat{\theta}(t-1)) + B(\hat{\theta}(t-1))z(k),\\
k&=0,1,\dots,t-1, \xi(0, \hat{\theta}(t-1))=\xi_0\\
\begin{pmatrix}\hat{y}(t\vert \hat{\theta}(t-1))\\
\text{col}~\psi(t,\hat{\theta}(t-1))
\end{pmatrix}&=C(\hat{\theta}(t-1))\xi(t,\hat{\theta}(t-1))\\
\varepsilon(t,\hat{\theta}(t-1))&=y(t)-\hat{y}(t\vert \hat{\theta}(t-1))
\label{eq:newton-kalman}
\end{align}
$$

The problem here is that $$\eqref{eq:newton-kalman}$$ is not recursive in general. To compute $$\xi (k+1, \hat{\theta}(t-1)) $$, we need the data from $$z(0)$$ to $$z(t-1)$$. It can be solved by first rewrite $$\xi (k+1, \hat{\theta}(t-1)) $$.

$$
\xi (k+1, \hat{\theta}(t-1)) = [A(\hat{\theta}(t-1)]^t\xi_0 + 
\sum_{k=0}^{t-1}[A(\hat{\theta}(t-1)]^{t-k-1}B(\hat{\theta}(t-1)z(k).
$$

If $$\hat{\theta}(t-1)\in D_s$$ where $$D_s=\{
\theta \vert A(\theta) \text{ has all eigenvalues strictly inside the unit circle}
\}$$, then the factor $$[A(\hat{\theta}(t-1)]^t$$ tends to zero exponentially. Consequently, the sum is dominated by the last terms $$k=t-K, t-K+1, \dots, t-1$$ for some value $$K$$. Also since $$\alpha(t)$$ is a small number for large $$t$$, the difference between $$\hat{\theta}(t-1)$$ and $$\hat{\theta}(t-K)$$ will be quite small for large $$t$$. Combining these two factors, 

$$
\xi(t, \hat{\theta}(t-1)) \approx \xi(t) \triangleq 
\sum_{k=1}^{t-1} \left [
\sum_{s=k+1}^{t-1}A(\theta({\hat{s}))}
\right]
B( \hat{\theta}(k)z(k)
$$

Then $$\xi(t)$$ can be computed recursively without all history data

$$
\begin{align}
\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t).
\label{eq:recursive_xi}
\end{align}
$$

Based on this, 

$$
\begin{pmatrix}
   \hat{y}(t | \hat{\theta}(t-1)) \\
   \text{col } \psi(t, \hat{\theta}(t-1))
   \end{pmatrix} \approx
   \begin{pmatrix}
   \hat{y}(t) \\
   \text{col } \psi(t)
   \end{pmatrix} \triangleq C(\hat{\theta}(t-1)) \xi(t).
$$

and 

$$
\begin{align}
\varepsilon(t, \hat{\theta}(t-1)) \approx \varepsilon(t) = y(t) - \hat{y}(t).
\label{eq:recursive_epsilon}
\end{align}
$$

These gives

$$
\hat{\theta}(t) = \hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t).
$$

This show shows that by using the approximations and understanding that the influence of older data diminishes exponentially, we can reformulate the original algorithm into a truly recursive one. This allows us to compute the required terms at each time step efficiently without needing the full history of data, thus making the algorithm more practical and computationally efficient.

### The Gauss-Newton Search Direction

Let's discuss how to choose $$R(t)$$.

The Hessian of $$V(\theta)$$  in $$\eqref{eq:V}$$ is given by
$$
\frac{d^2}{d\theta^2} V(\theta) = \frac{d^2}{d\theta^2} \mathbb{E} \frac{1}{2}\varepsilon^T(t,\theta)\Lambda^{-1}\varepsilon^T(t,\theta) \\
=  \mathbb{E}\frac{d}{d\theta}  -\varepsilon(t,\theta)  \Lambda^{-1}\psi^T(t, \theta)\\
=  \mathbb{E}\psi(t, \theta)\Lambda^{-1}\psi^T(t, \theta) + 
 \mathbb{E} \left \{ 
\left [ \frac{d^2}{d\theta^2}
\varepsilon^T(t,\theta)
\right ]  \Lambda^{-1}  \varepsilon(t,\theta) 
\right \}.
$$

Here the second derivative of $$\varepsilon$$ is a matrix whose $$i,j$$-component is given by

$$
\mathbb{E} \sum_{k,l=1}^p \left[ \frac{d}{d\theta_i}\frac{d}{d\theta_j} \varepsilon_k(t,\theta)\right](\Lambda^{-1})_{kl}\varepsilon_l(t,\theta).
$$


**[Example]**

$$
\varepsilon(t, \theta) = 
\begin{pmatrix}
\varepsilon_1(t, \theta) \\
\varepsilon_2(t, \theta)
\end{pmatrix}
$$

The first derivative of $$\varepsilon(t, \theta)$$ with respect to $$\theta$$ is:

$$
\frac{\partial \epsilon(t, \theta)}{\partial \theta} = 
\begin{pmatrix}
\frac{\partial \varepsilon_1(t, \theta)}{\partial \theta_1} & \frac{\partial \varepsilon_1(t, \theta)}{\partial \theta_2} \\
\frac{\partial \varepsilon_2(t, \theta)}{\partial \theta_1} & \frac{\partial \varepsilon_2(t, \theta)}{\partial \theta_2}
\end{pmatrix}
=
\begin{pmatrix}
t & 2\theta_2 \\
t \cos(\theta_1 t) & 1
\end{pmatrix}
$$


Second Derivative of $$\varepsilon(t, \theta)$$:

For $$\varepsilon_1(t, \theta) = \theta_1 \cdot t + \theta_2^2$$,

$$
\frac{\partial^2 \varepsilon_1(t, \theta)}{\partial \theta_1^2} = 0, \quad \frac{\partial^2 \varepsilon_1(t, \theta)}{\partial \theta_1 \partial \theta_2} = 0\\
\frac{\partial^2 \varepsilon_1(t, \theta)}{\partial \theta_2 \partial \theta_1} = 0, \quad \frac{\partial^2 \varepsilon_1(t, \theta)}{\partial \theta_2^2} = 2
$$

For $$\varepsilon_2(t, \theta) = \sin(\theta_1 \cdot t) + \theta_2$$

$$
\frac{\partial^2 \varepsilon_2(t, \theta)}{\partial \theta_1^2} = -t^2 \sin(\theta_1 t), \quad \frac{\partial^2 \varepsilon_2(t, \theta)}{\partial \theta_1 \partial \theta_2} = 0 \\
\frac{\partial^2 \varepsilon_2(t, \theta)}{\partial \theta_2 \partial \theta_1} = 0, \quad \frac{\partial^2 \varepsilon_2(t, \theta)}{\partial \theta_2^2} = 0
$$


So, the second derivative tensor is:

$$
\begin{pmatrix}
\frac{\partial^2 \varepsilon_1}{\partial \theta_1^2} & \frac{\partial^2 \varepsilon_1}{\partial \theta_1 \partial \theta_2} \\
\frac{\partial^2 \varepsilon_1}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 \varepsilon_1}{\partial \theta_2^2}
\end{pmatrix}
,
\begin{pmatrix}
\frac{\partial^2 \varepsilon_2}{\partial \theta_1^2} & \frac{\partial^2 \varepsilon_2}{\partial \theta_1 \partial \theta_2} \\
\frac{\partial^2 \varepsilon_2}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 \varepsilon_2}{\partial \theta_2^2}
\end{pmatrix}
=
\begin{pmatrix}
0 & 0 \\
0 & 2
\end{pmatrix}
,
\begin{pmatrix}
-t^2 \sin(\theta_1 t) & 0 \\
0 & 0
\end{pmatrix}
$$



Suppose that there exists a value $$\theta_\circ \in \mathcal{D}_\mathscr{M}$$ that gives a correct description of the system, so that $$\{\varepsilon(t, \theta)\}$$ is a sequence of independent random vectors each of zero mean. This implies that $$\varepsilon(t, \theta_\circ)$$ is independent of $$z^{t-1}$$ and hence of 

$$
\frac{d}{d\theta_i}\frac{d}{d\theta_j}\varepsilon(t,\theta) =
\frac{d}{d\theta_i}\frac{d}{d\theta_j} [y(t)-g_\mathscr{M}(\theta;t,z^{t-1})]\\
=-\frac{d}{d\theta_i}\frac{d}{d\theta_j}g_\mathscr{M}(\theta;t,z^{t-1}).
$$

At the true minimum $$\theta_\circ$$,

$$
\mathbb{E} \left \{ 
\left [ \frac{d^2}{d\theta^2}
\varepsilon^T(t,\theta)
\right ]  \Lambda^{-1}  \varepsilon(t,\theta) 
\right \}=0
$$

Then a suitable approximation of Heesion is

$$
\frac{d^2}{d\theta^2} V(\theta) = \mathbb{E}\psi(t, \theta)\Lambda^{-1}\psi^T(t, \theta)
$$

This approximation is good when close to the minimum where a true Hessian is more important than elsewhere. With an approximation of the Hessian, the algorithms are often referred to the Gauss-Newton direction.

A natural approximation of Hession at $$\theta=\hat{\theta}(t-1)$$, based on the observation $$z^{t}$$ is then obtained by 

$$
\begin{align}
R(t)=\frac{1}{t}
\sum_{k=1}^t \psi(k, \hat{\theta}(t-1))\Lambda^{-1}\psi^T(k, \hat{\theta}(t-1)).
\label{eq:rt}
\end{align}
$$



This is not recursive because $$\psi(k, \hat{\theta}(t-1))$$ can not be computed recursively. Then we have to use

$$
\begin{align}
R(t)=\frac{1}{t}
\sum_{k=1}^t \psi(k)\Lambda^{-1}\psi^T(k).
\label{eq:app_rt}
\end{align}
$$

where $$\psi(k)$$ are determined as ...
However, since the first terms of sum in $$\eqref{eq:app_rt}$$ are computed for parameter estimates far from $$\hat{\theta}(t-1)$$, $$\eqref{eq:app_rt}$$ is usually not a good approximation of $$\eqref{eq:rt}$$. It is better to use a weighted mean where more weight is put on the last values

$$
\begin{align}
R(t)=\frac{1}{t}
\sum_{k=1}^t \beta(t,k)\psi(k)\Lambda^{-1}\psi^T(k)+\delta(t)R_0
\label{eq:app_rr}
\end{align}
$$

where $$\delta(t)+\sum_{k=1}^t \beta(t,k)=1$$.  Also, we can added some prior infomation in $$R_0$$. We can also write this in a recursive way

$$
R(t)=R(t-1)+\alpha(t)[\psi(t)\Lambda^{-1}\psi^T(t) - R(t-1)],
R(0)=R_0.
$$

When $$\alpha(t)$$ is chosen larger than $$1/t$$, we put more weight on recent measurements. 




### The choice of Weighting Matrix

When the system has scalar output, then a constant $$\Lambda$$ acts only as scaling factor. For a multioutput system, the choice of $$\Lambda$$ will affect the accuracy of the estimates. The optimal choice is the covariance matrix of the true prediction errors

$$
\Lambda_\circ = \mathbb{E}[\varepsilon(t,\theta_\circ)\varepsilon^{T}(t,\theta_\circ) ]
$$

which gives the smallest covariance matrix of the parameter estimates in the offline case. Since $$\Lambda_\circ $$ is usually ubnknown, a reasonable choice of $$\Lambda$$,

$$
\hat{\Lambda}(t) = \hat{\Lambda}(t-1) + \alpha(t)[\varepsilon(t)\varepsilon^T(t) - \hat{\Lambda}(t-1)].
$$





### Projection into the Stability Region

The model $$\mathscr{M}$$ is obtained as $$\theta$$ ranges over the set $$\mathcal{D}_\mathscr{M}$$. The  generation of prediction is stable only for $$\theta\in \mathcal{D}_s$$ where 

$$
\begin{align}
D_{s}=\{
\theta \vert \mathscr{F}(\theta) \text{ has all eigenvalues strictly inside the unit circle}
\}.
\label{eq:Ds}
\end{align}
$$

In fact, in the derivation of the algorithm, we used an assumption that $$\hat{\theta}(t)\in D_{s}$$ to adjust $$\xi$$ and $$\varepsilon$$ in $$\eqref{eq:recursive_xi}$$ and $$\eqref{eq:recursive_epsilon}$$. This can be accomplished by a project facility of the type

$$
\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}},
$$

where

$$
\begin{align}
\left [x
\right ] _{D_\mathscr{M}}= \left \{\begin{array}{lr}
x & \text{if }x \in {D_\mathscr{M}}\\
\text{a value strctily interior to }{D_\mathscr{M}} \text{ if } x \notin {D_\mathscr{M}}
\end{array}
\right.
\end{align}
$$


### Summary of the Algorithm

Now we can summarize the general algorihtm for minizing the quadratic criterion.

$$
\begin{align}
&\varepsilon(t) = y(t)-\hat{y}(t) \\
&\hat{\Lambda}(t) = \hat{\Lambda}(t-1) + \alpha(t)[\varepsilon(t)\varepsilon^T(t) - \hat{\Lambda}(t-1)]\\
&R(t)=R(t-1)+\alpha(t)[\psi(t)\Lambda^{-1}\psi^T(t) - R(t-1)]\\
&\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}}\\
&\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t)\\
&\begin{pmatrix}
   \hat{y}(t) \\
   \text{col } \psi(t)
   \end{pmatrix} = C(\hat{\theta}(t-1)) \xi(t).
\label{eq:algo_summary_quad}
\end{align}
$$



### Equivalent Rerrangement

Introduce

$$
P(t)=\alpha(t)R^{-1}(t).
$$

Then,

$$
P(t)=\frac{1}{\gamma(t)}\left\{
P(t-1)-P(t-1)\psi(t)[\psi^T(t)P(t-1)\psi(t)+\gamma(t)\hat{\Lambda}(t)
]^{-1}\psi^T(t)P(t-1),
\right \}
$$

where

$$
\gamma(t)=\alpha(t-1)[1-\alpha(t-1)]/\alpha(t-1).
$$

Using this expression we can write

$$
\begin{align}
L(t)&\triangleq \alpha(t)R^{-1}(t)\psi(t)\hat{\Lambda}^{-1}(t)\\
&=P(t)\psi(t)\hat{\Lambda}^{-1}(t)\\
&=P(t-1)\psi(t)[\psi^T(t)P(t-1)\psi(t)+\gamma(t)\hat{\Lambda}(t)
]^{-1}.
\end{align}
$$

Hence the algorithm can be rewriiten as

$$
\begin{align}
&\varepsilon(t) = y(t)-\hat{y}(t) \\
&\hat{\Lambda}(t) = \hat{\Lambda}(t-1) + \alpha(t)[\varepsilon(t)\varepsilon^T(t) - \hat{\Lambda}(t-1)]\\
&S(t)= \psi^T(t)P(t-1)\psi(t)+\gamma(t)\hat{\Lambda}(t)\\
&L(t)=P(t-1)\psi(t)S^{-1}(t)\\
&\hat{\theta}(t) = \left [\hat{\theta}(t-1) + \alpha(t) R^{-1}(t) \psi(t) \Lambda^{-1} \varepsilon(t)\right ]_{D_\mathscr{M}}\\
&P(t)=[P(t-1)-L(t)S(t)L^T(t)]/\gamma(t)\\
&\xi (t+1)= A(\hat{\theta}(t))\xi(t) + B(\hat{\theta}(t))z(t)\\
&\begin{pmatrix}
   \hat{y}(t) \\
   \text{col } \psi(t)
   \end{pmatrix} = C(\hat{\theta}(t-1)) \xi(t).
\label{eq:algo_summary_quad2}
\end{align}
$$









