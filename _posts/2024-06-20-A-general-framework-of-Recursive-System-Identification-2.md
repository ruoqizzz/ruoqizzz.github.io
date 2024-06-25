---
layout: post
title: Offline Identification
date: 2024-06-19 12:05:00
description: 
tags: linear_model
categories: control
---

## Aspects of Offline Identification 

Offline identification is to identify a model of a system using data that has been collected and stored beforehand. For simplicity, we define $$z^{t}=[y(t)^T, u(t)^T]^T$$ as the data at the time step $$t$$ and  $$\{y_t\}, \{u_t\}$$ are output and input signals. Offline identification is a wide subject. Here we only try to point out some aspects that have useful implications for recursive identification 

### Identification as Criterion Minimization

Given a prediction model, 

$$
\hat{y}(t\vert \theta)=g_\mathscr{M}(\theta;t,z^{t-1})
$$

a natural measure of its validity is the prediction error,

$$
\varepsilon(t,\theta)=y(t)-\hat{y}(t\vert \theta)
$$

Since $$\varepsilon(t,\theta)$$ is a $$p$$-dimensional vector, it is useful to introduce a scalar measure


$$
l(t,\theta,\varepsilon(t,\theta))
$$

where $$l(\cdot,\cdot,\cdot)$$ is a function from $$\mathbb{R}\times \mathbb{R}^d \times \mathbb{R}^p$$ to $$\mathbb{R}$$. With stored data up to time $$t$$, a natural criterion of the validity of the model $$\mathscr{M}(\theta)$$ is 

$$
V_N(\theta, z^{N})=\frac{1}{N}\sum_{t=1}^N l (t,\theta, \varepsilon(t,\theta)).
$$


The offline estimate, denoted by $$\hat{\theta}_N$$ is obtained by minimization of $$V_N(\theta, z^{N})$$ over $$\theta \in \mathcal{D}_\mathscr{M}$$.

### Choices of Criterion 

#### A Quadratic Criterion

Quadratic criterion is a natural way of measuring the size of the prediction error,

$$
\begin{align}
l(t,\theta,\varepsilon) = \frac{1}{2}\varepsilon^T\Lambda^{-1}\varepsilon
\label{eq:quad_criterion}
\end{align}
$$

where $$\Lambda$$ is a positive definite matrix. A possible disadvantage is that it gives substantial penalties for large errors which might lead to sensitivity to outliers. One simple and common way in system identification is to filter all data before processing. 

#### The Maximum Likelihood Criterion

Here we assume that $$\varepsilon(t,\theta)$$ is a sequence of independent random vectors and its probability density function is $$\bar{f}(t, \theta, x)$$ such that

$$
P(\varepsilon(t,\theta)\in \mathcal{B}) = \int_\mathcal{B} \bar{f}(t, \theta, x) dx.
$$


The output at time $$t$$ can be written as 


$$
y(t) = g_\mathscr{M}(\theta;t,z^{t-1}) + \varepsilon (t,\theta).
$$

Under the assumption, we can write the conditional probability density function of $$y(t)$$ given $$z^{t-1}$$ as

$$
f(\theta; x_t \vert z^{t-1})=P\left (
y(t)=x_t \vert z^{t-1}, \theta
\right) = 
\bar{f}(t, \theta, y_t-g_\mathscr{M}(\theta;t,z^{t-1})).
$$


Using Bayes's rule, the joint density function of $$y(t)$$ and $$y(t-1)$$ given $$z^{t-2}$$ can be expressed as 

$$
\begin{align}
f(\theta; x_t, x_{t-1} \vert z^{t-2})&=P\left (
y(t)=x_t, y(t-1)=x_{t-1} \vert z^{t-2}, \theta
\right) \\
&= P\left (
y(t)=x_t \vert y(t-1)=x_{t-1} ,z^{t-2}, \theta
\right) 
\cdot
P\left (
y(t-1)=x_{t-1} \vert z^{t-2}, \theta
\right) \\
&= P\left (
y(t)=x_t \vert z^{t-1}, \theta
\right) 
\cdot
P\left (
y(t-1)=x_{t-1} \vert z^{t-2}, \theta
\right) 
\\
&=
\bar{f}(t, \theta, y_t-g_\mathscr{M}(\theta;t,z^{t-1}))
\cdot
\bar{f}(t-1, \theta, y_{t-1}-g_\mathscr{M}(\theta;t-1,z^{t-2}))
.
\end{align}
$$

Here we assume that $$\{u^t\}$$ is a deterministic sequence. Iterating the foregoing expression from $$t=N$$ to $$t=1$$ gives the joint probability density of $$y(N), y(N-1),\dots,y(1): f(\theta; x_t, x_{t-1},\dots,x_1)$$. By replacing the dummy variables $$xc_i$$ with corresponding $$y(i)$$, we can obtain

$$
\log (\theta;y(N), y(N-1),\dots,y(1)) = \sum_{t=1}^N \log \bar{f}(t,\theta,\varepsilon(t,\theta)).
$$

With $$l(t,\theta,\varepsilon)=-\log\bar{f}(t,\theta,\varepsilon)$$, $$\hat{\theta}_N$$ equals the maximum likelihood estimate (MLE).

If we assume that $$\varepsilon$$ has a Gaussian distribution with zero mean and covariance matrix $$\Lambda_t(\theta)$$ the 


$$
\begin{align}
l(t,\theta,\varepsilon)&=-\log f(t,\theta, \varepsilon)\\
&=\frac{p}{2}\log2\pi + \frac{1}{2}\log \det \Lambda_t(\theta)+\frac{1}{2}\varepsilon^T\Lambda_t^{-1}\varepsilon.
\label{eq:loss_log}
\end{align}
$$



Here if $$\Lambda_t$$ is known and independent of $$\theta$$, the we obtain the quadratic criterion as in $$\eqref{eq:quad_criterion}$$.

**[Relationship of MLE and MAP]**

MLE and Bayesian maximum a posteriori estimate (MAP) are closely related. Using Bayes's rule, 

$$
P(\theta \vert y^N)=\frac{P(y^N \vert \theta )P(\theta)}{P(y^N)}
$$

where $$P(y^N \vert \theta )$$ is the likelihood function and $$P(\theta)$$ is the prior distribution while $$P(y^N)$$ is independent of $$\theta$$. Thus, the MAP estimate differs from the MLE estimate only via the prior distribution.



### Asymptotic Properties of the Offline Estimate

Some of the results without proof are from [Ljung (1978c)](https://ieeexplore.ieee.org/abstract/document/1101840/) and [Ljung and Caines (1979)](https://ieeexplore.ieee.org/abstract/document/4046253/). 

Suppose that the limit

$$
\bar{V}(\theta)=\lim_{N\rightarrow \infty}\mathbb{E} V_N(\theta,z^N)
$$

exists, where $$\mathbb{E}$$ is the expectation operator with respect to $$z^N$$,  The function $$\bar{V}(\theta)$$  thus is the expected value of the criterion for a certain fixed $$\theta$$. Then, under weak regularity conditions,

$$
\hat{\theta}_N \text{ converges w.p. to } \theta^* \text{ a minimum of } \bar{V}(\theta)
$$

as $$N$$ tends to infinity. Notice that this is true whether or not the model set $$\mathscr{M}$$  contains the true model. Moreover, if  $$\hat{\theta}_N $$ converges to $$\theta^*$$, such that the matrix $$d^2\bar{V}(\theta^*)/d\theta^2$$ is incertible, then

$$
\sqrt{N}(\hat{\theta}_N -\theta^*)\rightarrow \mathcal{N}(0,P),
$$

where 

$$
\begin{align}
P=[\bar{V}^{''}(\theta^*)]^{-1} \left \{ 
\lim N \cdot \mathbb{E}[\bar{V}^{}(\theta^*, z^N)]^T \bar{V}^{'}(\theta^*, z^N)
\right \}
[\bar{V}^{''}(\theta^*)]^{-1}
\label{eq:app_P}
\end{align}
$$

where $$'$$ and $$''$$ denotes differentiation once and twice, respectively with respect to $$\theta$$. 

### Relation to Cramér-Rao Bound

Suppose there is a value $$\theta_\circ$$ in the model set such that with $$\theta=\theta_\circ$$ gives a correct description of the true data. Then it can be shown that $$\theta^*=\theta_\circ$$ and that the matrix P in $$\eqref{eq:app_P}$$ equals the Cramer-Rao lower bound provided that $$l$$ is chosen as in $$\eqref{eq:loss_log}$$.

#### [Cremer-Rao Bound and properties of estimators]

Consider a random vector in $$\mathbb{R}^n$$, $$y^n = \begin{pmatrix} y(1) & y(2) &\dots &y(n)
\end{pmatrix}$$. Let its joint density function be 

$$
f(\theta; x_1,\dots,x_n) = f(\theta;x^n),
\label{eq:f_density}
$$

which is known up to a finite-dimensional parameter vector $$\theta$$ with dimension $$d$$. The problem we face is to find an estimate of $$\theta$$  based on a sample of $$y^n$$.  
There are certain properties of estimators are desirable.  We shall drop the supercrip $$n$$ in $$y^n $$ when not essential. 

**[Unbiased]**

An estimator is thus said to be *unbiasd* if

$$
\mathbb{E}_\theta \hat{\theta}(y^n) = \theta
$$

 i.e., if its expected value is the true parameter value. For an unbiased estimator, it is of interest to know its variance or covariance around the mean

$$
\begin{align}
P_s=\mathbb{E}[\hat{\theta}_s(y)-\theta][\hat{\theta}_s(y)-\theta]^T.
\label{eq:cov}
\end{align}
$$

An estimator $$\hat{\theta}_1(y)$$ is said to be more efficient than an estimator $$\hat{\theta}_2(y)$$ if 

$$
P_1 \leq P_2.
$$

where $$\leq$$ is the matrix inequality. 

**[Consistent]**

The estimator is said to be *consistent* if 

$$
\begin{align}
\hat{\theta}(y^n) \rightarrow \theta \text{ as } n \rightarrow \infty.
\label{eq:consistent}
\end{align}
$$

Since $$\hat{\theta}(y^n)$$ is a random variable, we must specify in what sense this holds. Thus, if $$\eqref{eq:consistent}$$ hold w.p.1, it is *strongly consistent*.  

**[Asumpototic distribution]**

Central limit theorem can often be applied to $$\hat{\theta}(y^n) $$ to infer that $$\sqrt{N}(\hat{\theta}_N -\theta)$$ is asymptotically normal with zero mean as $$n\rightarrow \infty$$. 

**[The Cramér-Rao Inequality]**

Naturally, we hope that the estimators are as efficient as possible, ie. ones that make the covariance matrix $$\eqref{eq:cov}$$ as small as possible. There is a theoretical lower limit called Cramér-Rao inequality.

**[Theorem 1]** Suppose that $$y(i)$$ may take values in interval, whose limits do not depend on $$\theta$$. Suppose that $$\theta$$ is a real scalar and that $$f(\theta;\cdot)$$ in $$\eqref{eq:f_density}$$ is twice continuously differentiable with respect to $$\theta$$. Let $$\hat{\theta}(y^n)$$ be an estimator of $$\theta$$ with expected value $$\mathbb{E}_\theta \hat{\theta}(y^n)=\gamma(\theta)$$, which is assumed to be differentiable with respect to $$\theta$$. Then

$$
\mathbb{E}_\theta [\hat{\theta}(y^n)-\gamma(\theta)]^2 \leq 
-\frac{[d\gamma(\theta)/d\theta^2]}{\mathbb{E}_\theta \partial^2 \log f(\theta;\gamma^n)/\partial \theta^2}.
$$

*Proof* By definition
$$
\mathbb{E}_\theta \hat{\theta}(y^n) = \int  \hat{\theta}(x^n)f(\theta;x^n)dx^n=\gamma(\theta).
$$

Differentiate w.r.t $$\theta$$:

$$
\int  \hat{\theta}(x^n)\frac{\partial}{\partial \theta}f(\theta;x^n)dx^n \\
= \int  \hat{\theta}(x^n) \left \{
\frac{\partial}{\partial \theta}\log f(\theta;x^n)dx^n f(\theta;x^n)dx^n 
\right \}\\
=\mathbb{E}_\theta  \hat{\theta}(y^n) \frac{\partial}{\partial \theta}\log f(\theta;y^n)\\
=\frac{d}{d\theta}\gamma(\theta).
$$

Since 

$$
\int f(\theta;x^n)dx^n =1,
$$

we have 

$$
\begin{align}
\int \frac{\partial}{\partial \theta}f(\theta;x^n)dx^n = 0\\
\int \frac{\partial}{\partial \theta}\log f(\theta;x^n)\cdot f(\theta;x^n) dx^n = 0
\label{eq:first_dif}
\end{align}
$$

or

$$
\mathbb{E}_\theta \frac{\partial}{\partial \theta}\log f(\theta; y^n)=0
$$

and 

$$
\mathbb{E}_\theta \gamma(\theta) \frac{\partial}{\partial \theta}\log f(\theta; y^n)\\
= \gamma(\theta) \mathbb{E}_\theta\frac{\partial}{\partial \theta}\log f(\theta; y^n)
\\
=0
$$

Note that $$\gamma(\theta) $$ is a constant with respect to the random variable $$y^n$$.  
Substracting this from previous expression gives

$$
\mathbb{E}_\theta [\hat{\theta}(y^n) - \gamma(\theta)] \frac{\partial}{\partial \theta}\log f(\theta; y^n) = \frac{d}{d\theta}\gamma(\theta).
$$
The Schwartz inequality now gives 

$$
\begin{align}
\left [ \frac{d}{d\theta}\gamma(\theta) \right]^2 \leq 
\mathbb{E}_\theta [\hat{\theta}(y^n) - \gamma(\theta)]^2 \cdot \mathbb{E}_\theta \left [\frac{\partial}{\partial \theta}\log f(\theta; y^n)\right]^2. \\
\left(\mathbb{E}_\theta[XY])^2 \leq \mathbb{E}_\theta[X]^2 \mathbb{E}_\theta[Y]^2 \right)
\label{eq:proof1}
\end{align}
$$

Also by differentiating $$\eqref{eq:first_dif}$$,

$$
\begin{align}
\int \left \{
\frac{\partial^2}{\partial \theta^2}\log f(\theta;x^n)
+ \left [ \frac{\partial^2}{\partial \theta^2}\log f(\theta;x^n)  \right]^2
\right \} f(\theta;x^n)  = 0. \\
 \left[ \frac{\partial^2}{\partial \theta^2}\log f(\theta;x^n)  \right]^2 \Rightarrow 
-\frac{\partial^2}{\partial \theta^2}\log f(\theta;x^n) 
\label{eq:proof2}
\end{align}
$$

Combing $$\eqref{eq:proof1}$$ and $$\eqref{eq:proof2}$$, we get

$$
\mathbb{E}_\theta [\hat{\theta}(y^n)-\gamma(\theta)]^2 \leq 
-\frac{[d\gamma(\theta)/d\theta^2]}{\mathbb{E}_\theta \partial^2 \log f(\theta;\gamma^n)/\partial \theta^2}
$$

**[Corllary]** Let $$\hat{\theta}(y^n)$$ be a vector-valued unbiased estimator of $$\theta$$. Then $$\mathbb{E} [\hat{\theta}(y^n)-\theta] [\hat{\theta}(y^n)-\theta]^T \geq M^{-1}$$,
where
$$
M = \mathbb{E}_\theta \left[
\frac{\partial}{\partial \theta}\log f(\theta; y^n)
\right ] \left[
\frac{\partial}{\partial \theta}\log f(\theta; y^n)
\right ]^T \\
=\mathbb{E}_\theta \frac{\partial^2}{\partial \theta^2} \log f(\theta; y^n)
$$
is the expected value of the second derivative matrix (the Hessian) of the function $$-\log f$$, also known as the fisher information matrix.



### Optimal Choice of Weights in Quadratic Criteria

Now we want to know the optimal choice of the weighting matrix in 
$$
l(t,\theta,\varepsilon) = \frac{1}{2}\varepsilon^T\Lambda^{-1}\varepsilon.
$$
To do so, we make the assumption that $$\theta^*=\theta_\circ$$ where $$\{\varepsilon(t, \theta_\circ)\}$$ is a sequence of indenpendet random vectors with zero means and covariance matrices $$\Lambda_\circ$$.

With asymptotic convariance matrix $$\eqref{eq:app_P}$$, we find here

$$
\begin{align}
P = \left [ 
\mathbb{E} \psi(t, \theta_\circ) \Lambda^{-1} \psi(t, \theta_\circ)
\right]^{-1}
\mathbb{E} \psi(t, \theta_\circ) \Lambda^{-1} \Lambda_\circ \Lambda^{-1} \psi(t, \theta_\circ) \times 
\left [ 
\mathbb{E} \psi(t, \theta_\circ) \Lambda^{-1}  \psi(t, \theta_\circ)
\right]^{-1}.
\label{eq:P_lambda}
\end{align}
$$
where $$\left [ \frac{d}{d\theta} \hat{y}(t\vert \theta \right )] = \psi (t, \theta)$$. $$P$$ can be seen as  a function of $$\Lambda$$ and the minimal value of  $$P$$  is obtained for $$\Lambda=\Lambda_\circ$$. Then, we also get best estimates $$\hat{\theta}_N$$ in the sense that they are closet to $$\theta_\circ$$ according to $$\eqref{eq:P_lambda}$$.