---
layout: post
title: Convergence Analysis of Recursive Sysid - A Heuristic Discussion
date: 2024-08-26 16:36:00
description: 
tags: recursive_sysid, linear_model
categories: control
---

Here we consider the basic structure for algorithms related to quadratic criteria
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
where $$\eta(t)$$ is a vector related to the gradient of the predition $\hat{y}(t)$ w.r.t $$\hat{\theta}$$, for exawmple $$\psi, \phi, \zeta$$ discussed before and $$z(t) = \begin{pmatrix} y(t) \\ u(t) \end{pmatrix}$$. 

When the criteria is general $$\bar{\mathbb{E}}l(t,\theta,\varepsilon(t,\theta))$$, then the algorithm can changed to 
$$
\begin{align}
&R(t) = R(t-1) + \alpha(t)
H\left(t, R(t-1),\hat{\theta}(t-1),\varepsilon(t), \eta(t)\right)\\
&\hat{\theta}(t) = \hat{\theta}(t-1) + \alpha(t) R^{-1}(t) 
h\left(t,\hat{\theta}(t-1),\varepsilon(t), \eta(t)\right)\\
\end{align}
$$
where functions $$H$$ and $$h$$ are related to the criterion function $$l$$.

Convergence analysis is in general diffusion. A major reason is the coupling between $$\hat{\theta}(t)$$, $$\eta(t)$$ and $$\xi(t)$$ (like a feedback loop) which makes the mapping from $$z^t$$ to $$\hat{\theta}(t)$$ complex.

There are several tools for convergence analysis can be used:

1. Lyapunov Functions
2. Stochastic Approximation 
3. Martingale Convergence Theorems
4. Stabiliy of Differentail Equqations

Here we mainly focus on associating a deterministic differential equation with the recursive algorithm due to its general applicability. The stability of this differential equation can then be analyzed to infer the stability of the recursive algorithm.

## An Associated Differential Equation: A Heuristic Discussion

For ufficiently large $$t$$, the step size $\alpha(t)$ will be arbitartily small due to our assumption that $$\alpha(t)\rightarrow 0$$ as $$t\rightarrow 0$$. Then the estimiates $$\{\hat{\theta}(t)\}$$ will change more and more sloowly. Let take a look a this consequences, (Check book (3.51) to (3.54) for details here.)
$$
\xi (t)=\sum_{j=0}^{t-1} \prod_{k=j+1}^{t-1}\left [A(\hat{\theta}(k))\right] \xi(t)  B(\hat{\theta}(j))z(j)
\label{eq:xi_t+1}
$$
Suppose now that $$\hat{\theta}(k)$$ belongs to a small neighborhood of a value $$\bar{\theta}$$ for $$t-K \leq k \leq t-1$$, such that $$\bar{\theta}\in \mathcal{D}_s$$ where $$\forall \theta \in \mathcal{D}_s$$, $$A(\theta)$$ has all eighven values strictly inside the unit circle. Then, if the neighborhood is small enough, we can write
$$
\prod_{k=t-K}^{t-1}A(\hat{\theta}(k)) \approx A(\bar{\theta})^K,
$$
which has a norm smaller than $$C\cdot \lambda^K$$ for some $$\lambda <1$$. For large enough $$K$$, we may thus approximate $$\eqref{eq:xi_t+1}$$ as 
$$
\xi (t) \approx \sum_{j=0}^{t-1} A(\bar{\theta})^{t-j-1} \xi(t)  B(\bar{\theta})z(j)
$$
Now we can add terms cooresponding to $A(\bar{\theta})^{t-j-1} \xi(t)  B(\bar{\theta})z(j)$ for $j<t-K$ to this sum. Since $A(\bar{\theta})$ is stable, this wil only make an arbitarily small change. Then we have
$$
\xi (t) \approx \xi (t,\bar{\theta}) \triangleq \sum_{j=0}^{t-1} A(\bar{\theta})^{t-j-1} \xi(t)  B(\bar{\theta})z(j)
$$
Which can be written recursively as 
$$
 &\xi (t+1,\bar{\theta}) = A(\bar{\theta})  \xi (t,\bar{\theta})+B(\bar{\theta})z(t),\\
 &\xi (0,\bar{\theta})=0.
\label{xi_theta_bar}
$$
As a consequence we also have
$$
\begin{aligned}
\hat{y}(t) \approx y(t\vert\bar{\theta}), \eta (t)\approx \eta(t,\bar{\theta}),\\
\varepsilon (t)\approx \varepsilon (t,\bar{\theta})
\end{aligned}
\label{eq:approximation}
$$
where 
$$
\begin{align}
\begin{pmatrix} y(t\vert\bar{\theta}) \\ \text{col } \eta(t,\bar{\theta}) \end{pmatrix}&=C(\bar{\theta}) \xi(t,\bar{\theta}) ,\\
\varepsilon(t,\bar{\theta})&=y(t)-\hat{y}(t\vert\bar{\theta}) 
\end{align}
$$
When $$\hat{\theta}(t)$$ is close to $$\bar{\theta}$$ and $$R(t)$$ is close to $$\bar{R}$$ and $$t$$ is large we can consequentily use the approximation $\eqref{eq:approximation}$ to conclude that 
$$
&\hat{\theta}(t) \approx \hat{\theta}(t-1) + \alpha(t) \bar{R}^{-1}(t) \eta(t, \bar{\theta}) \Lambda^{-1} \varepsilon(t,\bar{\theta}), \\
&R(t) = R(t-1) + \alpha(t)\left [\eta(t,\bar{\theta})\Lambda^{-1}(t)\eta^T(t,\bar{\theta}) - \bar{R}\right]\\
$$
Introduce the expected values
$$
f(\bar{\theta}) \triangleq \mathbb{E}\eta(t, \bar{\theta}) \Lambda^{-1} \varepsilon(t,\bar{\theta}),\\
G(\bar{\theta}) \triangleq \mathbb{E}\eta(t, \bar{\theta}) \Lambda^{-1} \eta^T(t,\bar{\theta})
\label{eq:f_G}
$$
where expectation is over $$z^t$$. Since $$t$$ is large, we have neglected the transients in $\eqref{xi_theta_bar}$ and the RHS of $\eqref{eq:f_G}$ to be time-invariant. We thus have
$$
&\hat{\theta}(t) \approx \hat{\theta}(t-1) + \alpha(t) \bar{R}^{-1}(t) f(\bar{\theta}) + \alpha(t) v(t), \\
&R(t) \approx R(t-1) + \alpha(t)\left [G(\bar{\theta}) - \bar{R}\right] + \alpha(t)w(t)
\label{eq_theta_R_approx}
$$
where $$\{v(t)\}$$ and  $$\{w(t)\}$$ are zero-mean random variables. 

Let $\Delta\tau$ be a small number and let $t, t'$ be defined by 
$$
\sum_{k=t}^{t'}\alpha(k)=\Delta \tau
\label{eq:delta_tau}
$$
If $\hat{\theta}(t)=\bar{\theta}$ and $R(t)=\bar{R}$, we then have from $\eqref{eq_theta_R_approx}$
$$
&\theta(t')\approx \bar{\theta} + \Delta\tau \bar{R}^{-1} f(\bar{\theta}) + \sum_{k=t}^{t'}\alpha(k) v(k), \\
&R(t') \approx \bar{R} +\Delta\tau\left [G(\bar{\theta}) - \bar{R}\right] + \sum_{k=t}^{t'}\alpha(k)w(k).
$$
Since $$v(k)$$ and $$w(k)$$ have zero means, the contribution forom the third terms of RHS will be an order of magnitude less than those from second terms. Therefore
$$
&\theta(t')\approx \bar{\theta} + \Delta\tau \bar{R}^{-1} f(\bar{\theta}) \\
&R(t') \approx \bar{R} +\Delta\tau\left [G(\bar{\theta}) - \bar{R}\right]
\label{eq:t'_approx}
$$
With a change of time scale, according to $$\eqref{eq:delta_tau}$$ such that $$t \leftrightarrow \tau$$ and $$t' \leftrightarrow \tau +\Delta \tau$$, we could regard $$\eqref{eq:t'_approx}$$ as a scheme to solve the differential equation with small $\Delta \tau$,
$$
&\frac{d}{d\tau}\theta_D(\tau) = \bar{R}^{-1}f(\theta_D(\tau))\\
&\frac{d}{d\tau}R_D(\tau) =G(\bar{\theta}) - R_D(\tau)
\label{eq:ODE}
$$
Here the subscript $D$ is used to distinguish the solution of $$\eqref{eq:ODE}$$ from the variables in $$\eqref{eq:general_form}$$. The chain of arguments suggests that if for some large $t_0$
$$
\hat{\theta}(t_0) = \theta_D(t_0), R(t_0)=R_D(t_0), \sum_{k=1}^{t_0}\alpha(k)=\tau_0,
$$
then for $t>t_0$,
$$
\hat{\theta}(t) \approx \theta_D(\tau), R(t_0)\approx R_D(\tau), \sum_{k=1}^{t_0}\alpha(k)= \tau,
\label{eq:connect_t_tau}
$$
which can be interpreted as the solution of $\hat{\theta}(t)$ and $R(t)$ can be approximiated by the solution of differential equation $\eqref{eq:ODE}$. 

These arguments have of coiurse been entire heuristic. They point, however, to the results $\eqref{eq:connect_t_tau}$ that asymptotically the algorithm  $$\eqref{eq:general_form}$$  can be linked to the differential equation $$\eqref{eq:ODE}$$. The estmiate should in some sense follow the trajectories of $$\eqref{eq:ODE}$$ asymptotically.

### The connection between  $$\eqref{eq:general_form}$$  and  $$\eqref{eq:ODE}$$.

**(A)** Suppose $D_c$ is an invariant set of $$\eqref{eq:ODE}$$ and $$D_A$$ is its domain of attraction. Then if $\hat{\theta}(t)\in D_A$ sufficiently often, the estimate will tend to $D_c$ w.p.1 as $t$ approaches infinity.

**(B)** Only stable stationay points of  $$\eqref{eq:ODE}$$ are possible convergence points for algorithm  $$\eqref{eq:general_form}$$.

**(C)** The trajectories $$\theta_D(\tau)$$ of  $$\eqref{eq:ODE}$$ are the "asymptotic paths" of the estimates $$\hat{\theta}(t)$$, generated by  $$\eqref{eq:general_form}$$ .

All above conditions can be proven to be formally correct. See the later post.

## The recipe for convergence analysis of  $$\eqref{eq:general_form}$$
1. Compute the prediction errors $\varepsilon(t, \theta)$ and gradient approximations $$\eta(t,\theta)$$ that would be obtained for a fixed and constant model $\theta$
2. Evaluate the average updating direction for the algorithm, based on these variables, see $$\eqref{eq:f_G}$$
3. Define a differential equation that has this direction as the right hand side
   $$
   \dot{\theta_D} = \bar{R}^{-1}f(\theta_D)\\
   \dot{R_D} =G(\bar{\theta}) - R_D
   $$
4. Study the stability properties of this differential equation.

