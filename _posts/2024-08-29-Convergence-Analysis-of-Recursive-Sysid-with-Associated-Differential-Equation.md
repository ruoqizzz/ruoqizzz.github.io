---
layout: post
title: Convergence Analysis of Recursive Sysid with Associated Differential Equation
date: 2024-08-29 17:49:00
description: 
tags: recursive_sysid, linear_model
categories: control
---

Based on heuristic arguments in previous blogs, we describe a link between the recursive identification algorithms under discussion and a certain associated differential equation. The analysis suggests that the convergence of the algorithm could be studied in terms of stability properties of related differential equations as summarized in heuristic claims. 

Let's study the following algorithms

$$
x(t) = x (t-1) + \alpha Q(t, x(t-1),\phi(t) )\\
\phi(t) = A( x (t-1)) \phi (t-1) + B( x (t-1)) e(t)
$$

with assumptions

**C1:** The function $$ Q(t, x, \varphi) $$ is Lipschitz continuous in $$ x $$ and $$ \varphi $$ in any neighborhood of $$ (\bar{x}, \bar{\varphi}) $$, where $$ \bar{x} \in D_R $$ and $$ \bar{\varphi} $$ is arbitrary:

$$
|Q(t, x_1, \varphi_1) - Q(t, x_2, \varphi_2)| \leq K(\bar{x}, \bar{\varphi}, \rho, v)\left[|x_1 - x_2| + |\varphi_1 - \varphi_2|\right]
\label{eq:recursive_proof}
$$

for $$ |x_i - \bar{x}| \leq \rho $$, $$ |\varphi_i - \bar{\varphi}| \leq v $$,

where $$ \rho = \rho(\bar{x}) > 0 $$, $$ v = v(\bar{\varphi}) > 0 $$. The Lipschitz constant $$ K $$ may thus depend on the neighborhood.

**C2:** The matrix functions $$ A(x) $$ and $$ B(x) $$ are Lipschitz continuous in $$ x $$ for $$ x \in D_R $$.

**C3:** The sequence $$ \{e(t)\} $$ is such that

$$
\sum_{k=1}^{t} \beta(t, k) Q(k, \bar{x}, \bar{\varphi}(k, \bar{x})) \to f(\bar{x}) \text{ as } t \to \infty \text{ for all } \bar{x} \in D_R.
$$

Here $$ \beta(t, k) $$ are the weights corresponding to $$ \{ \alpha(t) \} $$ such that


$$
\begin{aligned}
\bar{\beta}(t, k) &= \prod_{i=k}^{t}\frac{\alpha(i-1)}{\alpha(i)}[1-\alpha(i)]\\
&=\frac{\alpha(k-1)}{\alpha(t)}\prod_{i=k}^{t}[1-\alpha(i)]
&\text{the normalized  mecumulative effect of the past}\\
\beta(t, k) &= \alpha(t)\bar{\beta}(t, k)
\end{aligned}
$$


and $$ \varphi(t, \bar{x}) $$ is defined by

$$
\varphi(t, \bar{x}) = A(\bar{x}) \varphi(t-1, \bar{x}) + B(\bar{x}) e(t), \quad \varphi(0, \bar{x}) = 0.
$$


**C4:** For all $$ x \in D_R $$ we have, for **some** $$ C(\bar{x}, \lambda, c) $$,

$$
\sum_{k=1}^{t} \beta(t, k)[1 + v(k, \lambda, c)] \cdot K(\bar{x}, \varphi(k, \bar{x}), \rho(\bar{x}), v(k, \lambda, c)) \to C(\bar{x}, \lambda, c) < \infty \text{ as } t \to \infty.
$$

Here $$ \lambda $$ is the maximum eigenvalue norm of $$ A(\bar{x}) $$, $$ v(t, \lambda, c) $$ is defined by

$$
v(t, \lambda, c) \triangleq c \sum_{k=1}^{t} \lambda^{t-k} |e(k)|,
$$

and $$ K $$ is the Lipschitz constant defined in C1.

**C5:** $$ \sum_{t=1}^{\infty} \alpha(t) = \infty $$.

**C6:** $$ \alpha(t) \to 0 \text{ as } t \to \infty $$.

**Interpretation of C1 and C2:** The function is smooth such that a small changes in $$x$$ or $$\phi$$ will not give a big changes in $$Q,A,B$$. 

**Interpretation of C3:** Recall that $$\sum\beta(t,k)=1$$ if $$\gamma(1)=1$$ (See PTRI 2.129), which gives

$$
\sum_{k=1}^T \beta(t,k)Q(k,\bar{x},\phi(k,\bar{x}))
$$

in C3 is a weighted average of the term $$\{Q(k,\bar{x},\phi(k,\bar{x}))\}$$. There in turn are the average asymptotic updating steps that the algorithm uses when $$\{x(k)\}$$ stays close enough to a certain value $$x$$. Hen the limit $$f(\bar{x})$$ assumed to exist in $$C3$$ represents an average asymptotic update direction for $$\eqref{eq:recursive_proof}$$. 

**Interpretation of C4:** The deviations from the average behavior is bounded by bounding the influence of past errors assuming the worst case with maximum eigenvalue.

**Theorem 4.1** Consider the algorithm  $$\eqref{eq:recursive_proof}$$ subject to the conditions C1–C6. Let $$\bar{D}$$ be a closed subset of $$D_R$$. Assume that there is a constant $$C < \infty$$ and a subsequence $$\{t_k\}$$ (that may depend on the particular sequence $$\{e(r)\}$$) such that

$$
x(t_k) \in \bar{D} \quad \text{and} \quad |\varphi(t_k)| < C.
$$

Assume also that there exists a twice-differentiable function $$V(x)$$ in $$D_R$$, such that, with $$f(x)$$ given by C3,

$$
V'(x)f(x) \leq 0, \quad x \in D_R.
$$

Then either

$$
x(t) \to D_c = \{x | x \in D_R \text{ and } V'(x)f(x) = 0 \} \text{ as } t \to \infty
$$

or

$$
\{x(t)\} \text{ has a cluster point on the boundary of } D_R.
$$

which means that there exists a subsequence $$\{t_j\}$$ such that $$x_j$$ tends to the boundary of $$D_R$$ as $$j$$ approcahces infinity.

The asymptotic average updating direction for the algorihtm is according to C3 given by $$f(x)$$, so the corresponding differential equation is 

$$
\frac{d}{d\tau}x(\tau) = f(x(\tau))
$$



The lyapunov function ensures that all trejectories of this differential equation that start in $$D_R$$ will either leave $$D_R$$ or converge to D_c as time tends to infinity.

The theorem 4.1. holds for any sequence $$\{e(t)\}$$ subject to C4 and C4. If $$\{e(t)\}$$ is regarded as a stochastic process, such that C3 and C4 holds w.p.1, then the conclusion of the theorem also hold w.p.1. 

Let's first check some examples.

**Example: Estimation of Mean** 

Consider the algorithm for estimating the mean of a random variable,

$$
x(t) = x(t - 1) + \frac{1}{t} [e(t) - x(t - 1)],
$$

where $$e(t)$$ represents a sequence of observations of the random variable. It can be seen as a special case of general form 

$$
Q(t, x, \varphi) = \varphi - x, \quad \varphi(t) = e(t) \quad \text{(i.e., } A(x) = 0, B(x) = 1\text{)}, \quad \text{and } \gamma(t) = \frac{1}{t}.
$$

**Verification of Conditions**

**C1 and C2**: 

   - Conditions C1 and C2 are trivially satisfied with $$D_R = \mathbb{R}$$

   - For C1,$$K(x, \varphi, \rho, v) $$can be taken as 1.

**C3**:
   - In C3, it is found that $$ \varphi(t, x) = e(t) $$ and that $$ \beta(t, k) = \frac{1}{t} $$ for $$ \alpha(t) = \frac{1}{t} $$.

   - The condition for convergence then reads:
     $$
     \frac{1}{t} \sum_{k=1}^{t} [e(k) - x] \text{ should converge.}
     $$
     
     This is satisfied if the sequence $$ \{e(t)\} $$ is such that:
     $$
     \frac{1}{t} \sum_{k=1}^{t} e(k) \to m \text{ as } t \to \infty.
     $$
     
     Then $$f(x)=m-x$$.

**C4**:

- For C4, since $$ A(x) = 0 $$, the maximum eigenvalue $$ \lambda $$ is also 0, which implies $$ v(t, \lambda, c) = 0 $$. The condition simplifies to:
   $$
   \sum_{k=1}^{t} \beta(t, k) \to C \text{ as } t \to \infty,
   $$
   
   which is satisfied with $$ C = 1 $$.

**C5 and C6**:

   - These conditions are satisfied for $$ \gamma(t) = \frac{1}{t} $$.

**Differential Equation**

The associated differential equation (d.e.) for this algorithm is:


$$
\dot{x} = m - x,
$$


which is globally asymptotically stable with $$ x = m $$ as the stationary point.

We can take a Lyapunov function $$ V(x) = \frac{1}{2}(m - x)^2 $$, which leads to:

$$
  V'(x)f(x) = -(m - x)^2,
$$

which is negative for all $$ x $$. This satisfies the conditions of Theorem 4.1, indicating that $$ x(t) $$ will tend to $$ m $$ as $$ t $$ approaches infinity.
