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
\begin{align}
x(t) = x (t-1) + \alpha Q(t, x(t-1),\phi(t) )\\
\phi(t) = A( x (t-1)) \phi (t-1) + B( x (t-1)) e(t)
\label{eq:algo}
\end{align}
$$

with assumptions

**C1:** The function $$ Q(t, x, \varphi) $$ is Lipschitz continuous in $$ x $$ and $$ \varphi $$ in any neighborhood of $$ (\bar{x}, \bar{\varphi}) $$, where $$ \bar{x} \in D_R $$ and $$ \bar{\varphi} $$ is arbitrary:
$$
\begin{align}
|Q(t, x_1, \varphi_1) - Q(t, x_2, \varphi_2)| \leq K(\bar{x}, \bar{\varphi}, \rho, v)\left[|x_1 - x_2| + |\varphi_1 - \varphi_2|\right]
\label{eq:recursive_proof}
\end{align}
$$

for $$ |x_i - \bar{x}| \leq \rho,  |\varphi_i - \bar{\varphi}| \leq v $$,
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

**Theorem 4.1** Consider the algorithm  $$\eqref{eq:recursive_proof}$$ subject to the conditions C1–C6. Let $$\bar{D}$$ be a closed subset of $$D_R$$. Assume that there is a constant $$C < \infty$$ and a subsequence $$\{t_k\}$$ (that may depend on the **particular** sequence $$\{e(r)\}$$) such that
$$
\begin{align}
x(t_k) \in \bar{D} \quad \text{and} \quad |\varphi(t_k)| < C.
\label{eq:x_close_to_DR}
\end{align}
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
\label{eq:de}
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



### Exist issues

- Regulation C1-C4 only holds for $$x\in D_R$$, Outside $$D_R$$ the d.e. is not defined and thus we do not know the behavior of the algorithm.  Thus we must  assume by $$\eqref{eq:x_close_to_DR}$$ that the estimates are inside $$\bar{D} \in D_R$$ infinitely often, so that they will eventually be captured by a trajectory.

- Lyapunov Function assures the stability of differential equation $$V$$ but it does not necessarily mean that every possible condition will lead to convergence to the equilibrium point. So $$D_R$$ is not necessarily the domain of attraction. 
- Once out of $$D_R$$, there is no control over the estimate sequence  except from $$\eqref {eq:x_close_to_DR}$$  we know that it might go back to $$D_R$$, but we do not know when and how often this could happen.

### Example: Nonlinear System with Multiple Equilibria

Consider a simple nonlinear system with the following dynamics:
$$
\dot{x} = -x(x - a)(x - b)
$$
where $$ a $$ and $$ b $$ are constants, with $$ 0 < a < b $$. This system has three equilibrium points:
- $$ x = 0 $$ (unstable equilibrium),
- $$ x = a $$ (stable equilibrium),
- $$ x = b $$ (unstable equilibrium).

Suppose the set $$ D_R $$ corresponds to the region where the system exhibits regular, predictable behavior, and it includes the stable equilibrium point $$ x = a $$. The boundary of $$ D_R $$ could correspond to the unstable equilibrium point $$ x = b $$. If the estimates $$ x(t) $$ are influenced by noise, errors, or model inaccuracies, they might be pushed out of the region of attraction for the stable point $$ x = a $$. For example:
- If $$ x(t) $$ is near $$ b $$ and experiences a perturbation that pushes it beyond $$ b $$, it might leave $$ D_R $$ (just as it might leave the region between $$ a $$ and $$ b $$ in the example).
- Once outside $$ D_R $$, the system could move towards an area where the stability conditions no longer apply (similar to how $$ x(t) $$ might move towards $$ x = 0 $$ in the example, where different dynamics govern the behavior).

## Projection Algorithms

To force the estimates to remain inside a compact subset $$\bar{D}$$,
$$
\bar{x}(t) = x(t - 1) + \gamma(t) Q(t, x(t - 1), \phi(t)), \\
x(t) = \begin{cases}
    \bar{x}(t) & \text{if } \bar{x}(t) \in \bar{D}, \\
    x(t-1) & \text{if } \bar{x}(t) \notin \bar{D}.
    \end{cases}
$$
**Corollary to Theorem 4.1:**

 It states that under the same assumptions as in Theorem 4.1, one of the following must hold:
  1. $$x(t) \rightarrow D_c$$ as $$t \rightarrow \infty$$, where $$D_c$$ is a stable set within $$\bar{D}$$.
  2. $$x(t) \rightarrow \delta \bar{D}$$ as $$t \rightarrow \infty$$, where $$\delta \bar{D}$$ is the boundary of $$\bar{D}$$.

Second situation may only hold if there is a trajectory of the differential equation that leaves $$\bar{D}$$. 

### Results

**Result 4.1 (Theorem 2 in Ljung, 1977b)**

Suppose that the sequence $$ x(t) $$, given by the recursive algorithm $$\eqref{eq:algo}$$ converges to some value $$ x^* $$ with a probability greater than zero. Then, two conditions must hold:

    1. **$$ f(x^*) = 0 $$**: This condition ensures that $$ x^* $$ is a fixed point of the function $$ f(x) $$ defined in condition C3.
        2. **Eigenvalues of $$ H(x^*) $$**: $$ H(x^*) $$ is defined as the derivative (Jacobian matrix) of $$ f(x) $$ with respect to $$ x $$ evaluated at $$ x^* $$. The condition requires that all the eigenvalues of $$ H(x^*) $$ must lie in the left half of the complex plane ensuring the stability of the equilibrium point $$ x^* $$.

**Result 4.2 (Theorem 3 in Ljung, 1977b)**

The upper bound on the probability that $$ x(t) $$ does not remain in an $$ \epsilon $$-neighborhood of the corresponding trajectory of $$\eqref{eq:de}$$ over the interval from $$ t = t_0 $$ to $$ t = N $$ is 
$$
\frac{K}{\epsilon^{4p}} \sum_{j=t_0}^{N} \alpha(j)^p,
$$
where $$ K $$ is a constant, $$ \alpha(j) $$ is the learning rate (or step size), and $$ p $$ is a positive exponent.
