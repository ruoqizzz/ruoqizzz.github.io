---
layout: post
title: Lyapunov Function and Theory
date: 2024-09-02 11:49:00
description: 
tags: recursive_sysid, linear_model
categories: control

---

To understand how to derive the condition $$V'(x)f(x) \leq 0$$, let's break it down step by step within the context of Lyapunov stability theory, which is commonly used to analyze the stability of dynamic systems.

#### 1. **Lyapunov Function $$V(x) $$:**
   - The function$$V(x)$$ is chosen or constructed to be a measure of the "energy" or "potential" of the system. For stability analysis, we typically require that:
     -$$V(x) > 0$$ for all$$x \neq 0$$ (positive definiteness).
     -$$V(x) = 0$$ at the equilibrium point$$x = 0$$.

#### 2. **System Dynamics $$\dot{x} = f(x)$$:**
   - Consider a dynamic system described by the differential equation$$\dot{x} = f(x)$$, where$$\dot{x}$$ is the time derivative of the state$$x$$, and$$f(x)$$ describes the system's dynamics.

#### 3. **Time Derivative of the Lyapunov Function:**
   - To analyze the stability of the system, we examine how the Lyapunov function$$V(x)$$ changes over time as the system evolves.
   - The rate of change of$$V(x)$$ with respect to time is given by the total derivative:
     $$
     \frac{dV(x)}{dt} = \frac{\partial V(x)}{\partial x} \cdot \frac{dx}{dt} = V'(x) \cdot \dot{x},
     $$
     where$$V'(x) = \frac{\partial V(x)}{\partial x}$$ is the gradient of$$V(x)$$ with respect to$$x$$, and$$\dot{x} = f(x)$$.

   - Substituting$$\dot{x} = f(x)$$ into this equation, we get:
     $$
     \frac{dV(x)}{dt} = V'(x) \cdot f(x).
     $$

#### 4. **Stability Condition:**
   - For the system to be stable, we require that the Lyapunov function$$V(x)$$ does not increase over time, meaning that$$\frac{dV(x)}{dt} \leq 0$$.
   - This condition implies:
     $$
     V'(x) \cdot f(x) \leq 0.
     $$
   - If$$V'(x)f(x) \leq 0$$, then$$V(x)$$ is non-increasing along the trajectories of the system. This non-increasing behavior suggests that the system's "energy" is either decreasing or staying the same, which corresponds to the system being stable or moving toward a stable state.

### 5. **Derivation Summary:**
   - The condition$$V'(x)f(x) \leq 0$$ is derived by considering the time derivative of the Lyapunov function$$V(x)$$.
   - If$$V'(x)f(x) \leq 0$$, it guarantees that$$V(x)$$ does not increase over time, meaning the system (or algorithm) is stable, as it tends to reduce or maintain its "energy" over time.

#### Example (Illustration):

Let's consider a simple system:

$$
\dot{x} = -x,
$$
and define the Lyapunov function$$V(x) = \frac{1}{2}x^2$$.

- The gradient$$V'(x) = x$$.
- The time derivative of$$V(x)$$ along the trajectory of the system is:
  $$
  \frac{dV(x)}{dt} = V'(x) \cdot \dot{x} = x \cdot (-x) = -x^2 \leq 0.
  $$
  

Here,$$V'(x)f(x) \leq 0$$ because$$-x^2$$ is always non-positive. This means$$V(x)$$ decreases over time, which indicates that the system is stable.

#### Conclusion:

The condition$$V'(x)f(x) \leq 0$$ is derived as a requirement for the Lyapunov function to be non-increasing, which in turn ensures the stability of the system. It indicates that the system is not gaining energy and is either stable or moving towards stability.