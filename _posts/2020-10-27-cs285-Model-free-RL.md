---
layout: post
title: Model-free Reinforcement Learning
date: 2020-10-27 11:04:00
categories: rl
---

Notes of course [CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) Lecture 05-08: 

- Policy Gradient
- Actor-Critic
- Value Function
- Deep RL with Q-functions



## Policy Gradient

The goal of RL: 


$$
\begin{align*}
\theta^* &= \arg \max_\theta E_{\tau\sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]\\
				 &= \arg \max_\theta J(\theta)
\end{align*}
$$


A natural thought is to use gradient of $$J(\theta)$$ to get the best policy.

### Direct policy differentiation

$$
\begin{align*}
J(\theta) &= E_{\tau\sim p_\theta(\tau)}[r(\tau)]  \\
					&= \int p_\theta(\tau)r(\tau)d\tau\\
\nabla_\theta J(\theta)&= \int \nabla_\theta p_\theta(\tau)r(\tau)d\tau \\
											&= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)r(\tau)d\tau \\
											&= E_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log p_\theta(\tau)r(\tau)d\tau] \\
			\nabla_\theta \log p_\theta(\tau) &= \nabla_\theta \big[\log p(s_1) + \sum_{t=1}^T \log \pi_\theta(a_t\vert s_t) + \log p(s_{t+1\vert s_t,a_t})\big]\\
								&= \nabla_\theta \big[\sum_{t=1}^T \log \pi_\theta(a_t\vert s_t)\big] \\
\text{Therefore, }\\
\nabla_\theta J(\theta)&=E_{\tau\sim p_\theta(\tau)}\Big[\Big (\sum_{t=1}^T \log \pi_\theta(a_t\vert s_t)\Big) +\Big(\sum_{t=1}^Tr(s_t,a_t) \Big) \Big] \\
						&\approx \frac{1}{N} \sum_{i=1}^N \Big(\sum_{t=1}^T\nabla_\theta \log\pi_\theta(a_{i,t}\vert s_{i,t}) \Big) \Big(\sum_{t=1}^T r(s_{i,t},a_{i,t})\Big)
\end{align*}
$$

### REINFORCE algorithm:

1. Sample {$$\tau^i$$} from $$\pi_\theta(a_t\vert s_t)$$ (run the policy)
2. Caculate $$\nabla_\theta J(\theta)$$
3. Update rule:$$\theta \leftarrow \theta + \alpha \nabla_\theta  J(\theta) $$
4. Back to step 1

### What is wrong with this:   HIGH VARIANCE!



### Reducing variance

-  **Causality**: policy at time $$t'$$ can not effect reward at time $$t$$ when $$t < t'$$

$$
\begin{align*}
\nabla_\theta J(\theta) &\approx  \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_{i,t}\vert s_{i,t}) \Big (\sum_{t'=t}^T r(s_{i,t'},a_{i,t'}) \Big)\\
										&= \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_{i,t}\vert s_{i,t}) \hat{Q}_{i,t}
\end{align*}
$$

-  **Baselines**

   $$
   \begin{align*}
   
   \nabla_\theta J(\theta) &\approx  \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau) r(\tau)\\
   					&=\frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau) [r(\tau) -b]\\
   				b&= \frac{1}{N} \sum_{i=1}^N r(\tau)
   \end{align*}
   $$
   
   Still **unbiased**! Although it is not the best baseline, but it's pretty good



### Off-policy Policy Gradients

What if we don't have samples from $$p_\theta(\tau)$$?

Idea: using **Importance sampling**
$$
\begin{align*}
J(\theta) &= E_{\tau\sim p_\theta(\tau)}[r(\tau)]  \\
					&= E_{\tau\sim \bar{p}(\tau)}\big[\frac{p_\theta(\tau)}{\bar{p}(\tau)}r(\tau)\big] \\
	\frac{p_\theta(\tau)}{\bar{p}(\tau)}&= \frac{p(s_1)\prod_{t=1}^T\pi_\theta(a_t\vert s_t)p(s_{t+1\vert s_t,a_t})}{p(s_1)\prod_{t=1}^T \bar{\pi}(a_t\vert s_t)p(s_{t+1\vert s_t,a_t})} \\
																	&=\frac{\prod_{t=1}^T\pi_\theta(a_t\vert s_t)}{\prod_{t=1}^T \bar{\pi}(a_t\vert s_t)} \\
\nabla_{\theta'} J(\theta')&=E_{\tau\sim p_\theta(\tau)} \Big[\frac{p_{\theta'}(\tau)}{p_{\theta}(\tau)} \nabla_{\theta'}\log \pi_{\theta'}(\tau)r(\tau) \Big]\\
&=E_{\tau\sim p_\theta(\tau)} \Big[
\Big(\prod_{t=1}^T\frac{\pi_{\theta'}(a_t\vert s_t)}{\pi_\theta(a_t\vert s_t)}\Big)
\Big(\sum_{t=1}^T \nabla_{\theta'}\log \pi _{\theta'}(a_t \vert s_t) \Big)
\Big(\sum_{t=1}^T r(s_t,a_t)\Big)
\Big]&\text{what about causality?}\\
&= E_{\tau\sim p_\theta(\tau)} \Big[
\sum_{t=1}^T \nabla_{\theta'}\log \pi _{\theta'}(a_t \vert s_t) 
\Big(\prod_{t'=1}^t\frac{\pi_{\theta'}(a_{t'}\vert s_{t'})}{ \pi_\theta(a_{t'}\vert s_{t'})}\Big)
\Big(\sum_{t'=t}^T r(s_{t'},a_{t'})
\Big(\prod_{t''=t}^{t'} \frac{\pi_{\theta'}(a_{t''}\vert s_{t''})}{\pi_\theta(a_{t''}\vert s_{t''})}\Big)
\Big)
\Big]
\end{align*}
$$

·

## Actor-Critic Algorithms

### Idea

Can we improve the policy gradient from
$$
\begin{align*}
\nabla_\theta J(\theta) &\approx\frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_{i,t}\vert s_{i,t}) \Big (\sum_{t'=t}^T r(s_{i,t'},a_{i,t'}) \Big)\\
				&= \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_{i,t}\vert s_{i,t}) \hat{Q}_{i,t}
\end{align*}
$$
Use **true** expected reward-to-go:
$$
Q(s_t,a_t) = \sum_{t'=t}^T E_{\pi_\theta}[r(s_{t'},a_{t'})\vert s_t,a_t]
$$
Also the **baseline**, use the value of state:
$$
V(s_t) = E_{a_t\sim \pi_\theta(a_t\vert s_t)}[Q(s_t,a_t)]
$$
Then the gradient can be rewritten as,
$$
\begin{align*}
\nabla_\theta J(\theta) &\approx \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_{i,t}\vert s_{i,t}) [Q(s_{i,t},a_{i,t}) - V(s_{i,t})] \\
&= \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta (a_{i,t}\vert s_{i,t}) A(s_{i,t},a_{i,t}) \\
\end{align*}
$$
The better we estimate the Advantage function $$A(s,a)$$, the lower the variance is.

Therefore, we introduce the critic to approximate the value functions.

### Policy evaluation (Monte Carlo with function approximation)

Use supervise learning with training data:
$$
\{\big(s_{i,t}, \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \big)\}
$$
Valuetion function with paramter $$\phi$$:
$$
\phi = \arg\min_\phi L(\phi) = \arg\min_\phi \frac{1}{2}\sum_i\Vert \hat{V}^\pi_{\phi}(s_i) - y_i \Vert^2
$$


Improve, use the privious fitted value function instead
$$
\{\big(s_{i,t}, r(s_{i,t}, a_{i,t})+\hat{V}_\phi^\pi(s_{i,t+1}) \big)\}
$$

### An actor-critic algorithm

1. Sample {$$s_i,a_i$$} from $$\pi_\theta(a_t\vert s_t)$$ (run the policy)
2. Fit $$\hat{V}_\phi^\pi$$ to sampled reward sums
3. Evaluate $$\hat{A}_\phi^\pi(s_i,a_i) = r(s_i,a_i) +\hat{V}_\phi^\pi(s_i') - \hat{V}_\phi^\pi(s_i)$$
4. Caculate $$\nabla_\theta J(\theta) \approx \sum_i \nabla_\theta \log \pi_\theta (a_{i}\vert s_{i}) A(s_{i},a_{i})$$
5. Update rule:$$\theta \leftarrow \theta + \alpha \nabla_\theta  J(\theta) $$
6. Back to step 1



### Discount factors

What if episode length is infinity?
$$
y_{i,t} \approx r(s_{i,t},a_{i,t}) + \gamma\hat{V}_\phi^\pi(s_{i,t})
$$
The gamma can be considered as, the agent has the probability $$1-\gamma$$ to die.

### An actor-critic algorithm with discount

**Batch actor-critic**:

1. Sample {$$s_i,a_i$$} from $$\pi_\theta(a_t\vert s_t)$$ (run the policy)
2. Fit $$\hat{V}_\phi^\pi$$ to sampled reward sums
3. Evaluate $$\hat{A}_\phi^\pi(s_i,a_i) = r(s_i,a_i) + \gamma \hat{V}_\phi^\pi(s_i') - \hat{V}_\phi^\pi(s_i)$$
4. Caculate $$\nabla_\theta J(\theta) \approx \sum_i \nabla_\theta \log \pi_\theta (a_{i}\vert s_{i}) A(s_{i},a_{i})$$
5. Update rule:$$\theta \leftarrow \theta + \alpha \nabla_\theta  J(\theta) $$
6. Back to step 1

**Online actor-critic:**

1. Take action $$a \sim \pi_\theta(a_t\vert s_t)$$, get $$(s,a,s',r')$$
2. Update $$\hat{V}_\phi^\pi$$ using target $$r + \gamma \hat{V}_\phi^\pi(s_i')$$
3. Evaluate $$\hat{A}_\phi^\pi(s_i,a_i) = r(s_i,a_i) + \gamma \hat{V}_\phi^\pi(s_i') - \hat{V}_\phi^\pi(s_i)$$
4. Caculate $$\nabla_\theta J(\theta) \approx \sum_i \nabla_\theta \log \pi_\theta (a_{i}\vert s_{i}) A(s_{i},a_{i})$$
5. Update rule:$$\theta \leftarrow \theta + \alpha \nabla_\theta  J(\theta) $$
6. Back to step 1

### Other imrpovement methods

- Q-prop
- Eligibility traces & n-step returns
- Generlized advantage estimation

## Value Function Methods

## Deep RL with Q-Functions

