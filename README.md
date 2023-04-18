# Reinforcement Learning

___

## Outline 
- [ 1 - Fundamentals ](#1)
  - [1.1 - Concepts and Components](#1.1) 
  - [1.2 - State-action value functions (Q-function)](#1.2) 
  - [1.3 - Continuous state spaces](#1.3) 
- [ 2 - Hands-on projects](#2)
  - [2.1 - Deep Q-Learning for Lunar Lander](#2.1)
  - [2.2 - Working in progress](#2.1)

<a name="1"></a>
## 1 - Fundamentals <img align="left" src="https://static.javatpoint.com/tutorial/reinforcement-learning/images/reinforcement-learning-markov-decision-process.png" style=" width:50px;">

To have an overview about *Reinforcement Learning*, I suggest you learning [*week 3 of "Unsupervised Learning, Recommenders, Reinforcement Learning" part*](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/home/week/3) in Machine Learing Specialization course, offered by Deeplearning.AI. Some images of this repository are captured in this course. 

<a name="1.1"></a>
#### 1.1 - **Concepts and Components**

> The key distinction between **reinforcement learning (RL)** and **standard deep learning (DL)** is that in standard deep learning the prediction of a trained model on one test datum *<u>does not affect the predictions on a future test datum</u>*; in reinforcement learning decisions at future instants (in RL, decisions are also called actions) are *<u>affected by what decisions were made in the past</u>*. [[Chaper 17. Reinforcement Learning - Dive into Deep Learing book]](https://d2l.ai/chapter_reinforcement-learning/index.html)

\- First, we figure out how many key components and their roles in RL:

+ *`Agent`*

+ *`Environment`*: represents a problem or a task to be solved. 

+ *`State`*

    + *Notation*: a set of states ($S$), a current state ($s$) and a new state ($s'$)

    + *Terminal state* nothing more happens

+ *`Action`*: at each state $s$, the agent can take an action $a$ in the set of actions $A$.

    + *Notation*: an action ($a$)

+ *`Reward`*:  the *agent* will get a reward $r(s)$ when it takes an action $a$ at state $s$ 

    + *Notation*: a reward of the action to another state ($r(s)$)

+ *`The return`*: is the estimate of the total long-term reward of a trajectory. On the other hand, it is sum of rewards the system got, and weighted by the *discount* factor.

    + *Notation*: a return $G_t$ and a discount factor ($\gamma$) that is used in the situation where the sequence of states and actions in a trajectory can be *infinitely long* and *the return of any such infinitely long trajectory will be infinite*.

    + *Discount* value: is a number close to 1.

    + *Formula*: given a state $S$ at the time $t$
    
    $$ G_t = R(\tau) = r_t + r_{(t+1)} \times \gamma + r_{(t+2)} \times \gamma ^ 2 + r_{(t+3)} \times \gamma ^ 3 + ...  $$

+ *`Policy`*: is a function mapping states to actions.

    + *Notation*: a function $\pi(s) = a$ tells us what action $a$ to take in a given state $s$.

    + ***The goal of Reinforcement learning*** is to find $\pi$ such that $a = \pi(s)$ and we will know what action $a$ needed to take in every state $s$ to maximize the *return*.

+ *`Markov Decision Process (MDP)`*: is a model for how the state of a system evolves as different actions are applied to the system. $$MDP: (S, A, r)$$

    + Let’s now consider the situation when the agent starts at a particular state and continues taking actions to result in a **trajectory** $\tau$: $$ \tau = (s_0, a_0, r_0, s_1, a_1, r_1,...)$$

![Overview](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2020/08/Reinforcement-Learning-in-ML-TV.jpg)

\- **Workdflow:** In the standard “agent-environment loop” formalism, an agent interacts with the environment in discrete time steps $t = 0,1,2,3,...$. At each time step $t$, the agent use a policy $\pi$ to select an action $a$ based on its observation of the environment's state $s_t$. The agent receives a numerical reward $r_t$ and on the next time step, moves to a new state $s_{t+1}$. 

\- **Applications:**

+ Controlling robots

+ Factory optimization

<a name="1.2"></a>
#### 1.2 - **State-action value function (Q-function)**

\- **Definition:**

+ The goal of $Q-function$ is to find the optimal policy.

+ *Notation*:  $Q(s, a)$

+ We calculate the $Q(s, a)$ like calculating the return $G_t$ with discount factor $\gamma$

![](./img/Q-function.png)


\- **Bellman equation:** sequence of rewards after you take an action $a$ at the state $s$.

![](./img/Bellman-equation.jpg)


\- **Random (stochastic) environment:** the sequence of different rewards and next state $s'$ is uncertain because we have a probability of going in the wrong direction which do not comply with the policy. $$ExpectedReturn = Average(r_1 + \tau \times r_2 + \tau ^ 2 \times r_3 + ...) \\ = E[r_1 + \tau \times r_2 + \tau ^ 2 \times r_3 + ...]$$

+ The goal of RL in this case is to find a policy to maximize the average value of the return. 

<a name="1.3"></a>
#### 1.3 - **Continuous state spaces**

\- Using *Deep RL* with a training dataset created by *Bellman equation*: 

+ Input: a pair of ($s, a$) - X

+ Output: $Q(s, a)$  - y

+ **Deep Q-Network (DQN)** algorithm: 

  + is used to approximate the action-value function $Q(s,a)\approx Q^*(s,a)$ or minimize the *mean-squared error* between them in the case of *<u>the state space is continuous</u>* (i.e., we cannot explore the entire state-action space and it is impossible to gradually update $Q(s,a)$ to $Q^*(s,a)$). 

  + uses a neural network to train a model to predict Q functions where guessing of $Q(s,a)$ constructed using Bellman equation as follows

$$
Q(s,a) = R + \gamma \max_{a'}Q(s',a')
$$



![](./img/Deep-Q-Network.png)

\- **$\epsilon$-greedy policy**: is to choose best actions while still learning. For instance, in some state $s$

+ *Option 1*: we pick the action $a$ that maximizes $Q(s, a)$. 

+ *Option 2*: called **$\epsilon$-greedy policy**

  + With probability of $1 - \epsilon$, we pick the action $a$ that maximizes $Q(s, a)$, named ***Greedy*** or ***Expoitation***. 

  + With probability of $\epsilon$, we pick the action $a$ <u>randomly</u> that maximizes $Q(s, a)$, named ***Exploration***. 

+ **NOTE**: at the beginning, we start with $\epsilon$ high and gradually decrease it. 

\- **Algorithm refinements**: *mini-batch* and *soft updates*

+ *Mini-batch learning or gradient descend*: if we have a large dataset, we need to divide our dataset into a number of smaller batches where parameters are updated each batch in stead of the entire dataset (*Batch learning*). 

+ *Soft updates*: make changes from the new value $Q$ to the old value $Q$ not too oscillating and unstable in the case of $Q$ new is worst than $Q$ old. For example, with $ \theta \ll 1 $ $$W_{current} = \theta W_{new} + (1 - \theta)W_{current} \\ b_{current} = \theta b_{new} + (1 - \theta) b_{current}$$


<a name="2"></a>
## 2 - Hands-on projects <img align="left" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQFrV7YddcdauSh0r01W58FYrho5pHZl93tA&usqp=CAU" style=" width:25px;">

<a name="2.1"></a>
#### [2.1 - Deep Q-Learning for Lunar Lander created by DeepLearning.AI](./src/Deep-Q-Network_Lunar_Lander.ipynb)

\- In this work, you should do the lab in the environment offered by Machine Learning Specialization course, Coursera. 

\- **Highlight:** in this lab we will learn how to implement *Deep Q-Learning* algorithm with two techniques, called ***target network*** and ***experience replay*** to avoid instabilities when using neural networks in reinforcement learning to estimate action-value functions or $Q(s, a)$.

![](./img/deep_q_algorithm_with_experience_replay.png)
