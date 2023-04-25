import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
from collections import deque, namedtuple
from settings import *

"""
*namedtuple* supports both access from key, value (or index) and iteration, the functionality that dictionaries lack.
*deque* is preferred over a *list* in the cases where we need quicker *append* and *pop* operations from both the ends of the container. Time complexity: O(1) for both operations
"""
# Memory representation of transition. It maps (state, action) pairs to their (next_state, reward) result
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Memory representation for our agent for training
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    # Save a transition
    def push(self, *arg):
        self.memory.append(Transition(*arg))
    
    # Select a random batch of saved trasitions in the memory deque
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # Return the length of memory
    def memory_len(self):
        return len(self.memory)
    
class Agent():
    def __init__(self, n_observations, n_actions, device, memory, lr) -> None:
        self.device = device
        self.policy_net = model.DQN(n_observations, n_actions).to(device)
        self.target_net = model.DQN(n_observations, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory)
        self.steps_done = 0

    def remember(self, *arg):
        self.memory.push(*arg)

    def select_action(self, env, state):
        """
        Choose an action based on an epsilon greedy policy approach

        For example: Pseudo code of epsilon greedy policy
            p = random()
            if p < epsilon:
                pull random action in the action space
            else:
                pull current best action (output of the neural network)

        Arguments:
            env: our environment
            state: the current state or the input of deep neural network
        Returns:
            tensor([[int]]): the index of the chosen action
        """
        sample = random.random() #[0, 1)
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if sample < epsilon_threshold:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
            
    def optimize_model(self):
        """
        Training out model
        """
        if self.memory.memory_len() < BATCH_SIZE:
            return
        
        # Sample our transition memory
        transitions = self.memory.sample(BATCH_SIZE)

        # Convert batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        """ torch.cat(tensors, dim=0, *, out=None) → Tensor
        Concatenates the given sequence of seq tensors in the given dimension. 
        All tensors must either have the same shape (except in the concatenating dimension) or be empty.
            >>> x = torch.randn(2, 3)
            >>> x
            tensor([[ 0.6580, -1.0969, -0.4614],
                    [-0.1034, -0.5790,  0.1497]])
            >>> torch.cat((x, x, x), 0)
            tensor([[ 0.6580, -1.0969, -0.4614],
                    [-0.1034, -0.5790,  0.1497],
                    [ 0.6580, -1.0969, -0.4614],
                    [-0.1034, -0.5790,  0.1497],
                    [ 0.6580, -1.0969, -0.4614],
                    [-0.1034, -0.5790,  0.1497]])
            - Ref: https://pytorch.org/docs/stable/generated/torch.cat.html
        """
        # Compute a mask of non-final states and concatenate the batch elements
        # Next states which are not None are mased with True
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        """ torch.gather(input, dim=int, index, *, sparse_grad=False, out=None) → Tensor
        Gathers values along an axis specified by dim.


        - Ref: https://machinelearningknowledge.ai/how-to-use-torch-gather-function-in-pytorch-with-examples/
        """
        # The model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all *next* states
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # COmpute the expected (predicted) Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss between our state-action values and expectations
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print("Loss: {}", loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        #
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
