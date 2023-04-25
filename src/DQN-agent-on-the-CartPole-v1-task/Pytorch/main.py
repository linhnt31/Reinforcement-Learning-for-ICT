import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import model, agent, environment, utils
from collections import deque, namedtuple
from settings import *
from itertools import count

# enviroment setup
env = environment.init()
# Initialize an agent
cart = agent.Agent(len(env.reset()), env.action_space.n, DEVICE, MEMORY, LEARNING_RATE)

for i_episode in range(NUM_EPISODE):
    done = False
    env.reset()
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    for t in count():
        action = cart.select_action(env, state)
        observation, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward], device=DEVICE)

        if i_episode % 20:
            print([env.step(action.item())])

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        # Store the transition in memory
        cart.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        cart.optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = cart.target_net.state_dict()
        policy_net_state_dict = cart.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            cart.target_net.load_state_dict(target_net_state_dict)

        
        if done:
            utils.episode_durations.append(t+1)
            utils.plot_durations(utils.episode_durations)
            break

print('Complete')
utils.plot_durations(show_result=True)
plt.ioff()
plt.show()
env.close()