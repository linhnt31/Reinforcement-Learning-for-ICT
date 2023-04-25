import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Resize frames we grab from gym and convert to tensor
# resizer = T.Compose([
# 	T.ToPILImage(),
# 	T.Resize(40, interpolation=Image.CUBIC),
# 	T.ToTensor()
# ])

# Start cartpole application through gym
# Reference: https://www.gymlibrary.dev/environments/classic_control/cart_pole/ [Online, Accessed on: Apr. 19, 2023]
def init():
	"""
	Because you want to access hidden attributes of a specific environment (i.e., CartPole), then we will use the unwrapped property.
    We can view attributes in the Cartpole env through: 
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    
    Returns:
		The environment object
    """
	return gym.make('CartPole-v1').unwrapped


# env = init()
# state = env.reset()
# print(len(state))
# print(env.action_space.n)
# state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
# print(state)
# print(env.observation_space)