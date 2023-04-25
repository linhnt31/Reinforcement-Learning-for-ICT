import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCREEN_WIDTH = 600
TARGET_UPDATE = 10
NUM_EPISODE = 100
BATCH_SIZE = 128 # is the number of transitions sampled from the replay buffer
GAMMA = 0.999 # is the discount factor
LEARNING_RATE = 1e-3
EPSILON_START = 0.9 # is the starting value of epsilon
EPSILON_END = 0.05 # is the final value of epsilon
EPSILON_DECAY = 200 # controls the rate of exponential decay of epsilon, higher means a slower decay
MEMORY = 10000 # is the maximum number of element in the trasition memory
TAU = 0.005 # is the update rate of the target network