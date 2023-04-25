import torch, random

# a = (True, True, True, False, True)
# print(type(a))

# print(torch.tensor(a))

t = torch.tensor([[1, 2], [3, 4]])

print(t)
print(t.gather(1, torch.tensor([[0, 0], [1, 0]])))