import torch

a = torch.arange(4)
for k, v in enumerate(a[:-1]):
    print(v)