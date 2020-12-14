import torch.nn as nn
import torch.functional as F
from torchvision import datasets, transforms
import os
import torch
import numpy as np

device_num = 2
device = torch.device(device_num if torch.cuda.is_available() else "cpu")

# change the path downloaded.
os.environ['TORCH_HOME'] = './models'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)

print("a")
