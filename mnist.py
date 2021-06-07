import torch as t
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

path = "./dataMNIST"
data_set = tv.datasets.MNIST(root = path,train = True,download = True)