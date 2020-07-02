# %%
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time
import copy
import syft as sy
from torch.utils.data import TensorDataset, DataLoader
from syft.generic import utils
from syft.workers.websocket_client import WebsocketClientWorker


# %%
class Parser:
    def __init__(self):
        self.epochs = 100
        self.lr = 0.001
        self.test_batch_size = 8
        self.batch_size = 8
        self.log_interval = 10
        self.seed = 1


# %%
args = Parser()
print(args)
torch.manual_seed(args.seed)
# %%
with open("D:/projects/pythonMachineLearnExamples/pytorch/towarddatascience/data/boston_housing.pickle", "rb") as f:
    ((x, y), (x_test, y_test)) = pickle.load(f)
# %%
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()
# %%
mean = x.mean(0, keepdim=True)
dev = x.std(0, keepdim=True)
mean[:, 3] = 0
dev[:, 3] = 1
x = (x - mean) / dev
x_test = (x_test - mean) / dev
train = TensorDataset(x, y)
test = TensorDataset(x_test, y_test)
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=True)


# %%
class Net(nn.Module):
    def __int__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.veiw(-1, 13)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc3(x)
        return x


# %%
remote_dataset = (list(), list())
train_distributed_dataset = []
for batch_index, (data, target) in enumerate(train_loader):
    data = data.send(compute_nodes[batch_index % len(compute_nodes)])
    target = target.send(compute_nodes[batch_index % len(compute_nodes)])
    remote_dataset[batch_index % len(compute_nodes)].append((data, target))

# %%
bobs_model = Net()
alices_model = Net()
bobs_optimizer = optim.SGD(bobs_model.parameters(), lr=args.lr)
alices_optimzer = optim.SGD(bobs_model.parameters(), lr=args.lr)
# %%
models = [bobs_model, alices_model]
optimizers = [bobs_optimizer, alices_model]
# %%
model = Net()
# %%
