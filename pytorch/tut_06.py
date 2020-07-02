# %%
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

import os
import cv2
import numpy as np
from tqdm import tqdm

# %%
REBUILD_DATA = True


# %%
class DvC():
    img_size = 50
    CatDir = "D:/School/Spring2020/themachine/programingAssignment3/data/training_set/cats"
    DogDir = "D:/School/Spring2020/themachine/programingAssignment3/data/training_set/dogs"
    labels = {CatDir: 0, DogDir: 1}
    training_data = []
    cat_count = 0
    dog_count = 1

    def make_training_data(self):
        for label in self.labels:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    self.training_data.append([np.array(img), np.eye(2)[self.labels[label]]])

                    if label == self.CatDir:
                        self.cat_count += 1
                    elif label == self.DogDir:
                        self.dog_count += 1

                except Exception as e:
                    pass
                    # print(str(e))
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("cats: ", self.cat_count, "dogs: ", self.dog_count)


# %%
if REBUILD_DATA:
    dvc = DvC()
    dvc.make_training_data()
# %%
training_data = np.load("training_data.npy", allow_pickle=True)


# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(torch.tanh(self.conv1(x)), (2, 2))
        x = F.max_pool2d(torch.tanh(self.conv2(x)), (2, 2))
        x = F.max_pool2d(torch.tanh(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = torch.tanh((self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# %%
net = Net()
print(net)
# %%
optimizer = optim.SGD(net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
# %%
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X / 255.
y = torch.Tensor([i[1] for i in training_data])
# %%
val_per = 0.1
val_size = int(len(X) * val_per)
# %%
train_x = X[:-val_size]
train_y = y[:-val_size]
test_x = X[-val_size:]
test_y = y[-val_size:]

print(len(train_x))
print(len(test_x))
# %%
batch_size = 100
epochs = 5
# %%
for eps in range(epochs):
    for i in tqdm(range(0, len(train_x), batch_size)):
        # print(i, i + batch_size)
        batch_x = train_x[i:i + batch_size].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + batch_size]

        net.zero_grad()
        outputs = net(batch_x)
        loss = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(loss)

# %%
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_x))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_x[i].view(-1, 2, 50, 50))[0]
        prediction_class = torch.argmax(net_out)
        if prediction_class == real_class:
            correct += 1
        total += 1
# %%
print("accuracy: ", round(correct / total, 3))
