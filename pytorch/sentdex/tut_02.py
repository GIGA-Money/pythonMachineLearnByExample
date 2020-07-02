import torch, torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

# %%
train = datasets.MNIST("",
                       train=True,
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",
                      train=False,
                      download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))
# %%
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# %%
x, y = 0, 0
for data in trainset:
    # print(data)
    x, y = data[0][0], data[1][0]
    print(y)
    break
# %%
plt.imshow(data[0][0].view(28, 28))
plt.show()

# %%
total = 0
couter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

# %%
xs, ys = 0, 0
for data in trainset:
    xs, ys = data
    for y in ys:
        couter_dict[int(y)] += 1
        total += 1
# %%
print(couter_dict)
# %%
for i in couter_dict:
    print(f"{i}: {couter_dict[i] / total * 100}")
# %%
