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
print(len(training_data))
# %%
plt.style.use("dark_background")
plt.title("Training and Testing accuracy", color="yellow")
plt.imshow(training_data[0][0], cmap="gray")
plt.show()