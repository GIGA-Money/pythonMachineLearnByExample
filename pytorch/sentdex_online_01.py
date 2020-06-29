import torch

# %%
x = torch.tensor([84, 39])
y = torch.tensor([10, 55])

# %%
print(x * y)

# %%
x = torch.zeros([2, 3])

print(x)

# %%
x.shape
# %%
y = torch.rand((2, 5))

# %%
y
# %%
# view is equivalent to reshape in tensorflow/numpy
y.view([1, 10])

# %%
y = y.view([1, 10])

# %%

