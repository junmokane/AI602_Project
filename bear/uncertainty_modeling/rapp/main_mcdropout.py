import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import Dummy
from calc_uncertainty import uncertainty
import random
import math


### Set random seed or not
'''
seed = 1001
torch.manual_seed(seed)
random.seed(seed)
'''

### Make meshgrid
end, d = 5, 101

x1 = torch.linspace(0, end, d)
y1 = torch.linspace(0, end, d)
xx, yy = np.meshgrid(np.arange(0, end, d), np.arange(0, end, d))
uncertainty_data = torch.from_numpy(np.dstack(np.meshgrid(x1, y1)))


### Make training data
ind_x = list(range(70, 75, 1))
ind_y = [a*2 for a in ind_x]
index = [a*d + b for (a, b) in zip(ind_x, ind_y)]
# index =  torch.randperm(d*d)[:20] --> If you want to select random location for training data

uncertainty_data = uncertainty_data.reshape((d*d, 2))
train_data = uncertainty_data[index, :]

label = torch.ones(train_data.size(0))

### Model and optimizer initialization
model = Dummy()
optim = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

### Training
for i in range(1000):  # change 1000 to 0 if you want to see not-trained version.
    out = model(train_data)
    loss = criterion(out, label)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 500 == 0:
        print(i, loss.item())

### Calc uncertainty
mean, var = uncertainty(uncertainty_data, model, k=500)
mean = mean.reshape((d, d)).detach().numpy()
var = var.reshape((d, d)).detach().numpy()

fig, (a2) = plt.subplots(1, 1)
im2 = a2.imshow(var, cmap="gray")
a2.set_title("var")

for ind in index:
    a2.scatter(ind % d, ind // d, c="red")

plt.show()
