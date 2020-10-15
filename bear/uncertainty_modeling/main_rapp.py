import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import RaPP
from calc_uncertainty import get_diffs
import random
import math


### Make meshgrid
end, d = 5, 101

x1 = torch.linspace(0, end, d)
y1 = torch.linspace(0, end, d)
xx, yy = np.meshgrid(np.arange(0, end, d), np.arange(0, end, d))
uncertainty_data = torch.from_numpy(np.dstack(np.meshgrid(x1, y1)))


### Make training data
z = random.randint(1, 10)
ind_x = list(range(z+30, z+60, 1))
# ind_y = [int((a-50)*(a-50)*0.2 + a + 1) for a in ind_x]
ind_y = [int(-1* a + 101) for a in ind_x]  # Change these things to make shape of training data
index = [a + b*d for (a, b) in zip(ind_x, ind_y)]
print(ind_x, ind_y, index)
# index =  torch.randperm(d*d)[:20] --> If you want to select random location for training data
# index =  torch.randperm(d*d)[:16]

uncertainty_data = uncertainty_data.reshape((d*d, 2))
train_data = uncertainty_data[index, :]

### Model and optimizer initialization
model = RaPP()
optim = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

### Training
for i in range(1500):  # change 1000 to 0 if you want to see not-trained version.
    td = train_data[torch.randperm(train_data.size(0))]
    for batch in td.split(4, dim=0):
        out = model(batch)  # train_data
        loss = criterion(out, batch)  # train_data

        optim.zero_grad()
        loss.backward()
        optim.step()
    if i % 500 == 0:
        print(i, loss.item())

### Calc uncertainty
'''
out = model(uncertainty_data)
dif = ((out - uncertainty_data)**2).mean(dim=1)
'''
with torch.no_grad():
    dif = get_diffs(uncertainty_data, model)
    difs = torch.cat([torch.from_numpy(i) for i in dif], dim=-1).numpy()
    
    dif = (difs**2).mean(axis=1)
    dif = dif.reshape(d, d)

    fig, (a2) = plt.subplots(1, 1)
    im2 = a2.imshow(dif, cmap="gray")
    a2.set_title("dif")

    for ind in index:
        a2.scatter(ind % d, ind // d, c="red")

    plt.show()