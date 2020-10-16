import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import RaPP
from calc_uncertainty import get_diffs
import random
import math
from gen_data import gen_datagrid, plot_meshgrid


start, end, step = -1, 1, 101

def dummy(a):
    return int(-0.1 * (a-10)**2 + 3 * a +10)
train_data, index, meshgrid_data_lin = gen_datagrid(start=start, end=end, step=step,
                                                    plot=False,
                                                    xrange=list(range(10, 71, 3)),
                                                    type='linear')
                                                    # function=dummy)
                                                    # type='random')



### Model and optimizer initialization
model = RaPP()  # It is just Auto-Encoder
optim = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

### Training
for i in range(1500):  # change 1000 to 0 if you want to see not-trained version.
    td = train_data[torch.randperm(train_data.size(0))]
    for batch in td.split(8, dim=0):
        # print(batch)
        out = model(batch)  # train_data
        loss = criterion(out, batch)  # train_data

        optim.zero_grad()
        loss.backward()
        optim.step()
    if i % 500 == 0:
        print(i, loss.item())


with torch.no_grad():
    # Calculate Uncertainty
    dif = get_diffs(meshgrid_data_lin, model)
    difs = torch.cat([torch.from_numpy(i) for i in dif], dim=-1).numpy()
    
    dif = (difs**2).mean(axis=1)
    dif = dif.reshape(step, step)

    fig, (a2) = plt.subplots(1, 1)
    im2 = a2.imshow(dif, cmap="gray", extent=[start, end, start, end], origin="lower")
    a2.set_title("dif")
    plot_meshgrid(a2, meshgrid_data_lin, index)
    
    plt.show()