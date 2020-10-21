import gym
import d4rl
import torch
from uncertainty_modeling.rapp.model import RaPP
from uncertainty_modeling.rapp.calc_uncertainty import get_diffs
import matplotlib.pyplot as plt
import random


# from bear directory, python -m uncertainty_modeling.d4rl.test_rapp
path = "/home/seungjae/Desktop/AI602/AI602_Project/bear/trained_hopper-medium-expert-v0_seungjae.pt"
model = torch.load(path)

environment = 'hopper-medium-expert-v0'
exp_name = "seungjae"

env = gym.make(environment)
print(f"ENV : {environment}, OBS_SPACE : {env.observation_space}, ACT_SPACE : {env.action_space}")
dataset = env.get_dataset()

rapp = RaPP(env.observation_space.shape[0] + env.action_space.shape[0]).cuda()

observations = torch.from_numpy(dataset['observations'])
actions = torch.from_numpy(dataset['actions'])


fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(8, 8))
for glo in range(10):
    loc = random.randint(0, observations.size(0)-1)
    test_data = torch.cat((observations[loc:loc+1], actions[loc:loc+1]), dim=1).cuda()
    for i in range(2 * len(axs)):
        ax = axs[i%7, i//7]
        gap = torch.linspace(-5*test_data[:, i].mean(), 5*test_data[:, i].mean(), steps=11).unsqueeze(1).cuda()
        plt_data = test_data.repeat(11, 1)
        plt_data[:, i] += gap[:, 0]

        dif = get_diffs(plt_data, rapp)
        difs = torch.cat([torch.from_numpy(i) for i in dif], dim=-1).numpy()
        dif = (difs**2).mean(axis=1)
        dif -= dif[5]

        if glo == 0:
            title = f"{i}th dimension"
            ax.set_title(title)
            
            if i == 0:
                print(dif)
        ax.plot(dif, alpha=0.5, color='b')


plt.tight_layout()
plt.show()