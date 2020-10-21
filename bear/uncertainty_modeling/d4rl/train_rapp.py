import gym
import d4rl
import torch
from uncertainty_modeling.rapp.model import RaPP


# from bear directory, python -m uncertainty_modeling.d4rl.train_rapp
environment = 'hopper-medium-expert-v0'
exp_name = "seungjae"

env = gym.make(environment)
print(f"ENV : {environment}, OBS_SPACE : {env.observation_space}, ACT_SPACE : {env.action_space}")
dataset = env.get_dataset()

rapp = RaPP(env.observation_space.shape[0] + env.action_space.shape[0]).cuda()
optim = torch.optim.Adam(rapp.parameters())
criterion = torch.nn.MSELoss()

observations = torch.from_numpy(dataset['observations'])
actions = torch.from_numpy(dataset['actions'])

batch_size = 8192
epochs = 100

for epoch in range(epochs):
    total_loss = 0.0
    shuffler = torch.randperm(observations.size(0))
    observations = observations[shuffler]
    actions = actions[shuffler]
    for (obs, act) in zip(observations.split(batch_size, dim=0), actions.split(batch_size, dim=0)):
        data = torch.cat((obs, act), dim=1).cuda()
        data_out = rapp(data)
        
        loss = criterion(data, data_out)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
    
    print(f"[{epoch}/{epochs}] : {total_loss}")

torch.save(rapp, f"trained_{environment}_{exp_name}.pt")