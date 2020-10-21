import gym
import d4rl
import torch
from uncertainty_modeling.rapp.model import RaPP
import argparse


def train_rapp(args):
    # from bear directory, python -m uncertainty_modeling.d4rl.train_rapp
    environment = args.e
    exp_name = args.x
    dataset_path = args.d

    env = gym.make(environment)
    print(f"ENV : {environment}, OBS_SPACE : {env.observation_space}, ACT_SPACE : {env.action_space}")
    
    if dataset_path is None:
        dataset = env.get_dataset()
        observations = torch.from_numpy(dataset['observations'])
        actions = torch.from_numpy(dataset['actions'])
    else:
        dataset = torch.load(dataset_path)
        observations = torch.from_numpy(dataset[0])
        actions = torch.from_numpy(dataset[1])
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

    # print(env.observation_space.shape, env.action_space.shape)
    # rapp = RaPP(observations.size(1) + actions.size(1)).cuda()
    rapp = RaPP(2).cuda()
    optim = torch.optim.Adam(rapp.parameters())
    criterion = torch.nn.MSELoss()

    batch_size = 4096
    epochs = 2000

    for epoch in range(epochs):
        total_loss = 0.0
        shuffler = torch.randperm(observations.size(0))
        observations = observations[shuffler]
        actions = actions[shuffler]
        for (obs, act) in zip(observations.split(batch_size, dim=0), actions.split(batch_size, dim=0)):
            data = torch.cat((obs, act.float()), dim=1).cuda()
            data = data[:, :2]
            data_out = rapp(data)

            loss = criterion(data, data_out)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
        
        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"[{epoch}/{epochs}] : {total_loss}")

    torch.save(rapp, f"trained_{environment}_{exp_name}.pt")

if __name__ == "__main__":
    # To run, refer README.md
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', type=str,
                        help='environment')
    parser.add_argument('--x', type=str,
                        help='experiment name')
    parser.add_argument('--d', type=str,
                        help='dataset path')
    args = parser.parse_args()
    train_rapp(args)
    # 'hopper-medium-expert-v0'
    # "seungjae"