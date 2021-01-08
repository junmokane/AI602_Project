import os
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from rlkit.core import logger


def simulate_policy(args):
    filename = args.file
    data = torch.load(filename)
    filename_token = filename.split('/')
    save_path = os.path.join(os.path.join(*filename_token[:-1]),
                             'offline_buffer_' + filename_token[-1].split('.')[0] + '.hdf5')
    print(save_path)
    print(data)
    '''
    I don't know why but they did not save the policy for evalutaion.
    Instead of that, I used trainer/policy
    '''
    policy = data['trainer/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    n = 0
    traj_list = []
    while n < args.buffer_size:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
        n = n + len(path['rewards'])
        print('Saving %d sequences' % n)
        traj_list.append(path)
        # Visualize one trajectory
        if args.visualize:
            states = path['observations']
            states = np.concatenate([states, path['next_observations'][-1:, :]], axis=0)
            gr = 0.1  # goal radius, for visualization purposes
            g = np.array([1.0, 1.0])
            plt.figure(figsize=(8, 8))
            axes = plt.axes()
            axes.set(aspect='equal')
            plt.axis([-0.25, 1.25, -0.25, 1.25])
            circle = plt.Circle((g[0], g[1]), radius=gr)
            axes.add_artist(circle)
            plt.plot(states[:-1, 0], states[:-1, 1], '-o')
            plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=20)
            plt.show()

    '''
    # Save trajectory
    observations = []
    actions = []
    rewards = []
    terminals = []
    f = h5py.File(save_path, 'w')
    for traj in traj_list:
        observations.append(traj['observations'])
        actions.append(traj['actions'])
        rewards.append(traj['rewards'])
        terminals.append(traj['terminals'])
    f['observations'] = np.concatenate(observations, axis=0)
    f['actions'] = np.concatenate(actions, axis=0)
    f['rewards'] = np.concatenate(rewards, axis=0)
    f['terminals'] = np.concatenate(terminals, axis=0)
    f.close()
    '''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--buffer_size', type=int, default=16000,
                        help='The size of saving buffer')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='Visualize the trajectory of point robot')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()
    simulate_policy(args)
