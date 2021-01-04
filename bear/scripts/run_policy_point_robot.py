import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    print(data)
    # policy = data['evaluation/policy']
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
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()
    print(args)
    simulate_policy(args)
