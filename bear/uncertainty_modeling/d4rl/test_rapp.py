import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from uncertainty_modeling.rapp.calc_uncertainty import get_diffs


def test_rapp_lunarlander(args):
    fig =  plt.figure(figsize=(10, 7))
    for i in range(4):
        path = f"{args.p}_{i}.pt"
        model = torch.load(path)

        extent = [-0.4, 0.4, 0.0, 1.5]

        x = np.linspace(-0.4, 0.4, 101)
        y = np.linspace(0.0, 1.5, 101)
        xv, yv = np.meshgrid(x, y)
        meshgrid_data = torch.from_numpy(np.dstack([xv, yv]))

        meshgrid_data_lin = meshgrid_data.reshape((101*101, 2)).cuda()

        dif = get_diffs(meshgrid_data_lin, model)
        difs = torch.cat([torch.from_numpy(i) for i in dif], dim=-1).numpy()
        
        dif = (difs**2).mean(axis=1)
        dif = dif.reshape(101, 101)

        fig.add_subplot(2, 2, i+1)
        im2 = plt.imshow(dif, extent=extent, origin="lower", cmap=plt.cm.jet, aspect='auto')
        plt.colorbar()
        plt.xlabel('horizontal displacement')
        plt.ylabel('vertical displacement')
        if i == 0:
            plt.title('action 0: do nothing')
        elif i == 1:
            plt.title('action 1: fire left engine')
        elif i == 2:
            plt.title('action 2: fire main engine')
        else:
            plt.title('action 3: fire right engine')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # python -m uncertainty_modeling.d4rl.test_rapp --p "/home/seungjae/Desktop/AI602/AI602_Project/bear/trained_LunarLander-v2_seungjae_horizontal"
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str,
                        help='path')
    args = parser.parse_args()
    test_rapp_lunarlander(args)

    # "/home/seungjae/Desktop/AI602/AI602_Project/bear/trained_LunarLander-v2_seungjae.pt"