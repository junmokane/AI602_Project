import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictive(data, trajectories, xs, mu=None, sigma=None, title=None):
    sns.set_style('darkgrid')
    blue = sns.color_palette()[0]
    red = sns.color_palette()[3]

    plt.figure(figsize=(9., 7.))
    
    plt.plot(data[:, 0], data[:, 1], "o", color=red, alpha=0.7, markeredgewidth=1., markeredgecolor="k")
    
    if mu is None:
        mu = np.mean(trajectories, axis=0)
    if sigma is None:
        sigma = np.std(trajectories, axis=0)

    plt.plot(xs, mu, "-", lw=2., color=blue)
    plt.plot(xs, mu-2 * sigma, "-", lw=0.75, color=blue)
    plt.plot(xs, mu+2 * sigma, "-", lw=0.75, color=blue)
    np.random.shuffle(trajectories)
    for traj in trajectories:
        plt.plot(xs, traj, "-", alpha=.5, color=blue, lw=1.)

    xs = xs[:,0]

    plt.fill_between(xs, mu-3*sigma, mu+3*sigma, alpha=0.35, color=blue)

    plt.xlim([np.min(xs), np.max(xs)])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if title:
        plt.title(title, fontsize=16)
    plt.show()