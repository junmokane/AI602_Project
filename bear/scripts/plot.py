import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from collections import defaultdict


def plot_one(exp_names, csv_slices, feature):
    print(feature)
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.set_window_title(feature)
    for csv_slice in csv_slices:
        plt.plot(csv_slice[feature].to_numpy())
    plt.legend(exp_names)
    plt.title(feature, fontsize=17)
    plt.xlabel("iteration", fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel(feature, fontsize=15)
    plt.yticks(fontsize=13)


def plot_data(args):
    path = args.file
    features = args.f
    style = args.s
    
    plt.style.use(style)
    features = features[0].split(",")

    for feature in features:
        path = path.rstrip('/').rstrip('\\')
        csv_paths = glob.glob(f"{path}/**/progress.csv")
        exp_names = [csv_path.split("/")[-2] for csv_path in csv_paths]

        csv_slices = []
        for csv_path in csv_paths:
            csv = pd.read_csv(csv_path)
            csv_slices.append(csv.loc[:, [feature]])
            del csv

        plot_one(exp_names, csv_slices, feature)
    plt.show()


if __name__ == "__main__":
    # To run, refer README.md
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the task directory')
    parser.add_argument('--f', type=str, nargs='+',
                        help='List of features to plot')
    parser.add_argument('--s', type=str, default='ggplot',
                        help='Style of plots, Look at (https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html)')
    args = parser.parse_args()
    plot_data(args)
