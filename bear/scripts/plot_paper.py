import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from collections import defaultdict


def plot_one(exp_names, csv_slices, feature, env_name, directory):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.canvas.set_window_title(feature)
    for (csv_slice, exp_name) in zip(csv_slices, exp_names):
        '''
        if exp_name == "BEAR":
            plt.plot(csv_slice[feature].to_numpy(), linewidth=3, color='red')
        elif exp_name == "UWAC":
            plt.plot(csv_slice[feature].to_numpy(), linewidth=3, color='blue')
        else:
            plt.plot(csv_slice[feature].to_numpy(), linewidth=3, color='violet')
        '''
        plt.plot(csv_slice[feature].to_numpy(), linewidth=3)
        ax.yaxis.get_offset_text().set_fontsize(30)
    plt.xlabel("Training Epochs", fontsize=30)
    plt.xticks(fontsize=30)
    if feature == "evaluation/Returns Mean":
        plt.title("Average Return", fontsize=30)
        plt.legend(exp_names, fontsize=30, fancybox=True)
    elif feature == "trainer/Q Targets Mean":
        plt.title("Q Target", fontsize=30)
    plt.ylabel(env_name, fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(f"""{directory}{env_name}_{feature.replace(" ", "-").replace("/", "-")}.png""")


def plot_data(args):
    path = args.file
    features = args.f
    style = args.s
    directory = args.d
    verbose = args.v
    
    plt.style.use(style)
    features = features[0].split(",")

    SORT_ORDER = {"0000": 0, "0001": 1, "0002": 2}

    for feature in features:
        path = path.rstrip('/').rstrip('\\')
        env_name = path.split('/')[-1]
        method = env_name.split('-')[0]
        env_name = env_name.replace(method+'-', '')
        csv_paths = glob.glob(f"{path}/**/progress.csv")
        try:
            csv_paths.sort(key=lambda x: (SORT_ORDER[x.split("/")[-2].split("_")[-3]], x.split("/")[-2].split("_")[-1]))
        except:
            print("Make sure your experiment number is one of {0000, 0001, 0002}!")
        exp_names = [csv_path.split("/")[-2] for csv_path in csv_paths]
        if not verbose:
            sim_exp_names = []
            for e in exp_names:
                if e.split("_")[-3] == "0000":
                    sim_exp_names.append("BEAR")
                elif e.split("_")[-3] == "0002":
                    if e.split("_")[-1] == "0":
                        sim_exp_names.append("AC-MUSAT 1")
                    elif e.split("_")[-1] == "3":
                        sim_exp_names.append("AC-MUSAT 1e-3")
                    elif e.split("_")[-1] == "6":
                        sim_exp_names.append("AC-MUSAT 1e-6")
                elif e.split("_")[-3] == "0001":
                    sim_exp_names.append("UWAC")
            exp_names = sim_exp_names
        print(env_name, exp_names, len(csv_paths))

        assert len(csv_paths) > 0, "There is no csv files"

        csv_slices = []
        for csv_path in csv_paths:
            try:
                csv = pd.read_csv(csv_path)
                csv_slices.append(csv.loc[:, [feature]])
            except:
                print(f"Failed to open : {csv_path}")
            del csv

        plot_one(exp_names, csv_slices, feature, env_name, directory)
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
    parser.add_argument('--d', type=str,
                        help="Save directory")
    parser.add_argument('--v', type=bool, default=False,
                        help="if True, legend will show full experiment name")
    args = parser.parse_args()
    plot_data(args)
