import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_replay_buffer(path):
    data = torch.load(path)
    location_list, action_list = [np.reshape(st[0][:2], (1, 2)) for st in data], [st[1] for st in data]
    location_list = np.concatenate(location_list, axis=0)
    action_list = np.asarray(action_list)
    
    action_0 = action_list == 0
    action_1 = action_list == 1
    action_2 = action_list == 2
    action_3 = action_list == 3
    
    location_0 = location_list[action_0, :]
    location_1 = location_list[action_1, :]
    location_2 = location_list[action_2, :]
    location_3 = location_list[action_3, :]
    
    fig =  plt.figure(figsize=(20, 3))
    fig.add_subplot(1, 4, 1)
    plt.hexbin(location_0[:, 0], location_0[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 0: do nothing')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    fig.add_subplot(1, 4, 2)
    plt.hexbin(location_1[:, 0], location_1[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 1: fire left engine')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    fig.add_subplot(1, 4, 3)
    plt.hexbin(location_2[:, 0], location_2[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 2: fire main engine')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    fig.add_subplot(1, 4, 4)
    plt.hexbin(location_3[:, 0], location_3[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 3: fire right engine')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    
    plt.tight_layout()
    plt.show()


def visualize_training_set(path):
    data = torch.load(path)
    state_list, action_list = data[0], data[1]

    action_0 = action_list == 0
    action_1 = action_list == 1
    action_2 = action_list == 2
    action_3 = action_list == 3
    
    location_0 = state_list[action_0, :2]
    location_1 = state_list[action_1, :2]
    location_2 = state_list[action_2, :2]
    location_3 = state_list[action_3, :2]
    
    fig =  plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 2, 1)
    plt.hexbin(location_0[:, 0], location_0[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 0: do nothing')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    fig.add_subplot(2, 2, 2)
    plt.hexbin(location_1[:, 0], location_1[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 1: fire left engine')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    fig.add_subplot(2, 2, 3)
    plt.hexbin(location_2[:, 0], location_2[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 2: fire main engine')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    fig.add_subplot(2, 2, 4)
    plt.hexbin(location_3[:, 0], location_3[:, 1], gridsize=(150, 150), bins='log', extent=(-0.4, 0.4, 0.0, 1.5), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('action 3: fire right engine')
    plt.xlabel('horizontal displacement')
    plt.ylabel('vertical displacement')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # path = "/home/seungjae/Desktop/lunarlander/replay_buffer.pt"
    # visualize_replay_buffer(path)
    path = "/home/seungjae/Desktop/lunarlander/replay_buffer_horizontal_0.pt" # horizontal.pt
    visualize_training_set(path)
    path = "/home/seungjae/Desktop/lunarlander/replay_buffer_vertical.pt"
    visualize_training_set(path)