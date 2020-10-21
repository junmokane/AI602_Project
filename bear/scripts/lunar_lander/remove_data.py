import torch
import numpy as np
import copy


def remove(path):
    data = torch.load(path)
    location_list, action_list = [np.reshape(st[0], (1, 8)) for st in data], [st[1] for st in data]
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

    action_0 = action_list[action_list == 0]
    action_1 = action_list[action_list == 1]
    action_2 = action_list[action_list == 2]
    action_3 = action_list[action_list == 3]

    action_l = copy.deepcopy([action_0, action_1, action_2, action_3])
    location_l = copy.deepcopy([location_0, location_1, location_2, location_3])

    a_hori, l_hori = [], []

    for a, l in zip(action_l, location_l):
        a = a[l[:, 0] > 0.1]
        l = l[l[:, 0] > 0.1]
        a_hori.append(a)
        l_hori.append(l)
    
    location_hori = np.concatenate(l_hori, axis=0)
    action_hori = np.concatenate(a_hori, axis=0)
    print("horizontal : ", location_hori.shape, action_hori.shape)

    # Vertical
    action_l = copy.deepcopy([action_0, action_1, action_2, action_3])
    location_l = copy.deepcopy([location_0, location_1, location_2, location_3])

    a_verti, l_verti = [], []

    for a, l in zip(action_l, location_l):
        a = a[l[:, 1] < 0.8]
        l = l[l[:, 1] < 0.8]
        a_verti.append(a)
        l_verti.append(l)
    
    location_verti = np.concatenate(l_verti, axis=0)
    action_verti = np.concatenate(a_verti, axis=0)
    print("vertical : ", location_verti.shape, action_verti.shape)

    # Save
    for i, (a, l) in enumerate(zip(a_verti, l_verti)):
        torch.save([l, a], f"/home/seungjae/Desktop/lunarlander/replay_buffer_vertical_{i}.pt")
    for i, (a, l) in enumerate(zip(a_hori, l_hori)):
        torch.save([l, a], f"/home/seungjae/Desktop/lunarlander/replay_buffer_horizontal_{i}.pt")
        
        
    # torch.save([location_hori, action_hori], "/home/seungjae/Desktop/lunarlander/replay_buffer_horizontal.pt")
    # torch.save([location_verti, action_verti], "/home/seungjae/Desktop/lunarlander/replay_buffer_vertical.pt")


if __name__ == "__main__":
    path = "/home/seungjae/Desktop/lunarlander/replay_buffer.pt"
    remove(path)