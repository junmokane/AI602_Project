import torch

# Unzip DDQN_lunar_rep_buf.zip

# Total data (replay_buffer)
data = torch.load("/home/seungjae/Desktop/lunarlander/DDQN_lunar_rep_buf_total.pt")
# list with [states, action]
print(data[0].shape, data[1].shape)  # (100000, 8), (100000, 1) --> np.array
xy_location = data[0][:, :2]  # xy is first 2 data of states.

# Masked data, 0, 1, 2, 3 indicated action!
data = torch.load("/home/seungjae/Desktop/lunarlander/DDQN_lunar_rep_buf_hori_masked_0.pt")
print(data[0].shape, data[1].shape)  # (9966, 8), (9966, 1)