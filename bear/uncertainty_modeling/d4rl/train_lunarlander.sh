#!/bin/bash/

python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_hori_0 --d /home/seungjae/Desktop/lunarlander/replay_buffer_horizontal_0.pt
python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_hori_1 --d /home/seungjae/Desktop/lunarlander/replay_buffer_horizontal_1.pt
python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_hori_2 --d /home/seungjae/Desktop/lunarlander/replay_buffer_horizontal_2.pt
python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_hori_3 --d /home/seungjae/Desktop/lunarlander/replay_buffer_horizontal_3.pt

python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_verti_0 --d /home/seungjae/Desktop/lunarlander/replay_buffer_vertical_0.pt
python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_verti_1 --d /home/seungjae/Desktop/lunarlander/replay_buffer_vertical_1.pt
python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_verti_2 --d /home/seungjae/Desktop/lunarlander/replay_buffer_vertical_2.pt
python -m uncertainty_modeling.d4rl.train_rapp --e LunarLander-v2 --x seungjae_verti_3 --d /home/seungjae/Desktop/lunarlander/replay_buffer_vertical_3.pt