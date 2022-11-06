import gym
from src.agent import SACAgent
from src.sac import DiscreteSAC
from src.replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.distributions as dist
import smarts_env_v4 as smt
import numpy as np


import warnings
warnings.filterwarnings("ignore")

def statistic_action(a_history):
    counter = [0 for _ in range(11)]
    for each in a_history:
        counter[int(each.item())] += 1
    return counter
        

if __name__ == '__main__':
    # env = gym.make("LunarLander-v2")
    env = smt.make(
        "loop", 
        is_endless_goal=True, 
        is_multi_scenario=True,
        visdom = True,
        average_sample=False,
    )
    replay_buffer= ReplayBuffer(env)
    agent = SACAgent(
        env , DiscreteSAC, device="cuda",
        target_entropy_coeff=0.1,
        actor_learning_rate=5E-4,
        critic_learning_rate=5E-4,
        temperature_learning_rate=5E-4
    )
    agent.load("./SSM_5_50k.pt")
    
    # agent.eval(env, 1)
    # agent.fit_online(env, replay_buffer, 
    #     total_steps=int(1E6)+1, 
    #     max_episode_length=500,
    #     start_learning=10000,
    #     random_steps=10000,
    #     save_freq=10000,
    #     batch_size=100,
    #     wandb_project_name="softmodule_sac",
    #     use_wandb_log=True
    # )
    
    # import numpy as np
    # x, ptr = np.empty((12000, 235)), 0
    # y = np.ones((12000,)) * 8
    # for epsilon in [i/5 for i in range(6)]:
    #     for k in range(4):
    #         print(f'Begin eps{epsilon} k{k+1}')
    #         o = env.reset()
    #         z_map = env.z_map
    #         o_f = agent.compress_raw_obs_feats(o, z_map)
    #         d = False
    #         while not d:
    #             Q = agent.collision_predictor(torch.from_numpy(o).reshape(1,-1).cuda().float())
    #             V = Q.mean()
                
    #             o = torch.from_numpy(o_f[:-9]).cuda().unsqueeze(0).float()
    #             z_map = torch.from_numpy(o_f[-9:]).cuda().reshape(1,-1).float()
    #             p = agent.net.P_net(o, z_map, V).cpu()
    #             a_d = p.argmax().item() if np.random.rand() < epsilon else env.action_space.sample()
                
    #             o, r, d, info = env.step(a_d)
    #             x[ptr] = env.env.preserved_info["Agent_0"].classifier_input
    #             ptr += 1
    #             z_map = env.z_map
    #             o_f = agent.compress_raw_obs_feats(o, z_map)
    # import pickle as pkl
    # with open("./classifier/dataset/8.pkl", "wb") as f:
    #     pkl.dump((y, x), f)


    a_history = []
    t_mean = 0
    r_mean = 0
    for _ in range(1):
        o = env.reset()
        z_map = env.z_map
        o_f = agent.compress_raw_obs_feats(o, z_map)
        d = False
        for i in range(2000):
            Q = agent.collision_predictor(torch.from_numpy(o).reshape(1,-1).cuda().float())
            V = Q.mean()
            A = Q - V
            
            o = torch.from_numpy(o_f[:-9]).cuda().unsqueeze(0).float()
            z_map = torch.from_numpy(o_f[-9:]).cuda().reshape(1,-1).float()
            p = agent.net.P_net(o, z_map, V).cpu()
            
            a = dist.Categorical(probs = p).sample().item()
            a_d = p.argmax().item()
            a_s = A.argmin().item()
            
            a_a = agent.act(o_f, V, 0.6)

            o, r, d, info = env.step(a_d)
            z_map = env.z_map
            o_f = agent.compress_raw_obs_feats(o, z_map)
            a_history.append((a,a_d,a_s,a_a))
            # print(f'{env.env.preserved_info["Agent_0"].road_all_wrong}')
            # print(f'{env.env.preserved_info["Agent_0"].np_obs["waypoints"]["pos"]}')
            
            # print(f'{env.env.preserved_info["Agent_0"].masked_all_lane_indeces}')
            # print(f'\r{V.item():.3f}',end="")
            t_mean += 1
            r_mean += r
            

# np_obs = env.env.preserved_info["Agent_0"].np_obs
# raw_obs = env.env.preserved_info["Agent_0"].raw_obs
# wrong_road_indeces = np.ones(4)
# # if not hasattr(raw_obs.ego_vehicle_state.mission.goal, "position"): return wrong_road_indeces
# goal_point = np_obs["mission"]["goal_pos"][:2].reshape(1, 1, -1)
# wp_end_points = np_obs["waypoints"]["pos"][:, -4:, :2]
# dist_to_goal = ((goal_point-wp_end_points)**2).sum(-1)
# for i in range(4):
#     if not (dist_to_goal[i, 0] > dist_to_goal[i, 1] and 
#             dist_to_goal[i, 1] > dist_to_goal[i, 2] and 
#             dist_to_goal[i, 2] > dist_to_goal[i, 3]):
#         wrong_road_indeces[i] = 0


