from cv2 import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import copy
import gym

from src.sac import DiscreteSAC, MLP
from src.replay_buffer import Episode, ReplayBuffer

import wandb


# ! EXPERIMENTAL
class CollisionPredictor(nn.Module):
    def __init__(self, s_dim=100, a_dim=11):
        super().__init__()
        self.net = MLP(s_dim, a_dim, [256], output_activation=nn.Sigmoid)
        self.apply(self._init_weights)
        
        self.net_targ = copy.deepcopy(self.net)
        for p in self.net_targ.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.net.parameters(), lr = 5E-4)
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
    def _update_target_parameters(self):
        for p, p_targ in zip(self.net.parameters(), self.net_targ.parameters()):
            p_targ.data.lerp_(p.data, 0.005)
    def update(self, s, a, R, s2, pi):
        batch_size = s.shape[0]
        # TD-target
        with torch.no_grad():
            y = R.clone().reshape(batch_size, 1)
            Q2 = self.net_targ(s2)
            # Q2 = self.net(s2)
            idx = torch.where(R == -1)
            y[idx] = 0.9 * ((Q2*pi).sum(-1, keepdims=True))[idx]
        Q = self.forward(s)[
            torch.arange(batch_size).reshape(batch_size, 1),
            a.long().reshape(batch_size, 1)
        ]
        loss = F.mse_loss(Q, y)
        # loss = (-y*(Q+1e-8).log()-(1-y)*(1-Q+1e-8).log()).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach().item()
    def forward(self, s):
        return self.net(s)

    
def np_one_hot(x, num, shape=None, dtype=np.float32):
    y = np.zeros(num)
    y[x] = 1.0
    if shape is not None: y = y.reshape(shape)
    return y.astype(dtype)
    
def linear_annealing(begin:float, end:float, duration:int, step:int):
    return max(begin - (step/duration)*(begin-end), end)

class SACAgent():
    def __init__(self, env, net: DiscreteSAC,
        actor_learning_rate = 3E-4,
        critic_learning_rate = 3E-4,
        temperature_learning_rate = 3E-4,
        tau = 0.995,
        n_critic = 2,
        target_entropy_coeff = 0.1,
        clip_grad_norm = 10,
        device = "cpu"
        ):
        
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.temperature_learning_rate = temperature_learning_rate
        
        self.tau = tau
        self.n_critic = n_critic
        self.target_entropy_coeff = target_entropy_coeff
        self.clip_grad_norm = clip_grad_norm
        
        self.idx_n_critic = torch.arange(n_critic).view(-1, 1, 1).to(device)
        self.device = device
        self.o_dim = env.observation_space.shape[-1]
        self.a_dim = env.action_space.n
        self.entropy_target = self.target_entropy_coeff * np.log(self.a_dim)
        
        self.net : DiscreteSAC = net(self.o_dim, self.a_dim, n_critic = n_critic)
        self.net_targ : DiscreteSAC = self._build_target_net(self.net)
        self.collision_predictor = CollisionPredictor(self.o_dim, self.a_dim)
        if device != "cpu":
            self.net.cuda()
            self.net_targ.cuda()
            self.collision_predictor.cuda()
        
        self.opt_actor = torch.optim.Adam(self.net.P_net.parameters(), lr = actor_learning_rate, weight_decay=1E-4)
        self.opt_critic = torch.optim.Adam(self.net.Q_net.parameters(), lr = critic_learning_rate, weight_decay=1E-4)
        
    def _build_target_net(self, net: DiscreteSAC) -> DiscreteSAC:
        net_targ = copy.deepcopy(net)
        for p in net_targ.parameters():
            p.requires_grad = False
        return net_targ
    
    def _update_target_parameters(self):
        for p, p_targ in zip(self.net.parameters(), self.net_targ.parameters()):
            p_targ.data.lerp_(p.data, 1 - self.tau)
        
    @torch.no_grad()
    def act(self, o_f, collision_prob, safty_bound = 0.0):
        o = torch.from_numpy(o_f[:-9]).to(self.device).unsqueeze(0).float()
        z_map = torch.from_numpy(o_f[-9:]).to(self.device).reshape(1,-1).float()
        prob = self.net.P_net(o, z_map, collision_prob)
        a = dist.Categorical(probs = prob).sample().item()
        
        # ! EXPERIMENTAL
        Q = self.collision_predictor(o)
        if collision_prob > safty_bound:
            a = ((1-Q) * prob.reshape(1,-1)**0.1).argmax().item()
        return a
    
    def update(self, o, next_o, a, r, indeces, IS_weights, o_R, next_o_R, a_R, R):
        # o:        [batch_size, o_dim]
        # next_o:   [seq_len, batch_size, o_dim]
        # a:        [seq_len, batch_size]
        # r:        [seq_len, batch_size, 1]
        # R:        [seq_len, batch_size, 1]
        batch_size = o.shape[0]
        
        z_map, next_z_map = o[:, -9:], next_o[:, :, -9:]
        next_z_map_R = next_o_R[:, :, -9:]
        o, next_o = o[:, :-9], next_o[:, :, :-9]
        o_R, next_o_R = o_R[:, :-9], next_o_R[:, :, :-9]

        # Update soft Q function
        with torch.no_grad():
            o = torch.cat([o.unsqueeze(0), next_o[:-1]], 0).reshape(3*batch_size, -1)
            o_R = torch.cat([o_R.unsqueeze(0), next_o_R[:-1]], 0).reshape(3*batch_size, -1)
            z_map = torch.cat([z_map.unsqueeze(0), next_z_map[:-1]], 0).reshape(3*batch_size, -1)
            
            o2 = next_o.reshape(3*batch_size, -1)
            o2_R = next_o_R.reshape(3*batch_size, -1)
            
            z_map2 = next_z_map.reshape(3*batch_size, -1)
            z_map2_R = next_z_map_R.reshape(3*batch_size, -1)
            
            a = a.reshape(-1,).long()
            a_R = a_R.reshape(-1).long()
            r = r.reshape(-1, 1)
            R = R.reshape(-1, 1)
            
            collision_prob = self.collision_predictor(o).mean()
            collision_prob2 = self.collision_predictor(o2).mean()
            collision_prob2_R = self.collision_predictor(o2_R).mean()
            
        
        Qs = self.net.cal_Q(o, z_map, collision_prob)[
            self.idx_n_critic,
            torch.arange(3*batch_size).view(1, -1, 1).to(self.device), 
            a.reshape(1, -1, 1)
        ]
        
        with torch.no_grad():
            Q2 = self.net_targ.cal_Q(o2, z_map2, collision_prob2).min(0).values
            P2 = self.net.P_net(o2, z_map2, collision_prob2)
            # Q2, P2 = self.net(o2)
            V2 = (P2 * (Q2 - self.net.temperature * (P2 + 1e-8).log())).sum(-1, keepdims=True)
            Q_targ = r + 0.99 * V2
            td_error = torch.abs(Qs - Q_targ.unsqueeze(0)).mean(0).reshape(3, batch_size).sum(0)
            
        lossQ = (((Qs - Q_targ.unsqueeze(0))**2).mean(0) * IS_weights).mean()
        
        self.opt_critic.zero_grad()
        lossQ.backward()
        torch.nn.utils.clip_grad_norm_(self.net.Q_net.parameters(), self.clip_grad_norm)
        self.opt_critic.step()
        
        # Update Collision Predictor
        with torch.no_grad():
            pi = self.net.P_net(o2_R, z_map2_R, collision_prob2_R)
        lossC = self.collision_predictor.update(o_R, a_R, R, o2_R, pi)
        
        # Update policy
        self.net.need_grad(self.net.Q_net, False)
        Q, P = self.net(o, z_map, collision_prob)
        lossP = (P * (self.net.temperature * (P + 1e-8).log() - Q)).sum(-1, keepdims=True).mean()
        
        self.opt_actor.zero_grad()
        lossP.backward()
        torch.nn.utils.clip_grad_norm_(self.net.P_net.parameters(), self.clip_grad_norm)
        self.opt_actor.step()
        self.net.need_grad(self.net.Q_net, True)
        
        # Update temperature
        with torch.no_grad():
            P = P.detach().cpu()
            grad_temperature = (P * ((P+1e-8).log() + self.entropy_target)).sum(-1).mean()
            self.net.temperature += self.temperature_learning_rate * grad_temperature

        self._update_target_parameters()
        return lossQ.detach().cpu().item(), lossP.detach().cpu().item(), td_error, lossC
    
    @torch.no_grad()
    def eval(self, env_eval: gym.Env, eval_num = 10, render = False):
        t_mean = 0
        r_mean = 0
        for _ in range(eval_num):
            o = env_eval.reset()
            z_map = env_eval.z_map
            o_f = self.compress_raw_obs_feats(o, z_map)
            d = False
            while not d:
                if render:
                    env_eval.render()
                a = self.act(o_f)
                o, r, d, _ = env_eval.step(a)
                t_mean += 1
                r_mean += r
        return t_mean / eval_num, r_mean / eval_num
    
    # def save_policy(self, path: str):
    #     torch.save(
    #         self.net.P_net.state_dict(), path
    #     )
    
    def param_linear_annealing(self, step:int):
        
        self.target_kl = linear_annealing(0.05, 0.01, int(3E5), step)
        self.clip_ratio = linear_annealing(0.2, 0.1, int(3E5), step)
        self.ent_coef = linear_annealing(0.03, 0, int(1E5), step)
    
    def save(self, path: str):
        params = {
            "net": self.net.state_dict(),
            "net_targ": self.net_targ.state_dict(),
            "collision_predictor": self.collision_predictor.state_dict()
        }
        torch.save(params, path)
        
    def load(self, path: str):
        params = torch.load(path)
        self.net.load_state_dict(params["net"])
        self.net_targ.load_state_dict(params["net_targ"])
        self.collision_predictor.load_state_dict(params["collision_predictor"])
        
    def compress_raw_obs_feats(self, o, z_map):
        z1 = np_one_hot(z_map, 9)
        return np.concatenate([o, z1])
    
    def fit_online(
        self, 
        env: gym.Env, 
        replay_buffer: ReplayBuffer,
        total_steps = 100000,
        batch_size = 64,
        max_episode_length = 1000,
        start_learning = 10000,
        random_steps = 4000,
        save_freq = 10000,
        wandb_project_name = "DiscreteSAC_smarts",
        use_wandb_log = True,
    ):
        if use_wandb_log:
            config = {
                "actor_learning_rate": self.actor_learning_rate,
                "critic_learning_rate": self.critic_learning_rate,
                "temperature_learning_rate": self.temperature_learning_rate,
                "target_entropy_coeff": self.target_entropy_coeff,
                "total_steps": total_steps,
                "batch_size": batch_size,
                "start_learning": start_learning,
                "random_steps": random_steps,
                "clip_grad_norm": self.clip_grad_norm
            }
            wandb.init(project=wandb_project_name, config=config)

        step = 0
        while step < total_steps:
            o = env.reset()
            z_map = env.z_map
            o_f = self.compress_raw_obs_feats(o, z_map)
            episode = Episode(o_f)
            humaness = 0
            for t in range(max_episode_length):
                collision_prob = self.collision_predictor(torch.from_numpy(o).reshape(1,-1).to(self.device).float()).mean()
                a = self.act(o_f, collision_prob) if step >= random_steps else env.action_space.sample()
                o, r, d, info = env.step(a)
                z_map = env.z_map
                o_f = self.compress_raw_obs_feats(o, z_map)
                
                # ! EXPERIMENTAL
                events = env.env.preserved_info["Agent_0"].np_obs["events"]
                if events["collisions"] or events["on_shoulder"]: R = 1.0
                elif info["reached_goal"]: R = 0.0
                else: R = -1.0
                
                step += 1
                humaness += info["humaness"]
                episode += o_f, a, r, d, R
                print(f'\r[TotalStep: {step}] Collecting episode: {(t+1)/max_episode_length*100:.2f}%', end="")
            replay_buffer += episode
            
            if use_wandb_log:
                wandb.log({
                    "reward": episode.cumulative_reward,
                    "length": max_episode_length,
                    "success_rate": info["success_rate"],
                    "temperature": self.net.temperature,
                    "humaness": humaness,
                    "total_episodes": info["total_episodes"],
                })
            print(" ")    
            print(f'[Step:{t+1}] reward:{episode.cumulative_reward:.2f} success_rate:{info["success_rate"]:.2f}')

            if step >= start_learning:
                for t in range(max_episode_length):
                    batch = replay_buffer.sample(batch_size)
                    lossQ, lossP, td_error, lossC = self.update(*batch)
                    replay_buffer.update_priorities(batch[-6], td_error)
                    print(f'\r[TotalStep: {step}] Training: {(t+1)/max_episode_length*100:.2f}%', end="")
                if use_wandb_log:
                    wandb.log({
                        "lossQ": lossQ,
                        "lossP": lossP,
                        "lossC": lossC,
                    })
                print(" ")
                print(f'[Step:{t+1}] LossQ:{lossQ:.2f} LossP:{lossP:.2f} LossC:{lossC:.2f} Temper:{self.net.temperature:.2f}')
                if step % save_freq == 0:
                    self.save(f'./log/DiscreteSAC_{step}.pt')
                    # self.save_policy(f'./PolicySwitcher/PolicyLib/DiscreteSAC_{step}.pt')
                    
            self.opt_actor.learning_rate = linear_annealing(self.actor_learning_rate, 1E-4, int(2E5), step)
            self.opt_critic.learning_rate = linear_annealing(self.critic_learning_rate, 1E-4, int(2E5), step)
            replay_buffer.per_beta = linear_annealing(0.4, 1.0, int(1E5), step)
                    
                

# def explore_and_train(
#     env: gym.Env, 
#     agent: DQN_Agent, 
#     replay_buffer: PrioritizedBuffer,
#     learning_start = 1000,
#     learning_freq = 1,
#     random_step = 1000,
#     eval_freq = 40,
# ):
#     ep_num = 0
#     t = 0
    
    
#     pass
        
        