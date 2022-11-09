from collections import deque
import numpy as np
import torch

class Episode(object):
    def __init__(
        self,
        init_obs,
        episode_length = 500,
        device = "cuda",
    ):
        self.device = device
        
        self.o = torch.empty((episode_length+1, *init_obs.shape), device=self.device)
        self.o[0] = torch.from_numpy(init_obs).to(self.device)
        self.a = torch.empty((episode_length,), device=self.device)
        self.r = torch.empty((episode_length,), device=self.device)
        self.R = torch.empty((episode_length,), device=self.device)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0
        
    def __len__(self):
        return self._idx
    
    @property
    def first(self):
        return len(self) == 0
    
    def add(self, o, a, r, d, R):
        self.o[self._idx + 1] = torch.from_numpy(o).cuda()
        self.a[self._idx] = a
        self.r[self._idx] = r
        self.R[self._idx] = R
        self.cumulative_reward += r
        self.done = d
        self._idx += 1
    
    def __add__(self, transition):
        self.add(*transition)
        return self

class ReplayBuffer():
    def __init__(
        self,
        env,
        per_alpha = 0.6,
        per_beta = 0.4,
        horizon = 3,
        device = "cuda",
        capacity = int(1E6),
        episode_length = 500,
    ):
        self.device = torch.device(device)
        self.capacity = capacity
        self.horizon = horizon
        self.episode_length = episode_length
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        dtype = torch.float32
        o_dim = env.observation_space.shape[-1] + 9
        self.o = torch.empty((self.capacity + 1, o_dim), dtype=dtype, device=self.device)
        self.last_o = torch.empty((self.capacity // episode_length, o_dim), dtype=dtype, device=self.device)
        self.a = torch.empty((self.capacity,), dtype=torch.long, device=self.device)
        self.r = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self.R = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self.priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self.priorities_R = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
        self.eps = 1e-6
        self.idx = 0
        self.full = False

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        self.o[self.idx:self.idx+self.episode_length] = episode.o[:-1]
        self.last_o[self.idx//self.episode_length] = episode.o[-1]
        self.a[self.idx:self.idx+self.episode_length] = episode.a
        self.r[self.idx:self.idx+self.episode_length] = episode.r
        self.R[self.idx:self.idx+self.episode_length] = episode.R
        if self.full:
            max_priority = self.priorities.max().cpu().item()
        else:
            max_priority = 1. if self.idx == 0 else self.priorities[:self.idx].max().cpu().item()
        mask = torch.arange(self.episode_length) >= self.episode_length-self.horizon
        new_priorities = torch.full((self.episode_length,), max_priority, device=self.device)
        new_priorities[mask] = 0
        self.priorities[self.idx:self.idx+self.episode_length] = new_priorities
        
        new_priorities_R = episode.R.clone()+1.1
        new_priorities_R[torch.where(new_priorities_R == 2.1)] = 1.0
        new_priorities_R[torch.where(new_priorities_R == 1.1)] = 2.0
        self.priorities_R[self.idx:self.idx+self.episode_length] = new_priorities_R
        
        self.idx = (self.idx + self.episode_length) % self.capacity
        self.full = self.full or self.idx == 0

    def update_priorities(self, idxs, priorities):
        self.priorities[idxs] = priorities.to(self.device) + self.eps

    def _get_obs(self, arr, idxs):
        return arr[idxs]


    def sample(self, batch_size = 512):
        probs = ((self.priorities if self.full else self.priorities[:self.idx]) ** self.per_alpha).clone()
        probs /= probs.sum()
        probs_R = (self.priorities_R if self.full else self.priorities_R[:self.idx]).clone()
        probs_R /= probs_R.sum()
        total = len(probs)
        idxs = torch.from_numpy(np.random.choice(total, batch_size, p=probs.cpu().numpy(), replace=not self.full)).to(self.device)
        idxs_R = torch.from_numpy(np.random.choice(total, batch_size, p=probs_R.cpu().numpy(), replace=not self.full)).to(self.device)
        
        weights = (total * probs[idxs]) ** (-self.per_beta)
        weights /= weights.max()

        obs = self._get_obs(self.o, idxs)
        obs_R = self._get_obs(self.o, idxs_R)
        next_obs = torch.empty((self.horizon, batch_size, self.o.shape[-1]), device=self.device)
        next_obs_R = torch.empty((self.horizon, batch_size, self.o.shape[-1]), device=self.device)
        action = torch.empty((self.horizon, batch_size), device=self.device)
        action_R = torch.empty((self.horizon, batch_size), device=self.device)
        reward = torch.empty((self.horizon, batch_size), device=self.device)
        Reward = torch.empty((self.horizon, batch_size), device=self.device)
        for t in range(self.horizon):
            _idxs = idxs + t
            _idxs_R = idxs_R + t
            next_obs[t] = self._get_obs(self.o, _idxs+1)
            next_obs_R[t] = self._get_obs(self.o, _idxs_R+1)
            action[t] = self.a[_idxs]
            action_R[t] = self.a[_idxs_R]
            reward[t] = self.r[_idxs]
            Reward[t] = self.R[_idxs_R]

        if not action.is_cuda:
            action, reward, idxs, weights, done, Reward, action_R = \
                action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda(), Reward.cuda(), action_R.cuda()
        # [horizon, batch_size, shape]
        return obs, next_obs, action, reward.unsqueeze(-1), idxs, weights, obs_R, next_obs_R, action_R, Reward
