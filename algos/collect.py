#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   collect.py
@Time    :   2023/04/19 10:14:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import ray
from copy import deepcopy
import torch
import time
from algos.buffer import Buffer
import numpy as np

@ray.remote
class Collect_Worker:
    def __init__(self, policy, critic, env_fn, gamma, lam):
        self.gamma = gamma
        self.lam = lam
        self.policy = deepcopy(policy)
        self.critic = deepcopy(critic)
        self.env = env_fn()

    def sync_policy(self, new_actor_params, new_critic_params):
        for p, new_p in zip(self.policy.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)

    def collect(self, max_traj_len, min_steps, anneal=1.0):
        torch.set_num_threads(1)
        with torch.no_grad():
            memory = Buffer(self.gamma, self.lam)
            num_steps = 0
            while num_steps < min_steps:
                state = torch.Tensor(self.env.reset())
                done = False
                value = 0
                traj_len = 0

                if hasattr(self.policy, 'init_hidden_state'):
                    self.policy.init_hidden_state()

                if hasattr(self.critic, 'init_hidden_state'):
                    self.critic.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action = self.policy(state, deterministic=False, anneal=anneal)
                    value = self.critic(state)

                    next_state, reward, done, _ = self.env.step(action.numpy())

                    memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                    state = torch.Tensor(next_state)
                    traj_len += 1
                    num_steps += 1

                value = self.critic(state)
                memory.finish_path(last_val=(not done) * value.numpy())
            return memory
        
    def evaluate(self, max_traj_len, trajs=1):
        torch.set_num_threads(1)
        with torch.no_grad():
            ep_returns = []
            for traj in range(trajs):
                state = torch.Tensor(self.env.reset())
                done = False
                traj_len = 0
                ep_return = 0

                if hasattr(self.policy, 'init_hidden_state'):
                    self.policy.init_hidden_state()

                while not done and traj_len < max_traj_len:
                    action = self.policy(state, deterministic=False, anneal=1.0)

                    next_state, reward, done, _ = self.env.step(action.numpy())

                    state = torch.Tensor(next_state)
                    ep_return += reward
                    traj_len += 1
                ep_returns += [ep_return]
            return np.mean(ep_returns)

