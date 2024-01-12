#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   buffer.py
@Time    :   2023/04/19 09:40:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence
import torch


def merge_buffers(buffers):
    merged = Buffer()
    for buf in buffers:
        offset = len(merged)

        merged.states  += buf.states
        merged.actions += buf.actions
        merged.rewards += buf.rewards
        merged.values  += buf.values
        merged.returns += buf.returns

        merged.ep_returns += buf.ep_returns
        merged.ep_lens    += buf.ep_lens

        merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
        merged.ptr += buf.ptr

    return merged

class Buffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.
    This container is intentionally not optimized w.r.t. to memory allocation
    speed because such allocation is almost never a bottleneck for policy
    gradient.

    On the other hand, experience buffers are a frequent source of
    off-by-one errors and other bugs in policy gradient implementations, so
    this code is optimized for clarity and readability, at the expense of being
    (very) marginally slower than some other implementations.
    (Premature optimization is the root of all evil).
    """
    def __init__(self, gamma=0.99, lam=0.95):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr = 0
        self.traj_idx = [0]

    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        # self.states  += [state.squeeze(0)]
        # self.actions += [action.squeeze(0)]
        # self.rewards += [reward.squeeze(0)]
        # self.values  += [value.squeeze(0)]
        self.states  += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values  += [value]
        self.ptr += 1

    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = last_val.squeeze(0).copy()  # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R)  # TODO: self.returns.insert(self.path_idx, R) ?
                                  # also technically O(k^2), may be worth just reversing list
                                  # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

    def get(self):
        return(
            np.array(self.states),
            np.array(self.actions),
            np.array(self.returns),
            np.array(self.values)
        )

    def sample(self, batch_size=64, recurrent=False):
        if recurrent:
            random_indices = SubsetRandomSampler(range(len(self.traj_idx)-1))
            sampler = BatchSampler(random_indices, batch_size, drop_last=False)
        else:
            random_indices = SubsetRandomSampler(range(self.ptr))
            sampler = BatchSampler(random_indices, batch_size, drop_last=True)
        observations, actions, returns, values = map(torch.Tensor, self.get())

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for indices in sampler:
            if recurrent:
                obs_batch       = [observations[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                action_batch    = [actions[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                return_batch    = [returns[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                advantage_batch = [advantages[self.traj_idx[i]:self.traj_idx[i+1]] for i in indices]
                mask            = [torch.ones_like(r) for r in return_batch]

                obs_batch       = pad_sequence(obs_batch, batch_first=False)
                action_batch    = pad_sequence(action_batch, batch_first=False)
                return_batch    = pad_sequence(return_batch, batch_first=False)
                advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                mask            = pad_sequence(mask, batch_first=False)
            else:
                obs_batch       = observations[indices]
                action_batch    = actions[indices]
                return_batch    = returns[indices]
                advantage_batch = advantages[indices]
                mask            = 1


            yield obs_batch, action_batch, return_batch, advantage_batch, mask
