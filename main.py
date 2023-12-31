import os
# os.environ['LD_LIBRARY_PATH']='$LD_LIBRARY_PATH:/home/HYK/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia'
# import gymnasium as gym
# from humanoid_wrapper import HumanoidTruncatedObsEnv
from cassie_env import CassieRunEnv

# 只能先gym.make，不然会报段错误，估计是导入的库冲突了(千万不要在创建环境之前导入mbrl的任何东西)
# env = HumanoidTruncatedObsEnv()
# test_env = HumanoidTruncatedObsEnv()
env = CassieRunEnv(traj= 'running', visual=False, dynamics_randomization=False)
test_env = CassieRunEnv(traj= 'running', visual=False, dynamics_randomization=False)
print(env)
print('------------create env OK----------------')

import mbrl.algorithms.pets as pets
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.planet as planet
import mbrl.env.termination_fns
import mbrl.env.reward_fns

import torch

def cassie(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2
    z = next_obs[:, 1]
    done = (z < 0.8)
    done = done[:, None]
    return done

# term_fn = mbrl.env.termination_fns.humanoid
term_fn = cassie # mbrl.env.termination_fns.humanoid
reward_fn = None
model_env = [env, test_env, term_fn, reward_fn]

import hydra
import numpy as np
import omegaconf
import torch


print('------Start Training------')
@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, test_env, term_fn, reward_fn = model_env[0], model_env[1], model_env[2], model_env[3]

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo" or cfg.algorithm.name == "sac":
        # test_env = mbrl.env.mujoco_envs.HumanoidTruncatedObsEnv()
        print('Start MBPO Training')
        test_env = model_env[1]
        return mbpo.train(env, test_env, term_fn, cfg)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)

if __name__ == "__main__":
    run()























