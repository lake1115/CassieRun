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

# term_fn = mbrl.env.termination_fns.humanoid
term_fn = mbrl.env.termination_fns.cassie
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
        return mbpo.train(env, test_env, term_fn, cfg,
                          model_dir =  r"/home/HYK/sim2real/cassie_running_test/cassie_mbrl/exp/mbpo/default/cassie/2023.04.19/105206",
                          load_dir =  r"/home/HYK/sim2real/cassie_running_test/cassie_mbrl/exp/mbpo/default/cassie/2023.04.19/105206",
                          continue_learning= True,
        )
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)

if __name__ == "__main__":
    run()























