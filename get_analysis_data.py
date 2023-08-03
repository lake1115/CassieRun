import os
os.environ['LD_LIBRARY_PATH']='$LD_LIBRARY_PATH:/home/HYK/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH:/usr/lib/nvidia'
from cassie_env import CassieRunEnv

# 只能先gym.make，不然会报段错误，估计是导入的库冲突了(千万不要在创建环境之前导入mbrl的任何东西)
env = CassieRunEnv(traj= 'running', visual=False, dynamics_randomization=False)
print(env)
print('------------create env OK----------------')
model_env = [env]

import hydra
import numpy as np
import pandas as pd
import omegaconf
import torch

import mbrl
import mbrl.util.common

@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    my_env = model_env[0]
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape, 
                                                              model_dir = r"/home/HYK/sim2real/cassie_running_test/cassie_mbrl/exp/mbpo/default/cassie/2023.04.19/105206")
    # nomalizer = dynamics_model.input_normalizer
    model = dynamics_model.model
    print(type(model))
    print("Create Model Success!")
    
    dtype = np.double
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=np.random.default_rng(seed=cfg.seed),
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
        load_dir= r"/home/HYK/sim2real/cassie_running_test/cassie_mbrl/exp/mbpo/default/cassie/2023.04.19/105206"
    )
    print("Create Buffer Success!")

    
    dataset = replay_buffer.get_all(shuffle=False)
    # (obs, act, next_obs, rewards, terminated, truncated) = dataset.astuple()
    print("Is delta target? ", dynamics_model.target_is_delta)
    (model_in, target) = dynamics_model._process_batch(dataset)

    mean_pred, logvar_pred = model._default_forward(model_in, False)

    # add to dataframe
    mean_pred = mean_pred.detach().cpu().numpy()
    logvar_pred = logvar_pred.detach().cpu().numpy()
    target = target.cpu().numpy()

    # 保存8个模型关于actor loss
    for i in range(mean_pred.shape[0]):
        diff = pd.DataFrame(np.abs(mean_pred[i] - target))
        des = diff.describe()
        row = pd.DataFrame(target.mean(axis = 0)[None],columns=des.columns,index = ['data_mean'])
        des = des.append(row)
        row2 = np.abs(pd.DataFrame(target.mean(axis = 0)[None],columns=des.columns,index = ['data_abs_mean']))
        des = des.append(row2)

        des.to_csv(f'/home/HYK/sim2real/cassie_running_test/cassie_mbrl/dataset/abs_model_loss{i+1}.csv')
        print('save abs OK!')

        diff = pd.DataFrame(np.abs(mean_pred[i] - target)/np.sqrt(np.exp(logvar_pred[i])))
        des = diff.describe()
        row = pd.DataFrame(target.mean(axis = 0)[None],columns=des.columns,index = ['data_mean'])
        des = des.append(row)
        row2 = np.abs(pd.DataFrame(target.mean(axis = 0)[None],columns=des.columns,index = ['data_abs_mean']))
        des = des.append(row2)

        des.to_csv(f'/home/HYK/sim2real/cassie_running_test/cassie_mbrl/dataset/norm_model_loss{i+1}.csv')
        print('save OK!')
    
if __name__=="__main__":
    run()