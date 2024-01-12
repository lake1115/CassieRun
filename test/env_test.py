import numpy as np
import time
from cassierun_env import CassieRunEnv
import gym
from gym import wrappers
import numpy as np
import os
from cassie_m.cassiemujoco import *
from cassie_m.udp import quaternion2euler,euler2quat
import random
import torch
# env = CassieRunEnv(traj="walking",speed=1.,visual=False, dynamics_randomization=False)

# if __name__ == '__main__':
#     obs = env.reset()
#     done = False
#     while not done:
#         action = np.random.normal(0,1,size=10)
#         obs, reward, done, info = env.step(action)
#         print("before", env.sim.x_position_before)
#         print("after", env.sim.x_position_after)
#         env.render()

##############################
env = CassieRunEnv(visual=True,traj="stepping",speed=0)


u = pd_in_t()

dt = 60/2000
env.phase = 0
ss = 0
obs = env.reset()

def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat
mirrored_pos = np.array([0.1,-1,2,3,4,5,6,21,22,23,24,25,26,27,28,29,30,31,32,33,34,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
mirrored_vel = np.array([0.1,-1,2,3,4,5,19,20,21,22,23,24,25,26,27,28,29,30,31,6,7,8,9,10,11,12,13,14,15,16,17,18])
#mirrored_obs = np.array([-0, 1, 2, -3, 4, -5, 13, 14, 15, 16, 17, 18, 19, 6, 7, 8, 9, 10, 11, 12, 20, -21, 22, 23, -24, 25, 33, 34, 35, 36, 37, 38, 39, 26, 27, 28, 29, 30, 31, 32])
pos_mirror_matrix = _get_symmetry_matrix(mirrored_pos)
vel_mirror_matrix = _get_symmetry_matrix(mirrored_vel)
def mirror_observation(pos, vel):
    pos = pos @ pos_mirror_matrix
    vel = vel @ vel_mirror_matrix
    return pos, vel

while True:


    pos, vel = env.get_ref_state(env.phase)
    #pos2, vel2 = env.get_ref_state(env.phase + 60)
    
    pos, vel = mirror_observation(pos,vel)
    euler = quaternion2euler(pos[3:7])
    euler[0] = -euler[0]
    euler[2] = -euler[2]
    pos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])
    env.phase += 60 #+ np.random.randint(-7, 8)

    if env.phase >= 1680:   # len(trajectory) = 1682 <=> 28 phase
        env.phase = env.phase % 1680
        env.counter += 1
        
    # print("x:",env.sim.qpos()[0])
    # print("pos",pos[0])
    #print("phase",env.phase)
    
    # print("counter",env.counter)
    print(env.phase//60+1)
    env.sim.set_qpos(pos)
    env.sim.set_qvel(vel)
    #env.mirror_state(pos)
   # env.sim.step_pd(u)
   # reward = env.compute_reward(u)
    env.time += 1

    env.render()
    # if (env.phase//60 +1) % 7 ==2:
    #   time.sleep(5)