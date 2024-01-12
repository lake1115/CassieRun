#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/04/14 15:27:01
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import torch
import sys, argparse
from util.log import parse_previous
from util.eval import Eval
import numpy as np
import json
import os
# os.system('nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > tmp.txt')
# memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
# os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax(memory_gpu)) 
# os.system('rm tmp.txt')
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    """
        General arguments for configuring the logger
    """
    
    parser.add_argument("--save_name", type=str, required=True, help="path to folder containing policy and run details")                                    # run name
    parser.add_argument("--logdir", type=str, default="./log/")          # Where to log diagnostics to
    parser.add_argument("--seed", default=0, type=int)                                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    if sys.argv[1] == 'eval':
        
        sys.argv.remove(sys.argv[1])
        parser.add_argument("--env_name", default="CassieRun-v0")                             # environment name
        parser.add_argument("--stats", default=True, action='store_false')
        parser.add_argument("--show", default=False, action='store_true')
        parser.add_argument("--policy", required=True)
        args = parser.parse_args()

        output_dir = os.path.join(args.logdir, args.policy, args.env_name, args.save_name)
        info_path = os.path.join(output_dir, "config.json")
        if  os.path.exists(info_path):
            with open(info_path, 'r') as f:
                run_args = json.load(f)
        else:
            print("can not find config")
        parser.set_defaults(**run_args)
        run_args= parser.parse_args()
        policy = torch.load(output_dir + "/actor.pt")
        policy.eval()

        eval = Eval(args, run_args, policy)
        eval.eval_policy()


    # Training Policy #
    """
    General arguments for configuring the environment
    """
    # command input, state input, env attributes
    parser.add_argument("--clock_based", default=False, action='store_true')
    parser.add_argument("--state_est", default=False, action='store_true')
    parser.add_argument("--simrate", default=60, type=int, help="simrate of environment")
    parser.add_argument("--history", default=0, type=int)                                         # number of previous states to use as input
    parser.add_argument("--dynamics_randomization", default=True, action='store_false')
    parser.add_argument("--impedance", default=False, action='store_true')             # learn PD gains or not
    parser.add_argument("--task", default="walking", type=str, help="select task for reward")
    parser.add_argument("--max_traj_len", type=int, default=150, help="Max episode horizon")
    parser.add_argument("--speed", type=float, default=1.0)
    # attributes for trajectory based environments
    parser.add_argument("--traj", default="walking", type=str, help="reference trajectory to use. options are 'aslip', 'walking', 'stepping'")
    parser.add_argument("--delta_action", default=True, action='store_false', dest='action as delta action')
    parser.add_argument("--ik_baseline", default=False, action='store_true', dest='ik_baseline')             # use ik as baseline for aslip + delta policies?
    # mirror loss and reward
    parser.add_argument("--mirror", default=True, action='store_false')             # mirror actions or not

    """
        General arguments for Curriculum Learning
    """
    ## TODO: need check
    parser.add_argument("--exchange_reward", default=None)                              # Can only be used with previous (below)
    parser.add_argument("--previous", type=str, default=None)                           # path to directory of previous policies for resuming training

    


    if sys.argv[1] == 'ppo':
        parser.add_argument("--env_name", default="CassieRun-v0")                             # environment name
        sys.argv.remove(sys.argv[1])
        """
            Utility for running Proximal Policy Optimization.

        """
        from algos.ppo import run_experiment

        # PPO algo args
        parser.add_argument("--input_norm_steps", type=int, default=10)
        parser.add_argument("--layers", type=int, nargs="*", default=(256, 256))
        parser.add_argument("--n_itr", type=int, default=10000, help="Number of iterations of the learning algorithm")
        parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
        parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
        parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")
        parser.add_argument("--learn_stddev", default=False, action='store_true', help="learn std_dev or keep it fixed")
        parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
        parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
        parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
        parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
        parser.add_argument("--num_steps", type=int, default=4096, help="Number of sampled timesteps per gradient estimate")
        parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
        parser.add_argument("--num_worker", type=int, default=4, help="Number of threads to train on")
        parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
        parser.add_argument("--recurrent",  default = True, action='store_true')
        parser.add_argument("--bounded",   type=bool, default=False)
        parser.add_argument("--policy",   type=str, default='ppo')
        parser.add_argument("--meta", default=False, action='store_false')
        args = parser.parse_args()

        args = parse_previous(args)

        run_experiment(args)

    elif sys.argv[1] == 'meta_ppo':
        sys.argv.remove(sys.argv[1])
        """
            Utility for running Meta RL.

        """
        from algos.meta import run_experiment

        parser.add_argument("--env_name", default="HalfCheetahMeta")                             # environment name
        parser.add_argument("--n_train_tasks", type=int, default=4)
        parser.add_argument("--n_eval_tasks", type=int, default=1)
        parser.add_argument("--n_tasks", type=int, default=5)
        parser.add_argument("--latent_size", type=int, default=5)
        parser.add_argument("--env_layers", type=int, default=(128,128))
        parser.add_argument("--recurrent",   default = True, action='store_true')
        parser.add_argument("--num_traj_sample", type=int, default=32, help="Number of trajectory to sample") 
        parser.add_argument("--num_worker", type=int, default=4, help="Number of threads to train on")
        parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
        parser.add_argument("--bounded",   type=bool, default=False)
        parser.add_argument("--kl_lambda",   type=float, default=0.1)
        parser.add_argument("--layers", type=int, nargs="*", default=(256, 256))
        parser.add_argument("--n_itr", type=int, default=10000, help="Number of iterations of the learning algorithm")
        parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
        parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
        parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")
        parser.add_argument("--learn_stddev", default=False, action='store_true', help="learn std_dev or keep it fixed")
        parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
        parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
        parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
        parser.add_argument("--num_steps", type=int, default=4096, help="Number of sampled timesteps per gradient estimate")
        parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
        parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
        parser.add_argument("--meta", default=True, action='store_false')
        parser.add_argument("--policy", default='meta')
        args = parser.parse_args()
        run_experiment(args)


    ## TODO: sac policy
    ## TODO: mbrl policy
    else:
        print("Invalid option '{}'".format(sys.argv[1]))

