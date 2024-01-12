"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim

from torch.distributions import kl_divergence


import time

import numpy as np
import os, sys

import ray

from util.env import WrapEnv,get_normalization_params

from policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from policies.critic import FF_V, LSTM_V
#from envs.normalize import get_normalization_params, PreNormalizer
from algos.buffer import Buffer, merge_buffers
import pickle
from algos.collect import Collect_Worker


class PPO:
    def __init__(self, args, env_fn, policy, critic, save_path):
        self.env_name       = args['env_name']
        self.gamma          = args['gamma']
        self.lam            = args['lam']
        self.lr             = args['lr']
        self.eps            = args['eps']
        self.entropy_coeff  = args['entropy_coeff']
        self.clip           = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.epochs         = args['epochs']
        self.num_steps      = args['num_steps']
        self.max_traj_len   = args['max_traj_len']
        self.use_gae        = args['use_gae']
        self.num_worker         = args['num_worker']
        self.grad_clip      = args['max_grad_norm']
        self.recurrent      = args['recurrent']
        self.device         = args['device']

        self.total_steps = 0
        self.highest_reward = -1
        self.limit_cores = 0
        
        self.save_path = save_path

        self.env_fn = env_fn
        self.policy = policy
        self.critic = critic
        # self.workers = [Collect_Worker(self.policy, self.critic, self.env_fn,self.gamma, self.lam) for _ in range(self.num_worker)]
        self.workers = [Collect_Worker.remote(self.policy, self.critic, self.env_fn,self.gamma, self.lam) for _ in range(self.num_worker)]

    def save(self):

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(self.policy, os.path.join(self.save_path, "actor" + filetype))
        torch.save(self.critic, os.path.join(self.save_path, "critic" + filetype))


    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, env_fn, mirror_observation=None, mirror_action=None):
        policy = self.policy.to(self.device)
        critic = self.critic.to(self.device)
        old_policy = self.old_policy.to(self.device)
        # policy.obs_mean, old_policy.obs_mean, critic.obs_mean, policy.obs_std, old_policy.obs_std, critic.obs_std = \
        #     policy.obs_mean.to(self.device), old_policy.obs_mean.to(self.device), critic.obs_mean.to(self.device), policy.obs_std.to(self.device), old_policy.obs_std.to(self.device), critic.obs_std.to(self.device)

        values = critic(obs_batch)
        pdf = policy.distribution(obs_batch)

        # TODO, move this outside loop?
        with torch.no_grad():
            old_pdf = old_policy.distribution(obs_batch)
            old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        ratio = (log_probs - old_log_probs).exp()

        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        critic_loss = 0.5 * ((return_batch - values) * mask).pow(2).mean()

        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        if mirror_observation is not None and mirror_action is not None:
            env = env_fn()
            deterministic_actions = policy(obs_batch)
            if env.clock_based:
                if self.recurrent:
                    mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:], env.clock_inds) for i in range(obs_batch.shape[0])])
                    mirror_actions = policy(mir_obs)
                else:
                    mir_obs = mirror_observation(obs_batch, env.clock_inds)
                    mirror_actions = policy(mir_obs)
            else:
                if self.recurrent:
                    mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:], env.ref_inds) for i in range(obs_batch.shape[0])])
                    mirror_actions = policy(mir_obs)
                else:
                    mir_obs = mirror_observation(obs_batch, env.ref_inds)
                    mirror_actions = policy(mir_obs)

            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = 0.4 * (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = 0

        self.actor_optimizer.zero_grad()
        (actor_loss + mirror_loss + entropy_penalty).backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        with torch.no_grad():
            kl = kl_divergence(pdf, old_pdf)

        if mirror_observation is not None and mirror_action is not None:
            mirror_loss_return = mirror_loss.item()
        else:
            mirror_loss_return = 0
        return actor_loss.item(), pdf.entropy().mean().item(), critic_loss.item(), ratio.mean().item(), kl.mean().item(), mirror_loss_return

    def train(self,
              n_itr,
              logger=None,
              anneal_rate=1.0):

        self.old_policy = deepcopy(self.policy)
        
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()

        env = self.env_fn()
        obs_mirr, act_mirr = None, None
        if hasattr(env, 'mirror_observation'):
            if env.clock_based:
                obs_mirr = env.mirror_clock_observation
            else:
                obs_mirr = env.mirror_phase_observation

        if hasattr(env, 'mirror_action'):
            act_mirr = env.mirror_action

        curr_anneal = 1.0

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))
            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            ## collecting ##
            sample_start = time.time()
            if curr_anneal > 0.5:
                curr_anneal *= anneal_rate
            # buffers = [w.collect(max_traj_len = self.max_traj_len, min_steps = self.num_steps// self.num_worker, anneal=curr_anneal) for w in self.workers]
            buffers = ray.get([w.collect.remote(max_traj_len = self.max_traj_len, min_steps = self.num_steps// self.num_worker, anneal=curr_anneal) for w in self.workers])
            memory = merge_buffers(buffers)
            total_steps = len(memory)
            
            samp_time = time.time() - sample_start
            print("\t{:3.2f}s to collect {:6n} timesteps | {:3.2f}sample/s.".format(samp_time, total_steps, (total_steps)/samp_time))
            self.total_steps += total_steps

            ## training ##
            optimizer_start = time.time()
            self.old_policy.load_state_dict(self.policy.state_dict())
            for epoch in range(self.epochs):
                losses = []
                for batch in memory.sample(self.minibatch_size,self.recurrent):
                    obs_batch, action_batch, return_batch, advantage_batch, mask = batch
                    #obs_batch, action_batch, return_batch, advantage_batch = obs_batch.to(self.device), action_batch.to(self.device), return_batch.to(self.device), advantage_batch.to(self.device)
                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask, self.env_fn, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy, critic_loss, ratio, kl, mirror_loss = scalars

   
                    losses.append([actor_loss, entropy, critic_loss, ratio, kl, mirror_loss])
                    mean_losses = np.mean(losses, axis=0)
                # Early stopping
                if np.mean(kl) > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break
            opt_time = time.time() - optimizer_start
            print("{:.2f} s to optimizer| actor loss {:6.3f}, critic loss {:6.3f}, kl div {:6.3f}, entropy {:6.3f}.".format(opt_time, mean_losses[0], mean_losses[2], mean_losses[4], mean_losses[1]))

            ## sync_policy ##
            policy_param_id  = ray.put(list(self.policy.parameters()))
            critic_param_id = ray.put(list(self.critic.parameters()))
            # policy_param_id  = list(self.policy.parameters())
            # critic_param_id = list(self.critic.parameters())
            for w in self.workers:
                w.sync_policy.remote(policy_param_id, critic_param_id)
                # w.sync_policy(policy_param_id, critic_param_id)

            ## eval_policy ##
            evaluate_start = time.time()
            eval_reward = np.mean(ray.get([w.evaluate.remote(max_traj_len=self.max_traj_len, trajs=1) for w in self.workers]))
            # eval_reward = np.mean([w.evaluate(max_traj_len=self.max_traj_len, trajs=1) for w in self.workers])
            eval_time = time.time() - evaluate_start
            print("{:.2f} s to evaluate.".format(eval_time))

            if logger is not None:
                evaluate_start = time.time()
                # self.policy = self.policy.cpu()
                # self.critic = self.critic.cpu()
                
                avg_eval_reward = np.mean(eval_reward)
                avg_batch_reward = np.mean(memory.ep_returns)
                avg_ep_len = np.mean(memory.ep_lens)
                

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Timesteps', self.total_steps) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                logger.add_scalar("Test/Return", avg_eval_reward, itr)
                logger.add_scalar("Train/Return", avg_batch_reward, itr)
                logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
                logger.add_scalar("Train/Mean KL Div", kl, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)

                logger.add_scalar("Misc/Critic Loss", mean_losses[2], itr)
                logger.add_scalar("Misc/Actor Loss", mean_losses[0], itr)
                logger.add_scalar("Misc/Mirror Loss", mean_losses[5], itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                print("Save best policy!")
                self.save()

def run_experiment(args):
    from util.env import env_factory
    from util.log import create_logger

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set up Parallelism
    if not ray.is_initialized():
        if args.device == 'cpu':
            os.environ['OMP_NUM_THREADS'] = '1'
            ray.init(num_cpus=args.num_worker)
        else:
            os.environ['OMP_NUM_THREADS'] = '1'
            ray.init(num_cpus=args.num_worker)
            #ray.init(num_cpus=12, num_gpus=2)
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        policy = torch.load(os.path.join(args.previous, "actor.pt"))
        critic = torch.load(os.path.join(args.previous, "critic.pt"))
        # TODO: add ability to load previous hyperparameters, if this is something that we event want
        # with open(args.previous + "experiment.pkl", 'rb') as file:
        #     args = pickle.loads(file.read())
        print("loaded model from {}".format(args.previous))
    else:
        if args.recurrent:
            policy = Gaussian_LSTM_Actor(obs_dim, action_dim, fixed_std=np.exp(-2), env_name=args.env_name)
            critic = LSTM_V(obs_dim)
        else:
            if args.learn_stddev:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, layers=args.layers, fixed_std=None, env_name=args.env_name, bounded=args.bounded)
            else:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, layers=args.layers, fixed_std=np.exp(args.std_dev), env_name=args.env_name, bounded=args.bounded)
            critic = FF_V(obs_dim, layers=args.layers)

        with torch.no_grad():
            obs_mean, obs_std = get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn, procs=args.num_worker)
        policy.obs_mean = critic.obs_mean = torch.Tensor(obs_mean)
        policy.obs_std = critic.obs_std = torch.Tensor(obs_std)

    
    policy.train()
    critic.train()

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    # create a tensorboard logging object
    logger = create_logger(args)

    algo = PPO(args=vars(args), env_fn=env_fn, policy=policy, critic=critic, save_path=logger.dir)

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print(" ├ recurrent:      {}".format(args.recurrent))
    print(" ├ run name:       {}".format(args.save_name))
    print(" ├ max traj len:   {}".format(args.max_traj_len))
    print(" ├ seed:           {}".format(args.seed))
    print(" ├ num worker:      {}".format(args.num_worker))
    print(" ├ lr:             {}".format(args.lr))
    print(" ├ eps:            {}".format(args.eps))
    print(" ├ lam:            {}".format(args.lam))
    print(" ├ gamma:          {}".format(args.gamma))
    print(" ├ learn stddev:  {}".format(args.learn_stddev))
    print(" ├ std_dev:        {}".format(args.std_dev))
    print(" ├ entropy coeff:  {}".format(args.entropy_coeff))
    print(" ├ clip:           {}".format(args.clip))
    print(" ├ minibatch size: {}".format(args.minibatch_size))
    print(" ├ epochs:         {}".format(args.epochs))
    print(" ├ num steps:      {}".format(args.num_steps))
    print(" ├ use gae:        {}".format(args.use_gae))
    print(" ├ max grad norm:  {}".format(args.max_grad_norm))
    print(" └ max traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(args.n_itr, logger=logger, anneal_rate=args.anneal)
