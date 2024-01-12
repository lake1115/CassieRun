#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   meta.py
@Time    :   2023/04/19 16:21:36
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''
from cassierun_env_meta import *
import ray
import os, sys
import torch
from policies.mlp import MlpEncoder, RecurrentEncoder
from policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from policies.critic import FF_V, LSTM_V
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from algos.collect import Collect_Worker
from algos.buffer import Buffer, merge_buffers, merge_meta_buffers
from torch.distributions import kl_divergence
from einops import repeat, rearrange
def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=1)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=1)
    return mu, sigma_squared




class Meta_Agent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 args
                 #**kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        #self.recurrent = kwargs['recurrent']
        #self.device = kwargs['device']
        self.recurrent = args.recurrent
        self.device = args.device
        self.use_ib = True
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self.latent_dim).to(self.device)
        var = torch.ones(num_tasks, self.latent_dim).to(self.device)

        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        # self.context_encoder.reset(num_tasks)

    def diffusion_context(self, inputs):

        self.update_context(inputs)

        params = self.context_encoder(self.context)  ## RNN , output 10dim mu, sigma  每次都得reset
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        mus = torch.stack([self.z_means, mu], dim=1)
        sigma_squareds = torch.stack([self.z_vars, sigma_squared],dim=1)
        z_params = _product_of_gaussians(mus, sigma_squareds)
        self.z_means = z_params[0]
        self.z_vars = z_params[1]

        self.sample_z()

    # def infer_posterior(self, transition):
    #     ''' compute q(z|c) as a function of input context and sample new z from it'''
    #     params = self.context_encoder(transition)
    #     params = params.unsqueeze(0)
    #     # with probabilistic z, predict mean and variance of q(z | c)
    #     if self.use_ib:
    #         mu = params[..., :self.latent_dim]
    #         sigma_squared = F.softplus(params[..., self.latent_dim:])
    #         z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))] ## mu 累乘， 
    #         self.z_means = torch.stack([p[0] for p in z_params])
    #         self.z_vars = torch.stack([p[1] for p in z_params])
    #     # sum rather than product of gaussians structure
    #     else:
    #         self.z_means = torch.mean(params, dim=1)
    #     self.sample_z()

    def get_z(self):
        return self.z_means, self.z_vars
    def set_z(self, z):
        self.z_means = z[:self.latent_dim]
        self.z_vars = z[self.latent_dim:]
        self.sample_z()

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no = inputs
        o = torch.Tensor(o[None, ...]).to(self.device)
        a = torch.Tensor(a[None, ...]).to(self.device)
        r = torch.Tensor(np.array([r])[None, ...]).to(self.device)
        no = torch.Tensor(no[None, ...]).to(self.device)
        data = torch.cat([o, a, r, no], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=0)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim).to(self.device), torch.ones(self.latent_dim).to(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_policy(self, obs_batch, transition_batch, policy):
        self.clear_z(obs_batch.shape[1])
        seq_len, batch_size, _ = obs_batch.size()
        params = self.context_encoder(transition_batch, return_last=False) ## 这里直接输入transition是不是就行了，出来直接就是各种z，然后直接得到多个 mu 和sigma
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        mus = torch.cat([self.z_means.unsqueeze(0), mu], dim=0)
        sigma_squareds = torch.cat([self.z_vars.unsqueeze(0), sigma_squared],dim=0)
        mus = rearrange(mus, 's b f -> b s f')
        sigma_squareds = rearrange(sigma_squareds, 's b f -> b s f')
        policy_z = torch.zeros([seq_len, batch_size, self.latent_dim]).to(self.device)
        for i in range(seq_len):
            mus_params, sigma_params = _product_of_gaussians(mus[:,:(i+1),:], sigma_squareds[:,:(i+1),:])
            self.z_means = mus_params
            self.z_vars = sigma_params
            self.sample_z()
            policy_z[i] = self.z
        
        in_ = torch.cat([obs_batch, policy_z], dim=-1)
        policy_pdf = self.policy.distribution(in_)
        return policy_pdf, policy_z
    # def infer_policy(self, obs_batch, context_batch, policy):
    #     in_ = torch.cat([obs_batch, context_batch], dim=-1)
    #     pdf = policy.distribution(in_)
    #     return pdf
    def sample_z(self):

        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        z = [d.rsample() for d in posteriors]
        self.z = torch.stack(z)

    def sample(self, mu, var):
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(mu), torch.unbind(var))]
        z = [d.rsample() for d in posteriors]
        return torch.stack(z)
    
    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = obs.to(self.device)
        in_ = torch.cat([obs, z], dim=1)
        return self.policy(in_, deterministic=deterministic)

    def forward(self, obs, transition, distribution=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        if transition is not None:
            self.diffusion_context(transition)
   
        task_z = self.z

        # b, _ = obs.size()    
        # task_z = repeat(task_z, 'b f -> (repeat b) f', repeat=b)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        if distribution:
            policy_outputs = self.policy.distribution(in_)
        else:
            policy_outputs = self.policy(in_, deterministic=True)

        return policy_outputs


    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(torch.get_numpy(self.z_means[0]).to(self.device)))
        z_sig = np.mean(torch.get_numpy(self.z_vars[0]).to(self.device))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]

class Meta_PPO:
    def __init__(self, args, meta_env, tasks, meta_policy,critic, save_path):
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
        self.num_worker     = args['num_worker']
        self.grad_clip      = args['max_grad_norm']
        self.recurrent      = args['recurrent']
        self.device         = args['device']
        self.n_itr          = args['n_itr']
        self.n_tasks        = args['n_tasks']
        self.kl_lambda      = args['kl_lambda']
        self.train_tasks = list(tasks[:args['n_train_tasks']])
        self.test_tasks = list(tasks[-args['n_eval_tasks']:])
        
        self.num_traj_sample= args['num_traj_sample']

        self.meta_env = meta_env
        self.meta_policy = meta_policy
        self.old_policy = deepcopy(self.meta_policy.policy)
        self.critic = critic
        self.save_path = save_path

        self.total_steps = 0 
        self.highest_reward = -1
        self.update_posterior_rate = np.inf


        self.actor_optimizer = optim.Adam(self.meta_policy.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)
        self.context_optimizer = optim.Adam(self.meta_policy.context_encoder.parameters(), lr=self.lr)

        self.workers = [Collect_Worker(self.meta_policy, self.critic, self.meta_env.env_fn, self.gamma, self.lam) for _ in range(self.num_worker)]
        #self.workers = [Collect_Worker.remote(self.meta_policy, self.critic, self.meta_env.env_fn, self.gamma, self.lam) for _ in range(self.num_worker)]
        ## separate replay buffers for training RL update and training encoder update
        self.replay_buffer = [Buffer(self.gamma, self.lam) for _ in range(self.n_tasks)]
        #self.encoder_buffer = [Buffer(self.gamma, self.lam) for _ in range(self.n_tasks)]
        self.context_buffer = [torch.zeros(self.num_traj_sample, 5) for _ in range(self.n_tasks)]
        self.task_mu = [torch.zeros(5) for _ in range(self.n_tasks)]
        self.task_var = [torch.zeros(5) for _ in range(self.n_tasks)]


    @property
    def networks(self):
        return self.meta_policy.networks + [self.meta_policy] + [self.critic]

    def to(self, device=None):
        if device == None:
            device = self.device
        for net in self.networks:
            net.to(device)
    def save(self):
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(self.meta_policy.policy, os.path.join(self.save_path, "actor" + filetype))
        torch.save(self.critic, os.path.join(self.save_path, "critic" + filetype))
        torch.save(self.meta_policy.context_encoder, os.path.join(self.save_path, "context_encoder" + filetype))

    # def sample_context(self, task_idx):
    #     ''' sample batch of context from a list of tasks from the replay buffer '''
    #     # make method work given a single task index, sample a traj context
    #     context = []
    #     for batch in self.encoder_buffer[task_idx].sample(batch_size=self.minibatch_size, recurrent=self.recurrent):
    #         o, a, r, _, _ = batch
    #         context.append(torch.cat([o,a,r],dim=1))
    #     context = torch.cat(context)
    #     return context
    
    def sync_policy(self, new_actor_params, new_critic_params):
        for p, new_p in zip(self.meta_policy.policy.parameters(), new_actor_params):
            p.data.copy_(new_p)

        for p, new_p in zip(self.critic.parameters(), new_critic_params):
            p.data.copy_(new_p)   

    def collect_data(self, task_idx, num_traj_sample, update_posterior_rate, add_to_encoder_buffer=True):
        self.meta_env.reset_task(task_idx)
        #task = self.meta_env.get_task()['speed']
        task = self.meta_env._goal_vel
        #if add_to_encoder_buffer:
        buffers_c = [w.meta_collect(max_traj_len = self.max_traj_len, min_steps = num_traj_sample// self.num_worker,task=task, anneal=1.0) for w in self.workers]


        buffers = merge_meta_buffers([buffers_c[i][0] for i in range(self.num_worker)]) ## 100 traj 
        policy_z = torch.cat([buffers_c[i][1] for i in range(self.num_worker)]) ## (100, 5) context for task_idx
        

        self.replay_buffer[task_idx] = buffers
        self.context_buffer[task_idx] = policy_z

        
       # buffers,context = ray.get([w.meta_collect.remote(max_traj_len = self.max_traj_len, min_steps = num_traj_sample// self.num_worker,task=task, anneal=1.0) for w in self.workers])
    
        #policy_z = np.mean(policy_z,0) # mean and var tend to 0, z ~ N(0, 1)
        # if add_to_encoder_buffer:
        #     self.encoder_buffer[task_idx] = buffers
        # else:
        #     self.replay_buffer[task_idx] = buffers
        # if update_posterior_rate:
        #     context = self.sample_context(task_idx)
        #     self.meta_policy.infer_posterior(context) #update policy.z
        #     self.task_mu[task_idx], self.task_var[task_idx] = self.meta_policy.get_z()

    def update_policy(self, context_batch, obs_batch, action_batch, return_batch, advantage_batch, mask,task_idx, mirror_observation=None, mirror_action=None):
        ## input transition_batch output pdf and task_z
        with torch.no_grad():
            old_pdf, _= self.meta_policy.infer_policy(obs_batch, context_batch, self.old_policy)
            old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        pdf, task_z = self.meta_policy.infer_policy(obs_batch, context_batch, self.meta_policy.policy)
        values = self.critic(torch.cat([obs_batch, task_z.detach()],dim=-1))
        
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)

        ratio = (log_probs - old_log_probs).exp()

        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()
  
        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        if mirror_observation is not None and mirror_action is not None:
            self.meta_env.reset_task(task_idx)
            env = self.meta_env.env 
            deterministic_actions = self.policy(obs_batch)
            if env.clock_based:
                if self.recurrent:
                    mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:], env.clock_inds) for i in range(obs_batch.shape[0])])
                    mirror_actions = self.policy(mir_obs)
                else:
                    mir_obs = mirror_observation(obs_batch, env.clock_inds)
                    mirror_actions = self.policy(mir_obs)
            else:
                if self.recurrent:
                    mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:], env.ref_inds) for i in range(obs_batch.shape[0])])
                    mirror_actions = self.policy(mir_obs)
                else:
                    mir_obs = mirror_observation(obs_batch, env.ref_inds)
                    mirror_actions = self.policy(mir_obs)

            mirror_actions = mirror_action(mirror_actions)
            mirror_loss = 0.4 * (deterministic_actions - mirror_actions).pow(2).mean()
        else:
            mirror_loss = 0

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.context_optimizer.zero_grad()

        # KL constraint on z if probabilistic
        kl_div = self.meta_policy.compute_kl_div() ## 产生的z与N(0,1)很接近
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        policy_loss = actor_loss + mirror_loss + entropy_penalty
        policy_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(self.meta_policy.policy.parameters(), self.grad_clip)

        critic_loss = 0.5 * ((return_batch - values) * mask).pow(2).mean()
        critic_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)


        with torch.no_grad():
            kl = kl_divergence(pdf, old_pdf)



        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.context_optimizer.step()

        if mirror_observation is not None and mirror_action is not None:
            mirror_loss_return = mirror_loss.item()
        else:
            mirror_loss_return = 0
        return actor_loss.item(), pdf.entropy().mean().item(), critic_loss.item(), ratio.mean().item(), kl_loss.item(), kl.mean().item()
    
    def train(self, logger=None):
        '''
        meta-training loop
        '''

        for itr in range(self.n_itr):
            self.old_policy.load_state_dict(self.meta_policy.policy.state_dict())
            self.meta_policy.policy = self.meta_policy.policy.to(self.device)
            self.critic = self.critic.to(self.device)
            self.old_policy = self.old_policy.to(self.device)           
            for idx in self.train_tasks:
                # 1. Sample train data by context 
                self.collect_data(idx, self.num_traj_sample, update_posterior_rate=False, add_to_encoder_buffer=False)

            training_buffer = merge_meta_buffers([self.replay_buffer[i] for i in range(len(self.train_tasks))])
            total_steps = len(training_buffer)
            self.total_steps += total_steps // len(self.train_tasks)
            for epoch in range(self.epochs):
                # 2. Contrastive Learning for context TODO
                #context_batch = self.sample_context(idx)
                #self.meta_policy.clear_z(num_tasks=len(idx))
              
                # 3. Training task policy
                losses = []
                # sample all replay_buffer to get training sample, 
                for batch in training_buffer.meta_sample(self.minibatch_size,self.recurrent):
                    obs_batch, action_batch, return_batch, advantage_batch, context_batch, mask = batch
                    obs_batch, action_batch, return_batch, advantage_batch, context_batch, mask = obs_batch.to(self.device), action_batch.to(self.device), return_batch.to(self.device), advantage_batch.to(self.device),context_batch.to(self.device), mask.to(self.device)

                    transition_batch = torch.cat([obs_batch[:-1,:,:], action_batch[:-1,:,:], return_batch[:-1,:,:], obs_batch[1:,:,:]],dim=2)

                    scalars = self.update_policy(transition_batch, obs_batch, action_batch, return_batch, advantage_batch, mask, idx)
                    actor_loss, entropy, critic_loss, ratio, kl_loss, kl = scalars
                    # kl_div = self.meta_policy.compute_kl_div() ## 产生的z与N(0,1)很接近
                    # kl_loss = self.kl_lambda * kl_div
                    # kl_loss.backward(retain_graph=True)

                    losses.append([actor_loss, entropy, critic_loss, ratio, kl_loss, kl])
                    mean_losses = np.mean(losses, axis=0)

                # Early stopping
                if np.mean(kl) > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

            print("timestep {:d} | actor loss {:6.3f}, critic loss {:6.3f}, kl div {:6.3f}, entropy {:6.3f}.".format(itr, mean_losses[0], mean_losses[2], mean_losses[4], mean_losses[1]))
            ## sync_policy ##
            # policy_param_id  = ray.put(list(worker_policy.parameters()))
            # critic_param_id = ray.put(list(worker_critic.parameters()))
            policy_param_id  = list(self.meta_policy.parameters())
            critic_param_id = list(self.critic.parameters())
            for w in self.workers:
                # w.sync_policy.remote(policy_param_id, critic_param_id)
                w.sync_policy(policy_param_id, critic_param_id)
            ## eval_policy ##
            eval_reward = []
            for idx in self.test_tasks:
                self.meta_env.reset_task(idx)
                task = self.meta_env._goal_vel
                #task = self.meta_env.get_task()['speed']
                eval_reward.append(np.mean([w.meta_evaluate(max_traj_len=self.max_traj_len,task=task, trajs=1) for w in self.workers]))
            if logger is not None:

                avg_eval_reward = np.mean(eval_reward)
                avg_batch_reward = np.mean(training_buffer.ep_returns)
                avg_ep_len = np.mean(training_buffer.ep_lens)
                

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
                logger.add_scalar("Train/Mean KL Div", kl_loss, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)

                logger.add_scalar("Misc/Critic Loss", mean_losses[2], itr)
                logger.add_scalar("Misc/Actor Loss", mean_losses[0], itr)
                #logger.add_scalar("Misc/Mirror Loss", mean_losses[5], itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                print("Save best policy!")
                self.save()


def run_experiment(args):
    from util.env import env_factory
    from util.log import create_logger

    #env = ENVS[args.env_name](args.n_tasks)
    env_fn = env_factory(args)
    meta_env = env_fn()
    tasks = meta_env.get_all_task_idx()
    print("tasks:", meta_env.tasks)
    # wrapper function for creating parallelized envs
    #env_fn = env_factory(args)
    #obs_dim = meta_env.observation_space.shape[0]
    obs_dim = meta_env.env._get_obs().shape[0]
    action_dim = meta_env.action_space.shape[0]
    reward_dim = 1
    latent_dim = args.latent_size
    context_encoder_input_dim = obs_dim*2 + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 # mu, std


    # Set up Parallelism
    if not ray.is_initialized():  
        os.environ['OMP_NUM_THREADS'] = '1'
        ray.init(num_cpus=args.num_worker)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        pass #TODO:
    else:
        #if args.recurrent:
            encoder = RecurrentEncoder(input_size = context_encoder_input_dim, output_size = context_encoder_output_dim, hidden_sizes=args.env_layers)     
        #else:
        #    encoder = MlpEncoder(input_size =context_encoder_input_dim,  output_size =context_encoder_output_dim, hidden_sizes=args.env_layers)
    
    # ppo actor and critic
    policy = Gaussian_FF_Actor(obs_dim + latent_dim, action_dim, layers=args.layers, fixed_std=np.exp(args.std_dev), env_name=args.env_name, bounded=args.bounded)
    meta_agent = Meta_Agent(latent_dim, encoder, policy, args)
    critic = FF_V(obs_dim + latent_dim, layers=args.layers)
    
    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    # create a tensorboard logging object
    logger = create_logger(args)

    algo = Meta_PPO(args=vars(args), meta_env=meta_env, tasks=tasks, meta_policy=meta_agent, critic = critic, save_path=logger.dir)

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

    algo.train(logger=logger)
