import os
import time 
import torch
import numpy as np
import ray

def env_factory(args, verbose=False, **kwargs):
    from functools import partial
    if args.env_name == "CassieRun-v0":
        from cassierun_env import CassieRunEnv
        env_fn = partial(CassieRunEnv, task=args.task, traj=args.traj, speed=args.speed, clock_based=args.clock_based, state_est=args.state_est, dynamics_randomization=args.dynamics_randomization, history=args.history, impedance=args.impedance)
        if args.mirror:
            from cassierun_env_mirror import CassieRunMirrorEnv
            env_fn = partial(CassieRunMirrorEnv, env_fn, mirrored_obs=env_fn().mirrored_obs, mirrored_act=env_fn().mirrored_acts)
        # if args.meta:
        #     from cassierun_env_meta import CassieRunMetaEnv
        #     env_fn = partial(CassieRunMetaEnv, env_fn, args.n_tasks)
    elif args.env_name == "HalfCheetahMeta":
        from half_cheetah_vel import HalfCheetahVelEnv
        env_fn = partial(HalfCheetahVelEnv,goal_vel=1.0)
        if args.meta:
            from half_cheetah_vel import CheetahMetaEnv
            env_fn = partial(CheetahMetaEnv, env_fn, n_tasks = args.n_tasks)
    else:
        import gym

        def make_env(env_name):
            def _f():
                env = gym.make(env_name)
                return env
            return _f
        env_fn = make_env(args.env_name)   
    
    if verbose:
      print("Created cassie env with arguments:")
      print("\tdynamics randomization: {}".format(args.dynamics_randomization))
      print("\tstate estimation:       {}".format(args.state_est))
      print("\tclock based:            {}".format(args.clock_based))
      print("\timpedance control:      {}".format(args.impedance))

    return env_fn

# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action):
        if action.ndim == 1:
            env_return = self.env.step(action)
        else:
            env_return = self.env.step(action[0])
        if len(env_return) == 4:
            state, reward, done, info = env_return
        else:
            state, reward, done, _, info = env_return
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])


    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        if isinstance(state, tuple):
            ## gym state is tuple type
            return np.array([state[0]])
        else:
            return np.array([state])


#@ray.remote
def _run_random_actions(iter, policy, env_fn, noise_std,device):

    env = WrapEnv(env_fn)
    states = np.zeros((iter, env.observation_space.shape[0]))

    state = env.reset()
    for t in range(iter):
        states[t, :] = state

        state = torch.Tensor(state).to(device)
        # state = torch.Tensor(state)
        action = policy(state).to("cpu")
        # action = policy(state)
        # add gaussian noise to deterministic action
        action = action + torch.randn(action.size()) * noise_std

        state, _, done, _ = env.step(action.data.numpy())

        if done:
            state = env.reset()
    
    return states

def get_normalization_params(iter, policy, env_fn, noise_std, procs=4,device="cpu"):
    print("Gathering input normalization data using {0} steps, noise = {1}...".format(iter, noise_std))

    #states_ids = [_run_random_actions.remote(iter // procs, policy, env_fn, noise_std,device) for _ in range(procs)]
    states_ids = [_run_random_actions(iter // procs, policy, env_fn, noise_std,device) for _ in range(procs)]
    states = []
    for _ in range(procs):
        ready_ids, _ = ray.wait(states_ids, num_returns=1)
        states.extend(ray.get(ready_ids[0]))
        states_ids.remove(ready_ids[0])

    print("Done gathering input normalization data.")

    return np.mean(states, axis=0), np.sqrt(np.var(states, axis=0) + 1e-8)

def eval_policy(policy, min_timesteps=1000, max_traj_len=1000, visualize=True, env=None, verbose=True):
  env_name = env
  with torch.no_grad():
    if env_name is None:
      env = env_factory(policy.env_name)()
    else:
      env = env_factory(env_name)()

    if verbose:
      print("Policy is a: {}".format(policy.__class__.__name__))
    reward_sum = 0
    env.dynamics_randomization = False
    total_t = 0
    episodes = 0

    obs_states = {}
    mem_states = {}

    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0
      eval_reward = 0
      episodes += 1

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      #speeds = [(0, 0), (0.5, 0), (2.0, 0)]
      speeds = list(zip(np.array(range(0, 350)) / 100, np.zeros(350)))
      pelvis_vel = 0
      while not done and timesteps < max_traj_len:
        if (hasattr(env, 'simrate') or hasattr(env, 'dt')) and visualize:
          start = time.time()

        action = policy.forward(torch.Tensor(state)).detach().numpy()
        state, reward, done, _ = env.step(action)
        if visualize:
          env.render()
        eval_reward += reward
        timesteps += 1
        total_t += 1

        if hasattr(policy, 'get_quantized_states'):
          obs, mem = policy.get_quantized_states()
          obs_states[obs] = True
          mem_states[mem] = True
          print(policy.get_quantized_states(), len(obs_states), len(mem_states))

        if visualize:
          if hasattr(env, 'simrate'):
            # assume 30hz (hack)
            end = time.time()
            delaytime = max(0, 1000 / 30000 - (end-start))
            time.sleep(delaytime)

          if hasattr(env, 'dt'):
            while time.time() - start < env.dt:
              time.sleep(0.0005)

      reward_sum += eval_reward
      if verbose:
        print("Eval reward: ", eval_reward)
    return reward_sum / episodes

def interactive_eval(policy_name, env=None):
    from copy import deepcopy
    import termios, sys
    import tty
    import select
    with torch.no_grad():
        policy = torch.load(policy_name)
        m_policy = torch.load(policy_name)
        #args, run_args = self.args, self.run_args
        #run_args = run_args

        print("GOT ENV", env)
        if env is None:
            env_name = policy.env_name
        else:
            env_name = env
        print("env name: ", env_name)

        env = env_factory(env_name)()
        env.dynamics_randomization = False
        env.evaluation_mode = True

        #if self.run_args.pca:
        #    from util.pca import PCA_Plot
        #    pca_plot = PCA_Plot(policy, env)

        #if self.run_args.pds:
        #    from util.pds import PD_Plot
        #    pd_plot = PD_Plot(policy, env)
        #    print("DOING PDS??")

        if hasattr(policy, 'init_hidden_state'):
            policy.init_hidden_state()
            m_policy.init_hidden_state()

        old_settings = termios.tcgetattr(sys.stdin)

        env.render()
        render_state = True
        slowmo = True

        try:
            tty.setcbreak(sys.stdin.fileno())

            state = env.reset()
            env.speed = 0
            env.side_speed = 0
            env.phase_add = 50
            env.period_shift = [0, 0.5]
            #env.ratio = [0.4, 0.6]
            env.eval_mode = True
            done = False
            timesteps = 0
            eval_reward = 0
            mirror = False

            def isData():
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

            while render_state:
            
                if isData():
                    c = sys.stdin.read(1)
                    if c == 'w':
                        env.speed = np.clip(env.speed + 0.1, env.min_speed, env.max_speed)
                    if c == 's':
                        env.speed = np.clip(env.speed - 0.1, env.min_speed, env.max_speed)
                    if c == 'q':
                        env.orient_add -= 0.005 * np.pi
                    if c == 'e':
                        env.orient_add += 0.005 * np.pi
                    if c == 'a':
                        env.side_speed = np.clip(env.side_speed + 0.05, env.min_side_speed, env.max_side_speed)
                    if c == 'd':
                        env.side_speed = np.clip(env.side_speed - 0.05, env.min_side_speed, env.max_side_speed)
                    if c == 'r':
                        state = env.reset()
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                            m_policy.init_hidden_state()
                        print("Resetting environment via env.reset()")
                        env.speed = 0
                        env.side_speed = 0
                        env.phase_add = env.simrate
                        env.period_shift = [0, 0.5]
                    if c == 't':
                        env.phase_add = np.clip(env.phase_add + 1, int(env.simrate * env.min_step_freq), int(env.simrate * env.max_step_freq))
                    if c == 'g':
                        env.phase_add = np.clip(env.phase_add - 1, int(env.simrate * env.min_step_freq), int(env.simrate * env.max_step_freq))
                    if c == 'y':
                        env.height = np.clip(env.height + 0.01, env.min_height, env.max_height)
                    if c == 'h':
                        env.height = np.clip(env.height - 0.01, env.min_height, env.max_height)
                    if c == 'm':
                        mirror = not mirror
                    if c == 'o':
                        # increase ratio of phase 1
                        env.ratio[0] = np.clip(env.ratio[0] + 0.01, 0, env.max_swing_ratio)
                        env.ratio[1] = 1 - env.ratio[0]
                    if c == 'l':
                        env.ratio[0] = np.clip(env.ratio[0] - 0.01, 0, env.max_swing_ratio)
                        env.ratio[1] = 1 - env.ratio[0]
                    if c == 'p':
                        # switch ratio shift from mirrored to matching and back
                        env.period_shift[0] = np.clip(env.period_shift[0] + 0.05, 0, 0.5)
                    if c == ';':
                        env.period_shift[0] = np.clip(env.period_shift[0] - 0.05, 0, 0.5)
                    if c == 'x':
                        if hasattr(policy, 'init_hidden_state'):
                            policy.init_hidden_state()
                            m_policy.init_hidden_state()

                start = time.time()

                if (not env.vis.ispaused()):
                    env.precompute_clock()

                    action   = policy(torch.Tensor(state)).numpy()
                    if hasattr(env, 'mirror_state') and hasattr(env, 'mirror_action'):
                        m_state = env.mirror_state(state)
                        m_action = env.mirror_action(m_policy(m_state).numpy())

                    if mirror:
                        pass
                        action = m_action

                    state, reward, done, _ = env.step(action)

                    eval_reward += reward
                    timesteps += 1
                    qvel = env.sim.qvel()
                    actual_speed = np.linalg.norm(qvel[0:2])

                    #if run_args.pca:
                    #    pca_plot.update(policy)
                    #if run_args.pds:
                    #    pd_plot.update(action, env.l_foot_frc, env.r_foot_frc)
                    print("Mirror: {} | Des. Spd. {:5.2f} | Speed {:5.1f} | Sidespeed {:4.2f} | Heading {:5.2f} | Freq. {:3d} | Coeff {},{} | Ratio {:3.2f},{:3.2f} | RShift {:3.2f},{:3.2f} | Height {:3.2f} \ {:20s}".format(mirror, env.speed, actual_speed, env.side_speed, env.orient_add, int(env.phase_add), *env.coeff, *env.ratio, *env.period_shift, env.height, ''), end='\r')

                render_state = env.render()
                if hasattr(env, 'simrate'):
                    # assume 40hz
                    end = time.time()
                    delaytime = max(0, 1000 / 40000 - (end-start))
                    if slowmo:
                        while(time.time() - end < delaytime*10):
                            env.render()
                            time.sleep(delaytime)
                    else:
                        time.sleep(delaytime)


            print("Eval reward: ", eval_reward)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def train_normalizer(args, policy, min_timesteps, max_traj_len=1000, noise=0.5):
  with torch.no_grad():
    env = env_factory(args)()
    env.dynamics_randomization = False

    total_t = 0
    while total_t < min_timesteps:
      state = env.reset()
      done = False
      timesteps = 0

      if hasattr(policy, 'init_hidden_state'):
        policy.init_hidden_state()

      while not done and timesteps < max_traj_len:
        state = torch.from_numpy(state).float()
        if noise is None:
          action = policy.forward(state, update_norm=True, deterministic=False).numpy()
        else:
          action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
        state, _, done, _ = env.step(action)
        timesteps += 1
        total_t += 1
