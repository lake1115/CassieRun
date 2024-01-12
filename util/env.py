import os
import time 
import torch
import numpy as np
import ray
from cassierun_env import CassieRunEnv
def env_factory(args, verbose=False, **kwargs):
    from functools import partial

    env_fn = partial(CassieRunEnv, task=args.task, traj=args.traj, speed=args.speed, clock_based=args.clock_based, state_est=args.state_est, dynamics_randomization=args.dynamics_randomization, history=args.history, impedance=args.impedance)
    if args.mirror:
      env_fn = partial(SymmetricEnv, env_fn, mirrored_obs=env_fn().mirrored_obs, mirrored_act=env_fn().mirrored_acts)

    
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
        state, reward, done, info = self.env.step(action[0])
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self):
        return np.array([self.env.reset()])

class SymmetricEnv:    
    def __init__(self, env_fn, mirrored_obs=None, mirrored_act=None, obs_fn=None, act_fn=None):

        assert (bool(mirrored_act) ^ bool(act_fn)) and (bool(mirrored_obs) ^ bool(obs_fn)), \
            "You must provide either mirror indices or a mirror function, but not both, for \
             observation and action."

        if mirrored_act:
            self.act_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_act))

        elif act_fn:
            assert callable(act_fn), "Action mirror function must be callable"
            self.mirror_action = act_fn

        if mirrored_obs:
            self.obs_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_obs))

        elif obs_fn:
            assert callable(obs_fn), "Observation mirror function must be callable"
            self.mirror_observation = obs_fn

        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def mirror_action(self, action):
        return action @ self.act_mirror_matrix.to(action.device)

    def mirror_observation(self, obs):
        return obs @ self.obs_mirror_matrix.to(obs.device)

    # To be used when there is a clock in the observation. In this case, the mirrored_obs vector inputted
    # when the SymmeticEnv is created should not move the clock input order. The indices of the obs vector
    # where the clocks are located need to be inputted.
    def mirror_clock_observation(self, obs, clock_inds):
        # print("obs.shape = ", obs.shape)
        # print("obs_mirror_matrix.shape = ", self.obs_mirror_matrix.shape)
        mirror_obs = obs @ self.obs_mirror_matrix.to(obs.device)
        clock = mirror_obs[:, self.clock_inds]
        # print("clock: ", clock)
        for i in range(np.shape(clock)[1]):
            mirror_obs[:, clock_inds[i]] = np.sin(np.arcsin(clock[:, i]) + np.pi)
        return mirror_obs
    def mirror_phase_observation(self, obs, ref_inds):
        mirror_obs = obs @ self.obs_mirror_matrix.to(obs.device)
        phase = mirror_obs[:, self.ref_inds]
        for i in range(np.shape(phase)[1]):
            mirror_obs[:, ref_inds[i]] = (phase[:,i]-1 + 14) % 28 +1 
        return mirror_obs

def _get_symmetry_matrix(mirrored):
    numel = len(mirrored)
    mat = np.zeros((numel, numel))

    for (i, j) in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])

    return mat



@ray.remote
def _run_random_actions(iter, policy, env_fn, noise_std):

    env = WrapEnv(env_fn)
    states = np.zeros((iter, env.observation_space.shape[0]))

    state = env.reset()
    for t in range(iter):
        states[t, :] = state

       # state = torch.Tensor(state).to(device)
        state = torch.Tensor(state)
        #action = policy(state).to("cpu")
        action = policy(state)
        # add gaussian noise to deterministic action
        action = action + torch.randn(action.size()) * noise_std

        state, _, done, _ = env.step(action.data.numpy())

        if done:
            state = env.reset()
    
    return states

def get_normalization_params(iter, policy, env_fn, noise_std, procs=4):
    print("Gathering input normalization data using {0} steps, noise = {1}...".format(iter, noise_std))

    states_ids = [_run_random_actions.remote(iter // procs, policy, env_fn, noise_std) for _ in range(procs)]

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