#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cassierun_env_meta.py
@Time    :   2023/03/29 15:43:35
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np

# ENVS = {}
# def register_env(name):
#     """Registers a env by name for instantiation in rlkit."""

#     def register_env_fn(fn):
#         if name in ENVS:
#             raise ValueError("Cannot register duplicate env {}".format(name))
#         if not callable(fn):
#             raise TypeError("env {} must be callable".format(name))
#         ENVS[name] = fn
#         return fn

#     return register_env_fn

# class RandomEnv(CassieRunEnv):
#     """
#     This class provides functionality for randomizing the physical parameters of a mujoco model
#     The following parameters are changed:
#         - get_body_mass
#         - body_inertia
#         - damping coeff at the joints
#     """
#     #RAND_PARAMS = ['body_mass', 'dof_damping',  'geom_friction','body_inertia']
#     RAND_PARAMS = ['speed']


#     def __init__(self, log_scale_limit=3.0, *args, rand_params=RAND_PARAMS, **kwargs):
#         CassieRunEnv.__init__(self)
#         self.log_scale_limit = log_scale_limit            
#         self.rand_params = rand_params
#         self.save_parameters()

#     def sample_tasks(self, n_tasks):
#         """
#         Generates randomized parameter sets for the mujoco env

#         Args:
#             n_tasks (int) : number of different meta-tasks needed

#         Returns:
#             tasks (list) : an (n_tasks) length list of tasks
#         """
#         param_sets = []

#         for _ in range(n_tasks):
#             # body mass -> one multiplier for all body parts

#             new_params = {}

#             if 'body_mass' in self.rand_params:
#                 body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.sim.get_body_mass().shape)
#                 new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

#             # # body_inertia
#             # if 'body_inertia' in self.rand_params:
#             #     body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=len(self.sim.centroid_inertia()))
#             #     new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

#             # damping -> different multiplier for different dofs/joints
#             if 'dof_damping' in self.rand_params:
#                 dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.sim.get_dof_damping().shape)
#                 new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

#             # friction at the body components
#             if 'geom_friction' in self.rand_params:
#                 dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.default_fric.shape)
#                 new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

#                 # fric_noise = []
#                 # translational = np.random.uniform(self.fric_low, self.fric_high)
#                 # torsional = np.random.uniform(1e-4, 5e-4)
#                 # rolling = np.random.uniform(1e-4, 2e-4)

#                 # for _ in range(int(len(self.default_fric)/3)):
#                 #     fric_noise += [translational, torsional, rolling]
#                 # new_params['geom_friction'] = np.clip(fric_noise, 0, None)
#             if 'speed' in self.rand_params:
#                 speed = np.random.uniform(0.0, 1.0)
#                 new_params['speed'] = speed

#             param_sets.append(new_params)

#         return param_sets

#     def set_task(self, task):
#         for param, param_val in task.items():
#             if 'body_mass' in param:
#                 self.sim.set_body_mass(param_val)
#             if 'dof_damping' in param:
#                 self.sim.set_dof_damping(param_val)
#             if 'geom_friction' in param:
#                 self.sim.set_geom_friction(param_val)
#             if 'speed' in param:
#                 self.speed = param_val

#         self.cur_params = task

#     def get_task(self):
#         return self.cur_params

#     def save_parameters(self):
#         self.init_params = {}
#         if 'body_mass' in self.rand_params:
#             self.init_params['body_mass'] = self.sim.get_body_mass()

#         # # body_inertia
#         # if 'body_inertia' in self.rand_params:
#         #     self.init_params['body_inertia'] = self.sim.centroid_inertia()

#         # damping -> different multiplier for different dofs/joints
#         if 'dof_damping' in self.rand_params:
#             self.init_params['dof_damping'] = self.sim.get_dof_damping()

#         # friction at the body components
#         if 'geom_friction' in self.rand_params:
#             self.init_params['geom_friction'] = self.sim.get_geom_friction()
#         if 'speed' in self.rand_params:
#             self.init_params['speed'] = self.speed
#         self.cur_params = self.init_params


# @register_env('CassieRun-v0_meta')
# class CassieRunMetaEnv(RandomEnv):

#     def __init__(self, n_tasks, log_scale_limit=3.0):
#         RandomEnv.__init__(self, log_scale_limit)
#         self.tasks = self.sample_tasks(n_tasks)
#         self.reset_task(0)

#     def get_all_task_idx(self):
#         return range(len(self.tasks))

#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         self._goal = idx # assume parameterization of task by single vector
#         self.set_task(self._task)
#         self.reset()



class CassieRunMetaEnv:
    def __init__(self, env_fn, n_tasks, rand_params=['speed'], log_scale_limit=3.0):
        self.env_fn = env_fn
        self.env = env_fn()
        self.rand_params = rand_params
        self.log_scale_limit = log_scale_limit  
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
   
        self.sim = self.env.sim
        self.save_parameters()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx # assume parameterization of task by single vector
        self.set_task(self._task)
        self.env.reset()

    def set_task(self, task):
        for param, param_val in task.items():
            if 'body_mass' in param:
                self.env.sim.set_body_mass(param_val)
            if 'dof_damping' in param:
                self.env.sim.set_dof_damping(param_val)
            if 'geom_friction' in param:
                self.env.sim.set_geom_friction(param_val)
            if 'speed' in param:
                self.env.speed = param_val

    def get_task(self):
        return self.cur_params
    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []

        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts

            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.sim.get_body_mass().shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.sim.get_dof_damping().shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.default_fric.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

                # fric_noise = []
                # translational = np.random.uniform(self.fric_low, self.fric_high)
                # torsional = np.random.uniform(1e-4, 5e-4)
                # rolling = np.random.uniform(1e-4, 2e-4)

                # for _ in range(int(len(self.default_fric)/3)):
                #     fric_noise += [translational, torsional, rolling]
                # new_params['geom_friction'] = np.clip(fric_noise, 0, None)
            if 'speed' in self.rand_params:
                speed = np.random.uniform(0.0, 1.0)
                new_params['speed'] = speed

            param_sets.append(new_params)

        return param_sets
    
    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.sim.get_body_mass()

        # # body_inertia
        # if 'body_inertia' in self.rand_params:
        #     self.init_params['body_inertia'] = self.sim.centroid_inertia()

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.sim.get_dof_damping()

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.sim.get_geom_friction()
        if 'speed' in self.rand_params:
            self.init_params['speed'] = self.env.speed
        self.cur_params = self.init_params
if __name__ == "__main__":
    from functools import partial
    from cassierun_env import CassieRunEnv
    env_fn = partial(CassieRunEnv,speed=1.0)
    aa = CassieRunMetaEnv(env_fn, n_tasks = 10)
    tasks = aa.tasks

    
    while True:
        aa.env.reset()
        aa.set_task(np.random.choice(tasks))
        env = aa.env
        print(env.speed)
        for _ in range(1):
            env.render()
            env.step(env.action_space.sample())  # take a random action
