from cassie_m.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from cassie_m.trajectory.trajectory import CassieTrajectory
from cassie_m.udp import euler2quat, quaternion_product, inverse_quaternion, quaternion2euler, rotate_by_quaternion
import os

from math import floor
import gym
from gym.spaces import Box
import numpy as np 

import random


class CassieRunEnv(gym.Env):
    """
    Cassie Env
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 60,
    }
    def __init__(self,simrate=60,
                 traj = None,
                 state_est=False,
                 clock_based=False,
                 dynamics_randomization=False,
                 history=0,
                 impedance=False,
                 speed=0,
                 render_mode=None,
                 visual = False,
                  **kwargs):
      dirname = os.path.dirname(__file__)
      self.config = dirname + '/cassie_m/model/cassie.xml'
      self.sim = CassieSim(self.config)
      self.visual = visual
      if self.visual:
        self.vis = CassieVis(self.sim)
      else:
        self.vis = None

      self.clock_based = clock_based
      self.dynamics_randomization = dynamics_randomization
      self.termination = False
      self.state_est = state_est
      self.record_forces = False

      clock_size     = 2  # sin and cos
      speed_size     = 2  # speed and side speed
      if state_est:
        self._obs = 38 + speed_size   # 20 motor's pos and vel + 8 joint's pos and vel + 10 pelvis
      else:
        #self._obs = 67 + speed_size  # 35 pos + 32 vel
        self._obs = 40 +1 # 20 pos + 20 vel + idx
      if clock_based: # Use clock inputs
        self._obs += clock_size
      
      # Adds option for state history for FF nets
      self.history = history

      self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self._obs + self._obs * self.history, ), dtype=np.float64)

      if impedance:   
        self.action_space = Box(low=-1, high=1, shape=(20,),dtype=np.float32)
      else:
        self.action_space = Box(low=-1, high=1, shape=(10,),dtype=np.float32)

      self.last_action = np.zeros(self.action_space.shape)
      if traj == "walking":
        traj_path = os.path.join(dirname, "cassie_m", "trajectory", "walking-trial.bin")
        self.trajectory = CassieTrajectory(traj_path)
      elif traj == "stepping":
        traj_path = os.path.join(dirname, "cassie_m", "trajectory", "stepping-trial.bin")
        self.trajectory = CassieTrajectory(traj_path)
      else:
        print("without trajectory path")
        self.trajectory = None
      self.len_traj = 1680


      

      # PD control
      self.P = np.array([100,  100,  88,  96,  50])  
      self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])


      self.u = pd_in_t() # action
      self.foot_pos = np.zeros(6)
      self.foot_quat = np.zeros(4)
      self.foot_force = np.zeros(6)
      self.init_foot_quat = np.array(
            [-0.24135508, -0.24244352, -0.66593612, 0.66294642]
        )


      self.cassie_state = state_out_t() 
      self.simrate = simrate # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
      self.time    = 0 # number of time steps in current episode
      self.phase   = 0 # portion of the phase the robot is in
      self.counter = 0 # number of phase cycles completed in episode
      self.time_limit = 600

      #motor: see include/cassiemujoco.h for meaning of these indices
      self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]  ## motor position index 
      self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]  ## motor velocities index

      self.j_pos_idx = [15, 16, 20, 29, 30, 34]  ## joint position index 
      self.j_vel_idx = [13, 14, 18, 26, 27, 31]  ## joint velocities index

      self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
      self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

      self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968]) ## 10个motor的偏移量

      # Dynamics Randomization
      self.max_orient_change = 0.1

      self.max_speed = 10
      self.min_speed = -0.1

      self.max_side_speed = 0.25
      self.min_side_speed = -0.25

      self.max_step_freq = 2.0
      self.min_step_freq = 0.9

      self.max_pitch_incline = 0.03
      self.max_roll_incline = 0.03

      self.encoder_noise = 0.01

      self.damping_low = 0.2
      self.damping_high = 5.5

      self.mass_low = 0.3
      self.mass_high = 1.7

      self.fric_low = 0.23
      self.fric_high = 1.1

      self.speed      = speed  # x-axis
      self.side_speed = 0  # y-axis
      self.orient_add = 0

      self.random_orientation = False
      self.random_speed = False
      self.random_side_speed = False
      self.random_step_freq = False

      # Record default dynamics parameters
      self.default_damping = self.sim.get_dof_damping()
      self.default_mass = self.sim.get_body_mass()
      self.default_ipos = self.sim.get_body_ipos()
      self.default_fric = self.sim.get_geom_friction()
      self.default_rgba = self.sim.get_geom_rgba()
      self.default_quat = self.sim.get_geom_quat()

      self.motor_encoder_noise = np.zeros(10)
      self.joint_encoder_noise = np.zeros(6)

      # rew_buf
      self.reward = 0
      self.reward_buf = 0
      self.time_buf = 0
      self.vel_buf = 0
      assert render_mode is None or render_mode in self.metadata["render_modes"]
      self.render_mode = render_mode
      self.window = None

    def step_simulation(self, action):
      
      if self.record_forces:
        self.sim_foot_frc.append(self.sim.get_foot_forces())

      target = action[:10] + self.offset
      p_add  = np.zeros(10)
      d_add  = np.zeros(10)

      if len(action) > 10:
        p_add = action[10:20]

      if len(action) > 20:
        d_add = action[20:30]

      if self.dynamics_randomization:
        target -= self.motor_encoder_noise
      
      self.u = pd_in_t()
      for i in range(5):
          self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + p_add[i]
          self.u.rightLeg.motorPd.pGain[i] = self.P[i] + p_add[i + 5]

          self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + d_add[i]
          self.u.rightLeg.motorPd.dGain[i] = self.D[i] + d_add[i + 5]

          self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
          self.u.rightLeg.motorPd.torque[i] = 0 

          self.u.leftLeg.motorPd.pTarget[i]  = target[i]
          self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

          self.u.leftLeg.motorPd.dTarget[i]  = 0
          self.u.rightLeg.motorPd.dTarget[i] = 0

      self.cassie_state = self.sim.step_pd(self.u)

    ## step for running
    def step(self, action):

        if self.dynamics_randomization:
          delay_rand = 5
          simrate = self.simrate + np.random.randint(-delay_rand, delay_rand+1) #[a, b)
        else:
          simrate = self.simrate

        if self.record_forces:
          self.sim_foot_frc = []


        self.x_position_before = (
            2.0 * self.sim.qpos()[0]
            + self.sim.xpos("left-foot")[0]
            + self.sim.xpos("right-foot")[0]
        )

        for _ in range(simrate):
            self.step_simulation(action)
            
        self.x_position_after = (
            2.0 * self.sim.qpos()[0]
            + self.sim.xpos("left-foot")[0]
            + self.sim.xpos("right-foot")[0]
        )

        self.time  += 1
        self.phase += self.phase_add
        
        if self.phase >= self.len_traj:   # len(trajectory) = 1682 <=> 28 phase
            self.phase = self.phase % self.len_traj 
            self.counter += 1

        state = self.get_full_state() 
        self.vel = self.qvel[0]
        #self.sim.foot_pos(self.foot_pos)
        #print("speed:", self.vel) 
        height = self.sim.qpos()[2]
        lfcf = np.zeros_like(self.foot_pos)
        self.sim.get_body_contact_force(self.foot_pos,'left-foot')
        lfcf += self.foot_pos
        self.sim.get_body_contact_force(self.foot_pos,'left-plantar-rod')
        lfcf += self.foot_pos
        left_foot_contact_force = np.sum(np.square( lfcf ))
        rfcf = np.zeros_like(self.foot_pos)
        self.sim.get_body_contact_force(self.foot_pos,'right-foot') 
        rfcf += self.foot_pos
        self.sim.get_body_contact_force(self.foot_pos,'right-plantar-rod')
        rfcf += self.foot_pos
        right_foot_contact_force = np.sum(np.square(rfcf))


        if self.visual:
          self.render()

        reward = self.compute_reward(action)
        if (
            self.sim.xpos("left-foot")[2] > 0.4
            or self.sim.xpos("right-foot")[2] > 0.4
            or np.abs(
                self.sim.xpos("left-foot")[0]
                - self.sim.xpos("right-foot")[0]
            )
            > 1.0
        ):  # constraint on step length:
            reward = reward - 20.0
        if left_foot_contact_force < 500.0 and right_foot_contact_force < 500.0:
            reward = reward - 20.0
        done = False
        if height < 0.8:
           reward = reward - 200.0
           done = True
        if self.time >= self.time_limit:
           done = True

        
        ## random changes to orientation
        if self.random_orientation: # random changes to orientation
         self.orient_add += np.random.uniform(-self.max_orient_change, self.max_orient_change)

        if self.random_speed: # random changes to speed
          self.speed = np.random.uniform(self.min_speed, self.max_speed)
          if not self.clock_based:
            new_freq = np.clip(self.speed, self.min_step_freq, self.max_step_freq)
            self.phase_add = int(self.simrate * new_freq)

        if self.random_side_speed: # random changes to sidespeed
          self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        if self.clock_based and self.random_step_freq: # random changes to clock speed
          new_freq = np.random.uniform(self.min_step_freq, self.max_step_freq)
          new_freq = np.clip(new_freq, 0.8 * np.abs(self.speed), None)
          self.phase_add = int(self.simrate * new_freq)

        self.last_action = action
        self.reward += reward
        return state, reward, done, False, {}
  
    def custom_footheight(self):
        h = 0.15
        h1 = max(0, h*np.sin(2*np.pi*self.phase / self.len_traj)-0.2*h) 
        h2 = max(0, h*np.sin(np.pi + 2*np.pi*self.phase / self.len_traj)-0.2*h) 
        return [h1,h2]

    @staticmethod
    def quat_distance(q1, q2):
        return 2.0 * np.arccos(max(min(np.sum(q1 * q2), 1 - 1e-10), -1 + 1e-10))
    
    ## step for running
    def compute_reward(self, action):

        orientation_error = 0


        x_velocity = 2.0 * (self.x_position_after - self.x_position_before) * self.simrate
        quat_error = self.quat_distance(self.init_foot_quat, self.sim.xquat('left-foot'))* 5 \
                    + self.quat_distance(self.init_foot_quat, self.sim.xquat('right-foot'))* 5
        # actual_q = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
        # target_q = [1, 0, 0, 0]
        # orientation_error = 6 * (1 - np.inner(actual_q, target_q) ** 2)


        # if self.last_action is None:
        #   ctrl_penalty = 0
        # else:
        #   ctrl_penalty = sum(np.abs(self.last_action - action)) / len(action)
 
        
        reward = x_velocity - np.abs(self.sim.qpos()[1]) * 10. -quat_error

                #-ctrl_penalty                  

    
        return reward   
 
    def rotate_to_orient(self, vec):
      quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
      iquaternion = inverse_quaternion(quaternion)

      if len(vec) == 3:
        return rotate_by_quaternion(vec, iquaternion)

      elif len(vec) == 4:
        new_orient = quaternion_product(iquaternion, vec)
        if new_orient[0] < 0:
          new_orient = -new_orient
        return new_orient
      
    def _get_obs(self):
        observation = self.get_full_state()  
        return observation
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.time != 0 :
           self.reward_buf = self.reward 
           self.time_buf = self.time
           self.vel_buf = self.vel
       # self.phase = random.randint(0, len(self.trajectory)) # random phase
        self.phase = 0
        self.reward = 0
    
        self.time = 0
        self.counter = 0
        self.termination = False

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]
        
        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping

            pelvis_damp_range = [[damp[0], damp[0]], 
                                [damp[1], damp[1]], 
                                [damp[2], damp[2]], 
                                [damp[3], damp[3]], 
                                [damp[4], damp[4]], 
                                [damp[5], damp[5]]]                 # 0->5

            hip_damp_range = [[damp[6]*self.damping_low, damp[6]*self.damping_high],
                              [damp[7]*self.damping_low, damp[7]*self.damping_high],
                              [damp[8]*self.damping_low, damp[8]*self.damping_high]]  # 6->8 and 19->21

            achilles_damp_range = [[damp[9]*self.damping_low,  damp[9]*self.damping_high],
                                  [damp[10]*self.damping_low, damp[10]*self.damping_high], 
                                  [damp[11]*self.damping_low, damp[11]*self.damping_high]] # 9->11 and 22->24

            knee_damp_range     = [[damp[12]*self.damping_low, damp[12]*self.damping_high]]   # 12 and 25
            shin_damp_range     = [[damp[13]*self.damping_low, damp[13]*self.damping_high]]   # 13 and 26
            tarsus_damp_range   = [[damp[14]*self.damping_low, damp[14]*self.damping_high]]             # 14 and 27

            heel_damp_range     = [[damp[15], damp[15]]]                           # 15 and 28
            fcrank_damp_range   = [[damp[16]*self.damping_low, damp[16]*self.damping_high]]   # 16 and 29
            prod_damp_range     = [[damp[17], damp[17]]]                           # 17 and 30
            foot_damp_range     = [[damp[18]*self.damping_low, damp[18]*self.damping_high]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass
            pelvis_mass_range      = [[self.mass_low*m[1],  self.mass_high*m[1]]]  # 1
            hip_mass_range         = [[self.mass_low*m[2],  self.mass_high*m[2]],  # 2->4 and 14->16
                                      [self.mass_low*m[3],  self.mass_high*m[3]], 
                                      [self.mass_low*m[4],  self.mass_high*m[4]]] 

            achilles_mass_range    = [[self.mass_low*m[5],  self.mass_high*m[5]]]  # 5 and 17
            knee_mass_range        = [[self.mass_low*m[6],  self.mass_high*m[6]]]  # 6 and 18
            knee_spring_mass_range = [[self.mass_low*m[7],  self.mass_high*m[7]]]  # 7 and 19
            shin_mass_range        = [[self.mass_low*m[8],  self.mass_high*m[8]]]  # 8 and 20
            tarsus_mass_range      = [[self.mass_low*m[9],  self.mass_high*m[9]]]  # 9 and 21
            heel_spring_mass_range = [[self.mass_low*m[10], self.mass_high*m[10]]] # 10 and 22
            fcrank_mass_range      = [[self.mass_low*m[11], self.mass_high*m[11]]] # 11 and 23
            prod_mass_range        = [[self.mass_low*m[12], self.mass_high*m[12]]] # 12 and 24
            foot_mass_range        = [[self.mass_low*m[13], self.mass_high*m[13]]] # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            self.pitch_bias = 0.0
            self.roll_bias = 0.0

            delta = 0.00001
            com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

            fric_noise = []
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for _ in range(int(len(self.default_fric)/3)):
              fric_noise += [translational, torsional, rolling]

            geom_plane = [np.random.uniform(-self.max_roll_incline, self.max_roll_incline), np.random.uniform(-self.max_pitch_incline, self.max_pitch_incline), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            geom_quat  = list(quat_plane) + list(self.default_quat[4:])

            self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
            self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
            self.sim.set_geom_quat(geom_quat)
        else:
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_geom_friction(self.default_fric)
            self.sim.set_geom_quat(self.default_quat)

            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

        self.sim.set_const()

        ## set random phase
        # qpos, qvel = self.get_ref_state(self.phase)
        # self.sim.set_qpos(qpos)
        # self.sim.set_qvel(qvel)

        #self.cassie_state = state_out_t() # 全为0了

        self.orient_add = 0
        self.speed      = self.speed #np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = self.side_speed #np.random.uniform(self.min_side_speed, self.max_side_speed)

        if self.clock_based:
          self.phase_add = 60#int(self.simrate * np.random.uniform(self.min_step_freq, self.max_step_freq))
        else:
          #new_freq = np.clip(self.speed, 1, 1.5)
          self.phase_add = 60#int(self.simrate * new_freq)

        #self.last_action = None
        self.last_action = np.zeros(self.action_space.shape)
        return self.get_full_state(), {}

    def get_dynamics(self):
      damping = self.sim.get_dof_damping()
      mass    = self.sim.get_body_mass()
      fric    = self.sim.get_geom_friction()[0]
      quat    = quaternion2euler(self.sim.get_geom_quat())[:2]

      motor_encoder_noise = np.copy(self.motor_encoder_noise)
      joint_encoder_noise = np.copy(self.joint_encoder_noise)

      return np.hstack([fric])

    def get_friction(self):
      return np.hstack([self.sim.get_geom_friction()[0]])
    
    def get_damping(self):
      return np.hstack([self.sim.get_dof_damping()])

    def get_mass(self):
      return np.hstack([self.sim.get_body_mass()])

    def get_quat(self):
      return np.hstack([quaternion2euler(self.sim.get_geom_quat())[:2]])
    
    def get_kin_state(self, phase):
        counter = self.counter
        if phase >= 1682:
            phase = phase % 1682
            counter += 1          
        pose = np.array([0.]*3)        
        vel = np.array([0.]*3)   
        pose[0] = self.speed  * (self.counter * 1682 + self.phase) / 2000
        pose[1] = self.side_speed  * (self.counter * 1682 + self.phase) / 2000
        pose[2] = 1.03 # 
        vel[0] = self.speed
        return pose, vel
    
    def get_ref_state(self, phase):
        counter = self.counter
        if phase >= self.len_traj:
            phase = phase % self.len_traj
            counter += 1

        pos = np.copy(self.trajectory.qpos[phase])
        vel = np.copy(self.trajectory.qvel[phase])
        if self.time != 0:
          ###### Setting variable speed, worse when speed is high #########
          length = (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) 
          ratio = length /(self.len_traj/2000)
          pos[0] = ( length* counter +pos[0])* self.speed/ratio
          
          pos[1] = self.side_speed  * (counter * 1682 + self.phase) / 2000

          vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):
        if self.clock_based:
          clock = [np.sin(2 * np.pi *  self.phase / self.len_traj),
                  np.cos(2 * np.pi *  self.phase / self.len_traj)]
          ext_state = np.concatenate((clock, [self.speed, self.side_speed]))

        else:
          ext_state = np.reshape(self.phase//60+1,-1)#np.concatenate(([self.speed], [self.side_speed]))
        if self.state_est:

          pelvis_quat = self.rotate_to_orient(self.cassie_state.pelvis.orientation)
          pelvis_vel = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
          pelvis_rvel = self.cassie_state.pelvis.rotationalVelocity[:]

          if self.dynamics_randomization:
            motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
            joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise
          else:
            motor_pos = self.cassie_state.motor.position[:]
            joint_pos = self.cassie_state.joint.position[:]

          motor_vel = self.cassie_state.motor.velocity[:]
          joint_vel = self.cassie_state.joint.velocity[:]
          
          # remove double-counted joint/motor positions
          joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
          joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])

          robot_state = np.concatenate([
              pelvis_quat[:],  # pelvis orientation
              motor_pos,       # actuated joint positions
              pelvis_vel,      # pelvis translational velocity
              pelvis_rvel,     # pelvis rotational velocity 
              motor_vel,       # actuated joint velocities
              joint_pos,       # unactuated joint positions
              joint_vel        # unactuated joint velocities
          ])

          state = np.concatenate([robot_state, ext_state])
        else:
          self.qpos = np.copy(self.sim.qpos()) # dim=35
          self.qvel = np.copy(self.sim.qvel()) # dim=32
          
          pos = self.qpos[self.pos_index]
          vel = self.qvel[self.vel_index]

          state = np.concatenate([pos, vel, ext_state])

        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history+1]

        return np.concatenate(self.state_history)

    def render(self):
        return self.vis.draw(self.sim)

    def close(self):
      if self.window is not None:
        self.vis.close()

    def mirror_state(self, state):
      state_est_indices = [0.01, 1, 2, 3,            # pelvis orientation
                          -9, -10, 11, 12, 13,      # left motor pos
                          -4,  -5,  6,  7,  8,      # right motor pos
                          14, -15, 16,              # translational vel
                          -17, 18, -19,             # rotational vel
                          -25, -26, 27, 28, 29,     # left motor vel
                          -20, -21, 22, 23, 24,     # right motor vel 
                          32, 33, 30, 31,           # joint pos
                          36, 37, 34, 35, ]         # joint vel

      return_as_1d = False
      if isinstance(state, list):
        return_as_1d = True
        statedim = len(state)
        batchdim = 1
        state = np.asarray(state).reshape(1, -1)

      elif isinstance(state, np.ndarray):
        if len(state.shape) == 1:
          return_as_1d = True
          state = np.asarray(state).reshape(1, -1)

        statedim = state.shape[-1]
        batchdim = state.shape[0]

      else:
        raise NotImplementedError

      sinclock, cosclock = None, None
      if statedim == 40: # state estimator with no clock and speed
        raise RuntimeError

      elif statedim == 42: # state estimator with clock and speed
        mirror_obs = state_est_indices + [len(state_est_indices) + i for i in range(4)]
        sidespeed  = mirror_obs[-1]
        sinclock   = mirror_obs[-3]
        cosclock   = mirror_obs[-4]

        new_orient       = state[:,:4]
        new_orient       = np.array(list(map(inverse_quaternion, [new_orient[i] for i in range(batchdim)])))
        new_orient[:,2] *= -1

        #if new_orient[:,0] < 0:
        #  new_orient = [-1 * x for x in new_orient]

        mirrored_state = np.copy(state)
        for idx, i in enumerate(mirror_obs):
          if i == sidespeed:
            mirrored_state[:,idx] = -1 * state[:,idx]
          elif i == sinclock or i == cosclock:
            mirrored_state[:,idx] = (np.sin(np.arcsin(state[:,i]) + np.pi))
          else:
            mirrored_state[:,idx] = (np.sign(i) * state[:,abs(int(i))])

        mirrored_state = np.hstack([new_orient, mirrored_state[:,4:]])
        if return_as_1d:
          return np.asarray(mirrored_state)[0]
        else:
          return np.asarray(mirrored_state)

      else:
        raise RuntimeError

    def mirror_action(self, action):
      return_as_1d = False
      if isinstance(action, list):
        return_as_1d = True
        actiondim    = len(action)
        batchdim     = 1
        action       = np.asarray(action).reshape(1, -1)

      elif isinstance(action, np.ndarray):
        if len(action.shape) == 1:
          return_as_1d = True
          action = np.asarray(action).reshape(1, -1)

        actiondim = action.shape[-1]
        batchdim  = action.shape[0]

      else:
        raise NotImplementedError
      mirror_act = np.copy(action)
      
      idxs = [-5, -6, 7, 8, 9, 
            -0.1, -1, 2, 3, 4]

      if actiondim > 10:
        idxs += [-15, -16, 17, 18, 19,
                -10, -11, 12, 13, 14]

      if actiondim > 20:
        idxs += [-25, -26, 27, 28, 29,
                -20, -21, 22, 23, 24]
  
      for idx, i in enumerate(idxs):
        mirror_act[:,idx] = (np.sign(i) * action[:,abs(int(i))])
      if return_as_1d:
        return mirror_act.reshape(-1)
      else:
        return mirror_act
      
