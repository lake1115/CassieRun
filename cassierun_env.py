# Consolidated Cassie environment.

__credits__ = ["Bin Hu"]


from cassie_m.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from cassie_m.trajectory.trajectory import CassieTrajectory
from cassie_m.udp import euler2quat, quaternion_product, inverse_quaternion, quaternion2euler, rotate_by_quaternion

from rewards import *
from math import floor
import gym
from gym.spaces import Box
import numpy as np 
import os
import random
import copy

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
                 task = "walking",
                 traj = None,
                 state_est=False,
                 clock_based=False,
                 dynamics_randomization=False,
                 history=0,
                 impedance=False,
                 speed=1.0,
                 period=28,
                 render_mode=None,
                  **kwargs):
        dirname = os.path.dirname(__file__)
        self.config = dirname + '/cassie_m/model/cassie.xml'
        self.sim = CassieSim(self.config)

        self.vis = None

        self.reward_func = task
        self.period = period
        self.clock_based = clock_based
        self.dynamics_randomization = dynamics_randomization
        self.termination = False
        self.state_est = state_est
        self.record_forces = False
        self.time_limit = 400

        
        # Adds option for state history for FF nets
        self.history = history
        self.observation_space, self.ref_inds, self.mirrored_obs = self.set_up_state_space()
        
        if impedance:  
            self.action_space = Box(low=-1, high=1, shape=(30,),dtype=np.float32)
            self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4,
                      -15, -16, 17, 18, 19, -10, -11, 12, 13, 14,
                      -25, -26, 27, 28, 29, -20, -21, 22, 23, 24]
        else:       
            self.action_space = Box(low=-1, high=1, shape=(10,),dtype=np.float32)
            self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]

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
        self.phaselen = self.period * simrate


        # PD control
        self.P = np.array([100,  100,  88,  96,  50])  
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])


        self.u = pd_in_t() # action
        self.foot_pos = np.zeros(6)
        self.foot_quat = np.zeros(4)
        self.foot_force = np.zeros(6)
        # global flat foot orientation, can be useful part of reward function:
        self.init_foot_quat = np.array(
              [-0.24135508, -0.24244352, -0.66593612, 0.66294642]
          )
        # various tracking variables for reward funcs
        self.stepcount = 0
        self.l_high = False  # only true if foot is above 0.2m 
        self.r_high = False
        self.l_swing = False  # these will be true even if foot is barely above ground
        self.r_swing = False
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0

        self.cassie_state = state_out_t() 
        self.simrate = simrate # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode
      

        #motor: see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]  ## motor position index 
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]  ## motor velocities index

        self.j_pos_idx = [15, 16, 20, 29, 30, 34]  ## joint position index 
        self.j_vel_idx = [13, 14, 18, 26, 27, 31]  ## joint velocities index

        self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        # CONFIGURE OFFSET for No Delta Policies
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

        # foot_pos = self.sim.foot_pos() 
        # prev_foot = copy.deepcopy(foot_pos)
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

        # foot_pos = self.sim.foot_pos()
        # self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        # self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
        # foot_forces = self.sim.get_foot_forces()
        # if self.l_high and foot_forces[0] > 0:
        #     self.l_high = False
        #     self.stepcount += 1
        # elif not self.l_high and foot_pos[2] >= 0.2:
        #     self.l_high = True
        # if self.r_high and foot_forces[0] > 0:
        #     self.stepcount += 1
        #     self.r_high = False
        # elif not self.r_high and foot_pos[5] >= 0.2:
        #     self.r_high = True

        # if self.l_swing and foot_forces[0] > 0:
        #     self.l_swing = False
        # elif not self.l_swing and foot_pos[2] >= 0:
        #     self.l_swing = True
        # if self.r_swing and foot_forces[0] > 0:
        #     self.r_swing = False
        # elif not self.r_swing and foot_pos[5] >= 0:
        #     self.r_swing = True
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
        if phase >= self.phaselen:
            phase = phase % self.phaselen
            counter += 1

        pos = np.copy(self.trajectory.qpos[phase])
        vel = np.copy(self.trajectory.qvel[phase])
        if self.time != 0:
            ###### Setting variable speed, worse when speed is high #########
            length = (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) 
            ratio = length /(self.phaselen/2000)
            pos[0] = ( length* counter +pos[0])* self.speed/ratio
            
            pos[1] = self.side_speed  * (counter * 1682 + self.phase) / 2000

            vel[0] *= self.speed

        return pos, vel
    def step(self, action):

        if self.dynamics_randomization:
            delay_rand = 5
            simrate = self.simrate + np.random.randint(-delay_rand, delay_rand+1) #[a, b)
        else:
            simrate = self.simrate

        if self.record_forces:
            self.sim_foot_frc = []

        if self.reward_func == "running":
            self.x_position_before = (
                2.0 * self.sim.qpos()[0]
                + self.sim.xpos("left-foot")[0]
                + self.sim.xpos("right-foot")[0]
            )

        for _ in range(simrate):
            self.step_simulation(action)
            
        if self.reward_func == "running": 
            self.x_position_after = (
                2.0 * self.sim.qpos()[0]
                + self.sim.xpos("left-foot")[0]
                + self.sim.xpos("right-foot")[0]
            )

        self.time  += 1
        self.phase += self.phase_add
        
        if self.phase >= self.phaselen:   # len(trajectory) = 1682 <=> 28 phase
            self.phase = self.phase % self.phaselen 
            self.counter += 1

        state = self.get_full_state() 
        self.vel = self.sim.qvel()[0]
        self.foot_pos = self.sim.foot_pos()
        #print("speed:", self.vel) 
        height = self.sim.qpos()[2]

        if self.reward_func == "running":
            self.termination = height < 0.8
        elif self.reward_func == "walking":
            if self.trajectory is None:
                ref_pos, ref_vel = self.get_kin_state(self.phase)
            else:
                ref_pos, ref_vel = self.get_ref_state(self.phase)

            xpos, ypos, height = self.sim.qpos()[0], self.sim.qpos()[1], self.sim.qpos()[2]
            xtarget, ytarget, ztarget = ref_pos[0], ref_pos[1], ref_pos[2] 
            pos2target = (xpos-xtarget)**2 + (ypos-ytarget)**2 + (height-ztarget)**2
            die_radii = 1 + (self.speed**2 + self.side_speed**2)**0.5
            self.termination = height < 0.6 or height > 1.2 or pos2target > die_radii**2  
        done = self.termination or self.time >= self.time_limit   

        reward = self.compute_reward(action)
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

        return  state, reward, done, {}

    def compute_reward(self, action):
        if self.reward_func == "walking":
            return walking_reward(self, action)
        elif self.reward_func == "running":
            return running_reward(self, action)

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

        #self.cassie_state = self.sim.step_pd(self.u) 

        self.orient_add = 0
        self.speed      = self.speed #np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = self.side_speed #np.random.uniform(self.min_side_speed, self.max_side_speed)

        if self.clock_based:
          self.phase_add = 60#int(self.simrate * np.random.uniform(self.min_step_freq, self.max_step_freq))
        else:
          #new_freq = np.clip(self.speed, 1, 1.5)
          self.phase_add = 60#int(self.simrate * new_freq)

        self.last_action = np.zeros(self.action_space.shape)
        return self.get_full_state()

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
    
    def get_full_state(self):
        if self.clock_based:
            clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                    np.cos(2 * np.pi *  self.phase / self.phaselen)]
            ext_state = np.concatenate((clock, [self.speed, self.side_speed]))

        else:
            ext_state = np.reshape(self.phase//self.simrate + 1,-1)#np.concatenate(([self.speed], [self.side_speed]))
        if self.state_est:
            # Update orientation
            pelvis_quat = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
            pelvis_vel = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
            pelvis_acc = self.rotate_to_orient(self.cassie_state.pelvis.translationalAcceleration[:])
            pelvis_rvel = self.cassie_state.pelvis.rotationalVelocity[:]

            if self.dynamics_randomization:
              motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
              joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise
            else:
              motor_pos = self.cassie_state.motor.position[:]
              joint_pos = self.cassie_state.joint.position[:]

            motor_vel = self.cassie_state.motor.velocity[:]
            joint_vel = self.cassie_state.joint.velocity[:]
            
            # # remove double-counted joint/motor positions
            # joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
            # joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])
            ## dim = 46
            robot_state = np.concatenate([
                [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],
                pelvis_quat[:],  # pelvis orientation
                motor_pos,       # actuated joint positions
                pelvis_vel,      # pelvis translational velocity
                pelvis_rvel,     # pelvis rotational velocity 
                motor_vel,       # actuated joint velocities
                pelvis_acc,     # pelvis translational acceleration
                joint_pos,       # unactuated joint positions
                joint_vel        # unactuated joint velocities
            ])

          
        else:
            qpos = np.copy(self.sim.qpos()) # dim=35
            qvel = np.copy(self.sim.qvel()) # dim=32
            
            pos = qpos[self.pos_index]
            vel = qvel[self.vel_index]
            robot_state = np.concatenate([pos, vel])
        state = np.concatenate([robot_state, ext_state]) 
        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history+1]

        return np.concatenate(self.state_history)

    def set_up_state_space(self):
        phase_size = 1 # index or swing duration, stance duration
        clock_size = 2 # sin, cos
        if self.state_est:
            self._obs = 46
            base_mir_obs = np.array([0.1, 1, -2, 3, -4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, -16, 17, -18, 19, -20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, -32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42])
        else:
            self._obs = 40
            ## this mirror obs not very right
            base_mir_obs = np.array([-0.1, 1, 2, -3, 4, -5, 13, 14, 15, 16, 17, 18, 19, 6, 7, 8, 9, 10, 11, 12, 20, -21, 22, 23, -24, 25, 33, 34, 35, 36, 37, 38, 39, 26, 27, 28, 29, 30, 31, 32])
        
        if self.clock_based == 'clock':
          append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size)])
          mirrored_obs = np.concatenate([base_mir_obs, append_obs])
          ref_inds = append_obs[0:clock_size].tolist()
          self._obs += clock_size
        else:
          append_obs = np.array([len(base_mir_obs) + i for i in range(phase_size)])
          mirrored_obs = np.concatenate([base_mir_obs, append_obs])
          ref_inds = append_obs[0:phase_size].tolist()
          self._obs += phase_size

        observation_space = Box(low=-np.inf, high=np.inf, shape=(self._obs + self._obs * self.history, ), dtype=np.float64)
        mirrored_obs = mirrored_obs.tolist()
        return observation_space, ref_inds, mirrored_obs

    def render(self):
        if self.vis is None:
           self.vis = CassieVis(self.sim)
        return self.vis.draw(self.sim)

    def close(self):
      if self.window is not None:
        self.vis.close()


#nbody layout:
# 0:  worldbody (zero)
# 1:  pelvis

# 2:  left hip roll 
# 3:  left hip yaw
# 4:  left hip pitch
# 5:  left achilles rod
# 6:  left knee
# 7:  left knee spring
# 8:  left shin
# 9:  left tarsus
# 10:  left heel spring
# 12:  left foot crank
# 12: left plantar rod
# 13: left foot

# 14: right hip roll 
# 15: right hip yaw
# 16: right hip pitch
# 17: right achilles rod
# 18: right knee
# 19: right knee spring
# 20: right shin
# 21: right tarsus
# 22: right heel spring
# 23: right foot crank
# 24: right plantar rod
# 25: right foot


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation wx
# [ 4] Pelvis orientation wy
# [ 5] Pelvis orientation wz
# [ 6] Left hip roll         (Motor [0])
# [ 7] Left hip yaw          (Motor [1])
# [ 8] Left hip pitch        (Motor [2])
# [ 9] Left achilles rod wx
# [10] Left achilles rod wy
# [11] Left achilles rod wz
# [12] Left knee             (Motor [3])
# [13] Left shin                        (Joint [0])
# [14] Left tarsus                      (Joint [1])
# [15] Left heel spring
# [16] Left foot crank
# [17] Left plantar rod
# [18] Left foot             (Motor [4], Joint [2])
# [19] Right hip roll        (Motor [5])
# [20] Right hip yaw         (Motor [6])
# [21] Right hip pitch       (Motor [7])
# [22] Right achilles rod wx
# [23] Right achilles rod wy
# [24] Right achilles rod wz
# [25] Right knee            (Motor [8])
# [26] Right shin                       (Joint [3])
# [27] Right tarsus                     (Joint [4])
# [28] Right heel spring
# [29] Right foot crank
# [30] Right plantar rod
# [31] Right foot            (Motor [9], Joint [5])