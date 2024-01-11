import numpy as np

@staticmethod
def custom_footheight(omega):
    h = 0.15
    h1 = max(0, h*np.sin(2*np.pi*omega)-0.2*h) 
    h2 = max(0, h*np.sin(np.pi + 2*np.pi*omega)-0.2*h) 
    return [h1,h2]

## step for walking or stepping
def walking_reward(self, action):
      omega = self.phase / self.phaselen
      ref_footheight = np.array(custom_footheight(omega))
      real_footheight = np.array([self.foot_pos[2],self.foot_pos[5]])
      ref_penalty = np.sum(np.square(ref_footheight - real_footheight))
      ref_penalty = ref_penalty/0.0025

      orientation_penalty = (self.sim.qpos()[3]-1.)**2 + (self.sim.qpos()[4])**2+(self.sim.qpos()[5])**2+(self.sim.qpos()[6])**2
      orientation_penalty = orientation_penalty/0.1

      vel_penalty = (self.speed - self.sim.qvel()[0])**2 + (self.side_speed - self.sim.qvel()[1])**2 
      vel_penalty = vel_penalty/max(0.5*(self.speed*self.speed+self.side_speed*self.side_speed),0.01)

      spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
      spring_penalty *= 1000

      rew_ref = 0.5*np.exp(-ref_penalty)
      rew_spring = 0.1*np.exp(-spring_penalty)
      rew_ori = 0.125*np.exp(-orientation_penalty)
      rew_vel = 0.375*np.exp(-vel_penalty) #
      rew_termin = -10 * self.termination

      R_star = 1
      Rp = (0.75 * np.exp(-vel_penalty) + 0.25 * np.exp(-orientation_penalty))/ R_star
      Ri = np.exp(-ref_penalty) / R_star
      Ri = (Ri-0.4)/(1.0-0.4)

      omega = 0.5 

      reward = (1 - omega) * Ri + omega * Rp + rew_spring + rew_termin

      return reward



