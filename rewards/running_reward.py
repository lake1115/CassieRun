
import numpy as np

@staticmethod
def quat_distance(q1, q2):
    return 2.0 * np.arccos(max(min(np.sum(q1 * q2), 1 - 1e-10), -1 + 1e-10))

## step for running
def running_reward(self, action):

    orientation_error = 0
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

    x_velocity = 2.0 * (self.x_position_after - self.x_position_before) * self.simrate
    quat_error = quat_distance(self.init_foot_quat, self.sim.xquat('left-foot'))* 5 \
                + quat_distance(self.init_foot_quat, self.sim.xquat('right-foot'))* 5
    # actual_q = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
    # target_q = [1, 0, 0, 0]
    # orientation_error = 6 * (1 - np.inner(actual_q, target_q) ** 2)


    # if self.last_action is None:
    #   ctrl_penalty = 0
    # else:
    #   ctrl_penalty = sum(np.abs(self.last_action - action)) / len(action)

    
    reward = x_velocity - np.abs(self.sim.qpos()[1]) * 10. -quat_error

            #-ctrl_penalty                  
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
    if self.sim.qpos()[2] < 0.8:
        reward = reward - 200.0
    return reward   
