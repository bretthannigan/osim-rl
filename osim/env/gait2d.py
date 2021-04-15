import math
import numpy as np
import os
from .utils.mygym import convert_to_gym
import gym
import opensim
import random
from .osim import OsimEnv

class Gait2DGenAct(OsimEnv):
    model = 'Generic'

    LENGTH0 = 1 # leg length

    footstep = {}
    footstep['n'] = 0
    footstep['new'] = False
    footstep['r_contact'] = 1
    footstep['l_contact'] = 1

    # state_variables = { 'pelvis_tilt_value': '/jointset/ground_pelvis/pelvis_tilt/value',
    #                     'pelvis_tilt_speed': '/jointset/ground_pelvis/pelvis_tilt/speed',
    #                     'pelvis_tx_value': '/jointset/ground_pelvis/pelvis_tx/value',
    #                     'pelvis_tx_speed': '/jointset/ground_pelvis/pelvis_tx/speed',
    #                     'pelvis_ty_value': '/jointset/ground_pelvis/pelvis_ty/value',
    #                     'pelvis_ty_speed': '/jointset/ground_pelvis/pelvis_ty/speed',
    #                     'hip_flexion_r_value': '/jointset/hip_r/hip_flexion_r/value',
    #                     'hip_flexion_r_speed': '/jointset/hip_r/hip_flexion_r/speed',
    #                     'knee_angle_r_value': '/jointset/knee_r/knee_angle_r/value',
    #                     'knee_angle_r_speed': '/jointset/knee_r/knee_angle_r/speed',
    #                     'ankle_angle_r_value': '/jointset/ankle_r/ankle_angle_r/value',
    #                     'ankle_angle_r_speed': '/jointset/ankle_r/ankle_angle_r/speed',
    #                     'hip_flexion_l_value': '/jointset/hip_l/hip_flexion_l/value',
    #                     'hip_flexion_l_speed': '/jointset/hip_l/hip_flexion_l/speed',
    #                     'knee_angle_l_value': '/jointset/knee_l/knee_angle_l/value',
    #                     'knee_angle_l_speed': '/jointset/knee_l/knee_angle_l/speed',
    #                     'ankle_angle_l_value': '/jointset/ankle_l/ankle_angle_l/value',
    #                     'ankle_angle_l_speed': '/jointset/ankle_l/ankle_angle_l/speed' }



    # INIT_POSE = { 'pelvis_tilt_value': 0.0*np.pi/180.0,
    #               'pelvis_tilt_speed': 0.0*np.pi/180.0,
    #               'pelvis_tx_value': 0.0*np.pi/180.0,
    #               'pelvis_tx_speed': 0.0,
    #               'pelvis_ty_value': 0.91,
    #               'pelvis_ty_speed': 0.0,
    #               'hip_flexion_r_value': 0.0*np.pi/180.0,
    #               'hip_flexion_r_speed': 0.0*np.pi/180.0,
    #               'knee_angle_r_value': 0.0*np.pi/180.0,
    #               'knee_angle_r_speed': 0.0*np.pi/180.0,
    #               'ankle_angle_r_value': 0.0*np.pi/180.0,
    #               'ankle_angle_r_speed': 0.0*np.pi/180.0,
    #               'hip_flexion_l_value': 0.0*np.pi/180.0,
    #               'hip_flexion_l_speed': 0.0*np.pi/180.0,
    #               'knee_angle_l_value': 0.0*np.pi/180.0,
    #               'knee_angle_l_speed': 0.0*np.pi/180.0,
    #               'ankle_angle_l_value': 0.0*np.pi/180.0,
    #               'ankle_angle_l_speed': 0.0*np.pi/180.0 }

    obs_body_space = np.array([[-1.0] * 27, [1.0] * 27])
    obs_body_space[:,0] = [0, 3] # pelvis height
    obs_body_space[:,1] = [-np.pi, np.pi] # pelvis pitch
    obs_body_space[:,2] = [-np.pi, np.pi] # pelvis roll
    obs_body_space[:,3] = [-20, 20] # pelvis vel (forward)
    obs_body_space[:,4] = [-20, 20] # pelvis vel (leftward)
    obs_body_space[:,5] = [-20, 20] # pelvis vel (upward)
    obs_body_space[:,6] = [-10*np.pi, 10*np.pi] # pelvis angular vel (pitch)
    obs_body_space[:,7] = [-10*np.pi, 10*np.pi] # pelvis angular vel (roll)
    obs_body_space[:,8] = [-10*np.pi, 10*np.pi] # pelvis angular vel (yaw)
    obs_body_space[:,[9 + x for x in [0, 9]]] = np.array([[-5, 5]]).transpose() # (r,l) ground reaction force normalized to bodyweight (forward)
    obs_body_space[:,[10 + x for x in [0, 9]]] = np.array([[-5, 5]]).transpose() # (r, l) ground reaction force normalized to bodyweight (rightward)
    obs_body_space[:,[11 + x for x in [0, 9]]] = np.array([[-10, 10]]).transpose() # (r, l) ground reaction force normalized to bodyweight (upward)
    obs_body_space[:,[12 + x for x in [0, 9]]] = np.array([[-180*np.pi/180, 45*np.pi/180]]).transpose() # (r, l) joint: (+) hip extension
    obs_body_space[:,[13 + x for x in [0, 9]]] = np.array([[-180*np.pi/180, 15*np.pi/180]]).transpose() # (r, l) joint: (+) knee extension
    obs_body_space[:,[14 + x for x in [0, 9]]] = np.array([[-45*np.pi/180, 90*np.pi/180]]).transpose() # (r, l) joint: (+) ankle extension (plantarflexion)
    obs_body_space[:,[15 + x for x in [0, 9]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) hip extension
    obs_body_space[:,[16 + x for x in [0, 9]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) knee extension
    obs_body_space[:,[17 + x for x in [0, 9]]] = np.array([[-5*np.pi, 5*np.pi]]).transpose() # (r, l) joint: (+) ankle extension (plantarflexion)

    def get_model_key(self):
        return self.model

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        if difficulty == 0:
            self.time_limit = 1000
        if difficulty == 1:
            self.time_limit = 1000
        if difficulty == 2:
            self.time_limit = 1000
            print("difficulty 2 for Round 1")
        if difficulty == 3:
            self.time_limit = 2500 # 25 sec
            print("difficulty 3 for Round 2")
        self.spec.timestep_limit = self.time_limit    

    def __init__(self, visualize=True, integrator_accuracy=5e-5, difficulty=3, seed=None, report=None, subject='Generic'):
        if difficulty not in [0, 1, 2, 3]:
            raise ValueError("difficulty level should be in [0, 1, 2, 3].")
        self.model_paths = {}
        self.model_paths['Generic'] = os.path.join(os.path.dirname(__file__), '../models/gait9dof7act.osim')
        self.model_paths['Brett'] = os.path.join(os.path.dirname(__file__), '../models/gait9dof7act_Brett.osim')
        self.model = subject
        self.model_path = self.model_paths[self.get_model_key()]
        super(Gait2DGenAct, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy)

        # Calculate mass by summing all bodies.
        self.G = np.abs(self.osim_model.model.getGravity().get(1))
        mass = 0.0
        for i in range(self.osim_model.bodySet.getSize()):
            mass = mass + self.osim_model.bodySet.get(i).getMass()
        self.MASS = mass
        self.state_variables = []
        for i in range(self.osim_model.model.getStateVariableNames().getSize()):
            self.state_variables.append(self.osim_model.model.getStateVariableNames().get(i))

        self.set_difficulty(difficulty)

        self.noutput = self.osim_model.noutput

        if report:
            bufsize = 0
            self.observations_file = open('%s-obs.csv' % (report,),'w', bufsize)
            self.actions_file = open('%s-act.csv' % (report,),'w', bufsize)
            self.get_headers()

    def reset(self, project=True, seed=None, init_pose=None, obs_as_dict=False):
        self.t = 0
        self.init_reward()

        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1

        # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        state = self.osim_model.get_state()
        for i in range(self.osim_model.model.getCoordinateSet().getSize()):
            for j in range(self.osim_model.model.getCoordinateSet().get(i).getStateVariableNames().getSize()):
                name = self.osim_model.model.getCoordinateSet().get(i).getStateVariableNames().get(j)
                kind = name.split('/')[-1]
                if kind=='value':
                    self.osim_model.model.setStateVariableValue(state, 
                                                                name,
                                                                self.osim_model.model.getCoordinateSet().get(i).getDefaultValue() )
                elif kind=='speed':
                    self.osim_model.model.setStateVariableValue(state, 
                                                                name,
                                                                self.osim_model.model.getCoordinateSet().get(i).getDefaultSpeedValue() )

        self.osim_model.set_state(state)

        self.osim_model.state.setTime(0)
        self.osim_model.istep = 0

        self.osim_model.reset_manager()

        d = super(Gait2DGenAct, self).get_state_desc()
        pose = np.array([d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])
        
        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        return self.get_observation()

    def load_model(self, model_path = None):
        super(Gait2DGenAct, self).load_model(model_path)
        observation_space = self.obs_body_space
        self.observation_space = convert_to_gym(observation_space)

    def step(self, action, project=True, obs_as_dict=False):

        _, reward, done, info = super(Gait2DGenAct, self).step(action, project=project, obs_as_dict=obs_as_dict)
        self.t += self.osim_model.stepsize
        self.update_footstep()

        d = super(Gait2DGenAct, self).get_state_desc()
        self.pose = np.array([d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])
        if project:
            if obs_as_dict:
                obs = self.get_observation_dict()
            else:
                obs = self.get_observation_clipped()
        else:
            obs = self.get_state_desc()
            
        return obs, reward, done, info

    def change_model(self, model='3D', difficulty=3, seed=0):
        if self.model != model:
            self.model = model
            self.load_model(self.model_paths[self.get_model_key()])
        self.set_difficulty(difficulty)
    
    def is_done(self):
        state_desc = self.get_state_desc()
        return state_desc['body_pos']['pelvis'][1] < 0.6

    def update_footstep(self):
        state_desc = self.get_state_desc()

        # update contact
        r_contact = True if state_desc['forces']['foot_r'][1] < -0.05*(self.MASS*self.G) else False
        l_contact = True if state_desc['forces']['foot_l'][1] < -0.05*(self.MASS*self.G) else False

        self.footstep['new'] = False
        if (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def get_observation_dict(self):
        state_desc = self.get_state_desc()

        obs_dict = {}

        yaw = 0.0
        # pelvis state (in local frame)
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['height'] = state_desc['body_pos']['pelvis'][1]
        obs_dict['pelvis']['pitch'] = -state_desc['joint_pos']['ground_pelvis'][0] # (+) pitching forward
        obs_dict['pelvis']['roll'] = state_desc['joint_pos']['ground_pelvis'][1] # (+) rolling around the forward axis (to the right)
        yaw = state_desc['joint_pos']['ground_pelvis'][2]
        dx_local, dy_local = rotate_frame(  state_desc['body_vel']['pelvis'][0],
                                            state_desc['body_vel']['pelvis'][2],
                                            yaw)
        dz_local = state_desc['body_vel']['pelvis'][1]
        obs_dict['pelvis']['vel'] = [   dx_local, # (+) forward
                                        -dy_local, # (+) leftward
                                        dz_local, # (+) upward
                                        -state_desc['joint_vel']['ground_pelvis'][0], # (+) pitch angular velocity
                                        state_desc['joint_vel']['ground_pelvis'][1], # (+) roll angular velocity
                                        state_desc['joint_vel']['ground_pelvis'][2]] # (+) yaw angular velocity

        # leg state
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            obs_dict[leg] = {}
            grf = [ f/(self.MASS*self.G) for f in state_desc['forces']['foot_{}'.format(side)][0:3] ] # forces normalized by bodyweight
            grm = [ m/(self.MASS*self.G) for m in state_desc['forces']['foot_{}'.format(side)][3:6] ] # forces normalized by bodyweight
            grfx_local, grfy_local = rotate_frame(-grf[0], -grf[2], yaw)
            if leg == 'r_leg':
                obs_dict[leg]['ground_reaction_forces'] = [ grfx_local, # (+) forward
                                                            grfy_local, # (+) lateral (rightward)
                                                            -grf[1]] # (+) upward
            if leg == 'l_leg':
                obs_dict[leg]['ground_reaction_forces'] = [ grfx_local, # (+) forward
                                                            -grfy_local, # (+) lateral (leftward)
                                                            -grf[1]] # (+) upward

            # joint angles
            obs_dict[leg]['joint'] = {}
            obs_dict[leg]['joint']['hip'] = -state_desc['joint_pos']['hip_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['joint']['knee'] = state_desc['joint_pos']['knee_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['joint']['ankle'] = -state_desc['joint_pos']['ankle_{}'.format(side)][0] # (+) extension
            # joint angular velocities
            obs_dict[leg]['d_joint'] = {}
            obs_dict[leg]['d_joint']['hip'] = -state_desc['joint_vel']['hip_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['d_joint']['knee'] = state_desc['joint_vel']['knee_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['d_joint']['ankle'] = -state_desc['joint_vel']['ankle_{}'.format(side)][0] # (+) extension

        return obs_dict

    def get_observation(self):
        obs_dict = self.get_observation_dict()

        # Augmented environment from the L2R challenge
        res = []

        res.append(obs_dict['pelvis']['height'])
        res.append(obs_dict['pelvis']['pitch'])
        res.append(obs_dict['pelvis']['roll'])
        res.append(obs_dict['pelvis']['vel'][0]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][1]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][2]/self.LENGTH0)
        res.append(obs_dict['pelvis']['vel'][3])
        res.append(obs_dict['pelvis']['vel'][4])
        res.append(obs_dict['pelvis']['vel'][5])

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
        return np.asarray(res)

    def get_observation_clipped(self):
        obs = self.get_observation()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    ## Values in the observation vector
    # 'pelvis': height, pitch, roll, 6 vel (9 values)
    # for each 'r_leg' and 'l_leg' (*2)
    #   'ground_reaction_forces' (3 values)
    #   'joint' (3 values)
    #   'd_joint' (3 values)
    # 9 + 2*(3 + 3 + 3) = 31
    def get_observation_space_size(self):
        return 27
        #return Gait2DGenAct._count_dict_elements(self.get_observation_dict())
        
    def get_state_desc(self):
        d = super(Gait2DGenAct, self).get_state_desc()
        #state_desc['joint_pos']
        #state_desc['joint_vel']
        #state_desc['joint_acc']
        #state_desc['body_pos']
        #state_desc['body_vel']
        #state_desc['body_acc']
        #state_desc['body_pos_rot']
        #state_desc['body_vel_rot']
        #state_desc['body_acc_rot']
        #state_desc['forces']
        #state_desc['muscles']
        #state_desc['markers']
        #state_desc['misc']
        return d

    def init_reward(self):
        self.d_reward = {}

        self.d_reward['weight'] = {}
        self.d_reward['weight']['footstep'] = 10
        self.d_reward['weight']['effort'] = (1.0/280000.0)*10
        self.d_reward['weight']['v_tgt'] = 5
        self.d_reward['weight']['v_tgt_R2'] = 3

        self.d_reward['alive'] = 0.1
        self.d_reward['effort'] = 0

        self.d_reward['footstep'] = {}
        self.d_reward['footstep']['effort'] = 0
        self.d_reward['footstep']['del_t'] = 0
        self.d_reward['footstep']['del_v'] = 0

    def get_reward(self):
        reward_footstep_0 = 0
        state_desc = self.get_state_desc()
        if not self.get_prev_state_desc():
            return 0

        reward = 0.0
        dt = self.osim_model.stepsize

        # alive reward
        # should be large enough to search for 'success' solutions (alive to the end) first
        reward += self.d_reward['alive']

        # effort ~ muscle fatigue ~ (muscle activation)^2 
        ACT2 = 0
        for actuator in ['pelvis', 'hip_r', 'hip_l', 'knee_r', 'knee_l', 'ankle_r', 'ankle_r']:
            ACT2 += np.square(state_desc['forces'][actuator])
        self.d_reward['effort'] += ACT2*dt
        self.d_reward['footstep']['effort'] += ACT2*dt

        self.d_reward['footstep']['del_t'] += dt

        # reward from velocity (penalize from deviating from v_tgt)

        p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        v_tgt = [1.0, 0.0, 0.0]

        self.d_reward['footstep']['del_v'] += (v_body[0] - v_tgt[0])*dt
        #reward += self.d_reward['footstep']['del_v']

        # footstep reward (when made a new step)
        if self.footstep['new']:
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
            reward_footstep_v = -self.d_reward['weight']['v_tgt']*np.linalg.norm(self.d_reward['footstep']['del_v'])/self.LENGTH0

            # panalize effort
            reward_footstep_e = -self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

            self.d_reward['footstep']['del_t'] = 0
            self.d_reward['footstep']['del_v'] = 0
            self.d_reward['footstep']['effort'] = 0

            reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

        # success bonus
        if not self.is_done() and (self.osim_model.istep >= self.spec.timestep_limit): #and self.failure_mode is 'success':
            # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
            #reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
            #reward += reward_footstep_0 + 100
            reward += reward_footstep_0 + 100

        # state_desc = self.get_state_desc()

        # reward = 0
        # reward += self.d_reward['alive']

        # # # effort ~ muscle fatigue ~ (muscle activation)^2 
        # # ACT2 = 0
        # # for actuator in ['pelvis', 'hip_r', 'hip_l', 'knee_r', 'knee_l', 'ankle_r', 'ankle_r']:
        # #     ACT2 += np.square(state_desc['forces'][actuator][0])

        # # reward -= ACT2*0.00001

        # pelvis_ty = np.square(state_desc['body_pos']['pelvis'][1] - 0.9)
        # pelvis_tilt = np.sum(np.square(state_desc['joint_pos']['ground_pelvis'] - np.asarray([-0.050387283520705727, -0.00014076261193743153, 0.9099458323860586])))

        # reward -= pelvis_ty
        # reward -= pelvis_tilt
        # reward -= pelvis_tx
        # state_desc = self.get_state_desc()
        # if not self.get_prev_state_desc():
        #     return 0

        # reward = 0
        # dt = self.osim_model.stepsize

        # # alive reward
        # # should be large enough to search for 'success' solutions (alive to the end) first
        # reward += self.d_reward['alive']

        # # effort ~ muscle fatigue ~ (muscle activation)^2 
        # ACT2 = 0
        # for actuator in ['pelvis', 'hip_r', 'hip_l', 'knee_r', 'knee_l', 'ankle_r', 'ankle_r']:
        #     ACT2 += np.square(state_desc['forces'][actuator])
        # self.d_reward['effort'] += ACT2*dt
        # self.d_reward['footstep']['effort'] += ACT2*dt

        # self.d_reward['footstep']['del_t'] += dt

        # # reward from velocity (penalize from deviating from v_tgt)

        # p_body = [state_desc['body_pos']['pelvis'][0], -state_desc['body_pos']['pelvis'][2]]
        # v_body = [state_desc['body_vel']['pelvis'][0], -state_desc['body_vel']['pelvis'][2]]
        # v_tgt = np.asarray([0.8, 0])

        # self.d_reward['footstep']['del_v'] += (v_body - v_tgt)*dt

        # # footstep reward (when made a new step)
        # if self.footstep['new']:
        #     # footstep reward: so that solution does not avoid making footsteps
        #     # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
        #     reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

        #     # deviation from target velocity
        #     # the average velocity a step (instead of instantaneous velocity) is used
        #     # as velocity fluctuates within a step in normal human walking
        #     #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
        #     reward_footstep_v = -self.d_reward['weight']['v_tgt']*np.linalg.norm(self.d_reward['footstep']['del_v'])/self.LENGTH0

        #     # panalize effort
        #     reward_footstep_e = -self.d_reward['weight']['effort']*self.d_reward['footstep']['effort']

        #     self.d_reward['footstep']['del_t'] = 0
        #     self.d_reward['footstep']['del_v'] = 0
        #     self.d_reward['footstep']['effort'] = 0

        #     reward += reward_footstep_0 + reward_footstep_v + reward_footstep_e

        # # success bonus
        # if not self.is_done() and (self.osim_model.istep >= self.spec.timestep_limit): #and self.failure_mode is 'success':
        #     # retrieve reward (i.e. do not penalize for the simulation terminating in a middle of a step)
        #     #reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']
        #     #reward += reward_footstep_0 + 100
        #     reward += reward_footstep_0 + 10

        return reward

    @staticmethod
    def _count_dict_elements(x):
        if isinstance(x, dict):
            return sum([Gait2DGenAct._count_dict_elements(_x) for _x in x.values()])
        elif isinstance(x, list):
            return len(x)
        else: return 1

def rotate_frame(x, y, theta):
    x_rot = np.cos(theta)*x - np.sin(theta)*y
    y_rot = np.sin(theta)*x + np.cos(theta)*y
    return x_rot, y_rot