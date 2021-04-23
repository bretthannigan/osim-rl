import math
import numpy as np
import pandas as pd
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

    obs_body_space = np.array([[-1.0] * 33, [1.0] * 33])
    # PELVIS:
    obs_body_space[:,0] = [0, 3] # pelvis height
    obs_body_space[:,1] = [-np.pi, np.pi] # pelvis pitch
    obs_body_space[:,2] = [-np.pi, np.pi] # pelvis roll
    obs_body_space[:,3] = [-20, 20] # pelvis vel (forward)
    obs_body_space[:,4] = [-20, 20] # pelvis vel (leftward)
    obs_body_space[:,5] = [-20, 20] # pelvis vel (upward)
    obs_body_space[:,6] = [-10*np.pi, 10*np.pi] # pelvis angular vel (pitch)
    obs_body_space[:,7] = [-10*np.pi, 10*np.pi] # pelvis angular vel (roll)
    obs_body_space[:,8] = [-10*np.pi, 10*np.pi] # pelvis angular vel (yaw)
    # RIGHT LEG:
    obs_body_space[:,9] = [-5, 5] # (r) ground reaction force normalized to bodyweight (forward)
    obs_body_space[:,10] = [-5, 5] # (r) ground reaction force normalized to bodyweight (rightward)
    obs_body_space[:,11] = [-10, 10] # (r) ground reaction force normalized to bodyweight (upward)
    obs_body_space[:,12] = [-180*np.pi/180, 45*np.pi/180] # (r) joint: (+) hip extension
    obs_body_space[:,13] = [-180*np.pi/180, 15*np.pi/180] # (r) joint: (+) knee extension
    obs_body_space[:,14] = [-45*np.pi/180, 90*np.pi/180] # (r) joint: (+) ankle extension (plantarflexion)
    obs_body_space[:,15] = [-5*np.pi, 5*np.pi] # (r) joint: (+) hip extension velocity
    obs_body_space[:,16] = [-5*np.pi, 5*np.pi] # (r) joint: (+) knee extension velocity
    obs_body_space[:,17] = [-5*np.pi, 5*np.pi] # (r) joint: (+) ankle extension (plantarflexion) velocity
    obs_body_space[:,18] = [-5*np.pi, 5*np.pi] # (r) joint: (+) hip extension acceleration
    obs_body_space[:,19] = [-5*np.pi, 5*np.pi] # (r) joint: (+) knee extension acceleration
    obs_body_space[:,20] = [-5*np.pi, 5*np.pi] # (r) joint: (+) ankle extension (plantarflexion) acceleration
    # LEFT LEG:
    obs_body_space[:,21] = [-5, 5] # (l) ground reaction force normalized to bodyweight (forward)
    obs_body_space[:,22] = [-5, 5] # (l) ground reaction force normalized to bodyweight (rightward)
    obs_body_space[:,23] = [-10, 10] # (l) ground reaction force normalized to bodyweight (upward)
    obs_body_space[:,24] = [-180*np.pi/180, 45*np.pi/180] # (l) joint: (+) hip extension
    obs_body_space[:,25] = [-180*np.pi/180, 15*np.pi/180] # (l) joint: (+) knee extension
    obs_body_space[:,26] = [-45*np.pi/180, 90*np.pi/180] # (l) joint: (+) ankle extension (plantarflexion)
    obs_body_space[:,27] = [-5*np.pi, 5*np.pi] # (l) joint: (+) hip extension velocity
    obs_body_space[:,28] = [-5*np.pi, 5*np.pi] # (l) joint: (+) knee extension velocity
    obs_body_space[:,29] = [-5*np.pi, 5*np.pi] # (l) joint: (+) ankle extension (plantarflexion) velocity
    obs_body_space[:,30] = [-5*np.pi, 5*np.pi] # (l) joint: (+) hip extension acceleration
    obs_body_space[:,31] = [-5*np.pi, 5*np.pi] # (l) joint: (+) knee extension acceleration
    obs_body_space[:,32] = [-5*np.pi, 5*np.pi] # (l) joint: (+) ankle extension (plantarflexion) acceleration

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
        self.data_path = 'D:\\Mohsen 2019 Running Study\\Participants\\Brett\\Angles\\Brett10_1.txt'
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

    def reset(self, project=True, random=False, seed=None, init_pose=None, obs_as_dict=False):
        self.t = 0
        self.init_reward()

        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1

        # load data
        self.joint_angles = pd.read_csv(self.data_path)
        self.joint_angles.drop('Index', axis=1, inplace=True)
        self.joint_angles.dropna(how='all', axis='columns', inplace=True)
        if random:
            np.random.seed(seed)
            self.data_position = np.random.randint(1, len(self.joint_angles)-self.time_limit)
        else:
            self.data_position = 1
        self.joint_angles = self.joint_angles.apply(np.deg2rad)
        self.joint_angles['dHipX'] = self.joint_angles['HipX'].diff()/self.osim_model.stepsize
        self.joint_angles['dKneeX'] = self.joint_angles['KneeX'].diff()/self.osim_model.stepsize
        self.joint_angles['dAnkleX'] = self.joint_angles['AnkleX'].diff()/self.osim_model.stepsize
        self.joint_angle_mapping = {}
        self.joint_angle_mapping['/jointset/hip_l/hip_flexion_l/value'] = 'HipX'
        self.joint_angle_mapping['/jointset/hip_l/hip_flexion_l/speed'] = 'dHipX'
        self.joint_angle_mapping['/jointset/knee_l/knee_angle_l/value'] = 'KneeX'
        self.joint_angle_mapping['/jointset/knee_l/knee_angle_l/speed'] = 'dKneeX'
        self.joint_angle_mapping['/jointset/ankle_l/ankle_angle_l/value'] = 'AnkleX'
        self.joint_angle_mapping['/jointset/ankle_l/ankle_angle_l/speed'] = 'dAnkleX'

        # initialize state
        self.osim_model.state = self.osim_model.model.initializeState()
        state = self.osim_model.get_state()
        for i in range(self.osim_model.model.getCoordinateSet().getSize()):
            for j in range(self.osim_model.model.getCoordinateSet().get(i).getStateVariableNames().getSize()):
                name = self.osim_model.model.getCoordinateSet().get(i).getStateVariableNames().get(j)
                kind = name.split('/')[-1]
                if name in self.joint_angle_mapping:
                    self.osim_model.model.setStateVariableValue(state,
                                                                name,
                                                                self.joint_angles[self.joint_angle_mapping[name]].loc[self.data_position])
                else:
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
            # joint angular accelerations
            obs_dict[leg]['d2_joint'] = {}
            obs_dict[leg]['d2_joint']['hip'] = -state_desc['joint_acc']['hip_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['d2_joint']['knee'] = state_desc['joint_acc']['knee_{}'.format(side)][0] # (+) extension
            obs_dict[leg]['d2_joint']['ankle'] = -state_desc['joint_acc']['ankle_{}'.format(side)][0] # (+) extension

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
            res += obs_dict[leg]['ground_reaction_forces'] # 3-element vector.
            for der in ['joint', 'd_joint', 'd2_joint']:
                for joint in ['hip', 'knee', 'ankle']:
                    res.append(obs_dict[leg][der][joint])
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
        self.d_reward['weight']['tracking'] = 0.1

        self.d_reward['alive'] = 0.1
        self.d_reward['effort'] = 0
        self.d_reward['tracking'] = 0

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
        reward += self.d_reward['footstep']['del_v']

        # Tracking motion capture
        reward_tracking = 0.
        reward_tracking += np.square(state_desc['joint_pos']['hip_l'][0] - self.joint_angles['HipX'].loc[self.osim_model.istep])
        reward_tracking += np.square(state_desc['joint_vel']['hip_l'][0] - self.joint_angles['dHipX'].loc[self.osim_model.istep])
        reward_tracking += np.square(state_desc['joint_pos']['knee_l'][0] - self.joint_angles['KneeX'].loc[self.osim_model.istep])
        reward_tracking += np.square(state_desc['joint_vel']['knee_l'][0] - self.joint_angles['dKneeX'].loc[self.osim_model.istep])
        reward_tracking += np.square(state_desc['joint_pos']['ankle_l'][0] - self.joint_angles['AnkleX'].loc[self.osim_model.istep])
        reward_tracking += np.square(state_desc['joint_vel']['ankle_l'][0] - self.joint_angles['dAnkleX'].loc[self.osim_model.istep])
        self.d_reward['tracking'] = -self.d_reward['weight']['tracking']*reward_tracking
        #reward = reward + self.d_reward['tracking']

        # footstep reward (when made a new step)
        if self.footstep['new']:
            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does not get higher rewards by making unnecessary (small) steps
            reward_footstep_0 = self.d_reward['weight']['footstep']*self.d_reward['footstep']['del_t']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            #reward_footstep_v = -self.reward_w['v_tgt']*(self.footstep['del_vx']**2)
            reward_footstep_v = -self.d_reward['weight']['v_tgt']*self.d_reward['footstep']['del_v'])/self.LENGTH0

            # penalize effort
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