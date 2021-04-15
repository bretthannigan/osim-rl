import opensim as osim
import numpy as np
import sys

import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.td3.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.env_checker import check_env
from stable_baselines import TD3, SAC
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from osim.env.gait2d import Gait2DGenAct

import argparse
import math

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=1000000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="gait2d_td3.h5f")
args = parser.parse_args()

# set to get observation in array
#def _new_step(self, action, project=True, obs_as_dict=False):
#    return super(Arm2DEnv, self).step(action, project=project, obs_as_dict=obs_as_dict)
#Arm2DEnv.step = _new_step
# Load walking environment
env = Gait2DGenAct(args.visualize, integrator_accuracy=3e-2)
eval_env = Gait2DGenAct(integrator_accuracy=3e-2)
#env = Arm2DVecEnv(visualize=True)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

n_actions = env.action_space.shape[-1]

param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions), theta=0.05)
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.287)

class CustomTD3Policy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs,
                                              layers=[400, 400],
                                              layer_norm=True,
                                              feature_extraction="mlp")

model = TD3(CustomTD3Policy, 
            env, 
            verbose=1, 
            action_noise=action_noise,
            learning_rate=0.001,
            gamma=0.99,
            buffer_size=1000000,
            batch_size=100,
            train_freq=1000,
            tensorboard_log="./gait2d_td3_tensorboard/"
            )

if args.train:
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(args.model)
else:
    model = TD3.load(args.model, env=env)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if args.visualize:
            env.render()