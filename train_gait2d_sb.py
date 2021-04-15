import opensim as osim
import numpy as np
import sys

import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines.ddpg.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.env_checker import check_env
from stable_baselines import DDPG, SAC
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
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
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

#env.reset(verbose=True, logfile='arm_log.txt')

n_actions = env.action_space.shape[-1]

param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions), theta=0.05)
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.287)

class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              layer_norm=True,
                                              feature_extraction="mlp")

model = DDPG(CustomDDPGPolicy, 
            env, 
            verbose=1, 
            action_noise=action_noise,
            param_noise=param_noise,
            gamma=0.995,
            buffer_size=1000000,
            normalize_observations=True,
            normalize_returns=False,
            actor_lr=0.000527,
            batch_size=256,
            random_exploration=0.0,
            tensorboard_log="./gait2d_tensorboard/"
            )

if args.train:
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    model.save(args.model)
else:
    model = DDPG.load(args.model, env=env)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if args.visualize:
            env.render()

# # Create networks for DDPG
# # Next, we build a very simple model.
# actor = Sequential()
# actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# actor.add(Dense(units=128, kernel_initializer=initializers.RandomNormal(stddev=0.01),
#     bias_initializer=initializers.Zeros()))
# actor.add(LeakyReLU(alpha=0.05))
# actor.add(Dense(units=128, kernel_initializer=initializers.RandomNormal(stddev=0.01),
#     bias_initializer=initializers.Zeros()))
# actor.add(LeakyReLU(alpha=0.05))
# actor.add(Dense(nb_actions))
# actor.add(Activation('tanh'))
# #actor.add(Rescaling(150., offset=0.))
# print(actor.summary())

# action_input = Input(shape=(nb_actions,), name='action_input')
# observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
# flattened_observation = Flatten()(observation_input)
# x = concatenate([action_input, flattened_observation])
# x = Dense(256)(x)
# x = LeakyReLU(alpha=0.05)(x)
# x = Dense(256)(x)
# x = LeakyReLU(alpha=0.05)(x)
# x = Dense(1)(x)
# x = Activation('tanh')(x)
# critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

# # Set up the agent for training
# memory = SequentialMemory(limit=100000, window_length=1)
# random_process = OrnsteinUhlenbeckProcess(theta=.1, mu=0., sigma=0.01, size=env.noutput)
# agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
#                   memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
#                   random_process=random_process, gamma=.995, target_model_update=1e-3,
#                   delta_clip=1.)
# # agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
# #                             memory=memory, nb_steps_warmup=1000, random_process=random_process,
# #                             gamma=.99, target_model_update=0.1)
# agent.compile(Adam(lr=1e-3, clipnorm=1.), metrics=['mae'])

# # Okay, now it's time to learn something! We visualize the training here for show, but this
# # slows down training quite a lot. You can always safely abort the training prematurely using
# # Ctrl + C.
# if args.train:
#     history_cb = agent.fit(env, nb_steps=nallsteps, visualize=args.visualize, verbose=1, nb_max_episode_steps=None, log_interval=1000)
#     # After training is done, we save the final weights.
#     agent.save_weights(args.model, overwrite=True)
#     reward_history = history_cb.history["episode_reward"]
#     np.savetxt("episode_reward.txt", reward_history, delimiter=",")

# if not args.train:
#     agent.load_weights(args.model)
#     # Finally, evaluate our algorithm for 1 episode.
#     agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=1000)
