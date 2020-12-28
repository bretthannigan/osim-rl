from osim.env.gait2d import Gait2DGenAct

env = Gait2DGenAct(visualize=True)
observation = env.reset()
od = env.get_observation_dict()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())