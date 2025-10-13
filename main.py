# import os
# os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
  
# reset the environment
env.reset()
  
for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

# def get_policy_action(obs):
    # # random action
    # low, high = env.action_spec
    # return np.random.uniform(low, high)
# 
# obs = env.reset()
# 
# done = False
# ret = 0.
# while not done:
    # action = get_policy_action(obs)
    # obs, reward, done, _ = env.step(action) # play action
    # ret += reward

env.close()
# print("rollout completed with return {}".format(ret))
print("Success!")
