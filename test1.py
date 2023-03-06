import gym
from gym import envs
# print(envs.registry)
env = gym.make("MyEnv-v0", render_mode="human")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()