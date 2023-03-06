import random
import gym
import numpy as np
import torch
import torch.nn as nn
from agent import Agent
import cv2

env = gym.make('CartPole-v1', render_mode="human")
s, _ = env.reset()
print(s)
EPSION_DECAY = 100000
EPSION_START = 1.0
EPSION_END = 0.02

TARGET_UPDATE_FREQUENCY = 10
n_episode = 5000
n_time_step = 500

n_state = len(s)
n_action = env.action_space.n
REWARD_BUFFER = np.empty(shape=n_episode)
agent = Agent(n_input=n_state, n_output=n_action)
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSION_DECAY], [EPSION_START, EPSION_END])
        random_sample = random.random()

        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s) # TODO

        s_, r, done, _, info = env.step(a)
        agent.memo.add_memo(s, a, r, done, s_) #TODO
        s = s_
        episode_reward += r

        if done:
            s, _ = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        # if np.mean(REWARD_BUFFER[:episode_i]) >= 100:
        #     while True:
        #         a = agent.online_net.act(s)
        #         s, r, done, _, info = env.step(a)
        #         img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        #         cv2.imshow('test',img)
        #         cv2.waitKey(1)
        #         if done:
        #             env.reset()

        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample() #TODO

        # Compute targets
        target_q_values = agent.target_net(batch_s_) #TODO
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values #TODO

        # Compute q_values
        q_values = agent.online_net(batch_s) #TODO
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

        # Compute loss
        loss = nn.functional.smooth_l1_loss(targets, a_q_values)

        # Gradient descent
        agent.optimizer.zero_grad() #TODO
        loss.backward()
        agent.optimizer.step() #TODO

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict()) #TODO

        # Show the training process
        print('Episode: {}'.format(episode_i))
        print('Avg. Reward: {}'.format(np.mean(REWARD_BUFFER[:episode_i])))
