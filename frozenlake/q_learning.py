import gym
import numpy as np
import random

env = gym.make("FrozenLake-v0")

alpha = 0.5
discount = 0.5
epi = 2000
rand_prob = 0.3
Q = np.zeros((env.observation_space.n, env.action_space.n))
tot = 0

for i in range(epi):

    state = env.reset()

    while True:
        # get action with some randomness
        rand = random.uniform(0, 1)
        if rand_prob > rand:
            action = env.action_space.sample()
        else:
            max_val = Q[state, :].max()
            actions = np.where(Q[state, :] == max_val)[0]
            action = np.random.choice(actions)

        # take one step
        state_new, reward, done, _ = env.step(action)

        Q[state, action] += alpha * (reward + discount * Q[state_new, :].max() - Q[state, action])

        state = state_new

        tot += reward

        if done:
            print("epi: " + str(i) + ", tot: " + str(tot))
            break
