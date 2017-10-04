import cliff
import random
import numpy as np

env = cliff.Cliff()

alpha = 0.5
discount = 0.5
epi = 5000
rand_prob = 0.3
Q = np.zeros((len(env.observation_space), len(env.action_space)))
tot = 0

for i in range(epi):

    state = env.reset()
    tot = 0

    while True:
        # get action with some randomness
        rand = random.uniform(0, 1)
        if rand_prob > rand:
            action = np.random.choice(env.action_space)
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

print(Q)

