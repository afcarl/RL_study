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


def epsilon_greedy(state_):
    # get action with some randomness
    rand = random.uniform(0, 1)
    if rand_prob > rand:
        action_ = np.random.choice(env.action_space)
    else:
        max_val = Q[state_, :].max()
        actions = np.where(Q[state_, :] == max_val)[0]
        action_ = np.random.choice(actions)
    return action_


for i in range(epi):

    state = env.reset()
    action = epsilon_greedy(state)
    tot = 0

    while True:

        # take one step
        state_new, reward, done, _ = env.step(action)

        # get an action for new state
        action_new = epsilon_greedy(state_new)

        Q[state, action] += alpha * (reward + discount * Q[state_new, action_new] - Q[state, action])

        state = state_new
        action = action_new

        tot += reward

        if done:
            print("epi: " + str(i) + ", tot: " + str(tot))
            break

print(Q)

