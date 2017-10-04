import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Cliff:
    """
    implementation of the Cliff Walking.
    Adapted Open AI gym API style

    The board is a 4x12 matrix, with (using Numpy matrix indexing):
        [3, 0] as the start at bottom-left
        [3, 11] as the goal at bottom-right
        [3, 1..10] as the cliff at bottom-center

    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    and a reset to the start.
    An episode terminates when the agent reaches the goal or fall off from cliff.
    Goal rewards 100
    """

    def __init__(self):
        self._shape = (4, 12)
        self._pos_row = None
        self._pos_col = None
        self._env = None
        self._start = (3, 0)
        self._goal = (3, 11)
        self._cliff = (3, range(1, 11))

        self.action_space = np.array((UP, RIGHT, DOWN, LEFT), dtype=int)
        self.observation_space = range(self._shape[0] * self._shape[1])

    def reset(self):
        self._initialize_env()
        self._pos_row = self._start[0]
        self._pos_col = self._start[1]
        return self._calc_observation(self._pos_row, self._pos_col)

    def _calc_observation(self, row, col):
        return self._shape[1] * row + col

    def _initialize_env(self):
        self._env = np.zeros(shape=self._shape)
        self._env[self._goal] = 3
        self._env[self._start] = 2
        self._env[3, 1:-1] = -1

    def step(self, action):
        # get the new position
        if action == UP:
            self._pos_row = max(self._pos_row-1, 0)
        elif action == DOWN:
            self._pos_row = min(self._pos_row+1, self._shape[0]-1)
        elif action == LEFT:
            self._pos_col = max(self._pos_col-1, 0)
        elif action == RIGHT:
            self._pos_col = min(self._pos_col+1, self._shape[1]-1)

        # get the reward
        obs = self._calc_observation(self._pos_row, self._pos_col)
        done = False
        reward = -1
        if self._pos_row == self._goal[0] and self._pos_col == self._goal[1]:
            reward = 100
            done = True
        elif self._pos_row == self._cliff[0] and self._pos_col in self._cliff[1]:
            reward = -100
            done = True

        # return observation(position), reward, if episode is done, and dummy (to match openai api style)
        return obs, reward, done, None

    def render(self):
        self._initialize_env()
        self._env[(self._pos_row, self._pos_col)] = 1
        return self._env

