import gym
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, noise=0.2, terminal_reward=10, start_state=0, width=5, height=5):
        self.height = height
        self.width = width
        self.noise = noise
        self.terminal_reward = terminal_reward
        self.done = False
        self.start_state = start_state

        #Number of spaces plus one absorbing state
        self.num_states = self.width * self.height + 1
        self.terminal_state = (self.width-1, self.height-1)

        #4 possible actions: North, South, East, West
        self.actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.width),spaces.Discrete(self.height)))

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        #If the agent reaches the terminal state
        if self.state == self.terminal_state:
            self.done = True
            return self.state, self._get_reward(), self.done, None

        col = self.state[0]
        row = self.state[1]
        #There is a probability to take a random action rather than the intended action
        if np.random.rand() < self.noise:
            action = self.action_space.sample()

        if action == 0:     #North
            row = max(row-1, 0)
        elif action == 1:   #South
            row = min(row+1, self.height-1)
        elif action == 2:   #East
            col = min(col+1, self.width-1)
        elif action == 3:   #West
            col = max(col-1, 0)

        reward = self._get_reward(new_state=(col, row))
        self.state = (col, row)
        return self.state, reward, self.done, None


    def _get_reward(self, new_state=None):
        if self.done:
            return self.terminal_reward
        reward = self.step_reward
        return reward

    def reset(self):
        self.state = self.start_state
        self.done = False
        return self.state

    def render(self, mode='human', close=False):
        pass
