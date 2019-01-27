import gym
import numpy as np
import graphics as gr
import random

from gym import error, spaces, utils
from gym.utils import seeding

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, noise=0.2, terminal_reward=10, start_state=(0,0), width=5, height=5, rendering=True):
        self.height = height
        self.width = width
        self.noise = noise
        self.terminal_reward = terminal_reward
        self.done = False
        self.start_state = start_state
        self.past_state = start_state
        self.rendering = rendering

        #Number of spaces plus one absorbing state
        self.num_states = self.width * self.height + 1
        self.terminal_state = (self.width-1, self.height-1)
        self.loosing_state = (0,0)
        while(self.loosing_state == self.start_state or self.loosing_state == self.terminal_state):
            self.loosing_state = (random.randint(0,self.height-1), random.randint(0, self.width-1))

        print("Loosing state: ", self.loosing_state)

        #4 possible actions: North, South, East, West
        self.actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(self.width),spaces.Discrete(self.height)))

        #Graphics rendering configuration
        self.w = WINDOW_WIDTH/self.width
        self.h = WINDOW_HEIGHT/self.height
        self.grid = [[0]*self.height for n in range(self.width)]
        self.grid[self.height-self.terminal_state[1]-1][self.terminal_state[0]] = 1
        self.grid[self.height-self.loosing_state[1]-1][self.loosing_state[0]] = 2

        if self.rendering == True:
            self.win = gr.GraphWin('Gridworld', WINDOW_WIDTH, WINDOW_HEIGHT)

            circle_x, circle_y = (start_state[0]+(1/2))*self.w, WINDOW_HEIGHT-(start_state[1]+(1/2))*self.w
            x,y = 0,0
            for row in self.grid:
                x = 0
                for col in row:
                    rect = gr.Rectangle(gr.Point(x, y), gr.Point(x+self.w, y+self.h))
                    rect.setFill('white')
                    if col == 1:        #Terminal State
                        rect.setFill('green')
                    elif col == 2:      #Pit
                        rect.setFill('red')
                    elif col == 3:      #Wall
                        rect.setFill('gray')
                    rect.setOutline('black')
                    rect.draw(self.win)
                    x = x + self.w
                y = y + self.h
            self.agent = gr.Circle(gr.Point(circle_x, circle_y), 20)
            self.agent.setFill('blue')
            self.agent.draw(self.win)

        #self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        self.past_state = self.state

        #If the agent reaches a terminal state
        if self.state == self.terminal_state or self.state == self.loosing_state:
            self.done = True
            return self.state, self._get_reward(), self.done, None

        col = self.state[0]
        row = self.state[1]

        #There is a probability to take a random action rather than the intended action
        if np.random.rand() < self.noise:
            action = self.action_space.sample()

        if action == 0:     #North
            row = min(row+1, self.height-1)
        elif action == 1:   #South
            row = max(row-1, 0)
        elif action == 2:   #East
            col = min(col+1, self.width-1)
        elif action == 3:   #West
            col = max(col-1, 0)

        reward = self._get_reward(new_state=(col, row))
        self.state = (col, row)
        return self.state, reward, self.done, None


    def _get_reward(self, new_state=None):
        if self.done:
            if self.state == self.terminal_state:
                return self.terminal_reward
            elif self.state == self.loosing_state:
                return -self.terminal_reward
        #reward = self.step_reward
        return 0

    def reset(self):
        self.state = self.start_state
        self.done = False
        return self.state

    def render(self, mode='human', close=False):

        if self.rendering == True:
            (x, y) = self.state
            (currX, currY) = self.past_state
            dx, dy = (x-currX)*self.w, (currY-y)*self.h
            self.agent.move(dx, dy)

        return
