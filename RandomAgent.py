import gym
import gym_gridworld
import time

env = gym.make('gridworldHard-v0')

for i in range(100):    #100 episodes
    env.reset()
    done = False
    while done == False:
        time.sleep(0.05)
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
    time.sleep(1)
    print("Terminal state reached: ", state)
    print("Episode Complete! Reward: ", reward)
    print("======================================")
