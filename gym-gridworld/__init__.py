from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
)
register(
    id='gridworld-extrahard-v0',
    entry_point='gym_gridworld.envs:GridWorldExtraHardEnv',
)
