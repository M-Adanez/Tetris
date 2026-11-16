import gymnasium as gym
import ale_py
import random

gym.register_envs(ale_py)

env=gym.make("ALE/Tetris-v5",render_mode="human",obs_type="ram")

state,_=env.reset()

for _ in range(500):
    action=random.randint(0,4)
    state, reward, terminated,truncated,info = env.step(action)
    print(state)

