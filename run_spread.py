from pettingzoo.mpe import simple_spread_v3
import numpy as np

env = simple_spread_v3.env()
env.reset()

for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)
    print(f"{agent}: reward={reward}, done={termination or truncation}")
