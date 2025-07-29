from mpe2 import simple_spread_v3
import time

env = simple_spread_v3.env(render_mode='human', max_cycles=25, local_ratio=0.5)
env.reset()

# Force landmarks to be non-movable (redundant but explicit)
for landmark in env.unwrapped.world.landmarks:
    landmark.movable = False

print("Starting episode...\n")

step_count = 0
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    print(agent, reward)

    # Print landmark positions to see that they are actually not moving-- just rendering makes them look weirdly moving
    # for i, landmark in enumerate(env.unwrapped.world.landmarks):
    #     print(f"  Landmark {i} position: {landmark.state.p_pos}")

    if terminated or truncated:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)
    time.sleep(0.05)
    step_count += 1

print(f"\nEpisode ended after {step_count} total steps.")
env.close()

# from mpe2 import simple_spread_v3
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# MAX_CYCLES = 25

# # Create parallel environment with rendering
# env = simple_spread_v3.parallel_env(render_mode="rgb_array", max_cycles=MAX_CYCLES)
# observations = env.reset()

# # Set up fixed display with matplotlib
# plt.ion()
# fig, ax = plt.subplots()
# img = env.render()
# im = ax.imshow(img)
# ax.set_title("Simple Spread (Parallel API, Static View)")
# ax.axis("off")

# # Track rewards per step
# print("\nStep-by-step rewards:")
# for step in range(MAX_CYCLES):
#     actions = {
#         agent: env.action_space(agent).sample()
#         for agent in env.agents
#     }

#     observations, rewards, terminations, truncations, infos = env.step(actions)

#     # Render and update plot
#     img = env.render()
#     im.set_data(img)
#     fig.canvas.draw()
#     fig.canvas.flush_events()

#     # Print rewards for each agent
#     reward_str = " | ".join([f"{agent}: {rewards[agent]:+.2f}" for agent in env.agents])
#     print(f"Step {step:>2} → {reward_str}")

#     time.sleep(0.05)

# plt.ioff()
# plt.show()
# env.close()

