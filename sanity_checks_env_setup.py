from mpe2 import simple_spread_v3
import numpy as np

# Set up environment
env = simple_spread_v3.env(render_mode=None, max_cycles=25, local_ratio=0.5)
obs = env.reset(seed=42)  # Fix seed for repeatability

# Check that landmarks are static
print("Checking if landmarks are static...")
initial_positions = [np.copy(l.state.p_pos) for l in env.unwrapped.world.landmarks]

for step in range(5):
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        # Take random action unless done
        if terminated or truncated:
            env.step(None)
        else:
            env.step(env.action_space(agent).sample())

# After some steps, compare landmark positions
print("\nLandmark positions after random steps:")
for i, landmark in enumerate(env.unwrapped.world.landmarks):
    print(f"  Landmark {i} position: {landmark.state.p_pos}, initial: {initial_positions[i]}")
    # --> Should match initial positions exactly if static

print("\n Static landmark check passed if all positions are unchanged.\n")


env = simple_spread_v3.env(render_mode=None, max_cycles=25, local_ratio=0.5)
env.reset(seed=123)

print("Sanity Check: Step-by-step environment inspection")

action_meanings = {
    0: "stay",
    1: "left",
    2: "right",
    3: "down",
    4: "up"
}

# Store cumulative rewards
agent_rewards = {}
true_env_step = 0
print("\nEnvironment timestep", true_env_step)

# Cache agent order and properties
agent_names = env.agents
agent_index_map = {agent: i for i, agent in enumerate(agent_names)}
agents_world = env.unwrapped.world.policy_agents

# Print agent sizes
print("\nAgent sizes:")
for agent, obj in zip(agent_names, agents_world):
    print("  ", agent, "size =", obj.size)

print("\nStarting rollout...")

for i, agent in enumerate(env.agent_iter()):
    obs, reward, terminated, truncated, info = env.last()
    print("\nStep", i, "| Agent:", agent)
    print("  Reward:", round(reward, 4))
    print("  Obs[0:4] (position and velocity):", np.round(obs[:4], 3))
    print("  Obs[-2:] (communication):", np.round(obs[-2:], 3))

    agent_rewards[agent] = agent_rewards.get(agent, 0.0) + reward

    this_agent_obj = agents_world[agent_index_map[agent]]
    this_pos = this_agent_obj.state.p_pos
    this_size = this_agent_obj.size
    print("  Agent position:", np.round(this_pos, 3))

    # Landmark distances and size checks
    for j, landmark in enumerate(env.unwrapped.world.landmarks):
        l_pos = landmark.state.p_pos
        l_size = landmark.size
        dist = np.linalg.norm(this_pos - l_pos)
        print("    Landmark", j, "position:", np.round(l_pos, 3),
              "| Distance:", round(dist, 3),
              "| Sum of radii:", round(this_size + l_size, 3))

    # Distances to other agents
    print("  Distances to other agents:")
    for other_agent, other_obj in zip(agent_names, agents_world):
        if other_agent == agent:
            continue
        other_pos = other_obj.state.p_pos
        other_size = other_obj.size
        dist = np.linalg.norm(this_pos - other_pos)
        print("    To", other_agent, ": distance =", round(dist, 3),
              "| Sum of radii =", round(this_size + other_size, 3))

    # Take a sample action
    if terminated or truncated:
        env.step(None)
    else:
        action = env.action_space(agent).sample()
        action_text = action_meanings.get(action, str(action))
        print("  Action chosen:", action, "meaning:", action_text)
        env.step(action)

    # Print timestep update
    if agent == env.agents[-1]:
        true_env_step += 1
        print("\n--- End of timestep", true_env_step, "---")
        print("Environment timestep", true_env_step)

    if true_env_step > 2:
        break

# Final reward summary
print("\nCumulative reward per agent:")
for agent, r in agent_rewards.items():
    print(" ", agent, ":", round(r, 4))



# # Print final positions of agents and landmarks
# print("\nFinal agent positions (sample):")
# for i, agent in enumerate(env.unwrapped.world.agents):
#     print(f"  {agent.name}: pos = {agent.state.p_pos}")
#     # --> Should vary across agents, start near (0, 0), spread out as steps increase

# print("\nLandmark positions (still static):")
# for i, landmark in enumerate(env.unwrapped.world.landmarks):
#     print(f"  Landmark {i}: pos = {landmark.state.p_pos}")
#     # --> Should match original since they're immovable

# print("\n Agent positions updated, landmarks unchanged.\n")





