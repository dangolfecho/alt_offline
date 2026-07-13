import minari

dataset = minari.load_dataset('cartpole/expert-v0')
print("Observation space:",  dataset.observation_space)
print("Action space: ", dataset.action_space)
print("Total episodes:", dataset.total_episodes)
print("Total steps:", dataset.total_steps)
