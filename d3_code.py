import d3rlpy
import gymnasium as gym
from gym.wrappers import RecordVideo

#dataset, env = d3rlpy.datasets.get_minari('D4RL/door/human-v2')
dataset, env = d3rlpy.datasets.get_cartpole()

cql = d3rlpy.algos.DiscreteCQLConfig().create()

cql.fit(
        dataset,
        n_steps=1000,
        n_steps_per_epoch=100,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)}
)

d3rlpy.notebook_utils.start_virtual_display()

env = RecordVideo(gym.make("D4RL/door/human-v2", render_mode="rgb_array"),
        './video')

d3rlpy.metrics.evaluate_qlearning_with_environment(cql, env)


d3rlpy.notebook_utils.render_video("video/rl-video-episode-0.mp4")
