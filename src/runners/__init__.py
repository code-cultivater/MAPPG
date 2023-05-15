REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_runner_1 import  EpisodeRunner as EpisodeRunner_1
REGISTRY["episode_1"] = EpisodeRunner_1

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .new_episode_runner import NewEpisodeRunner
REGISTRY["new_episode"]=NewEpisodeRunner
