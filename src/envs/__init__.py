from functools import partial

from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
from .ma_simple_env import  MeetEnv
from smac.env import MultiAgentEnv, StarCraft2Env
from .matrix_game.matrix_game_simple import Matrixgame

from .quadratic_game.quadratic_simple import  QuadraticGame
#from smac_plus import  Tracker1Env, Join1Env
# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["matrix_game"] = partial(env_fn, env=Matrixgame)

REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["simple_game"] = partial(env_fn, env=MeetEnv)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["tracker1"] = partial(env_fn, env=Tracker1Env)
# REGISTRY["join1"] = partial(env_fn, env=Join1Env)
REGISTRY["quadratic"] = partial(env_fn, env=QuadraticGame)