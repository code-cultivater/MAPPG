from .dist_agent import DistAgent
from .ensemble_agent import EnsembleAgent
from .pr_q_agent import PRQAgent
from .polar_q_agent import  PolarQAgent
REGISTRY = {}
REGISTRY["dist_agent"]=DistAgent
REGISTRY["ensemble_agent"]=EnsembleAgent
REGISTRY["pr_q_agent"]=PRQAgent
REGISTRY["polar_q_agent"]=PolarQAgent