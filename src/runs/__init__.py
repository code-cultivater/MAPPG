from .run import run  as BasicRun
from .policy_run import run as PolicyRun
from .policy_run_v2 import run as PolicyRunV2
REGISTRY={}



REGISTRY["basic_run"]=BasicRun
REGISTRY["policy_run"]=PolicyRun
REGISTRY["policy_run_v2"]=PolicyRunV2

