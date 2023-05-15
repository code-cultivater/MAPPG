REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_1 import BasicMAC as BasicMAC_1
from .basic_controller_policy import BasicMAC as PolicyMAC
from .central_basic_controller import CentralBasicMAC


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_1"] = BasicMAC_1
REGISTRY["policy"] = PolicyMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC

