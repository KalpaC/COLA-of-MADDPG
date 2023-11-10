# __init__.py 2023/10/19 17:04
from .multiagentenv import MultiAgentEnv
from .v2x.mode_4_in_3gpp import Mode_4_in_3GPP


REGISTRY = {
    "v2x": Mode_4_in_3GPP
}


