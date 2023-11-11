REGISTRY = {}

from .basic_controller import BasicMAC
from .fingerprint_controller import FingerPrintMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["fingerprint_mac"] = FingerPrintMAC

