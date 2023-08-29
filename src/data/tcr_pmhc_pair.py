from src.data.tcr import Tcr
from src.data.pmhc import Pmhc


class TcrPmhcPair:
    def __init__(self, tcr: Tcr, pmhc: Pmhc) -> None:
        self.tcr = tcr
        self.pmhc = pmhc

    def __repr__(self) -> str:
        return f"({self.tcr}) - ({self.pmhc})"