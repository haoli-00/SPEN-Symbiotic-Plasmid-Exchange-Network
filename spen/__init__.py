"""
SPEN: Symbiotic Plasmid Exchange Network

A bio-inspired deep learning paradigm where neural network modules
(plasmids) evolve independently and flow horizontally across a
population of host models through quorum-gated conjugation.
"""

from .plasmid import Plasmid, PlasmidConfig
from .host import HostCell, HostConfig
from .conjugation import ConjugationBridge
from .quorum import QuorumSensor
from .crispr import CRISPRBank
from .mycorrhiza import MycorrhizalSpace
from .ecosystem import Ecosystem

__version__ = "0.4.0"

__all__ = [
    "Plasmid",
    "PlasmidConfig",
    "HostCell",
    "HostConfig",
    "ConjugationBridge",
    "QuorumSensor",
    "CRISPRBank",
    "MycorrhizalSpace",
    "Ecosystem",
]
