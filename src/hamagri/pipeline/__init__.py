"""Event pipeline: FFT scan over an atlas, battle-royale search, VBM polish."""

from .battle import Settings, solve
from .hills import Landscape
from .objective import EventModel, lm
from .parallax import Parallax
from .run import solve_event
from .scan import GeometryProfiles, geometry_stencil, scan_profiles

__all__ = [
    "EventModel",
    "GeometryProfiles",
    "Landscape",
    "Parallax",
    "Settings",
    "geometry_stencil",
    "lm",
    "scan_profiles",
    "solve",
    "solve_event",
]
