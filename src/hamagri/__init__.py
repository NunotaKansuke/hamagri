"""hamagri: fast binary-microlensing map building and seed searches."""

from .build import build_maps, merge_maps
from .config import MapConfig, ParameterGrid, Partition, SearchConfig
from .maps import Atlas, Maps
from .search import Candidate, Dataset, Geometry, SearchResult, search
from .fit import (
    CandidateFitResult,
    DirectCandidateScore,
    FitConfig,
    FitParameters,
    FitResult,
    fit_candidates,
)

__all__ = [
    "MapConfig",
    "Atlas",
    "Maps",
    "Candidate",
    "CandidateFitResult",
    "DirectCandidateScore",
    "FitConfig",
    "FitParameters",
    "FitResult",
    "Dataset",
    "Geometry",
    "ParameterGrid",
    "Partition",
    "SearchConfig",
    "SearchResult",
    "build_maps",
    "fit_candidates",
    "merge_maps",
    "search",
]
