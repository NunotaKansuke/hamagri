"""High-level, memory-mapped map collection."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ._core.atlas import PolarAtlas
from ._core.bucketed_atlas import BucketedPolarAtlas
from ._core.radial import interpolate_coefficients


class Atlas:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        manifest = json.loads((self.path / "manifest.json").read_text())
        self._bucketed = manifest.get("format") == "hamagri-bucketed-fourier-maps"
        self._core = (
            BucketedPolarAtlas(self.path) if self._bucketed else PolarAtlas(self.path)
        )
        self.coordinate_frame = self._read_coordinate_frame(manifest)

    def _read_coordinate_frame(self, manifest: dict) -> str | None:
        """Read explicit frame metadata or the legacy atlas angle convention."""
        declared = manifest.get("coordinate_frame")
        if declared is not None and declared not in {"map", "native"}:
            raise ValueError(
                "atlas coordinate_frame must be either 'map' or 'native'"
            )
        conventions = {
            "hamagri-map-frame-radians": "map",
            "hamagri-native-frame-radians": "native",
        }
        if self._bucketed:
            angle_conventions = {
                bucket.atlas.manifest.get("angle_convention")
                for bucket in self._core.buckets
            }
            if len(angle_conventions) > 1:
                raise ValueError(
                    "bucketed atlas mixes angle conventions; a single search "
                    "coordinate frame cannot be inferred"
                )
            inferred = conventions.get(next(iter(angle_conventions), None))
        else:
            inferred = conventions.get(manifest.get("angle_convention"))
        if declared is not None and inferred is not None and declared != inferred:
            raise ValueError(
                "atlas coordinate_frame conflicts with its angle_convention"
            )
        return declared or inferred

    @classmethod
    def open(cls, path: str | Path) -> "Atlas":
        return cls(path)

    @property
    def parameters(self) -> np.ndarray:
        """Rows ``(log10(s), log10(q), log10(rho))`` aligned with ``map_ids``."""
        return np.asarray(self._core.map_parameters)

    @property
    def map_ids(self) -> np.ndarray:
        return np.asarray(self._core.map_ids)

    @property
    def n_maps(self) -> int:
        return int(self.map_ids.size)

    @property
    def radial_nodes(self) -> np.ndarray:
        if self._bucketed:
            raise AttributeError(
                "radial nodes vary by (s,q) bucket; use radial_nodes_for(map_id)"
            )
        return np.asarray(self._core.radial_nodes)

    def radial_nodes_for(self, map_id: int) -> np.ndarray:
        if self._bucketed:
            nodes, _, _ = self._core.coefficient_row(int(map_id), m_max=0)
            return nodes
        self._core.parameters_for(int(map_id))
        return np.asarray(self._core.radial_nodes)

    @property
    def m_max(self) -> int:
        return self._core.m_max

    def magnification(
        self,
        map_id: int,
        x: np.ndarray,
        y: np.ndarray,
        *,
        m_max: int | None = None,
        radial_order: int = 1,
    ) -> np.ndarray:
        """Reconstruct a magnification map at arbitrary Cartesian points."""
        modes = self.m_max if m_max is None else int(m_max)
        if self._bucketed:
            nodes, _, coefficients = self._core.coefficient_row(map_id, m_max=modes)
        else:
            nodes = np.asarray(self._core.radial_nodes)
            coefficients = self._core.coefficient_rows([map_id], m_max=modes)[
                int(map_id)
            ][1]
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        radius = np.hypot(x, y).ravel()
        phase = np.arctan2(y, x).ravel()
        local = interpolate_coefficients(
            coefficients, radius, nodes, order=radial_order
        )
        mode = np.arange(modes + 1)
        excess = np.real(
            local[:, 0]
            + 2.0 * np.sum(local[:, 1:] * np.exp(1j * phase[:, None] * mode[None, 1:]), axis=1)
        )
        return (1.0 + excess).reshape(x.shape)

    def search(self, datasets, geometries, *, config=None):
        from .search import search

        return search(self, datasets, geometries, config=config)


# ``Atlas`` is the release-facing name; ``Maps`` remains as a compatibility alias.
Maps = Atlas
