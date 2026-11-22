"""Per-geometry alpha-FFT profiles over a whole atlas.

The battle royale starts from the full ``(map, geometry)`` chi-square table:
for every map and every point-lens geometry of a small stencil, the best
chi-square over the alpha grid and its alpha bin.  Unlike :func:`hamagri.search`
nothing is ranked or discarded here, so neighbouring maps keep their basin
shape for the watershed in :mod:`hamagri.pipeline.hills`.

Light curves that leave the stored radial range (long seasons, small tE) get a
point-lens tail appended to every map in memory; the atlas is never modified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .._core.config import PSPLGeometry
from .._core.jax_backend import JAXConsistentGeometryBatchScanner
from .._core.multires import _padded_scan_batch
from .._core.trajectory import ConsistentKernelFactory, PhotometricDataset, radial_coordinates
from ..maps import Atlas

# Wide maps (s above this) are centred on the primary in its own Einstein
# units, as in ``_core.map_adapter``; their far field is a point lens of mass
# 1 / (1 + q) in total-mass units.
WIDE_PRIMARY_S = 4.25


def geometry_stencil(t0: float, u0: float, tE: float, *, frame: str = "map",
                     dt0: float | None = None, du0: float | None = None,
                     dlog_tE: float = 0.15) -> list[PSPLGeometry]:
    """16 point-lens geometries around a PSPL fit ``(t0, u0, tE)``.

    The centre, +-1 step on each axis, the eight corners of the unit cube and
    one off-lattice point (-1/2, -1/2, +1/2).  Steps default to
    ``max(2, 0.1 tE)`` days, ``max(0.03, 0.1 |u0|)`` and 0.15 in ln tE.
    """
    dt0 = max(2.0, 0.1 * float(tE)) if dt0 is None else float(dt0)
    du0 = max(0.03, 0.1 * abs(float(u0))) if du0 is None else float(du0)
    offsets = [(0.0, 0.0, 0.0)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            offset = [0.0, 0.0, 0.0]
            offset[axis] = sign
            offsets.append(tuple(offset))
    offsets += [(a, b, c) for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)]
    offsets.append((-0.5, -0.5, 0.5))
    geometries = [
        PSPLGeometry(float(t0) + a * dt0, float(u0) + b * du0,
                     float(tE) * float(np.exp(c * dlog_tE)), coordinate_frame=frame)
        for a, b, c in offsets
    ]
    for geometry in geometries:
        geometry.validate()
    return geometries


@dataclass
class GeometryProfiles:
    """Best alpha-FFT chi-square of every (map, geometry) pair.

    ``parameters`` rows are the maps' ``(log10 s, log10 q, log10 rho)``;
    ``alpha_index_by_geometry`` counts bins of a ``n_alpha`` grid over 2 pi.
    """

    map_ids: np.ndarray
    parameters: np.ndarray
    chi2_by_geometry: np.ndarray
    alpha_index_by_geometry: np.ndarray
    geometries: np.ndarray  # (n_geometry, 3): t0, u0, tE
    n_alpha: int
    coordinate_frame: str = "map"
    scan_seconds: float = 0.0

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, map_ids=self.map_ids, parameters=self.parameters,
            chi2_by_geometry=self.chi2_by_geometry,
            alpha_index_by_geometry=self.alpha_index_by_geometry,
            geometries=self.geometries, n_alpha=np.int64(self.n_alpha),
            coordinate_frame=np.str_(self.coordinate_frame),
            scan_seconds=np.float64(self.scan_seconds),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeometryProfiles":
        with np.load(path) as data:
            return cls(
                map_ids=np.asarray(data["map_ids"], dtype=np.int64),
                parameters=np.asarray(data["parameters"], dtype=np.float64),
                chi2_by_geometry=np.asarray(data["chi2_by_geometry"], dtype=np.float64),
                alpha_index_by_geometry=np.asarray(data["alpha_index_by_geometry"], dtype=np.int64),
                geometries=np.asarray(data["geometries"], dtype=np.float64),
                n_alpha=int(data["n_alpha"]),
                coordinate_frame=str(data["coordinate_frame"]) if "coordinate_frame" in data else "map",
                scan_seconds=float(data["scan_seconds"]) if "scan_seconds" in data else 0.0,
            )


def _max_radius(datasets, geometries) -> float:
    radius = 0.0
    for dataset in datasets:
        for geometry in geometries:
            r, _ = radial_coordinates(dataset.time, geometry, dataset.trajectory_offset)
            radius = max(radius, float(np.max(r)))
    return radius


def _tail_nodes(nodes: np.ndarray, radius: float, radial_order: int) -> np.ndarray | None:
    epsilon = max(1.0e-8, 1.0e-6 * max(1.0, abs(float(nodes[-1]))))
    if not np.isfinite(radius) or radius <= float(nodes[-1]) + epsilon:
        return None
    return np.linspace(float(nodes[-1]) + epsilon, radius, max(int(radial_order) + 1, 2))


def _append_tail(coefficients: np.ndarray, tail: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Append point-lens excess rows (m = 0 only) beyond the stored radii."""
    r2 = np.broadcast_to(tail[None, :] ** 2, (len(coefficients), len(tail))).copy()
    wide = np.power(10.0, parameters[:, 0]) > WIDE_PRIMARY_S
    r2[wide] *= 1.0 + np.power(10.0, parameters[wide, 1])[:, None]
    rows = np.zeros((len(coefficients), len(tail), coefficients.shape[-1]), dtype=coefficients.dtype)
    rows[:, :, 0] = (r2 + 2.0) / np.sqrt(r2 * (r2 + 4.0)) - 1.0
    return np.concatenate((np.asarray(coefficients), rows), axis=1)


def _sources(maps: Atlas, m_max: int):
    """Yield ``(radial_nodes, shards)`` for every bucket of ``maps``."""
    if maps._bucketed:
        for bucket in maps._core.iter_buckets():
            yield np.asarray(bucket.atlas.radial_nodes), bucket.atlas.iter_shards(m_max=m_max)
    else:
        yield np.asarray(maps._core.radial_nodes), maps._core.iter_shards(m_max=m_max)


def scan_profiles(maps: Atlas, datasets: Sequence[PhotometricDataset],
                  geometries: Sequence[PSPLGeometry], *, m_max: int = 128,
                  n_alpha: int = 2048, radial_order: int = 1,
                  batch_size: int = 1024, progress=None) -> GeometryProfiles:
    """Scan every map at every geometry, keeping the alpha-minimum per geometry."""
    datasets, geometries = tuple(datasets), tuple(geometries)
    if maps.coordinate_frame is None:
        raise ValueError("the atlas does not declare its coordinate frame")
    for geometry in geometries:
        geometry.validate()
        if geometry.coordinate_frame != maps.coordinate_frame:
            raise ValueError(f"geometries must use the atlas {maps.coordinate_frame!r} frame")
    if m_max > maps.m_max:
        raise ValueError("m_max exceeds the modes stored in these maps")
    started = time.perf_counter()
    factory = ConsistentKernelFactory(datasets, geometries, m_max, radial_order=radial_order)
    radius = _max_radius(datasets, geometries)
    scanner = None
    ids, params, chi2, alpha = [], [], [], []
    for bucket_index, (nodes, shards) in enumerate(_sources(maps, m_max)):
        tail = _tail_nodes(nodes, radius, radial_order)
        full_nodes = nodes if tail is None else np.concatenate((nodes, tail))
        kernels = factory(full_nodes, m_max)
        if scanner is None or scanner_nodes != len(full_nodes):
            scanner = JAXConsistentGeometryBatchScanner(kernels, n_alpha=n_alpha)
            scanner_nodes = len(full_nodes)
        else:
            scanner.set_kernels(kernels)
        # Shards can hold a handful of maps; scanning them one by one would pad
        # every call to the scan quantum, so rows are gathered into batches.
        pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def scan(final: bool):
            held = sum(len(p[0]) for p in pending)
            if not pending or (held < batch_size and not final):
                return
            batch = [np.concatenate([p[k] for p in pending]) for k in range(3)]
            pending.clear()
            for start in range(0, held, batch_size):
                stop = start + batch_size
                if stop > held and not final:
                    pending.append(tuple(b[start:] for b in batch))
                    return
                rows_ids, rows_params, coefficients = (b[start:stop] for b in batch)
                if tail is not None:
                    coefficients = _append_tail(coefficients, tail, rows_params)
                padded, count = _padded_scan_batch(coefficients)
                c, a = scanner.scan_minima_by_geometry(padded)
                ids.append(rows_ids)
                params.append(rows_params)
                chi2.append(np.asarray(c[:count], dtype=np.float64))
                alpha.append(np.asarray(a[:count], dtype=np.int64))

        for shard in shards:
            pending.append((np.asarray(shard.map_ids, dtype=np.int64),
                            np.asarray(shard.parameters, dtype=np.float64), np.asarray(shard.x_coeff)))
            scan(final=False)
        scan(final=True)
        if progress is not None:
            progress(bucket_index, sum(len(x) for x in ids))
    return GeometryProfiles(
        map_ids=np.concatenate(ids), parameters=np.concatenate(params),
        chi2_by_geometry=np.concatenate(chi2), alpha_index_by_geometry=np.concatenate(alpha),
        geometries=np.asarray([[g.t0, g.u0, g.tE] for g in geometries], dtype=np.float64),
        n_alpha=int(n_alpha), coordinate_frame=maps.coordinate_frame,
        scan_seconds=time.perf_counter() - started,
    )
