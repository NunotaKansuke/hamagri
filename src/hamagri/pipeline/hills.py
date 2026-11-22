"""FFT hills and the search regions around them.

Every row of a :class:`~hamagri.pipeline.scan.GeometryProfiles` table (atlas
map x trial geometry) is a point in the map-frame vector

    v = (t0, u0 tE, ln tE, log10 s, log10 q, ln(rho tE), alpha)

with its FFT chi-square.  A *hill* is the minimum of a strict downhill basin
of the (log s, log q, log rho) grid inside one trial geometry; geometry is
categorical, so basins never cross geometries.

Around one hill the rows of a (log s, log q) window are weighted by a tempered
likelihood ``exp(-(chi2 - chi2_min) / (2 T))`` with ``T`` chosen so that the
effective number of weighted rows is ``TARGET_NEFF``.  Their weighted
covariance, floored by half a grid cell per axis, is the hill's *region*: its
short axes are what the FFT already pins down and its long axes the valley
directions a search should spread along.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._core.caustics import adamgrid_map_origin_shift, native_pspl_parameters_from_map
from .scan import GeometryProfiles

NAMES = ("t0", "u0tE", "lntE", "logs", "logq", "lnrhotE", "alpha")
WINDOW_CELLS = 4  # half-width of the (log s, log q) window in grid cells
TARGET_NEFF = 40.0


def to_vector(t0, u0, tE, s, q, rho, alpha) -> np.ndarray:
    """Map-frame vector of map-frame parameters."""
    return np.array([t0, u0 * tE, np.log(tE), np.log10(s), np.log10(q), np.log(rho * tE), alpha],
                    dtype=np.float64)


def from_vector(v) -> dict[str, float]:
    """Map-frame parameters of a vector (alpha wrapped to [0, 2 pi))."""
    tE = float(np.exp(v[2]))
    return {
        "t0": float(v[0]),
        "u0": float(v[1]) / tE,
        "tE": tE,
        "s": float(10.0 ** v[3]),
        "q": float(10.0 ** v[4]),
        "rho": float(np.exp(v[5])) / tE,
        "alpha": float(np.mod(v[6], 2.0 * np.pi)),
    }


def native_parameters(v) -> dict[str, float]:
    """VBM-frame (centre-of-mass) parameters of a map-frame vector."""
    p = from_vector(v)
    p["t0"], p["u0"] = native_pspl_parameters_from_map(
        p["t0"], p["u0"], p["tE"], p["s"], p["q"], p["alpha"])
    return p


def map_vector(p: dict[str, float]) -> np.ndarray:
    """Map-frame vector of VBM-frame parameters."""
    shift = adamgrid_map_origin_shift(p["s"], p["q"])
    t0 = p["t0"] - shift * p["tE"] * np.cos(p["alpha"])
    u0 = p["u0"] - shift * np.sin(p["alpha"])
    return to_vector(t0, u0, p["tE"], p["s"], p["q"], p["rho"], p["alpha"])


def watershed_basins(parameters: np.ndarray, chi2: np.ndarray, map_ids: np.ndarray) -> list[dict[str, Any]]:
    """Group grid points into strict 6-neighbour downhill basins."""
    axes = [np.unique(parameters[:, axis]) for axis in range(3)]
    cells = np.column_stack([np.searchsorted(axes[axis], parameters[:, axis]) for axis in range(3)])
    row_by_cell: dict[tuple[int, int, int], int] = {}
    for row, cell in enumerate(map(tuple, cells.tolist())):
        if cell in row_by_cell:
            raise ValueError(f"duplicate atlas coordinate {parameters[row].tolist()}")
        row_by_cell[cell] = row
    parent = np.arange(len(chi2), dtype=np.int64)
    for row, cell in enumerate(cells.tolist()):
        lower = []
        for dimension in range(3):
            for step in (-1, 1):
                neighbour = list(cell)
                neighbour[dimension] += step
                other = row_by_cell.get(tuple(neighbour))
                if other is not None and chi2[other] < chi2[row]:
                    lower.append(other)
        if lower:
            parent[row] = min(lower, key=lambda index: (chi2[index], map_ids[index]))

    def root_of(row: int) -> int:
        trail = []
        while parent[row] != row:
            trail.append(row)
            row = int(parent[row])
        parent[trail] = row
        return row

    roots: dict[int, int] = {}
    for row in range(len(chi2)):
        root = root_of(row)
        roots[root] = roots.get(root, 0) + 1
    return [{"root_row": root, "chi2": float(chi2[root]), "map_id": int(map_ids[root]), "size": size}
            for root, size in roots.items()]


class Landscape:
    """All finite FFT rows of one event as map-frame vectors with their chi2."""

    def __init__(self, profiles: GeometryProfiles):
        if profiles.coordinate_frame != "map":
            raise ValueError("search regions need map-frame geometries")
        n_map, n_geometry = profiles.chi2_by_geometry.shape
        self.profiles = profiles
        self.n_alpha = int(profiles.n_alpha)
        self.map_ids = np.repeat(profiles.map_ids, n_geometry)
        self.parameters = np.repeat(profiles.parameters, n_geometry, axis=0)
        self.geometry_index = np.tile(np.arange(n_geometry), n_map)
        chi2 = profiles.chi2_by_geometry.reshape(-1)
        alpha_index = profiles.alpha_index_by_geometry.reshape(-1)
        keep = np.isfinite(chi2)
        if not keep.any():
            raise ValueError("no finite FFT rows")
        self.map_ids, self.parameters = self.map_ids[keep], self.parameters[keep]
        self.geometry_index, self.chi2 = self.geometry_index[keep], chi2[keep]
        self.alpha_index = alpha_index[keep]
        geometries = profiles.geometries
        geometry = geometries[self.geometry_index]
        tE = geometry[:, 2]
        self.v = np.column_stack([
            geometry[:, 0], geometry[:, 1] * tE, np.log(tE),
            self.parameters[:, 0], self.parameters[:, 1],
            np.log(10.0 ** self.parameters[:, 2] * tE),
            2.0 * np.pi * self.alpha_index / self.n_alpha,
        ])
        self.cell = np.array([_median_step(self.parameters[:, k]) for k in range(3)])
        spread = np.array([
            np.ptp(geometries[:, 0]),
            np.ptp(geometries[:, 1] * geometries[:, 2]),
            np.ptp(np.log(geometries[:, 2])),
        ])
        # Half a grid step per axis: the FFT cannot resolve anything finer.
        self.floor = 0.5 * np.array([
            max(spread[0] / 2.0, 1e-3),
            max(spread[1] / 2.0, 1e-4),
            max(spread[2] / 2.0, 1e-3),
            self.cell[0],
            self.cell[1],
            self.cell[2] * np.log(10.0),
            2.0 * np.pi / self.n_alpha * 4.0,
        ])

    def hills(self, count: int = 24) -> list[dict[str, Any]]:
        """The lowest basin minima; the best basin of the two best geometries first."""
        basins = []
        for geometry in np.unique(self.geometry_index):
            rows = np.flatnonzero(self.geometry_index == geometry)
            for basin in watershed_basins(self.parameters[rows], self.chi2[rows], self.map_ids[rows]):
                basins.append({**basin, "root_row": int(rows[basin["root_row"]])})
        basins.sort(key=lambda basin: (basin["chi2"], basin["map_id"]))
        best_by_geometry: dict[int, dict[str, Any]] = {}
        for basin in basins:
            best_by_geometry.setdefault(int(self.geometry_index[basin["root_row"]]), basin)
        selected = list(best_by_geometry.values())[: min(2, count)]
        chosen = {basin["root_row"] for basin in selected}
        selected += [basin for basin in basins if basin["root_row"] not in chosen][: count - len(selected)]
        return [{
            "map_id": int(self.map_ids[basin["root_row"]]),
            "geometry_index": int(self.geometry_index[basin["root_row"]]),
            "fft_chi2": float(self.chi2[basin["root_row"]]),
            "vector": self.v[basin["root_row"]].copy(),
        } for basin in selected]

    def region(self, center: np.ndarray, *, window_cells: int = WINDOW_CELLS,
               target_neff: float = TARGET_NEFF) -> dict[str, Any]:
        """Tempered weighted mean and covariance of the rows around ``center``."""
        v = self.v
        inside = ((np.abs(v[:, 3] - center[3]) <= window_cells * self.cell[0] + 1e-9)
                  & (np.abs(v[:, 4] - center[4]) <= window_cells * self.cell[1] + 1e-9))
        sub = v[inside].copy()
        chi2 = self.chi2[inside]
        sub[:, 6] = center[6] + np.mod(sub[:, 6] - center[6] + np.pi, 2.0 * np.pi) - np.pi
        delta = chi2 - chi2.min()
        temperature = _temperature_for_neff(delta, target_neff)
        weight = np.exp(-0.5 * delta / temperature)
        weight /= weight.sum()
        mean = weight @ sub
        centered = sub - mean
        cov = (centered * weight[:, None]).T @ centered + np.diag(self.floor ** 2)
        return {"mean": mean, "cov": cov, "temperature": float(temperature),
                "n_rows": int(inside.sum()), "neff": float(1.0 / np.sum(weight ** 2))}


def _median_step(values: np.ndarray) -> float:
    unique = np.unique(values)
    return float(np.median(np.diff(unique))) if len(unique) > 1 else 0.1


def _temperature_for_neff(delta: np.ndarray, target: float) -> float:
    target = min(target, 0.5 * len(delta))
    lo, hi = -6.0, 9.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        w = np.exp(-0.5 * delta / 10.0 ** mid)
        if w.sum() ** 2 / np.sum(w ** 2) < target:
            lo = mid
        else:
            hi = mid
    return float(10.0 ** hi)
