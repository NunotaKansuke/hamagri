"""One event from photometry to solutions: FFT scan, battle royale, polish."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

from .._core.trajectory import PhotometricDataset
from ..maps import Atlas
from .battle import Settings, solve
from .hills import Landscape
from .objective import EventModel
from .parallax import Parallax
from .scan import GeometryProfiles, geometry_stencil, scan_profiles


def solve_event(
    maps: Atlas | str | Path,
    datasets: Sequence[PhotometricDataset],
    pspl: tuple[float, float, float],
    *,
    parallax: Parallax | None = None,
    piE: tuple[float, float] = (0.0, 0.0),
    workers: int = 4,
    seed: int = 0,
    settings: Settings | None = None,
    m_max: int = 128,
    n_alpha: int = 2048,
    profiles: GeometryProfiles | str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Search one event over a whole atlas.

    ``pspl`` is a point-lens fit ``(t0, u0, tE)``; the atlas is scanned on a
    16-geometry stencil around it.  With ``parallax`` the scan and the search
    hold ``piE`` (a host or point-lens value) fixed, and the final step frees
    it.  ``profiles`` reuses a saved scan (path or object); a path that does
    not exist yet receives the new scan.
    """
    say = log or (lambda message: None)
    datasets = tuple(datasets)
    started = time.perf_counter()
    if parallax is not None:
        # The atlas scan sees parallax as a per-epoch trajectory offset at fixed piE.
        datasets = tuple(replace(d, trajectory_offset=parallax.trajectory_offset(d.time, *piE))
                         for d in datasets)
    save_to = None
    if isinstance(profiles, (str, Path)):
        if Path(profiles).exists():
            profiles = GeometryProfiles.load(profiles)
        else:
            save_to, profiles = Path(profiles), None
    if profiles is None:
        atlas = maps if isinstance(maps, Atlas) else Atlas(maps)
        geometries = geometry_stencil(*pspl, frame=atlas.coordinate_frame or "map")
        say(f"scanning {len(atlas.map_ids)} maps at {len(geometries)} geometries")
        profiles = scan_profiles(atlas, datasets, geometries, m_max=m_max, n_alpha=n_alpha)
        say(f"scan done in {profiles.scan_seconds:.0f}s")
        if save_to is not None:
            profiles.save(save_to)
    model = EventModel(datasets, parallax, piE)
    result = solve(model, Landscape(profiles), workers=workers, seed=seed, settings=settings, log=say)
    result["scan_seconds"] = float(profiles.scan_seconds)
    result["n_maps"] = int(len(profiles.map_ids))
    result["pspl"] = [float(v) for v in pspl]
    result["total_seconds"] = time.perf_counter() - started
    return result
