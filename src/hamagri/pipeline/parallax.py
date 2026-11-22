"""Annual and space-site parallax through VBMicrolensing.

Everything here is VBMicrolensing's own parallax: the sky position, the
reference time ``t_ref`` (VBM ``t0_par``, held fixed) and, for a satellite, a
JPL Horizons position table in VBM's ``satellite<N>.txt`` format.  The source
offset is linear in ``(piEN, piEE)``, so one basis per dataset gives the
per-epoch ``(dtau, dbeta)`` of any parallax vector; at fixed ``piE`` this is the
``trajectory_offset`` the atlas scan uses.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _sexagesimal(ra_deg: float, dec_deg: float) -> str:
    hours = (float(ra_deg) % 360.0) / 15.0
    hh = int(hours)
    mm = int((hours - hh) * 60.0)
    ss = ((hours - hh) * 60.0 - mm) * 60.0
    sign = "-" if dec_deg < 0.0 else "+"
    dec = abs(float(dec_deg))
    dd = int(dec)
    dm = int((dec - dd) * 60.0)
    ds = ((dec - dd) * 60.0 - dm) * 60.0
    return f"{hh:02d}:{mm:02d}:{ss:07.4f} {sign}{dd:02d}:{dm:02d}:{ds:07.4f}"


@dataclass(frozen=True)
class Parallax:
    """Parallax set-up of one event.

    ``satellite`` is VBM's satellite index (0 = geocentric observer); for
    ``satellite > 0`` the table ``satellite_dir/satellite<N>.txt`` must cover
    every epoch.  Times use the data's own convention (VBM accepts HJD - 2450000).
    """

    ra_deg: float
    dec_deg: float
    t_ref: float
    satellite: int = 0
    satellite_dir: str | None = None

    def __post_init__(self) -> None:
        if not all(np.isfinite((self.ra_deg, self.dec_deg, self.t_ref))):
            raise ValueError("ra_deg, dec_deg and t_ref must be finite")
        if self.satellite < 0:
            raise ValueError("satellite must be >= 0")
        if self.satellite > 0:
            if self.satellite_dir is None:
                raise ValueError("a satellite needs satellite_dir")
            table = Path(self.satellite_dir) / f"satellite{self.satellite}.txt"
            if not table.is_file():
                raise FileNotFoundError(table)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Parallax":
        return cls(
            ra_deg=float(payload["ra_deg"]),
            dec_deg=float(payload["dec_deg"]),
            t_ref=float(payload["t_ref"]),
            satellite=int(payload.get("satellite", 0)),
            satellite_dir=payload.get("satellite_dir"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "t_ref": self.t_ref,
            "satellite": self.satellite,
            "satellite_dir": self.satellite_dir,
        }

    def configure(self, vbm: Any) -> Any:
        """Set ``vbm``'s coordinates, satellite and fixed ``t0_par``."""
        # VBM reads the coordinates from a file when it also needs the
        # satellite directory.
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "coordinates.txt")
            with open(path, "w", encoding="ascii") as stream:
                stream.write(_sexagesimal(self.ra_deg, self.dec_deg))
            directory = self.satellite_dir if self.satellite_dir is not None else folder
            vbm.SetObjectCoordinates(path, str(directory))
        vbm.satellite = int(self.satellite)
        vbm.parallaxsystem = 1  # (piEN, piEE)
        vbm.parallaxephemeris = 1
        vbm.t0_par_fixed = 1
        vbm.t0_par = float(self.t_ref)
        return vbm

    def offset_basis(self, time: np.ndarray, vbm: Any | None = None) -> np.ndarray:
        """``(n, 2, 2)`` array B with ``(dtau, dbeta) = B @ (piEN, piEE)``."""
        time = np.asarray(time, dtype=np.float64)
        if vbm is None:
            import VBMicrolensing

            vbm = self.configure(VBMicrolensing.VBMicrolensing())
        times = time.tolist()
        # alpha = 0: VBM's source position is the negated (tau, beta).
        base = [0.0, np.log(1.0e-6), 0.0, 0.0, np.log(1.0e-4), 0.0, float(self.t_ref)]
        _, x0, y0 = vbm.BinaryLightCurveParallax(base + [0.0, 0.0], times)
        basis = np.empty((time.size, 2, 2), dtype=np.float64)
        for column, pi in enumerate(([1.0, 0.0], [0.0, 1.0])):
            _, x, y = vbm.BinaryLightCurveParallax(base + pi, times)
            basis[:, 0, column] = -(np.asarray(x) - np.asarray(x0))
            basis[:, 1, column] = -(np.asarray(y) - np.asarray(y0))
        if not np.all(np.isfinite(basis)):
            raise ValueError("VBM parallax is not finite (satellite table range?)")
        return basis

    def trajectory_offset(self, time: np.ndarray, piEN: float, piEE: float) -> np.ndarray:
        """``(n, 2)`` per-epoch ``(dtau, dbeta)`` at fixed ``(piEN, piEE)``."""
        return self.offset_basis(time) @ np.array([piEN, piEE], dtype=np.float64)
