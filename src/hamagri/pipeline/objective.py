"""Profiled direct-VBM residuals and Levenberg-Marquardt polish.

Every light curve is VBMicrolensing's own (``BinaryLightCurve``, or
``BinaryLightCurveParallax`` when the event has parallax), in VBM's native
centre-of-mass frame.  Source and blend fluxes are profiled per dataset.  The
Jacobian is a central finite difference in the optimizer coordinates with
steps tied to the source radius, so it resolves caustic-crossing structure.

Optimizer coordinates ``z`` (all zero at the start point ``initial``)::

    z0 = (t0 - t0_i) / t0_scale      z3 = log(s / s_i)
    z1 = (u0 - u0_i) / u0_scale      z4 = log(q / q_i)
    z2 = log(tE / tE_i)              z5 = log(rho / rho_i)
    z6 = alpha - alpha_i             z7, z8 = piEN, piEE shifts (free parallax)

LM itself runs in the timescale chart (:func:`timescale_chart`), where y1
moves ``u0 tE`` and y5 moves ``log(rho tE)``: those are the combinations a
light curve pins down, so ``tE`` moves without dragging the caustic timing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .._core.reference import profile_flux
from .._core.trajectory import PhotometricDataset
from .parallax import Parallax

PARAMETERS = ("t0", "u0", "tE", "s", "q", "rho", "alpha")
COARSE = (1.0e-3, 1.0e-4)  # VBM (Tol, RelTol) of the search and LM
FINE = (1.0e-5, 1.0e-6)  # VBM (Tol, RelTol) of every reported chi2
JACOBIAN = (1.0e-4, 1.0e-5)  # VBM (Tol, RelTol) of finite-difference columns
INVALID_RESIDUAL = 1.0e6


def _new_vbm(tolerance: tuple[float, float], parallax: Parallax | None) -> Any:
    try:
        import VBMicrolensing
    except ImportError as error:  # pragma: no cover - VBM is a dependency
        raise ImportError("the pipeline requires VBMicrolensing") from error
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol, vbm.RelTol = tolerance
    vbm.a1 = 0.0
    if parallax is not None:
        parallax.configure(vbm)
    return vbm


@dataclass
class EventModel:
    """Datasets of one event and, optionally, their parallax set-up.

    ``parallax`` is one :class:`Parallax` for every dataset, or one entry
    (``None`` for no parallax) per dataset.  ``piE`` is the host ``(piEN,
    piEE)`` used whenever a parameter dictionary does not carry its own.
    """

    datasets: Sequence[PhotometricDataset]
    parallax: Parallax | Sequence[Parallax | None] | None = None
    piE: tuple[float, float] = (0.0, 0.0)
    _vbm: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.datasets = tuple(self.datasets)
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        if self.parallax is None or isinstance(self.parallax, Parallax):
            self.parallax = (self.parallax,) * len(self.datasets)
        self.parallax = tuple(self.parallax)
        if len(self.parallax) != len(self.datasets):
            raise ValueError("one parallax entry per dataset is required")
        self.piE = (float(self.piE[0]), float(self.piE[1]))
        self.n_points = int(sum(len(d.time) for d in self.datasets))

    def __getstate__(self) -> dict:
        # VBM instances do not pickle; a worker process builds its own.
        return {**self.__dict__, "_vbm": {}}

    @property
    def has_parallax(self) -> bool:
        return any(p is not None for p in self.parallax)

    def vbm(self, index: int, tolerance: tuple[float, float]) -> Any:
        key = (index, tolerance)
        if key not in self._vbm:
            self._vbm[key] = _new_vbm(tolerance, self.parallax[index])
        return self._vbm[key]

    def magnifications(self, p: dict[str, float], tolerance=COARSE) -> list[np.ndarray]:
        """VBM magnification of every dataset at native parameters ``p``."""
        base = [
            float(np.log(p["s"])), float(np.log(p["q"])), float(p["u0"]), float(p["alpha"]),
            float(np.log(p["rho"])), float(np.log(p["tE"])), float(p["t0"]),
        ]
        piE = [float(p.get("piEN", self.piE[0])), float(p.get("piEE", self.piE[1]))]
        out = []
        for index, dataset in enumerate(self.datasets):
            vbm = self.vbm(index, tolerance)
            times = dataset.time.tolist()
            if self.parallax[index] is None:
                values = vbm.BinaryLightCurve(base, times)[0]
            else:
                values = vbm.BinaryLightCurveParallax(base + piE, times)[0]
            out.append(np.asarray(values, dtype=np.float64))
        return out

    def residual(self, p: dict[str, float], tolerance=COARSE) -> np.ndarray:
        pieces = []
        for dataset, magnification in zip(self.datasets, self.magnifications(p, tolerance)):
            profile = profile_flux(magnification, dataset)
            if not np.isfinite(profile.chi2):
                raise ValueError(f"profiled flux fit failed for dataset {dataset.name!r}")
            model = profile.source_flux * magnification + profile.blend_flux
            piece = (dataset.flux - model) / dataset.error
            if not np.all(np.isfinite(piece)):
                raise ValueError(f"non-finite residual for dataset {dataset.name!r}")
            pieces.append(piece)
        return np.concatenate(pieces)

    def chi2(self, p: dict[str, float], tolerance=FINE) -> float:
        r = self.residual(p, tolerance)
        return float(r @ r)

    def fluxes(self, p: dict[str, float], tolerance=FINE) -> list[dict[str, float]]:
        rows = []
        for dataset, magnification in zip(self.datasets, self.magnifications(p, tolerance)):
            profile = profile_flux(magnification, dataset)
            rows.append({"name": dataset.name, "source_flux": profile.source_flux,
                         "blend_flux": profile.blend_flux, "chi2": profile.chi2})
        return rows


class Objective:
    """Residual and finite-difference Jacobian in ``z`` around ``initial``."""

    def __init__(self, model: EventModel, initial: dict[str, float], *,
                 tolerance=COARSE, jacobian_tolerance=JACOBIAN, central: bool = False,
                 free_parallax: bool = False) -> None:
        self.model = model
        self.initial = {k: float(v) for k, v in initial.items()}
        self.initial.setdefault("piEN", model.piE[0])
        self.initial.setdefault("piEE", model.piE[1])
        if free_parallax and not model.has_parallax:
            raise ValueError("free parallax needs a parallax set-up")
        self.free_parallax = bool(free_parallax)
        self.n = 9 if free_parallax else 7
        self.tolerance = tolerance
        # Finite differences need far less VBM noise than the step: at the
        # COARSE tolerance some gradient signs flip at a 1e-3 rho step.  One
        # tighter tolerance with forward differences reached the same minima
        # as FINE central differences at a third of the cost.
        self.jacobian_tolerance = jacobian_tolerance
        self.central = bool(central)
        self.t0_scale = max(abs(self.initial["tE"]), 1.0)
        self.u0_scale = max(abs(self.initial["u0"]), self.initial["rho"], 0.01)
        self.calls = 0
        self.jacobian_calls = 0
        self.last_error: str | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def unpack(self, z) -> dict[str, float] | None:
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (self.n,) or not np.all(np.isfinite(z)):
            return None
        # Numerical guards only: runaway LM probes become an invalid residual
        # before exp() or VBM sees them.
        if np.any(np.abs(z[2:6]) > 30.0) or abs(float(z[0])) > 1.0e6:
            return None
        i = self.initial
        p = {
            "t0": i["t0"] + self.t0_scale * float(z[0]),
            "u0": i["u0"] + self.u0_scale * float(z[1]),
            "tE": i["tE"] * float(np.exp(z[2])),
            "s": i["s"] * float(np.exp(z[3])),
            "q": i["q"] * float(np.exp(z[4])),
            "rho": i["rho"] * float(np.exp(z[5])),
            "alpha": i["alpha"] + float(z[6]),
            "piEN": i["piEN"] + (float(z[7]) if self.free_parallax else 0.0),
            "piEE": i["piEE"] + (float(z[8]) if self.free_parallax else 0.0),
        }
        if not all(np.isfinite(v) for v in p.values()):
            return None
        if not (1.0e-6 <= p["tE"] <= 1.0e4 and 1.0e-4 <= p["s"] <= 1.0e2
                and 1.0e-8 <= p["q"] <= 1.0e4 and 1.0e-8 <= p["rho"] <= 1.0):
            return None
        return p

    def _evaluate(self, z, tolerance=None) -> np.ndarray:
        p = self.unpack(z)
        if p is None:
            raise ValueError("invalid trial parameter transform")
        return self.model.residual(p, self.tolerance if tolerance is None else tolerance)

    def __call__(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if self._cache is not None and np.array_equal(z, self._cache[0]):
            return self._cache[1].copy()
        self.calls += 1
        try:
            residual = self._evaluate(z)
            self.last_error = None
        except Exception as error:  # one bad trial must not end the fit
            self.last_error = f"{type(error).__name__}: {error}"
            residual = np.full(self.model.n_points, INVALID_RESIDUAL)
        self._cache = (z.copy(), residual.copy())
        return residual

    def steps(self, z) -> np.ndarray:
        """Central-difference steps: about 1e-3 source radii of source motion."""
        p = self.unpack(z)
        rho = p["rho"] if p is not None else self.initial["rho"]
        s = p["s"] if p is not None else self.initial["s"]
        shift = max(1.0e-3 * rho, 1.0e-8)
        step = np.empty(self.n)
        step[0] = shift * self.initial["tE"] / self.t0_scale
        step[1] = shift / self.u0_scale
        step[2] = shift
        # Each step moves the source or a caustic by about ``shift``: the
        # planetary caustic sits near s - 1/s, and dq/q changes a caustic of
        # size below one Einstein radius by less than dq/q.
        step[3] = shift / (s + 1.0 / s)
        step[4] = shift
        step[5] = 1.0e-3
        step[6] = shift
        if self.free_parallax:
            step[7:] = max(1.0e-2 * shift, 1.0e-6)
        return step

    def jacobian(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        self.jacobian_calls += 1
        jac = np.zeros((self.model.n_points, self.n))
        step = self.steps(z)
        tol = self.jacobian_tolerance
        base = None if self.central else self._evaluate(z, tol)
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = step[k]
            plus, minus = self.unpack(z + e) is not None, self.unpack(z - e) is not None
            if base is not None and plus:
                jac[:, k] = (self._evaluate(z + e, tol) - base) / step[k]
                continue
            try:
                if plus and minus:
                    jac[:, k] = (self._evaluate(z + e, tol) - self._evaluate(z - e, tol)) / (2.0 * step[k])
                elif plus:  # one-sided at a guard
                    jac[:, k] = (self._evaluate(z + e, tol) - self._evaluate(z, tol)) / step[k]
                elif minus:
                    jac[:, k] = (self._evaluate(z, tol) - self._evaluate(z - e, tol)) / step[k]
            except Exception:
                jac[:, k] = 0.0
        return jac


def timescale_chart(objective: Objective):
    """``(convert, dz_dy)``: chart y where y1 moves u0 tE and y5 moves log(rho tE)."""
    base, u0_scale = objective.initial, objective.u0_scale
    n = objective.n

    def convert(y):
        z = np.array(y, dtype=np.float64, copy=True)
        if not np.any(z):
            return z
        if not np.all(np.isfinite(z)) or np.any(np.abs(z[2:6]) > 20.0):
            z[2] = 31.0  # rejected by Objective.unpack
            return z
        tE = base["tE"] * np.exp(z[2])
        u0 = (base["u0"] * base["tE"] + u0_scale * base["tE"] * z[1]) / tE
        rho = base["rho"] * base["tE"] * np.exp(z[5]) / tE
        z[1] = (u0 - base["u0"]) / u0_scale
        z[5] = np.log(rho / base["rho"])
        return z

    def dz_dy(y):
        matrix = np.eye(n)
        decay = np.exp(-y[2])
        matrix[1, 1] = decay
        matrix[1, 2] = -(base["u0"] + u0_scale * y[1]) * decay / u0_scale
        matrix[5, 2] = -1.0
        return matrix

    return convert, dz_dy


class BestTracker:
    """Residual function in the chart that remembers its best valid point."""

    def __init__(self, objective: Objective, convert) -> None:
        self.objective, self.convert = objective, convert
        residual = objective(convert(np.zeros(objective.n)))
        if objective.last_error is not None:
            raise RuntimeError(f"initial objective failed: {objective.last_error}")
        self.initial_residual = residual
        self.chi2 = float(residual @ residual)
        self.y = np.zeros(objective.n)

    def __call__(self, y):
        residual = self.objective(self.convert(y))
        chi2 = float(residual @ residual)
        if self.objective.last_error is None and chi2 < self.chi2:
            self.chi2, self.y = chi2, np.array(y, dtype=np.float64, copy=True)
        return residual

    def parameters(self) -> dict[str, float]:
        return self.objective.unpack(self.convert(self.y))


class _TimeBudget(Exception):
    pass


X_SCALE_FLOOR = 1.0e-4


def _x_scale(jacobian: np.ndarray) -> np.ndarray:
    """Fixed LM variable scales from the start Jacobian's column norms.

    A coordinate the data hardly constrain (rho of a near point-source
    crossing, say) has a column norm many decades below the rest.  Scaling by
    that norm, as ``x_scale="jac"`` does, lets LM propose steps of e^20 in it,
    which the guards reject until the trust region has collapsed.  The norms
    are floored at ``X_SCALE_FLOOR`` of the largest one instead.
    """
    norms = np.linalg.norm(jacobian, axis=0)
    floor = X_SCALE_FLOOR * float(np.max(norms)) if np.max(norms) > 0 else 1.0
    return 1.0 / np.maximum(norms, max(floor, 1e-300))


def lm(model: EventModel, start: dict[str, float], *, max_nfev: int = 100,
       max_seconds: float = float("inf"), fixed: str | None = None,
       free_parallax: bool = False, tolerance=COARSE, **options) -> dict[str, Any]:
    """LM polish from native ``start``; the best point so far is kept.

    ``fixed`` ("u0" or "rho") holds that chart coordinate (u0 tE or
    log(rho tE)) at its start value.  The result's ``fine_chi2`` is
    evaluated at the FINE tolerance.
    """
    from scipy.optimize import least_squares

    started = time.perf_counter()
    try:
        objective = Objective(model, start, tolerance=tolerance, free_parallax=free_parallax,
                              **options)
        convert, dz_dy = timescale_chart(objective)
        track = BestTracker(objective, convert)
        free = [k for k in range(objective.n) if k != {"u0": 1, "rho": 5, None: -1}[fixed]]
        deadline = started + float(max_seconds)

        def full(x):
            y = np.zeros(objective.n)
            y[free] = x
            return y

        def residual(x):
            if time.perf_counter() > deadline:
                raise _TimeBudget
            return track(full(x))

        def jacobian(x):
            y = full(x)
            return (objective.jacobian(convert(y)) @ dz_dy(y))[:, free]

        try:
            x_scale = _x_scale(jacobian(np.zeros(len(free))))
            fit = least_squares(residual, np.zeros(len(free)), jac=jacobian, method="lm",
                                max_nfev=int(max_nfev), ftol=1e-8, xtol=1e-8, gtol=1e-8,
                                x_scale=x_scale)
            nfev, message = int(fit.nfev), str(fit.message)
        except _TimeBudget:
            nfev, message = objective.calls, "time budget reached"
        parameters = track.parameters()
        if not free_parallax and not model.has_parallax:
            parameters.pop("piEN"), parameters.pop("piEE")
        return {
            "status": "ok",
            "parameters": parameters,
            "coarse_chi2": track.chi2,
            "fine_chi2": model.chi2(parameters, FINE),
            "nfev": nfev,
            "jacobians": objective.jacobian_calls,
            "message": message,
            "wall_seconds": time.perf_counter() - started,
        }
    except Exception as error:  # one bad start must not end the event
        return {"status": "error", "error": f"{type(error).__name__}: {error}",
                "wall_seconds": time.perf_counter() - started}
