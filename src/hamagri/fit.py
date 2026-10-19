"""Direct VBMicrolensing re-evaluation and Levenberg-Marquardt refinement."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal, Sequence

import numpy as np

from ._core.reference import profile_flux
from .search import Candidate, Dataset

CoordinateFrame = Literal["map", "native"]
FitStage = Literal["short", "long"]


@dataclass(frozen=True)
class FitConfig:
    """Controls the direct-VBM screening and two LM refinement stages.

    Defaults follow the candidate handoff used in the paper: directly score
    every seed, short-refine the best 500, then long-refine the best short fit
    for each input geometry.
    """

    short_count: int = 500
    short_max_nfev: int = 80
    long_max_nfev: int | None = 1200
    diff_step: float = 1.0e-3
    vbm_tolerance: float = 1.0e-3
    vbm_relative_tolerance: float = 1.0e-4
    ftol: float = 1.0e-8
    xtol: float = 1.0e-8
    gtol: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("short_count", "short_max_nfev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.long_max_nfev is not None and (
            isinstance(self.long_max_nfev, bool)
            or not isinstance(self.long_max_nfev, Integral)
            or self.long_max_nfev < 1
        ):
            raise ValueError("long_max_nfev must be a positive integer or None")
        for name in (
            "diff_step",
            "vbm_tolerance",
            "vbm_relative_tolerance",
            "ftol",
            "xtol",
            "gtol",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class FitParameters:
    """Fitted binary-lens parameters in the requested coordinate frame."""

    t0: float
    u0: float
    tE: float
    s: float
    q: float
    rho: float
    alpha: float
    coordinate_frame: CoordinateFrame

    def as_dict(self) -> dict[str, float | str]:
        return {
            "t0": self.t0,
            "u0": self.u0,
            "tE": self.tE,
            "s": self.s,
            "q": self.q,
            "rho": self.rho,
            "alpha": self.alpha,
            "coordinate_frame": self.coordinate_frame,
        }


@dataclass(frozen=True)
class DirectCandidateScore:
    """One seed scored with direct VBMicrolensing before LM refinement."""

    candidate: Candidate
    parameters: FitParameters
    chi2: float
    error: str | None


@dataclass(frozen=True)
class CandidateFitResult:
    """One candidate's result from a short or long LM stage."""

    candidate: Candidate
    initial_parameters: FitParameters
    parameters: FitParameters
    initial_chi2: float
    chi2: float
    stage: FitStage
    success: bool
    status: int
    message: str
    nfev: int | None
    njev: int | None
    optimality: float | None
    last_objective_error: str | None


@dataclass(frozen=True)
class FitResult:
    """Direct scores, short fits, per-geometry long fits, and the best fit."""

    direct_scores: tuple[DirectCandidateScore, ...]
    short_fits: tuple[CandidateFitResult, ...]
    long_fits: tuple[CandidateFitResult, ...]
    best: CandidateFitResult


def _new_vbm(config: FitConfig) -> Any:
    try:
        import VBMicrolensing
    except ImportError as error:
        raise ImportError(
            "direct fitting requires VBMicrolensing; install hamagri[fit]"
        ) from error
    vbm = VBMicrolensing.VBMicrolensing()
    vbm.Tol = config.vbm_tolerance
    vbm.RelTol = config.vbm_relative_tolerance
    vbm.a1 = 0.0
    return vbm


class _DirectVBMObjective:
    """Profiled multi-dataset residual using VBM's vector light-curve API."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        initial: FitParameters,
        *,
        config: FitConfig,
        vbm: Any | None = None,
    ) -> None:
        if initial.coordinate_frame not in ("map", "native"):
            raise ValueError("coordinate_frame must be 'map' or 'native'")
        self.datasets = tuple(datasets)
        self.initial = initial
        self.config = config
        self.t0_scale = max(abs(initial.tE), 1.0)
        self.u0_scale = max(abs(initial.u0), initial.rho, 0.01)
        self.n_residuals = int(sum(len(dataset.time) for dataset in datasets))
        if self.n_residuals < 7:
            raise ValueError("LM fitting requires at least seven photometric points")
        self._vbm = vbm if vbm is not None else _new_vbm(config)
        self._vbm.Tol = config.vbm_tolerance
        self._vbm.RelTol = config.vbm_relative_tolerance
        self._vbm.a1 = 0.0
        self.calls = 0
        self.last_error: str | None = None
        self._cached_z: np.ndarray | None = None
        self._cached_residual: np.ndarray | None = None

    def unpack(self, z: np.ndarray) -> FitParameters | None:
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (7,) or not np.all(np.isfinite(z)):
            return None
        if (
            np.any(np.abs(z[2:6]) > 30.0)
            or abs(float(z[0])) > 1.0e6
            or abs(float(z[1])) > 1.0e6
            or abs(float(z[6])) > 1.0e6
        ):
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            values = FitParameters(
                t0=self.initial.t0 + self.t0_scale * float(z[0]),
                u0=self.initial.u0 + self.u0_scale * float(z[1]),
                tE=self.initial.tE * float(np.exp(z[2])),
                s=self.initial.s * float(np.exp(z[3])),
                q=self.initial.q * float(np.exp(z[4])),
                rho=self.initial.rho * float(np.exp(z[5])),
                alpha=self.initial.alpha + float(z[6]),
                coordinate_frame=self.initial.coordinate_frame,
            )
        physical = (values.tE, values.s, values.q, values.rho)
        if not np.all(np.isfinite((values.t0, values.u0, *physical, values.alpha))):
            return None
        if any(value <= 0.0 for value in physical):
            return None
        # Numerical guardrails keep runaway LM probes in a computable range.
        if not (
            1.0e-6 <= values.tE <= 1.0e4
            and 1.0e-4 <= values.s <= 1.0e2
            and 1.0e-8 <= values.q <= 1.0e4
            and 1.0e-8 <= values.rho <= 1.0
        ):
            return None
        return values

    def _invalid_residual(self) -> np.ndarray:
        return np.full(self.n_residuals, 1.0e6, dtype=np.float64)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if self._cached_z is not None and np.array_equal(z, self._cached_z):
            assert self._cached_residual is not None
            return self._cached_residual.copy()
        self.calls += 1
        parameters = self.unpack(z)
        if parameters is None:
            self.last_error = "invalid trial parameter transform"
            residual = self._invalid_residual()
            self._cached_z = z.copy()
            self._cached_residual = residual.copy()
            return residual
        try:
            if parameters.coordinate_frame == "map":
                from ._core.caustics import native_pspl_parameters_from_map

                native_t0, native_u0 = native_pspl_parameters_from_map(
                    parameters.t0,
                    parameters.u0,
                    parameters.tE,
                    parameters.s,
                    parameters.q,
                    parameters.alpha,
                )
            else:
                native_t0, native_u0 = parameters.t0, parameters.u0
            vbm_parameters = [
                float(np.log(parameters.s)),
                float(np.log(parameters.q)),
                float(native_u0),
                float(parameters.alpha),
                float(np.log(parameters.rho)),
                float(np.log(parameters.tE)),
                float(native_t0),
            ]
            pieces = []
            for dataset in self.datasets:
                magnification = np.asarray(
                    self._vbm.BinaryLightCurve(vbm_parameters, dataset.time)[0],
                    dtype=np.float64,
                )
                profile = profile_flux(magnification, dataset)
                if not np.isfinite(profile.chi2):
                    raise ValueError(
                        f"profiled flux fit failed for dataset {dataset.name!r}"
                    )
                model = profile.source_flux * magnification + profile.blend_flux
                residual = (dataset.flux - model) / dataset.error
                if residual.shape != dataset.flux.shape or not np.all(
                    np.isfinite(residual)
                ):
                    raise ValueError(
                        f"direct-VBM residual is invalid for dataset {dataset.name!r}"
                    )
                pieces.append(residual)
            output = np.concatenate(pieces)
            self.last_error = None
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            output = self._invalid_residual()
        self._cached_z = z.copy()
        self._cached_residual = output.copy()
        return output


def _candidate_parameters(
    candidate: Candidate, coordinate_frame: CoordinateFrame
) -> FitParameters:
    values = FitParameters(
        t0=float(candidate.t0),
        u0=float(candidate.u0),
        tE=float(candidate.tE),
        s=float(candidate.s),
        q=float(candidate.q),
        rho=float(candidate.rho),
        alpha=float(candidate.alpha),
        coordinate_frame=coordinate_frame,
    )
    all_values = (
        values.t0,
        values.u0,
        values.tE,
        values.s,
        values.q,
        values.rho,
        values.alpha,
    )
    if not np.all(np.isfinite(all_values)) or any(
        value <= 0.0 for value in (values.tE, values.s, values.q, values.rho)
    ):
        raise ValueError("candidate contains non-finite or non-positive fit parameters")
    return values


def _fit_seed(
    candidate: Candidate,
    initial: FitParameters,
    datasets: Sequence[Dataset],
    *,
    config: FitConfig,
    vbm: Any,
    least_squares: Any,
    stage: FitStage,
    max_nfev: int,
    known_initial_chi2: float | None = None,
) -> CandidateFitResult:
    objective = _DirectVBMObjective(datasets, initial, config=config, vbm=vbm)
    z0 = np.zeros(7, dtype=np.float64)
    start_residual = objective(z0)
    start_chi2 = (
        float(known_initial_chi2)
        if known_initial_chi2 is not None
        else float(np.dot(start_residual, start_residual))
    )
    if objective.last_error is not None:
        return CandidateFitResult(
            candidate=candidate,
            initial_parameters=initial,
            parameters=initial,
            initial_chi2=start_chi2,
            chi2=start_chi2,
            stage=stage,
            success=False,
            status=-1,
            message=f"initial direct-VBM evaluation failed: {objective.last_error}",
            nfev=0,
            njev=None,
            optimality=None,
            last_objective_error=objective.last_error,
        )
    try:
        fit = least_squares(
            objective,
            z0,
            method="lm",
            jac="2-point",
            diff_step=config.diff_step,
            max_nfev=max_nfev,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
            x_scale="jac",
        )
        final_parameters = objective.unpack(fit.x)
        if final_parameters is None:
            raise ValueError("LM returned an invalid transformed parameter vector")
        final_residual = objective(fit.x)
        final_chi2 = float(np.dot(final_residual, final_residual))
        return CandidateFitResult(
            candidate=candidate,
            initial_parameters=initial,
            parameters=final_parameters,
            initial_chi2=start_chi2,
            chi2=final_chi2,
            stage=stage,
            success=bool(fit.success),
            status=int(fit.status),
            message=str(fit.message),
            nfev=int(fit.nfev),
            njev=None if fit.njev is None else int(fit.njev),
            optimality=float(fit.optimality),
            last_objective_error=objective.last_error,
        )
    except Exception as error:
        return CandidateFitResult(
            candidate=candidate,
            initial_parameters=initial,
            parameters=initial,
            initial_chi2=start_chi2,
            chi2=start_chi2,
            stage=stage,
            success=False,
            status=-1,
            message=f"{type(error).__name__}: {error}",
            nfev=None,
            njev=None,
            optimality=None,
            last_objective_error=objective.last_error,
        )


def fit_candidates(
    candidates: Sequence[Candidate],
    datasets: Sequence[Dataset],
    *,
    coordinate_frame: CoordinateFrame,
    config: FitConfig | None = None,
) -> FitResult:
    """Screen search seeds with direct VBM, then perform short and long LM fits.

    ``coordinate_frame`` must match the geometry frame used by ``Atlas.search``.
    Map-frame trajectories are converted to VBM's native frame at every LM
    evaluation while the fitted ``t0`` and ``u0`` remain in the map frame.

    The function serially evaluates all candidates. Pass a larger search pool
    (for example ``SearchConfig(candidate_count=3000)``) to reproduce the
    paper's 3,000-direct / 500-short-LM / per-geometry-long-LM handoff.
    """
    if coordinate_frame not in ("map", "native"):
        raise ValueError("coordinate_frame must be 'map' or 'native'")
    candidates = tuple(candidates)
    datasets = tuple(datasets)
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not datasets:
        raise ValueError("at least one photometric dataset is required")
    if sum(len(dataset.time) for dataset in datasets) < 7:
        raise ValueError("LM fitting requires at least seven photometric points")
    config = config or FitConfig()
    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise ImportError(
            "LM fitting requires SciPy; install hamagri[fit]"
        ) from error
    vbm = _new_vbm(config)

    direct_scores = []
    for candidate in candidates:
        initial = _candidate_parameters(candidate, coordinate_frame)
        objective = _DirectVBMObjective(
            datasets, initial, config=config, vbm=vbm
        )
        residual = objective(np.zeros(7, dtype=np.float64))
        direct_scores.append(
            DirectCandidateScore(
                candidate=candidate,
                parameters=initial,
                chi2=float(np.dot(residual, residual)),
                error=objective.last_error,
            )
        )
    direct_scores.sort(key=lambda row: (row.error is not None, row.chi2))
    valid_scores = [
        row for row in direct_scores if row.error is None and np.isfinite(row.chi2)
    ]
    if not valid_scores:
        detail = direct_scores[0].error or "no finite direct-VBM score"
        raise RuntimeError(
            f"direct VBMicrolensing failed for all {len(direct_scores)} candidates; "
            f"first error: {detail}"
        )

    short_fits = tuple(
        _fit_seed(
            row.candidate,
            row.parameters,
            datasets,
            config=config,
            vbm=vbm,
            least_squares=least_squares,
            stage="short",
            max_nfev=config.short_max_nfev,
            known_initial_chi2=row.chi2,
        )
        for row in valid_scores[: config.short_count]
    )
    long_fits: tuple[CandidateFitResult, ...] = ()
    if config.long_max_nfev is not None:
        best_by_geometry: dict[int, CandidateFitResult] = {}
        for fit in short_fits:
            geometry_index = int(fit.candidate.geometry_index)
            previous = best_by_geometry.get(geometry_index)
            if previous is None or fit.chi2 < previous.chi2:
                best_by_geometry[geometry_index] = fit
        long_fits = tuple(
            _fit_seed(
                fit.candidate,
                fit.parameters,
                datasets,
                config=config,
                vbm=vbm,
                least_squares=least_squares,
                stage="long",
                max_nfev=config.long_max_nfev,
                known_initial_chi2=fit.chi2,
            )
            for _, fit in sorted(best_by_geometry.items())
        )
    best_pool = long_fits or short_fits
    best = min(best_pool, key=lambda fit: fit.chi2)
    return FitResult(
        direct_scores=tuple(direct_scores),
        short_fits=short_fits,
        long_fits=long_fits,
        best=best,
    )
