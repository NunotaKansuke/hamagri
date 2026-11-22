from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hamagri import Dataset, Geometry, Maps, SearchConfig
from hamagri._core.atlas import PolarAtlas
from hamagri._core.atlas_builder import MapBuildSpec
from hamagri._core.caustics import (
    map_frame_pspl_parameters,
    native_pspl_parameters_from_map,
)
from hamagri._core.direct_vbm import (
    DirectSpectrumConfig,
    DirectVBMPolarAtlasBuilder,
    VBMBinaryLensEvaluator,
    adaptive_ring_spectrum,
)
from hamagri._core.map_adapter import trajectory_xy


@dataclass
class SmoothRing:
    source_radius: float = 0.01
    caustic_components: tuple[np.ndarray, ...] = ()

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        angle = np.arctan2(y, x)
        return 1.4 + 0.2 * np.cos(3.0 * angle) - 0.1 * np.sin(2.0 * angle)


@dataclass
class NarrowCausticRing:
    source_radius: float = 0.002

    @property
    def caustic_components(self) -> tuple[np.ndarray, ...]:
        angle = 0.37
        direction = np.asarray([np.cos(angle), np.sin(angle)])
        tangent = np.asarray([-direction[1], direction[0]])
        return (
            np.stack(
                (
                    0.96 * direction - 0.02 * tangent,
                    1.04 * direction + 0.02 * tangent,
                )
            ),
        )

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        angle = np.arctan2(y, x)
        distance = np.angle(np.exp(1j * (angle - 0.37)))
        return 1.2 + 3.0 * np.exp(-0.5 * (distance / self.source_radius) ** 2)


@dataclass
class RadialRing:
    source_radius: float = 0.01
    caustic_components: tuple[np.ndarray, ...] = ()

    def magnification(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        radius = np.hypot(x, y)
        angle = np.arctan2(y, x)
        return (
            1.25
            + 0.18 * radius**2
            + 0.07 * np.sin(2.3 * radius) * np.cos(3.0 * angle)
        )


def _config(**updates) -> DirectSpectrumConfig:
    values = {
        "m_max": 8,
        "core_m_max": 4,
        "min_n_phi": 32,
        "caustic_base_n_phi": 64,
        "max_n_phi": 128,
        "coefficient_rtol": 1.0e-5,
        "coefficient_atol": 1.0e-10,
        "diagnostic_m_max": 16,
        "caustic_local_levels": 6,
    }
    values.update(updates)
    return DirectSpectrumConfig(**values)


def test_vbm_coordinate_frames_are_physically_equivalent(monkeypatch) -> None:
    class FakeVbm:
        def __init__(self):
            self.Tol = None
            self.RelTol = None
            self.a1 = None

        def Caustics(self, separation, mass_ratio):
            return [np.asarray([[0.1, 0.0], [0.1, 0.2], [0.1, 0.0]])]

        def BinaryMag2(self, separation, mass_ratio, x, y, source_radius):
            return 2.0 + x + 3.0 * y

    monkeypatch.setitem(
        sys.modules,
        "VBMicrolensing",
        SimpleNamespace(VBMicrolensing=FakeVbm),
    )
    separation = 2.0
    mass_ratio = 0.5
    shift = -(separation - 1.0 / separation) * mass_ratio / (1.0 + mass_ratio)
    x = np.asarray([-0.4, 0.0, 0.8])
    y = np.asarray([0.2, -0.3, 0.5])
    native = VBMBinaryLensEvaluator(
        separation, mass_ratio, 0.01, coordinate_frame="native", method="binary"
    )
    map_frame = VBMBinaryLensEvaluator(
        separation, mass_ratio, 0.01, coordinate_frame="map", method="binary"
    )

    np.testing.assert_allclose(
        native.magnification(x, y), map_frame.magnification(x - shift, y)
    )
    np.testing.assert_allclose(
        native.caustic_components[0][:, 0] - shift,
        map_frame.caustic_components[0][:, 0],
    )


def test_vbm_auto_switches_only_rings_failing_the_probe(monkeypatch) -> None:
    class FakeVbm:
        Method = SimpleNamespace(Multipoly="multipoly")

        def __init__(self):
            self.Tol = self.RelTol = self.a1 = None

        def Caustics(self, separation, mass_ratio):
            return [np.asarray([[0.0, 0.01], [0.01, 0.0], [0.0, -0.01], [0.0, 0.01]])]

        def SetMethod(self, method):
            pass

        def SetLensGeometry(self, geometry):
            pass

        def BinaryMag2(self, separation, mass_ratio, x, y, source_radius):
            # Round-off-like failure inside r < 0.1 only.
            return 2.0 + (0.5 if np.hypot(x, y) < 0.1 else 0.0)

        def MultiMag2(self, x, y, source_radius):
            return 2.0

    monkeypatch.setitem(
        sys.modules, "VBMicrolensing", SimpleNamespace(VBMicrolensing=FakeVbm)
    )
    evaluator = VBMBinaryLensEvaluator(
        1.0, 0.5, 0.01, coordinate_frame="native", method="auto"
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    for radius in (0.05, 1.0):
        values = evaluator.magnification(
            radius * np.cos(angles), radius * np.sin(angles)
        )
        np.testing.assert_allclose(values, 2.0)
    assert evaluator.ring_uses_multipoly(0.05)
    assert not evaluator.ring_uses_multipoly(1.0)


def test_vbm_multipoly_survives_a_crashing_point(monkeypatch) -> None:
    import os
    import signal

    crash_x = 0.3

    class FakeVbm:
        Method = SimpleNamespace(Multipoly="multipoly")

        def __init__(self):
            self.Tol = self.RelTol = self.a1 = None

        def Caustics(self, separation, mass_ratio):
            return [np.asarray([[0.0, 0.01], [0.01, 0.0], [0.0, -0.01], [0.0, 0.01]])]

        def SetMethod(self, method):
            pass

        def SetLensGeometry(self, geometry):
            pass

        def BinaryMag2(self, separation, mass_ratio, x, y, source_radius):
            return 5.0

        def MultiMag2(self, x, y, source_radius):
            # Like the VBM 5.6.1 use-after-free: only the exact position aborts.
            if x == crash_x and y == 0.0:
                os.kill(os.getpid(), signal.SIGKILL)
            return 2.0 + x

    monkeypatch.setitem(
        sys.modules, "VBMicrolensing", SimpleNamespace(VBMicrolensing=FakeVbm)
    )
    evaluator = VBMBinaryLensEvaluator(
        1.0, 0.5, 0.01, coordinate_frame="native", method="multipoly"
    )
    assert evaluator.isolate_multipoly
    x = np.asarray([0.1, 0.2, crash_x, 0.4])
    values = evaluator.magnification(x, np.zeros_like(x))
    np.testing.assert_allclose(values, 2.0 + x, rtol=0.0, atol=1.0e-9)
    assert evaluator.crash_recoveries == 1


def test_builder_retries_unconverged_auto_ring_with_multipoly() -> None:
    class ProbeMissingEvaluator:
        """Binary is noisy on ring 0.5 away from the probe angles; multipoly is smooth."""

        source_radius = 0.01
        caustic_components: tuple[np.ndarray, ...] = ()

        def __init__(self):
            self.multipoly: dict[float, bool] = {}

        def ring_uses_multipoly(self, radius):
            return self.multipoly.get(round(float(radius), 12), False)

        def force_multipoly(self, radius, enabled=True):
            key = round(float(radius), 12)
            if self.multipoly.get(key, False) == enabled:
                return False
            self.multipoly[key] = enabled
            return True

        def magnification(self, x, y):
            x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            radius = float(np.hypot(x.ravel()[0], y.ravel()[0]))
            smooth = 2.0 + 0.1 * x
            if self.ring_uses_multipoly(radius) or abs(radius - 0.5) > 1e-9:
                return smooth
            noise = np.sin(1.0e4 * np.arctan2(y, x)) ** 2
            return smooth + 0.3 * noise

    config = _config(m_max=4, core_m_max=2, diagnostic_m_max=8, build_radial_certificate=False)
    builder = DirectVBMPolarAtlasBuilder(
        np.asarray([0.1, 0.5, 1.0]), spectrum_config=config, shard_size=2
    )
    evaluator = ProbeMissingEvaluator()
    builder.sample(evaluator)
    diagnostics = builder._map_diagnostics[-1]

    assert diagnostics["converged"].all()
    assert diagnostics["vbm_multipoly"].tolist() == [False, True, False]


def test_builder_widens_guard_for_caustic_just_outside_it() -> None:
    class NearCausticRing:
        """A caustic 0.05 outside ring 1.0 gives a peak the uniform grid cannot resolve."""

        source_radius = 0.001
        caustic_components = (np.asarray([[1.05, -0.01], [1.05, 0.01]]),)

        def magnification(self, x, y):
            angle = np.arctan2(y, x)
            return 1.4 + 0.02 / (angle**2 + 0.01**2) * 1.0e-3

    config = _config(m_max=8, core_m_max=4, diagnostic_m_max=16, build_radial_certificate=False)
    nodes = np.asarray([0.98, 1.0, 1.02])
    builder = DirectVBMPolarAtlasBuilder(nodes, spectrum_config=config, shard_size=2)
    plain = adaptive_ring_spectrum(NearCausticRing(), 1.0, config, radial_guard_distance=0.01)
    ring = builder._ring(NearCausticRing(), 1.0, 0.01)

    assert not plain.converged and not plain.local_refined
    assert ring.converged and ring.local_refined


def test_map_frame_pspl_translation_matches_pointwise_trajectory() -> None:
    time = np.linspace(-3.0, 4.0, 31)
    t0, u0, tE, s, q, alpha = 2.3, 0.17, 4.2, 2.0, 0.5, 1.1
    t0_map, u0_map = map_frame_pspl_parameters(t0, u0, tE, s, q, alpha)
    native_x, native_y = trajectory_xy(time, t0, u0, tE, alpha)
    map_x, map_y = trajectory_xy(time, t0_map, u0_map, tE, alpha)
    shift = -(s - 1.0 / s) * q / (1.0 + q)

    np.testing.assert_allclose(map_x, native_x - shift)
    np.testing.assert_allclose(map_y, native_y)
    np.testing.assert_allclose(
        native_pspl_parameters_from_map(t0_map, u0_map, tE, s, q, alpha),
        (t0, u0),
    )


def test_nested_fft_recovers_smooth_low_modes() -> None:
    result = adaptive_ring_spectrum(SmoothRing(), 1.0, _config())

    assert result.converged
    assert not result.caustic_guarded
    assert result.uniform_n_phi == 64
    np.testing.assert_allclose(result.x_coeff[0], 0.4, atol=1.0e-14)
    np.testing.assert_allclose(result.x_coeff[2], 0.05j, atol=1.0e-14)
    np.testing.assert_allclose(result.x_coeff[3], 0.1, atol=1.0e-14)
    assert result.reconstruction_error < 1.0e-12


def test_caustic_local_refinement_recovers_narrow_feature() -> None:
    evaluator = NarrowCausticRing()
    config = _config(coefficient_rtol=2.0e-3)
    result = adaptive_ring_spectrum(evaluator, 1.0, config)
    dense_angles = np.arange(131072) * (2.0 * np.pi / 131072)
    dense_values = (
        evaluator.magnification(np.cos(dense_angles), np.sin(dense_angles)) - 1.0
    )
    reference = np.fft.rfft(dense_values)[: config.m_max + 1] / len(dense_angles)

    assert result.caustic_guarded
    assert result.caustic_angles
    assert result.evaluations < 512
    np.testing.assert_allclose(result.x_coeff, reference, rtol=6.0e-3, atol=2.0e-4)
    assert result.certified_error <= result.certified_error_core
    reconstructed = np.real(
        result.x_coeff[0]
        + 2.0
        * np.sum(
            result.x_coeff[None, 1:]
            * np.exp(
                1j
                * dense_angles[:, None]
                * np.arange(len(result.x_coeff))[None, 1:]
            ),
            axis=1,
        )
    )
    assert np.max(np.abs(dense_values - reconstructed)) <= result.certified_error


def test_direct_builder_writes_compatible_atlas_and_provenance(tmp_path) -> None:
    config = _config(
        m_max=4,
        core_m_max=2,
        diagnostic_m_max=8,
        build_radial_certificate=True,
    )
    builder = DirectVBMPolarAtlasBuilder(
        np.asarray([0.1, 0.5, 1.0]),
        spectrum_config=config,
        shard_size=2,
    )
    output = builder.build(
        tmp_path / "atlas",
        [MapBuildSpec(7, 0.0, -3.0, -2.0, SmoothRing())],
    )
    atlas = PolarAtlas(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert atlas.map_ids.tolist() == [7]
    assert atlas.m_max == 4
    assert manifest["builder"] == "direct-vbm-adaptive-radial-fourier"
    assert manifest["direct_diagnostics"]["unconverged_rings"] == 0
    assert manifest["error_certificate"] == (
        "piecewise-linear-vbm-angular-and-radial-reference"
    )
    assert manifest["direct_diagnostics"]["radial_holdout_evaluations"] > 0
    with np.load(output / "direct_diagnostics.npz") as diagnostics:
        assert diagnostics["radial_certificate_order1"].shape == (1, 3)
        assert diagnostics["radial_certificate_order3"].shape == (1, 3)
    shard = next(atlas.iter_shards(m_max=4))
    assert shard.certified_error is not None
    assert np.all(shard.certified_error >= shard.reconstruction_error)


def test_direct_builder_can_skip_radial_certificate(tmp_path) -> None:
    config = _config(
        m_max=4,
        core_m_max=2,
        diagnostic_m_max=8,
        build_radial_certificate=False,
    )
    output = DirectVBMPolarAtlasBuilder(
        np.asarray([0.1, 0.5, 1.0]),
        spectrum_config=config,
        shard_size=1,
    ).build(
        tmp_path / "angular-only",
        [MapBuildSpec(3, 0.0, -3.0, -2.0, SmoothRing())],
    )
    atlas = PolarAtlas(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["error_certificate"] == "adaptive-vbm-angular-reference-only"
    assert manifest["direct_diagnostics"]["radial_holdout_evaluations"] == 0
    with np.load(output / "direct_diagnostics.npz") as diagnostics:
        assert diagnostics["radial_holdout_evaluations"] == 0
    shard = next(atlas.iter_shards(m_max=4))
    assert shard.certified_error is not None


def test_public_search_returns_score_without_chi2_bounds(tmp_path) -> None:
    evaluator = RadialRing()
    config = _config(
        m_max=4,
        core_m_max=2,
        diagnostic_m_max=8,
        build_radial_certificate=True,
    )
    nodes = np.asarray([0.02, 0.16, 0.38, 0.7, 1.05, 1.4])
    output = DirectVBMPolarAtlasBuilder(
        nodes, spectrum_config=config, shard_size=1
    ).build(
        tmp_path / "radial-maps",
        [MapBuildSpec(0, 0.0, -3.0, -2.0, evaluator)],
    )
    geometry = Geometry(t0=0.0, u0=0.12, tE=1.0, coordinate_frame="map")
    time = np.linspace(-1.2, 1.2, 81)
    alpha = 0.43
    tau = time
    x = geometry.u0 * np.sin(alpha) - tau * np.cos(alpha)
    y = -geometry.u0 * np.cos(alpha) - tau * np.sin(alpha)
    flux = 2.1 * evaluator.magnification(x, y) + 0.35
    dataset = Dataset(time, flux, np.full_like(time, 0.01))
    n_alpha = 24
    result = Maps.open(output).search(
        [dataset],
        [geometry],
        config=SearchConfig(
            base_m_max=1,
            full_m_max=4,
            base_n_alpha=8,
            full_n_alpha=n_alpha,
            top_count=1,
            risk_count=1,
            candidate_count=1,
            refine=False,
        ),
    )
    candidate = result.candidates[0]

    assert np.isfinite(candidate.chi2)
    assert not hasattr(candidate, "chi2_lower")
    assert not hasattr(candidate, "chi2_upper")


@pytest.mark.parametrize("isolate", [True, False])
def test_vbm_multipoly_retries_a_nan_point(monkeypatch, isolate) -> None:
    nan_x = 0.3

    class FakeVbm:
        Method = SimpleNamespace(Multipoly="multipoly")

        def __init__(self):
            self.Tol = self.RelTol = self.a1 = None

        def Caustics(self, separation, mass_ratio):
            return [np.asarray([[0.0, 0.01], [0.01, 0.0], [0.0, -0.01], [0.0, 0.01]])]

        def SetMethod(self, method):
            pass

        def SetLensGeometry(self, geometry):
            pass

        def BinaryMag2(self, separation, mass_ratio, x, y, source_radius):
            return 5.0

        def MultiMag2(self, x, y, source_radius):
            # VBM 5.6.1 also returns NaN, without aborting, at rare exact positions.
            if x == nan_x and y == 0.0:
                return float("nan")
            return 2.0 + x

    monkeypatch.setitem(
        sys.modules, "VBMicrolensing", SimpleNamespace(VBMicrolensing=FakeVbm)
    )
    evaluator = VBMBinaryLensEvaluator(
        1.0, 0.5, 0.01, coordinate_frame="native", method="multipoly",
        isolate_multipoly=isolate,
    )
    x = np.asarray([0.1, 0.2, nan_x, 0.4])
    values = evaluator.magnification(x, np.zeros_like(x))
    np.testing.assert_allclose(values, 2.0 + x, rtol=0.0, atol=1.0e-9)
    assert evaluator.crash_recoveries == 1
