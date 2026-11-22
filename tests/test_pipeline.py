import json
import math

import numpy as np

from hamagri._core.trajectory import PhotometricDataset
from hamagri.pipeline import EventModel, geometry_stencil, lm
from hamagri.pipeline.battle import same_solution
from hamagri.pipeline.event import load_event
from hamagri.pipeline.hills import map_vector, native_parameters, watershed_basins

TRUE = {"t0": 0.0, "u0": 0.05, "tE": 20.0, "s": 1.1, "q": 3e-3, "rho": 2e-3, "alpha": 1.0}


def _dataset(seed=1):
    rng = np.random.default_rng(seed)
    time = np.linspace(-40.0, 40.0, 600)
    model = EventModel((PhotometricDataset(time, np.ones_like(time), np.full_like(time, 0.01)),))
    flux = 2.0 * model.magnifications(TRUE)[0] + 1.0
    error = np.full_like(time, 0.01)
    return PhotometricDataset(time, flux + error * rng.standard_normal(time.size), error, "W146")


def test_map_vector_round_trip():
    back = native_parameters(map_vector(TRUE))
    for key, value in TRUE.items():
        assert math.isclose(back[key], value, rel_tol=1e-9, abs_tol=1e-12)


def test_same_solution_maps_large_q_to_its_inverse():
    a = {"s": 1.0, "q": 0.5, "u0": 0.1, "alpha": 0.3}
    b = {"s": 1.0, "q": 2.0, "u0": 0.1, "alpha": 0.3 + math.pi}
    assert same_solution(a, b)
    assert not same_solution(a, {**a, "u0": -0.1})
    assert not same_solution(a, {**a, "s": 1.5})


def test_watershed_finds_two_basins():
    s, q, r = np.meshgrid(np.arange(9.0), np.arange(3.0), np.arange(2.0), indexing="ij")
    parameters = np.column_stack([s.ravel(), q.ravel(), r.ravel()])
    chi2 = np.minimum((parameters[:, 0] - 1) ** 2, (parameters[:, 0] - 7) ** 2 + 0.5) + parameters[:, 1]
    basins = watershed_basins(parameters, chi2 + 0.1 * parameters[:, 2], np.arange(len(chi2)))
    assert sorted(round(b["chi2"], 6) for b in basins) == [0.0, 0.5]


def test_geometry_stencil_has_sixteen_points():
    stencil = geometry_stencil(0.0, 0.1, 20.0)
    assert len(stencil) == 16 and stencil[0].t0 == 0.0 and stencil[0].tE == 20.0


def test_lm_recovers_a_binary(tmp_path):
    model = EventModel((_dataset(),))
    assert model.chi2(TRUE) < 700
    start = {**TRUE, "s": 1.105, "q": 3.3e-3, "alpha": 1.01, "t0": 0.05}
    fit = lm(model, start, max_nfev=60)
    assert fit["fine_chi2"] < model.chi2(TRUE) + 1.0


def test_load_event(tmp_path):
    data = _dataset()
    np.savetxt(tmp_path / "w146.txt", np.column_stack([data.time, data.flux, data.error]))
    (tmp_path / "event.json").write_text(json.dumps({
        "pspl": [0.0, 0.05, 20.0], "datasets": [{"path": "w146.txt"}]}))
    event = load_event(tmp_path / "event.json")
    assert event["pspl"] == (0.0, 0.05, 20.0) and "parallax" not in event
    np.testing.assert_allclose(event["datasets"][0].flux, data.flux)
    assert event["datasets"][0].name == "w146"
