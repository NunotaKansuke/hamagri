import json

import numpy as np
import pytest

from hamagri import MapConfig, Maps, combine_maps, plan_additions
from hamagri._core.atlas_builder import MapBuildSpec, PolarAtlasBuilder
from hamagri._core.bucketed_atlas import write_bucket_manifest


class ConstantEvaluator:
    def __init__(self, value):
        self.value = value

    def magnification(self, x, y):
        return np.full(np.broadcast(x, y).shape, self.value)


def write_collection(root, buckets, *, nodes=None):
    """buckets: {name: (logs, logq, [map ids])}"""
    entries = []
    for name, (logs, logq, ids) in buckets.items():
        builder = PolarAtlasBuilder(
            np.linspace(0.0, 2.0, 8) if nodes is None else nodes,
            n_phi_build=8, m_max=2, core_m_max=1, shard_size=1)
        builder.build(root / name, [MapBuildSpec(i, logs, logq, -3.0, ConstantEvaluator(1.0 + i)) for i in ids])
        entries.append({"name": name, "path": name, "count": len(ids), "logs": [logs, logs],
                        "logq": [logq, logq], "map_ids": ids})
    write_bucket_manifest(root, entries)
    return root


def test_combine_extends_buckets_and_adds_new_ones(tmp_path):
    base = write_collection(tmp_path / "base", {"s000_q000": (0.0, -3.0, [0, 1]), "s001_q000": (0.5, -3.0, [2])})
    extra = write_collection(tmp_path / "extra", {"s000_q000": (0.0, -3.0, [10]), "x1-s000_q000": (0.0, -7.0, [11])})
    combined = combine_maps(tmp_path / "combined", [base, extra])
    maps = Maps.open(combined)
    np.testing.assert_array_equal(sorted(maps.map_ids), [0, 1, 2, 10, 11])
    names = [b.name for b in maps._core.buckets]
    assert names == ["s000_q000", "s001_q000", "x1-s000_q000"]
    first = maps._core.buckets[0]
    assert sorted(int(i) for i in first.map_ids) == [0, 1, 10]
    rows = {int(s.map_ids[0]): np.asarray(s.x_coeff)[0] for s in first.atlas.iter_shards(m_max=2)}
    np.testing.assert_allclose(rows[10], next(extra_shard for extra_shard in [
        np.asarray(s.x_coeff)[0] for s in Maps.open(extra)._core.buckets[0].atlas.iter_shards(m_max=2)]))
    # the sources are untouched
    assert json.loads((base / "manifest.json").read_text())["n_maps"] == 3


def test_combine_rejects_shared_ids_and_different_nodes(tmp_path):
    base = write_collection(tmp_path / "base", {"s000_q000": (0.0, -3.0, [0, 1])})
    same = write_collection(tmp_path / "same", {"s000_q000": (0.0, -3.0, [1])})
    with pytest.raises(ValueError, match="share map IDs"):
        combine_maps(tmp_path / "out1", [base, same])
    other = write_collection(tmp_path / "other", {"s000_q000": (0.0, -3.0, [5])}, nodes=np.linspace(0.0, 3.0, 8))
    with pytest.raises(ValueError, match="radial nodes"):
        combine_maps(tmp_path / "out2", [base, other])
    assert not (tmp_path / "out2").exists()


def test_plan_reuses_base_layouts_inside_the_hull(tmp_path):
    base = write_collection(tmp_path / "base", {"s000_q000": (0.0, -3.0, [0]), "s001_q000": (0.5, -3.0, [1])})
    config = MapConfig(m_max=2, core_m_max=1, radial_nodes=8)
    rows = [(100, 0.2, -3.0, -3.0), (101, 0.3, -3.0, -3.0), (102, 0.2, -7.0, -3.0)]
    plan = {b["name"]: b for b in plan_additions(base, rows, config)}
    assert sorted(plan) == ["s000_q000", "s001_q000", "x1-s000_q000"]
    assert plan["s000_q000"]["rows"][:, 0].tolist() == [100]  # 0.2 is nearer 0.0 than 0.5
    np.testing.assert_array_equal(plan["s001_q000"]["radial_nodes"], np.linspace(0.0, 2.0, 8))
    assert plan["x1-s000_q000"].get("radial_nodes") is None
    with pytest.raises(ValueError, match="already exist"):
        plan_additions(base, [(0, 0.2, -3.0, -3.0)], config)
    with pytest.raises(ValueError, match="does not match"):
        plan_additions(base, rows, MapConfig())
