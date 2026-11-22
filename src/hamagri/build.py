"""Map construction, partitioning and merge operations."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np

from .config import MapConfig, ParameterGrid, Partition
from ._core.adaptive_grid import (
    build_adaptive_radial_nodes,
    sharp_qs_pilot_indices,
)
from ._core.atlas_builder import MapBuildSpec
from ._core.bucketed_atlas import BucketedPolarAtlas, write_bucket_manifest
from ._core.direct_vbm import (
    DirectSpectrumConfig,
    DirectVBMPolarAtlasBuilder,
    VBMBinaryLensEvaluator,
)


def _spectrum_config(config: MapConfig) -> DirectSpectrumConfig:
    return DirectSpectrumConfig(
        m_max=config.m_max,
        core_m_max=config.core_m_max,
        min_n_phi=config.min_n_phi,
        caustic_base_n_phi=config.caustic_base_n_phi,
        max_n_phi=config.max_n_phi,
        coefficient_rtol=config.coefficient_rtol,
        coefficient_atol=config.coefficient_atol,
        caustic_guard_rho=config.caustic_guard_rho,
        caustic_local_levels=config.caustic_local_levels,
        caustic_points_per_side=config.caustic_points_per_side,
        diagnostic_m_max=config.diagnostic_m_max,
        radial_certificate_levels=1,
        build_radial_certificate=False,
        store_certificate_arrays=False,
    )


def _specs(rows: np.ndarray, config: MapConfig) -> Iterable[MapBuildSpec]:
    for map_id_value, logs, logq, logrho in rows:
        yield MapBuildSpec(
            map_id=int(map_id_value),
            logs=float(logs),
            logq=float(logq),
            logrho=float(logrho),
            evaluator=VBMBinaryLensEvaluator(
                10.0**logs,
                10.0**logq,
                10.0**logrho,
                tolerance=config.vbm_tolerance,
                relative_tolerance=config.vbm_relative_tolerance,
                coordinate_frame=config.coordinate_frame,
                method=config.vbm_method,
            ),
        )


def _parameter_buckets(table: np.ndarray, config: MapConfig) -> list[dict[str, object]]:
    """Group complete rho axes into deterministic `(s,q)` radial-layout bins."""
    logs_values = np.unique(table[:, 1])
    logq_values = np.unique(table[:, 2])
    logs_groups = np.array_split(
        logs_values, min(int(config.radial_s_buckets), len(logs_values))
    )
    logq_groups = np.array_split(
        logq_values, min(int(config.radial_q_buckets), len(logq_values))
    )
    buckets: list[dict[str, object]] = []
    for logs_index, logs_group in enumerate(logs_groups):
        for logq_index, logq_group in enumerate(logq_groups):
            mask = np.isin(table[:, 1], logs_group) & np.isin(table[:, 2], logq_group)
            rows = table[mask]
            if not len(rows):
                continue
            buckets.append(
                {
                    "name": f"s{logs_index:03d}_q{logq_index:03d}",
                    "rows": rows,
                    "logs": [float(logs_group[0]), float(logs_group[-1])],
                    "logq": [float(logq_group[0]), float(logq_group[-1])],
                }
            )
    return buckets


def _pilot_evaluators(rows: np.ndarray, config: MapConfig):
    local = sharp_qs_pilot_indices(
        rows[:, 1:], min(int(config.radial_pilot_maps), len(rows))
    )
    pilot_rows = rows[local]
    evaluators = [
        VBMBinaryLensEvaluator(
            10.0**logs,
            10.0**logq,
            10.0**logrho,
            tolerance=config.vbm_tolerance,
            relative_tolerance=config.vbm_relative_tolerance,
            coordinate_frame=config.coordinate_frame,
            method=config.vbm_method,
        )
        for _, logs, logq, logrho in pilot_rows
    ]
    return pilot_rows, evaluators


def _build_buckets(temporary: Path, buckets, config: MapConfig, resume: bool, progress_every: int):
    """Build every bucket below ``temporary``; a bucket may bring its own radial nodes."""
    entries = []
    selected_ids: list[int] = []
    for bucket in buckets:
        rows = np.asarray(bucket["rows"], dtype=np.float64)
        child = temporary / str(bucket["name"])
        if bucket.get("radial_nodes") is None:
            pilot_rows, evaluators = _pilot_evaluators(rows, config)
            radial_nodes, pilot_radii, difficulty = build_adaptive_radial_nodes(
                evaluators,
                max_radius=config.radial_max,
                n_nodes=config.radial_nodes,
                pilot_bins=config.radial_pilot_bins,
                n_phi=config.radial_pilot_phi,
                m_max=config.radial_pilot_m_max,
                map_quantile=1.0,
                adaptive_fraction=config.radial_adaptive_fraction,
            )
        else:
            radial_nodes, pilot_rows = np.asarray(bucket["radial_nodes"], dtype=np.float64), None
        builder = DirectVBMPolarAtlasBuilder(
            radial_nodes,
            spectrum_config=_spectrum_config(config),
            shard_size=config.shard_size,
            coordinate_frame=config.coordinate_frame,
        )
        builder.build(child, _specs(rows, config), progress_every=progress_every, resume=resume)
        if pilot_rows is not None:
            np.save(child / "radial_pilot_map_ids.npy", pilot_rows[:, 0].astype(np.int64))
            np.save(child / "radial_pilot_radii.npy", pilot_radii)
            np.save(child / "radial_pilot_difficulty.npy", difficulty)
        map_ids = rows[:, 0].astype(int).tolist()
        selected_ids.extend(map_ids)
        entries.append({
            "name": bucket["name"],
            "path": bucket["name"],
            "count": len(rows),
            "logs": bucket["logs"],
            "logq": bucket["logq"],
            "map_ids": map_ids,
        })
    return entries, selected_ids


def build_maps(
    output: str | Path,
    grid: ParameterGrid,
    *,
    config: MapConfig | None = None,
    partition: Partition | None = None,
    progress_every: int = 1,
) -> Path:
    """Generate one complete map collection or one schedulable part.

    For ``Partition(index=i, count=n)``, every machine receives the same grid
    and config and writes ``output/parts/part-iiiii-of-nnnnn``.  A private
    temporary directory is atomically renamed only after the part succeeds.
    """
    config = config or MapConfig()
    buckets = _parameter_buckets(grid.table(), config)
    return _build_collection(output, buckets, config, partition or Partition(), progress_every,
                             {"grid": grid.to_dict()})


def _build_collection(output, all_buckets, config: MapConfig, partition: Partition,
                      progress_every: int, record: dict[str, object]) -> Path:
    output = Path(output).resolve()
    buckets = all_buckets[partition.rows(len(all_buckets))]
    if not buckets:
        raise ValueError("this partition contains no radial-layout buckets")
    destination = (
        output
        if partition.count == 1
        else output / "parts" / f"part-{partition.index:05d}-of-{partition.count:05d}"
    )
    if destination.exists():
        raise FileExistsError(f"map destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partials = sorted(destination.parent.glob(f".{destination.name}.partial-*"))
    if len(partials) > 1:
        raise FileExistsError(
            f"multiple partial outputs exist for {destination}: {partials}"
        )
    if partials:
        temporary = partials[0]
        resume = True
    else:
        temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
        resume = False
        temporary.mkdir(parents=True)

    # A failure keeps the partial tree as a checkpoint: the next invocation
    # reuses durable map shards and only evaluates map IDs not yet present.
    entries, selected_ids = _build_buckets(temporary, buckets, config, resume, progress_every)
    write_bucket_manifest(
        temporary,
        entries,
        {
            "builder": "direct-vbm-rho-aware-adaptive-radial-fourier",
            "coordinate_frame": config.coordinate_frame,
            "radial_layout": {
                "s_buckets": config.radial_s_buckets,
                "q_buckets": config.radial_q_buckets,
                "pilot_maps": config.radial_pilot_maps,
                "pilot_bins": config.radial_pilot_bins,
                "pilot_phi": config.radial_pilot_phi,
                "pilot_m_max": config.radial_pilot_m_max,
                "adaptive_fraction": config.radial_adaptive_fraction,
            },
        },
    )
    metadata = {
        **record,
        "config": config.to_dict(),
        "partition": asdict(partition),
        "selected_map_ids": sorted(selected_ids),
    }
    (temporary / "hamagri-build.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


def _rectangle_distance(value: float, bounds) -> float:
    return max(float(bounds[0]) - value, value - float(bounds[1]), 0.0)


def plan_additions(base: str | Path, rows, config: MapConfig | None = None) -> list[dict[str, object]]:
    """Assign added map rows ``(map_id, log10 s, log10 q, log10 rho)`` to radial layouts.

    A row inside the ``(log s, log q)`` hull of the base atlas joins the
    nearest base bucket (distance to its bounds, ties to the lower bucket) and
    reuses that bucket's radial nodes, so the bucket can later be extended in
    place by :func:`combine_maps`.  Rows outside the hull are grouped like a
    fresh build, with their own pilot layout and names that do not collide
    with the base.
    """
    config = config or MapConfig()
    collection = BucketedPolarAtlas(base)
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if rows.shape[1] != 4 or not len(rows):
        raise ValueError("rows must be (map_id, log10 s, log10 q, log10 rho)")
    ids = rows[:, 0].astype(np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("added map IDs must be unique")
    clash = set(ids.tolist()) & {int(i) for i in collection.map_ids}
    if clash:
        raise ValueError(f"added map IDs already exist in the base atlas: {sorted(clash)[:5]}")
    buckets = sorted(collection.buckets, key=lambda b: (b.logs[0], b.logq[0], b.name))
    for bucket in buckets:
        manifest = bucket.atlas.manifest
        if (int(manifest["m_max"]), int(manifest.get("core_m_max", manifest["m_max"])),
                len(bucket.atlas.radial_nodes)) != (config.m_max, config.core_m_max, config.radial_nodes):
            raise ValueError(f"config does not match base bucket {bucket.name!r} (m_max, core_m_max, radial nodes)")
    hull_s = (min(b.logs[0] for b in buckets), max(b.logs[1] for b in buckets))
    hull_q = (min(b.logq[0] for b in buckets), max(b.logq[1] for b in buckets))
    eps = 1.0e-9
    assigned: dict[str, list[np.ndarray]] = {}
    outside = []
    for row in rows:
        if _rectangle_distance(row[1], hull_s) > eps or _rectangle_distance(row[2], hull_q) > eps:
            outside.append(row)
            continue
        best = min(buckets, key=lambda b: (_rectangle_distance(row[1], b.logs) + _rectangle_distance(row[2], b.logq)))
        assigned.setdefault(best.name, []).append(row)
    by_name = {b.name: b for b in buckets}
    plan = []
    for name, members in sorted(assigned.items()):
        members = np.asarray(members)
        plan.append({
            "name": name,
            "rows": members,
            "logs": [float(members[:, 1].min()), float(members[:, 1].max())],
            "logq": [float(members[:, 2].min()), float(members[:, 2].max())],
            "radial_nodes": np.asarray(by_name[name].atlas.radial_nodes, dtype=np.float64),
        })
    if outside:
        taken = {b.name for b in buckets}
        generation = 1
        while any(n.startswith(f"x{generation}-") for n in taken):
            generation += 1
        for bucket in _parameter_buckets(np.asarray(outside), config):
            plan.append({**bucket, "name": f"x{generation}-{bucket['name']}"})
    return plan


def build_additions(
    output: str | Path,
    base: str | Path,
    rows,
    *,
    config: MapConfig | None = None,
    partition: Partition | None = None,
    progress_every: int = 1,
) -> Path:
    """Build added maps for ``base`` (or one partition), laid out by :func:`plan_additions`.

    ``config`` must be the configuration the base was built with.  The
    result is an ordinary map collection; :func:`merge_maps` joins its parts
    and :func:`combine_maps` joins it with the base.
    """
    config = config or MapConfig()
    buckets = plan_additions(base, rows, config)
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    return _build_collection(output, buckets, config, partition or Partition(), progress_every,
                             {"grid": {"additions_to": str(Path(base).resolve()), "n_rows": int(len(rows))}})


# Per-build fields of a child manifest that do not describe the coefficients.
_CHILD_LOCAL = {"shards", "n_maps", "direct_diagnostics", "error_certificate", "builder"}


def _link_tree(source: Path, target: Path) -> None:
    """Hard-link every file of ``source`` below ``target`` (copy across devices)."""
    for path in sorted(source.rglob("*")):
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)


def combine_maps(destination: str | Path, sources: Iterable[str | Path]) -> Path:
    """Join map collections into a new one without touching the sources.

    Buckets with the same name are concatenated and must share their radial
    nodes and coefficient format; all map IDs must be distinct.  Shard files
    are hard links where the file system allows (copies otherwise), so the
    sources stay valid and the new collection costs no extra space.
    """
    sources = [Path(source).resolve() for source in sources]
    if len(sources) < 2:
        raise ValueError("combine_maps needs at least two collections")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"combine destination already exists: {destination}")
    collections = [BucketedPolarAtlas(source) for source in sources]
    grouped: dict[str, list] = {}
    for collection in collections:
        for bucket in collection:
            grouped.setdefault(bucket.name, []).append(bucket)
    all_ids = np.concatenate([np.asarray(c.map_ids, dtype=np.int64) for c in collections])
    if len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError("the collections share map IDs")
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        entries = []
        for name, members in sorted(grouped.items()):
            first = members[0]
            nodes = np.asarray(first.atlas.radial_nodes)
            reference = {k: v for k, v in first.atlas.manifest.items() if k not in _CHILD_LOCAL}
            child = temporary / name
            child.mkdir()
            shards, ids, params = [], [], []
            for index, bucket in enumerate(members):
                if not np.array_equal(np.asarray(bucket.atlas.radial_nodes), nodes):
                    raise ValueError(f"bucket {name!r} has different radial nodes in {bucket.path}")
                fields = {k: v for k, v in bucket.atlas.manifest.items() if k not in _CHILD_LOCAL}
                if fields != reference:
                    raise ValueError(f"bucket {name!r} has a different coefficient format in {bucket.path}")
                for shard_index, shard in enumerate(bucket.atlas.manifest["shards"]):
                    path = f"shards/src{index:02d}-{shard_index:06d}"
                    _link_tree(bucket.path / shard["path"], child / path)
                    shards.append({**shard, "path": path})
                ids.append(np.asarray(bucket.atlas.map_ids, dtype=np.int64))
                params.append(np.asarray(bucket.atlas.map_parameters))
            ids, params = np.concatenate(ids), np.concatenate(params)
            np.save(child / "map_ids.npy", ids)
            np.save(child / "map_parameters.npy", params)
            np.save(child / "radial_nodes.npy", nodes)
            manifest = {**reference, "builder": "combined", "n_maps": int(len(ids)), "shards": shards}
            (child / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            entries.append({
                "name": name, "path": name, "count": int(len(ids)),
                "logs": [min(b.logs[0] for b in members), max(b.logs[1] for b in members)],
                "logq": [min(b.logq[0] for b in members), max(b.logq[1] for b in members)],
                "map_ids": ids.tolist(),
            })
        metadata = {"builder": "combined", "sources": [str(source) for source in sources]}
        frames = {c.manifest.get("coordinate_frame") for c in collections} - {None}
        if len(frames) > 1:
            raise ValueError(f"the collections use different coordinate frames: {sorted(frames)}")
        if frames:
            metadata["coordinate_frame"] = frames.pop()
        write_bucket_manifest(temporary, entries, metadata)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def merge_maps(output: str | Path, *, destination: str | Path | None = None) -> Path:
    """Validate and merge all generated parts without recomputing coefficients."""
    output = Path(output).resolve()
    parts = sorted((output / "parts").glob("part-*-of-*"))
    if not parts:
        raise FileNotFoundError(f"no parts found below {output / 'parts'}")
    records = [json.loads((part / "hamagri-build.json").read_text()) for part in parts]
    expected = int(records[0]["partition"]["count"])
    indices = [int(record["partition"]["index"]) for record in records]
    if len(parts) != expected or indices != list(range(expected)):
        raise ValueError(f"expected all partition indices 0..{expected - 1}; got {indices}")
    if any(record["grid"] != records[0]["grid"] for record in records[1:]):
        raise ValueError("parts were generated from different parameter grids")
    def merge_config(record: dict[str, object]) -> dict[str, object]:
        # Shard size changes only the on-disk packing/checkpoint granularity;
        # it does not change the generated coefficients.
        config = dict(record["config"])
        config.pop("shard_size", None)
        return config

    if any(
        merge_config(record) != merge_config(records[0])
        for record in records[1:]
    ):
        raise ValueError("parts were generated with different map configs")

    destination = Path(destination or output / "maps").resolve()
    if destination.exists():
        raise FileExistsError(f"merge destination already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        entries = []
        selected_ids: list[int] = []
        m_max: int | None = None
        for part in parts:
            collection = BucketedPolarAtlas(part)
            if m_max is None:
                m_max = collection.m_max
            elif collection.m_max != m_max:
                raise ValueError("parts use different stored mode budgets")
            for bucket in collection:
                target_name = bucket.name
                target = temporary / target_name
                if target.exists():
                    raise ValueError(f"duplicate radial bucket: {target_name}")
                shutil.copytree(bucket.path, target)
                ids = [int(value) for value in bucket.map_ids]
                selected_ids.extend(ids)
                entries.append(
                    {
                        "name": target_name,
                        "path": target_name,
                        "count": len(ids),
                        "logs": list(bucket.logs),
                        "logq": list(bucket.logq),
                        "map_ids": ids,
                    }
                )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("parts contain duplicate map IDs")
        entries.sort(key=lambda item: str(item["name"]))
        write_bucket_manifest(
            temporary,
            entries,
            {
                "builder": "merged-direct-vbm-rho-aware-adaptive-radial-fourier",
                "radial_layout": json.loads(
                    (parts[0] / "manifest.json").read_text()
                ).get("radial_layout"),
            },
        )
        (temporary / "hamagri-build.json").write_text(
            json.dumps(
                {
                    **records[0],
                    "partition": {"index": 0, "count": 1},
                    "selected_map_ids": sorted(selected_ids),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination
