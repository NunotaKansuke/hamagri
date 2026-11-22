"""Command-line entry point for schedulers and multi-machine map builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_additions, build_maps, combine_maps, merge_maps
from .config import MapConfig, ParameterGrid, Partition
from ._core.map_queue import MapQueue


def _config(path: Path):
    content = json.loads(path.read_text(encoding="utf-8"))
    grid_data = content["grid"]
    if grid_data.get("preset") == "adamgrid-default":
        grid = ParameterGrid.from_adamgrid_default()
    elif {"log_s", "log_q", "log_rho"}.issubset(grid_data):
        grid = ParameterGrid.from_log10(**grid_data)
    else:
        grid = ParameterGrid(**grid_data)
    return grid, MapConfig(**content.get("maps", {}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hamagri")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-maps", help="build maps or one partition")
    build.add_argument("config", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--part-index", type=int, default=0)
    build.add_argument("--part-count", type=int, default=1)
    build.add_argument("--progress-every", type=int, default=1)
    merge = commands.add_parser("merge-maps", help="merge a complete set of parts")
    merge.add_argument("output", type=Path)
    merge.add_argument("--destination", type=Path)
    add = commands.add_parser("add-maps", help="build maps added to an existing collection (or one partition)")
    add.add_argument("config", type=Path, help="the base collection's build config")
    add.add_argument("base", type=Path)
    add.add_argument("rows", type=Path, help="text table: map_id log10_s log10_q log10_rho")
    add.add_argument("output", type=Path)
    add.add_argument("--part-index", type=int, default=0)
    add.add_argument("--part-count", type=int, default=1)
    add.add_argument("--progress-every", type=int, default=1)
    combine = commands.add_parser("combine-maps", help="join collections into a new one (hard links)")
    combine.add_argument("destination", type=Path)
    combine.add_argument("sources", type=Path, nargs="+")
    solve = commands.add_parser("solve", help="search one event (event JSON) and write the solutions")
    solve.add_argument("maps", type=Path)
    solve.add_argument("event", type=Path)
    solve.add_argument("output", type=Path)
    solve.add_argument("--workers", type=int, default=4)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--profiles", type=Path, help="saved FFT scan to reuse, or where to save it")
    status = commands.add_parser("queue-status", help="inspect a map-level queue")
    status.add_argument("queue", type=Path)
    status.add_argument("--stale-after", type=float, default=600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-maps":
        grid, config = _config(args.config)
        result = build_maps(
            args.output,
            grid,
            config=config,
            partition=Partition(args.part_index, args.part_count),
            progress_every=args.progress_every,
        )
    elif args.command == "merge-maps":
        result = merge_maps(args.output, destination=args.destination)
    elif args.command == "add-maps":
        import numpy as np

        _, config = _config(args.config)
        result = build_additions(
            args.output,
            args.base,
            np.loadtxt(args.rows, ndmin=2),
            config=config,
            partition=Partition(args.part_index, args.part_count),
            progress_every=args.progress_every,
        )
    elif args.command == "combine-maps":
        result = combine_maps(args.destination, args.sources)
    elif args.command == "solve":
        from .pipeline.event import load_event
        from .pipeline.run import solve_event

        event = load_event(args.event)
        solution = solve_event(args.maps, event.pop("datasets"), workers=args.workers, seed=args.seed,
                               profiles=args.profiles, log=print, **event)
        args.output.write_text(json.dumps(solution, indent=1) + "\n", encoding="utf-8")
        best = solution["best"]
        print(f"best chi2 {best['chi2']:.2f}" if best else "no solution")
        return 0 if best else 1
    else:
        queue = MapQueue.open(args.queue)
        print(json.dumps(queue.status(stale_after=args.stale_after).to_dict(), indent=2))
        return 0
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
