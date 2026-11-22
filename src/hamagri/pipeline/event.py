"""Event description files for ``hamagri solve``.

An event is a JSON object::

    {
      "pspl": [t0, u0, tE],
      "datasets": [{"path": "w146.txt", "name": "W146"}, ...],
      "parallax": {"ra_deg": ..., "dec_deg": ..., "t_ref": ...,
                   "satellite": 0, "satellite_dir": null},
      "piE": [piEN, piEE]
    }

Each dataset file holds whitespace-separated columns ``time flux error``
(``columns`` picks other column indices).  Relative paths are resolved from
the event file's directory.  ``parallax`` and ``piE`` are optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._core.trajectory import PhotometricDataset
from .parallax import Parallax


def load_event(path: str | Path) -> dict[str, Any]:
    """Keyword arguments of :func:`hamagri.pipeline.solve_event` from an event file."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    datasets = []
    for entry in payload["datasets"]:
        source = Path(entry["path"])
        source = source if source.is_absolute() else root / source
        table = np.loadtxt(source, ndmin=2)
        t, f, e = (table[:, int(k)] for k in entry.get("columns", (0, 1, 2)))
        datasets.append(PhotometricDataset(t, f, e, str(entry.get("name", source.stem))))
    event: dict[str, Any] = {"datasets": datasets, "pspl": tuple(float(v) for v in payload["pspl"])}
    if payload.get("parallax") is not None:
        parallax = dict(payload["parallax"])
        directory = parallax.get("satellite_dir")
        if directory is not None and not Path(directory).is_absolute():
            parallax["satellite_dir"] = str(root / directory)
        event["parallax"] = Parallax.from_dict(parallax)
        event["piE"] = tuple(float(v) for v in payload.get("piE", (0.0, 0.0)))
    return event
