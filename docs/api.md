# Python API

## Open an atlas

```python
from hamagri import Atlas

atlas = Atlas.open("hamagri-atlas")
print(atlas.n_maps, atlas.m_max, atlas.coordinate_frame)
```

`Atlas.open()` supports single and bucketed atlases. Coefficient arrays are
memory-mapped. For a bucketed atlas, rows of `atlas.parameters` follow the map
IDs listed in the root manifest.

## Reconstruct magnification

```python
A = atlas.magnification(map_id, x, y, m_max=96)
```

Coordinates are Cartesian in the atlas frame and must be within the map's
radial range. Use `atlas.radial_nodes_for(map_id)` to inspect a bucketed map's
radial nodes.

## Search light curves

```python
import numpy as np
from hamagri import Dataset, Geometry

samples = np.loadtxt("lightcurve.dat")
dataset = Dataset(samples[:, 0], samples[:, 1], samples[:, 2])
geometry = Geometry(
    t0=2459000.0,
    u0=0.08,
    tE=24.0,
    coordinate_frame=atlas.coordinate_frame,
)
result = atlas.search([dataset], [geometry])
best = result.candidates[0]
print(best.s, best.q, best.rho, best.alpha, best.chi2)
```

`Geometry.coordinate_frame` defaults to `"native"` and must match
`atlas.coordinate_frame`. Search does not convert frames automatically; see
[Coordinate frames](coordinate-frames.md).

### SearchConfig

| Field | Default | Meaning |
|---|---:|---|
| `base_m_max` | 32 | modes in the first pass |
| `full_m_max` | 128 | modes in the detailed pass |
| `base_n_alpha` | 144 | angle samples in the first pass |
| `full_n_alpha` | 576 | angle samples in the detailed pass |
| `top_count` | 500 | best first-pass maps to rescan |
| `risk_count` | 500 | highest-risk maps to rescan |
| `candidate_count` | 300 | maximum returned seeds |
| `full_radial_order` | 3 | interpolation order in the detailed pass |
| `refine` | true | refine seeds using stored map coefficients |

For each pass, `n_alpha` must be even and at least `4 * m_max + 2`.
`SearchConfig` rejects grids that do not meet this limit.

The first pass scans all maps. The detailed pass rescans the union of the
best-scoring and highest-risk maps, then clusters and refines candidate seeds
using the stored atlas.

## Fit candidates with direct VBMicrolensing

From a source checkout, install the fitting dependencies:

```bash
python -m pip install '.[fit]'
```

For searching and fitting in one environment, install both extras:

```bash
python -m pip install '.[search,fit]'
```

At least seven photometric points are required. The fitter scores every seed
with direct VBMicrolensing, short-refines the best-scoring seeds, and optionally
runs a longer fit from the best short fit for each input geometry. These are
local fits from the supplied seeds, not a global search.

```python
from hamagri import FitConfig, fit_candidates

fit_result = fit_candidates(
    result.candidates,
    [dataset],
    coordinate_frame=atlas.coordinate_frame,
    config=FitConfig(),
)
best_fit = fit_result.best
print(best_fit.parameters.as_dict())
print(best_fit.chi2, best_fit.success, best_fit.message)
```

For map-frame fits, each evaluation converts the trajectory to VBMicrolensing's
native frame. Candidates are processed serially. Set
`FitConfig(long_max_nfev=None)` to skip the longer stage.

`FitResult` contains direct scores, short fits, long fits, and the best fit.
Each fit includes its parameters, `chi2`, optimizer status, and evaluation
counts.

### FitConfig

| Field | Default | Meaning |
|---|---:|---|
| `short_count` | 500 | direct-scored seeds to short-refine |
| `short_max_nfev` | 80 | function evaluations per short fit |
| `long_max_nfev` | 1200 | evaluations per geometry; `None` skips this stage |
| `diff_step` | `1e-3` | relative finite-difference step |
| `vbm_tolerance` | `1e-3` | VBMicrolensing absolute tolerance |
| `vbm_relative_tolerance` | `1e-4` | VBMicrolensing relative tolerance |
| `ftol`, `xtol`, `gtol` | `1e-8` | optimizer tolerances |

## Build maps

```python
from hamagri import MapConfig, ParameterGrid, build_maps

grid = ParameterGrid(s=(0.8, 1.0, 1.2), q=(1e-4, 1e-3), rho=(3e-4, 1e-3))
output = build_maps("maps", grid, config=MapConfig())
```

Map generation requires VBMicrolensing and uses adaptive angular and radial
sampling. Existing destinations are not overwritten; interrupted builds can
resume from their partial output. See [Map generation](map-generation.md).

Map reconstruction diagnostics and the meaning of search scores are described
in [Accuracy and limitations](accuracy.md).
