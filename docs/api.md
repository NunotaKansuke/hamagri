# Python API

## Open an atlas

```python
from hamagri import Atlas

atlas = Atlas.open('hamagri-atlas')
print(atlas.n_maps, atlas.m_max, atlas.coordinate_frame)
```

`Atlas.open()` reads the bucketed version-1 manifest distributed with the
curated atlas, as well as the single-atlas version-1 format. Array files are
memory-mapped. `Maps` remains as a compatibility alias; new code should use
`Atlas`.

`atlas.map_ids` and rows of `atlas.parameters` are aligned. For a bucketed
atlas, their order follows the bucket entries in the root manifest; it is not
necessarily ascending by numeric map ID.

Version-1 manifests use `hamagri-*` format and angle-convention identifiers.
The software and distributed atlas use the same manifest schema.

## Reconstruct magnification

```python
A = atlas.magnification(map_id, x, y, m_max=96)
```

Coordinates are Cartesian in the atlas coordinate frame and must lie within its
stored radial range. For bucketed atlases, use `atlas.radial_nodes_for(map_id)`
to inspect the radial grid for a map.

## Search light curves

```python
import numpy as np
from hamagri import Dataset, Geometry, SearchConfig

samples = np.loadtxt('lightcurve.dat')
dataset = Dataset(samples[:, 0], samples[:, 1], samples[:, 2])
geometry = Geometry(
    t0=2459000.0,
    u0=0.08,
    tE=24.0,
    coordinate_frame=atlas.coordinate_frame,
)
result = atlas.search([dataset], [geometry], config=SearchConfig())
best = result.candidates[0]
```

`Geometry.coordinate_frame` defaults to `"native"`. It must match
`atlas.coordinate_frame`; this release raises an error on a mismatch. The
curated atlas is in the AdaMGrid map frame. Conversion from native geometry is
map-specific and varies with `(s, q, alpha)`, so the shared-geometry FFT path
does not apply that conversion automatically.

### SearchConfig

| Field | Default | Meaning |
|---|---:|---|
| `base_m_max` | 32 | mode limit for the all-map ranking pass |
| `full_m_max` | 128 | mode limit for the selected-map pass |
| `base_n_alpha` | 144 | alpha samples in the ranking pass |
| `full_n_alpha` | 576 | alpha samples in the selected-map pass |
| `top_count` | 500 | low-chi-square maps considered for rescanning |
| `risk_count` | 500 | high spectral-risk maps considered for rescanning |
| `candidate_count` | 300 | maximum downstream candidate seeds |
| `full_radial_order` | 3 | cubic interpolation order in the selected-map pass |
| `refine` | true | refine candidate seeds in coefficient space |

For every stage, `n_alpha` must be even and satisfy `n_alpha >= 4*m_max + 2`.
The quadratic light-curve moment reaches mode `2*m_max`; putting that mode on
the real-FFT Nyquist bin is unsafe. A `ValueError` reports the parameter,
unsafe value, reason, minimum safe grid, and a radix-2 suggestion.

For example, `base_m_max=32` with `base_n_alpha=128` puts mode 64 exactly on
the Nyquist bin. `SearchConfig` rejects it before a scan and reports that
`base_n_alpha` must be at least 130, with 256 as the radix-2 suggestion.

The default search ranks every map at `(M, n_alpha) = (32, 144)`, rescans
selected maps at `(128, 576)`, then refines retained seeds up to the modes
stored in the atlas. Selection uses central chi-square and reconstruction-risk
ranking. It does not use or report certified chi-square intervals.

A `Candidate` exposes the physical map parameters, profiled `chi2`, FFT score,
spectral-risk diagnostic, refined trajectory parameters, and whether
refinement changed the map ID. It has no `chi2_lower` or `chi2_upper` fields.

## Direct physical fitting

Install the `fit` extra to use `fit_candidates()`:

```bash
python -m pip install '.[fit]'
```

The fitting function takes search candidates and the same photometric datasets
used for the search. `coordinate_frame` is required and must match the atlas
frame. At least seven photometric points are needed to fit the seven physical
parameters `(t0, u0, tE, s, q, rho, alpha)`.

```python
from hamagri import FitConfig, SearchConfig, fit_candidates

search_result = atlas.search(
    [dataset],
    [geometry],
    config=SearchConfig(candidate_count=3000),
)
fit_result = fit_candidates(
    search_result.candidates,
    [dataset],
    coordinate_frame=atlas.coordinate_frame,
    config=FitConfig(),
)
best_fit = fit_result.best
print(best_fit.parameters.as_dict())
print(best_fit.chi2, best_fit.success, best_fit.message)
```

The fitter first scores every seed with direct VBMicrolensing and profiles the
source and blend flux for each dataset. It short-refines up to the best 500
direct scores with Levenberg-Marquardt for at most 80 function evaluations.
It then starts a longer fit, capped at 1,200 evaluations, from the best short
fit for each `geometry_index`. These stages are local optimization from the
provided seeds; inspect the result status and chi-square before using a fit.

For map-frame candidates, every LM evaluation converts the map-frame
trajectory to VBMicrolensing's native frame using the current `(s, q, alpha)`.
This keeps `t0` and `u0` in the map frame throughout the optimization. Candidates
are processed serially. Set `FitConfig(long_max_nfev=None)` to skip the longer
stage.

`FitResult.direct_scores` contains the direct score for every input candidate;
`short_fits` and `long_fits` contain optimizer results for each stage; and
`best` is the lowest-chi-square result in the last stage that ran. Each fit
result includes its initial and final parameters and chi-square, optimizer
status, evaluation counts, and the last objective error if one occurred.

### FitConfig

| Field | Default | Meaning |
|---|---:|---|
| `short_count` | 500 | number of best direct scores to short-refine |
| `short_max_nfev` | 80 | maximum short-stage function evaluations per seed |
| `long_max_nfev` | 1200 | maximum long-stage evaluations per geometry; `None` skips it |
| `diff_step` | `1e-3` | relative finite-difference step used by SciPy |
| `vbm_tolerance` | `1e-3` | VBMicrolensing absolute tolerance |
| `vbm_relative_tolerance` | `1e-4` | VBMicrolensing relative tolerance |
| `ftol`, `xtol`, `gtol` | `1e-8` | LM convergence tolerances |

## Build maps

```python
from hamagri import MapConfig, ParameterGrid, Partition, build_maps

grid = ParameterGrid(s=(0.8, 1.0, 1.2), q=(1e-4, 1e-3), rho=(3e-4, 1e-3))
build_maps('maps', grid, config=MapConfig(), partition=Partition())
```

`build_maps` requires the optional VBMicrolensing dependency. Its default
adaptive angular and radial sampling targets map reconstruction quality. The
default does not write `certified_error*.npy` arrays, and reconstruction
diagnostics do not provide a certified chi-square bound. The destination is
never overwritten; interrupted builds can be resumed from their partial output
as described in [Distributed generation](distributed-generation.md).
