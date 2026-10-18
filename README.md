# hamagri

`hamagri` builds and reads Fourier atlases for binary microlensing
magnification maps, then uses them to produce candidate seeds for a downstream
physical fit. Its search score is a profiled light-curve chi-square; a returned
candidate is a seed, not a final scientific fit.

The precomputed curated atlas is a separate data release because it is about
72 GB. It is not stored in this software repository. You can generate maps
with the code here without downloading that atlas; see [Map generation](docs/map-generation.md)
for a small example and the full-grid workflow. The precomputed atlas and its
format are described in [Data release](docs/data-release.md).

## Install

Requires Python 3.10 or later. From a checkout of the `hamagri` repository,
install the search dependencies:

```bash
python -m pip install '.[search]'
```

Map generation additionally requires VBMicrolensing:

```bash
python -m pip install '.[generate]'
```

To install both sets of dependencies at once:

```bash
python -m pip install '.[search,generate]'
```

## Open the atlas and search

```python
import numpy as np
from hamagri import Atlas, Dataset, Geometry, SearchConfig

samples = np.loadtxt('lightcurve.dat')  # time, flux, flux uncertainty
dataset = Dataset(samples[:, 0], samples[:, 1], samples[:, 2])
atlas = Atlas.open('hamagri-atlas')

# Geometry values must use the same coordinate frame as the atlas.
geometry = Geometry(
    t0=2459000.0,
    u0=0.08,
    tE=24.0,
    coordinate_frame=atlas.coordinate_frame,
)
result = atlas.search([dataset], [geometry], config=SearchConfig())
best = result.candidates[0]
print(best.s, best.q, best.rho, best.alpha, best.chi2)
```

`Atlas.open()` reads both bucketed and single-atlas directories. It memory-maps
coefficient arrays and exposes `n_maps`, `map_ids`, `parameters`, `m_max`, and
`coordinate_frame`. `Atlas.magnification(map_id, x, y)` reconstructs a map at
Cartesian points.

The curated atlas uses the map coordinate frame. A geometry obtained in the
native lens frame must not be passed as if it were already in the map frame:
for wide binaries, the conversion depends on `(s, q, alpha)`. `hamagri` raises a
clear error when the declared geometry frame and atlas frame differ; a single
conversion for an atlas-wide FFT scan is not implemented yet.

## Search behavior

The default two-stage scan uses `(M, n_alpha) = (32, 144)` and `(128, 576)`.
The alpha grid must place the highest quadratic light-curve harmonic strictly
below the real-FFT Nyquist bin. For a map truncated at mode `M`, this requires
an even `n_alpha >= 4*M + 2`. Unsafe settings fail before the scan starts with
the required minimum and a radix-2 suggestion.

Search returns the profiled `chi2` and a `spectral_risk` ranking diagnostic.
Re-evaluate retained seeds with direct VBMicrolensing before scientific
inference.

## Build maps

For a custom grid, install the generation extra and call `build_maps()`. This
example builds a small 12-map collection:

```python
from hamagri import Atlas, MapConfig, ParameterGrid, build_maps

grid = ParameterGrid(
    s=(0.8, 1.0, 1.2),
    q=(1e-4, 1e-3),
    rho=(3e-4, 1e-3),
)
output = build_maps('maps', grid, config=MapConfig())
atlas = Atlas.open(output)
print(atlas.n_maps)
```

To generate the full AdaMGrid default parameter set, run this command from the
repository root:

```bash
python -m pip install '.[generate]'
hamagri build-maps examples/adamgrid_default_60.json ./hamagri-atlas
```

The included configuration defines 33,993 maps with `m_max=512`. A full build
is a large compute job and produces an atlas of about 72 GB; allow additional
free space for checkpoints and intermediate files. The command above runs one
partition. For a scheduler or multiple machines, follow the
[distributed generation guide](docs/distributed-generation.md). The generated
directory can be opened directly with `Atlas.open('./hamagri-atlas')`.

## Documentation

- [Map generation](docs/map-generation.md)
- [Python API](docs/api.md)
- [Data release and atlas format](docs/data-release.md)
- [Theory and search method](docs/theory.md)
- [Distributed generation](docs/distributed-generation.md)
- [Accuracy and limitations](docs/accuracy.md)
- [Numerical validation plan](docs/validation-plan.md)
- [Development](docs/development.md)
- [Map coordinate frame](docs/pspl-coordinate-frame-fft-search.md)

## License

MIT. Please also follow the licenses and citation requirements of
VBMicrolensing and JAX.
