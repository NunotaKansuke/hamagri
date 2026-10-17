# hamagri

`hamagri` builds and reads Fourier atlases for binary microlensing
magnification maps, then uses them to produce candidate seeds for a downstream
physical fit. Its search score is a profiled light-curve chi-square; a returned
candidate is a seed, not a final scientific fit.

The software repository and the large map atlas are separate releases. The
curated atlas is distributed separately because it is about 72 GB. See
[Data release](docs/data-release.md) for its manifest format and packaging
status.

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
It does not return certified chi-square lower or upper limits. The atlas has
adaptive reconstruction-error metadata for map-quality screening, but it does
not include chi-square certificate arrays and `hamagri` does not claim a
chi-square error guarantee. Re-evaluate retained seeds with direct
VBMicrolensing before scientific inference.

## Build maps

```python
from hamagri import MapConfig, ParameterGrid, build_maps

grid = ParameterGrid(s=(0.8, 1.0, 1.2), q=(1e-4, 1e-3), rho=(3e-4, 1e-3))
build_maps('maps', grid, config=MapConfig())
```

Adaptive angular sampling and radial-node placement target magnification-map
quality. Builds retain reconstruction diagnostics but do not write
`certified_error*.npy` arrays. These diagnostics do not guarantee a bound on a
light-curve chi-square. `Atlas.search()` returns a profiled score and risk
diagnostic without a certified chi-square interval. See [Python API](docs/api.md)
and [Theory](docs/theory.md) for details.

## Documentation

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
