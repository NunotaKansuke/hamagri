# Map generation

The software repository includes the map generator. A precomputed atlas is
useful when you want to search immediately, but it is not required to generate
maps. Generation evaluates the requested parameters with VBMicrolensing and
writes an atlas directory that hamagri can open.

## Install

From the repository root, install the generation dependency:

```bash
python -m pip install '.[generate]'
```

If you also plan to run light-curve searches, install both optional dependency
sets:

```bash
python -m pip install '.[search,generate]'
```

## Generate a custom grid

Use physical values for `s`, `q`, and `rho`. This example builds one map:

```python
from hamagri import Atlas, MapConfig, ParameterGrid, build_maps

grid = ParameterGrid(
    s=(1.0,),
    q=(1e-3,),
    rho=(1e-3,),
)
output = build_maps('my-maps', grid, config=MapConfig())
atlas = Atlas.open(output)
print(atlas.n_maps)  # 1
```

The output directory contains its manifest and coefficient arrays. It can be
used directly with `Atlas.open()`; no conversion or atlas download is needed.

## Generate the full AdaMGrid default set

The repository includes a production configuration for the complete default
parameter grid:

```bash
hamagri build-maps examples/adamgrid_default_60.json ./hamagri-atlas
```

This grid contains 33,993 maps and uses `m_max=512`. The full atlas is about
72 GB, and generating it is a substantial compute job. Keep additional free
space for checkpoints and intermediate files. The command above runs one
partition on one process. For Slurm or multiple machines, use the
[distributed generation guide](distributed-generation.md), which includes the
partition and merge commands.

A user who only needs a smaller or different parameter range can pass a custom
`ParameterGrid` to `build_maps()` instead of running the full preset.

## Precomputed atlas

The curated, precomputed atlas is distributed separately from the software
because of its size (about 72 GB). It is not a prerequisite for map generation.
See [Atlas data](atlas-data.md) for information about the data product and its
format.
