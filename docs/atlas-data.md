# Atlas data

## Software and data are separate

This repository contains the hamagri software, examples, and documentation;
it does not contain the Fourier coefficient arrays. The curated atlas is a
separate, large data product (about 72 GB). Keep atlas archives and build
intermediates outside the source repository.

The software is licensed under MIT. That license does not automatically apply
to a separately distributed atlas. Follow the license, attribution, and
citation terms supplied with the specific atlas you use, as well as the
licenses of VBMicrolensing and other dependencies.

## Release status and provenance

The curated atlas contains 33,993 maps (~72 GB) and is packaged as four
checksum-verified archives. Public download hosting is being arranged; a
download link and citation details will be added here when the data release is
available.

The curated atlas was generated over several months.

## Atlas format

A bucketed atlas has a root `manifest.json` and a `buckets/` directory.
The root manifest identifies the collection; each bucket has its own manifest
and the coefficient and radial-layout arrays used by the reader. hamagri
recognizes the `hamagri-bucketed-fourier-maps` root format and
`hamagri-fourier-maps` bucket format. A compatible single-atlas directory
can also be opened.

Open an existing atlas with:

```python
from hamagri import Atlas

atlas = Atlas.open("/path/to/atlas")
print(atlas.n_maps, atlas.m_max, atlas.coordinate_frame)
```

To generate an atlas from a parameter grid, see
[distributed generation](distributed-generation.md). The atlas is not
downloaded or modified by `Atlas.open()`.
