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

### Roman additions

The atlas is aimed at the Nancy Grace Roman Space Telescope's Galactic
Bulge Time Domain Survey. End-to-end tests of the
[event pipeline](pipeline.md) on simulated Roman light curves showed where
the base grid was too coarse or too narrow. In most of the failed searches,
the correct solution fell between grid points in the resonant-caustic zone,
or below the smallest mass ratio. An addition set of 5,781 maps
(39,774 maps in total) closes these two gaps:

| set | maps | what it adds |
|---|---:|---|
| resonant s refinement | 4,401 | log s midpoints (+0.025 dex) inside the resonant-caustic zone of each q, widened by 0.1 dex on both sides; every active log q row and the nine base log rho levels |
| low-q extension | 1,380 | log q = -7.0 ... -6.1 on the base s lattice (plus the midpoints above), with the same active (s, q) rule continued |

A new map is kept only while its finite-source planetary signal
`2 q / rho^2` is at least 0.3 %, that is, `rho <= sqrt(667 q)`. Larger
sources at that q would add maps that are indistinguishable from a point
lens. The additions were built with `hamagri add-maps` and joined to the
base atlas with `hamagri combine-maps` (see [Event pipeline](pipeline.md)).
They will be distributed with the data release.

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
