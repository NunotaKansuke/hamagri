# Precomputed atlas data

## Separate data release

This repository contains the hamagri software, not the full precomputed map
arrays. The curated atlas is about 72 GB and is prepared as a separate data
release. Its public record and DOI will be linked here when available.

You do not need to download the precomputed atlas to generate maps. See
[Map generation](map-generation.md) for the custom-grid and full-grid workflows.

## Atlas contents and format

The curated integrated version-1 atlas contains 33,993 maps in 161 radial
buckets with `m_max=512`. Its root manifest uses
`hamagri-bucketed-fourier-maps`; each bucket has a
`hamagri-fourier-maps` manifest. The atlas stores Fourier coefficients,
reconstruction-error metadata, and radial-layout metadata.

`Atlas.open(path)` reads the version-1 bucket layout. The loaded atlas exposes
map IDs, parameters, mode count, and magnification reconstruction through the
Python API.

## License

The software in this repository is licensed under MIT. The precomputed atlas
is intended for release under CC BY 4.0; use the data record's citation and
license text when that record becomes public.
