# Accuracy and limitations

## Map reconstruction

`reconstruction_error` and `deviation_envelope` describe magnification-map
reconstruction. `spectral_risk` uses these diagnostics to prioritize maps for
the detailed search.

## Search scores

`Candidate.chi2` is a profiled score from the stored Fourier map at the
reported seed. It is a seed-ranking score, not a final physical fit. The
search does not establish a global best fit. Re-evaluate promising candidates
with direct VBMicrolensing and use `fit_candidates()` before scientific
inference.

## Coordinate frames

The curated atlas uses the AdaMGrid map frame. `Atlas.search()` requires input
geometries in the atlas frame and reports an error when the frames differ. It
does not convert geometries during the search; see
[Coordinate frames](coordinate-frames.md).

## Angular sampling

For each search pass, `n_alpha` must be even and at least
`4 * m_max + 2`. The code rejects unsafe values before scanning.
