# Accuracy and limitations

## Map reconstruction

The atlas stores angular Fourier coefficients on radial nodes. Angular sampling
adapts to measured coefficient convergence and caustic structure. Radial nodes
are placed using pilot maps selected to expose sharp finite-source structure.
The stored `reconstruction_error` and `deviation_envelope` arrays summarize map
reconstruction behavior and support risk-aware rescanning.

These arrays are useful diagnostics for deciding which maps to rescan. They are
not a formal error bound on every light curve or on its fitted chi-square.
The curated atlas does not contain `certified_error*.npy` arrays. The
public search API neither requires such arrays nor exposes chi-square lower and
upper limits.

## Search scores

A candidate `chi2` is the profiled score calculated from the stored Fourier
representation at the reported seed. `spectral_risk` is a screening diagnostic
for maps whose reconstruction metadata suggests greater risk. Neither value
certifies that the candidate is the global minimum or that its score differs
from a direct VBMicrolensing evaluation by a bounded amount.

Always re-evaluate promising seeds with direct VBMicrolensing and perform the
physical fit used for scientific inference. Validate recall and score errors on
a representative suite of caustic and smooth events before changing defaults.

## Coordinate frames

The curated atlas uses the AdaMGrid map coordinate frame. Native source
geometries and map-frame geometries differ for wide binaries. Because the
translation depends on `(s, q, alpha)`, the current atlas-wide FFT search cannot
convert one native geometry once and reuse it for every map and angle. The API
tags geometry and atlas frames and rejects mismatches. Map reconstruction at
a caller-supplied map-frame point remains available.

## Angular Nyquist condition

For map coefficients through mode `M`, the quadratic light-curve moment reaches
mode `2*M`. A real FFT must keep that mode strictly below its Nyquist bin.
With an even grid, the safe condition is

```text
n_alpha >= 4*M + 2
```

Equality at `n_alpha = 4*M` places the quadratic term exactly on Nyquist.
`hamagri` rejects that setting before scanning and reports the required minimum.
