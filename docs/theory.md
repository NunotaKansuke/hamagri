# Search method

`hamagri` builds Fourier atlases of excess magnification (magnification minus
one) for binary lenses.

## Map generation

VBMicrolensing evaluates magnification on radial rings. Angular sampling adapts
to the Fourier coefficients and caustic structure. Radial nodes are selected
using pilot maps. The atlas stores the Fourier coefficients and reconstruction
diagnostics.

## Search

For each input geometry, `hamagri` interpolates the atlas coefficients at the
observed source positions. An FFT evaluates the trajectory angles together.
The source and blend fluxes are profiled for each trial, producing a
light-curve `chi2` score.

Search uses two passes:

1. A coarse pass ranks every map.
2. A detailed pass rescans the best-scoring maps and maps with higher
   reconstruction risk.
3. Nearby results are clustered and refined using the stored atlas
   coefficients.

This refinement works on the atlas. It is separate from
`fit_candidates()`, which evaluates seeds with direct VBMicrolensing and runs
local physical fits.

See [Accuracy and limitations](accuracy.md) for the meaning of map diagnostics
and search scores, and [Coordinate frames](coordinate-frames.md) for the
required input frame.
