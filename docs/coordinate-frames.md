# Coordinate frames

An atlas and the geometries passed to `Atlas.search()` must use the same
coordinate frame. The atlas declares its frame as `atlas.coordinate_frame`;
set the corresponding frame on each `Geometry`.

The native lens frame and the map frame are related by a translation. For the
current map convention,

```text
shift_x(s, q) = -(s - 1/s) q / (1 + q),  s > 1
shift_x(s, q) = 0,                        s <= 1

x_map = x_native - shift_x(s, q)
y_map = y_native
```

The translation depends on the candidate separation and mass ratio. When
represented by trajectory parameters, its projection also depends on the
trajectory angle. A single conversion applied to a whole atlas search is
therefore not generally valid.

The current search checks frame labels but does not yet perform a
candidate-dependent native-to-map conversion inside an atlas-wide FFT scan.
Pass geometries already expressed in the atlas frame; do not bypass a frame
mismatch by relabeling the geometry. The implementation limitation and its
regression-test requirements are tracked in
[issue #4](https://github.com/NunotaKansuke/hamagri/issues/4).
