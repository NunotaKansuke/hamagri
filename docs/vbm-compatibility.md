# VBMicrolensing compatibility

Reading an existing atlas and reconstructing magnification maps require NumPy.
Map generation uses VBMicrolensing through its Python bindings; install the
optional `generate` dependency group.

For a reproducible data build, record the Python, NumPy, VBMicrolensing, and
compiler/runtime versions alongside the map configuration. The resulting
manifest records the builder settings and atlas schema. Before generating a
large atlas, build a small representative grid and compare its magnification
values with direct VBMicrolensing evaluations.

The released atlas is already built, so users who only search or
reconstruct maps do not need VBMicrolensing installed. Scientific finalists
should be reevaluated with a compatible direct VBMicrolensing installation.
