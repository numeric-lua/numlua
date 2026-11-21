# numlua 0.12.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.take_along_axis(t, indices[, axis])` adds `np.take_along_axis`-shaped
  per-slice selection. Indices are 1-based integer tensors with the same rank
  as the input; non-axis size-1 dimensions broadcast in either operand, the
  selection length along the axis comes from the indices, and `axis=nil`
  flattens both operands. Out-of-range indices, including `0` and negatives,
  raise a clear `numlua.take_along_axis:` error, and the result owns independent
  storage. It composes with `argsort`, `argmin`/`argmax(..., keepdims)`, and
  `take`;
- `numlua.load` now reads NPY v2.0 and v3.0 headers (the 4-byte header-length
  field and v3.0's UTF-8 header encoding) in addition to v1.0. `numlua.save`
  still writes v1.0, so output stays byte-compatible with existing readers.
  `.npz` archives whose members use any of the three header versions load
  through the same path;
- docs/API reference now cover `take_along_axis`, its view/copy boundary and
  error behavior, the reader's version support, and the fixed negative-index
  difference from NumPy;
- release-consistency tests keep `numlua.VERSION`, the rockspec version/tag,
  the CHANGELOG entry, the packaged module list, the README status/rockspec,
  and the documented selection and NPY-version surface in sync.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it. No runtime
dependency was added.

## Verification

- 156 tests passed with system zlib
- 156 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
