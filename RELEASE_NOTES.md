# numlua 0.13.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.save(path, tensor[, options])` can write NPY v2.0 and v3.0 with
  `{ version = 2 }` / `{ version = 3 }` (the dotted strings `"2.0"`/`"3.0"` are
  accepted too). The default remains v1.0, so existing output stays
  byte-identical. v2.0/v3.0 replace v1.0's 2-byte header length with a 4-byte
  one while keeping the 64-byte alignment rule; invalid versions and unknown
  options fail before the output file is opened;
- `numlua.take` now accepts 0-D and N-D integer index tensors following
  NumPy's result-shape rule. The selected axis is replaced by the index shape,
  a 0-D index removes the axis (a scalar result when `axis` is omitted), and
  the flattened-input form returns exactly the index shape. Out-of-range
  values raise a `numlua.take:` error naming the offending 1-based index, the
  axis or flattened input, and its size, instead of forwarding ATen's 0-based
  message;
- `numlua.put` rejects NaN and infinite indices with a clear
  `numlua.put: indices must be finite integers` error in every mode, before any
  write is attempted;
- docs/API reference now cover the writer versions, scalar/N-D `take` shapes,
  explicit range errors, and finite `put` indices. A dedicated byte-level NPY
  conformance suite checks all three header versions, alignment, length widths,
  every supported dtype, and 0-D/empty/non-contiguous tensors. Release tests
  also verify that the CI smoke job names the current rockspec and version;
- release consistency keeps `numlua.VERSION`, the rockspec version/tag, the
  CHANGELOG entry, the packaged module list, the README status/rockspec, and the
  documented writer and selection surface in sync.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored archives
and all non-compression features work without it. No runtime dependency was
added.

## Verification

- 169 tests passed with system zlib
- 169 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
