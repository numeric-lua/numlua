# numlua 0.15.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.pad` gains the `"wrap"` mode: padded axes repeat cyclically with
  NumPy semantics, widths may exceed the axis length, per-axis before/after
  widths are independent, and zero-size axes can keep a zero-width wrap but
  cannot be extended by it;
- `numlua.pad` constant values now accept NumPy's `constant_values` forms. In
  addition to the historical scalar, a `{before, after}` pair broadcasts over
  every axis and a table with one entry per axis gives per-axis values (each a
  number or pair). Scalar calls and the default zero fill are unchanged, and
  fills keep the input dtype;
- `numlua.pad` now rejects negative and fractional numeric `pad_width` forms
  instead of silently ignoring them;
- `numlua.savez` and `numlua.savez_compressed` reject keys that normalize to
  the same ZIP member name (`x` and `x.npy`, for example) and the bare key
  `.npy` before writing payloads or creating a file. Previously duplicate
  members silently overwrote each other on read;
- the test suite adds dedicated pad parity tests plus selection and I/O edge
  coverage: empty/0-D `select` operands, numeric truth testing, non-contiguous
  defaults, Unicode/spaced and colliding `.npz` names, EOCD archive comments,
  invalid `.npy` members, multi-character text delimiters, and zero-width text
  tensors. Docs and CHANGELOG document all of it.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it. No runtime
dependency was added.

## Verification

- 193 tests passed with system zlib
- 193 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
