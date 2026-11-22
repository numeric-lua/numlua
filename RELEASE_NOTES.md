# numlua 0.17.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.apply_along_axis(func, axis, t[, ...])` brings
  `np.apply_along_axis` to Lua callbacks. The callback receives a 1-D luatorch
  view of each slice along the 1-based axis (negative axes accepted) and may
  return a scalar or 1-D tensor, Lua number, boolean, or Lua table. Scalar
  results remove the axis, 1-D results append their length after the remaining
  dimensions, and extra positional arguments are forwarded. Slices are visited
  in C order, the output dtype follows the first callback result, and the
  result owns independent storage. A zero-size selected axis is supported;
  because a zero-size non-axis dimension has no first slice, that case raises a
  clear error instead of guessing a dtype or result length;
- the test suite adds a dedicated `apply_along_axis` file covering 1-D/2-D/3-D
  inputs, every axis, C-order visitation, scalar and 1-D callbacks, dtype
  propagation, output ownership, empty axes, non-contiguous inputs, and the
  validation/error paths;
- selection and IO edge coverage expands: `compress` and `flatnonzero`
  behavior, 0-D `argsort` (now a clear `numlua.argsort:` error), zero-size
  gather dimensions, non-tensor `take` indices, all numeric `loadtxt` dtype
  ids, literal delimiters, comment-marker validation, `savetxt` special floats
  and one-row/one-column shapes, invalid format strings, and empty or
  prefix-only NPY files;
- docs, README, and CHANGELOG describe the new API and release.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it. No runtime
dependency was added.

## Verification

- 213 tests passed with system zlib
- 213 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
