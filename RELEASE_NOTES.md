# numlua 0.18.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.apply_over_axes(func, t, axes[, ...])` brings
  `np.apply_over_axes` to Lua callbacks. `axes` is a 1-based axis or a table of
  axes (negative and repeated axes accepted), applied in order to the current
  value. A callback result that keeps the input rank is used unchanged, so
  keepdims reductions and same-shape callbacks such as `cumsum` work; a result
  that drops one dimension is restored with a size-1 axis at the same position.
  Any other rank change raises a clear `numlua.apply_over_axes:` error, and 0-D
  input is rejected because it has no axis. Callbacks may return tensors, Lua
  numbers, booleans, or Lua tables, and extra positional arguments are
  forwarded;
- the test suite adds a dedicated `apply_over_axes` file covering 1-D/2-D/3-D
  inputs, every axis and order, negative and repeated axes, same-rank and
  reducing callbacks, callback argument forwarding, scalar/boolean/table
  results, zero-size reductions, non-contiguous input, dtype preservation, and
  the validation/rank-error paths;
- selection and IO edge coverage expands: descending `argsort` values and flag
  validation, non-contiguous `argsort`/`take`/`take_along_axis`/`compress`/
  `flatnonzero` inputs, empty broadcast selections, CR-only and mixed text line
  endings, strict unknown text option keys, multi-character save delimiters,
  integer and float16 text round-trips, trailing empty fields, and stored NPZ
  members that use ZIP data descriptors;
- `numlua.argsort` now rejects a non-boolean `descending` flag instead of
  silently sorting ascending, and `loadtxt`/`savetxt` reject unknown option keys
  instead of ignoring them;
- docs, README, and CHANGELOG describe the new API and release, including the
  `int64` text-output precision limit beyond the exact double range.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it. No runtime
dependency was added.

## Verification

- 225 tests passed with system zlib
- 225 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.18.0-1.rockspec`

No tag or release is created by this preparation.
