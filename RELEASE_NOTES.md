# numlua 0.23.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.var` and `numlua.std` add NumPy's population variance/standard
  deviation default (`ddof = 0`), arbitrary finite `ddof`, axis/`keepdims`
  reductions, and `float64` promotion for integer/bool input.
- `numlua.nansum`, `numlua.nanmean`, `numlua.nanmin`, `numlua.nanmax`,
  `numlua.nanvar`, and `numlua.nanstd` ignore NaNs, with all-NaN slices
  returning NaN and `nansum` treating NaN as zero.
- `numlua.meshgrid` builds coordinate grids with NumPy's `indexing = "xy"` or
  `"ij"`, `sparse`, and `copy` options, from tensors or Lua tables.
- `numlua.gradient` computes first-order edge and central interior differences
  with uniform spacing, coordinate arrays, or per-axis spacing.
- `numlua.genfromtxt` extends the numeric text reader with `skip_header`,
  `skip_footer`, `missing_values`, and `filling_values`; missing floats default
  to NaN and missing integers to zero. Existing `loadtxt` behavior is unchanged.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives, raw binary and text I/O, and all non-compression features work
without it.

## Verification

- 346 tests passed with system zlib
- 346 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.23.0-1.rockspec`

Tag and GitHub release: `v0.23.0`.
