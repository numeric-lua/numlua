# numlua 0.26.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- New elementwise ufuncs: `isinf`, `isfinite`, `sign`, `floor`, `ceil`,
  `trunc`, `round`, `reciprocal`, `log1p`, and the logical family
  `logical_and`/`logical_or`/`logical_xor`/`logical_not`.
- `nan_to_num`, `nancumsum`, and `nancumprod` add NaN replacement and
  NaN-aware cumulative reductions.
- `numlua.unwrap` removes phase jumps along a configurable axis, period, and
  discontinuity threshold.
- `numlua.interp` provides shape-preserving linear interpolation over a
  strictly increasing 1-D grid.
- `loadtxt`, `savetxt`, and `genfromtxt` now support bool tensors, including
  `True`/`False` and numeric spellings, custom formats, converters, and
  missing/filling values.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives, ZIP64 reading, raw binary and text I/O, and all non-compression
features work without it.

## Verification

- 421 tests passed with system zlib
- 421 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.26.0-1.rockspec`

Tag and GitHub release: `v0.26.0`.
