# numlua 0.25.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.dot` now follows NumPy's shape rules for vector, matrix, and batched
  products.
- `numlua.inner` sums over the last axes, `numlua.outer` flattens both
  operands, and `numlua.cross` computes 3-vector cross products over a
  1-based axis with ATen batch broadcasting.
- `numlua.trapezoid` and the `numlua.trapz` alias integrate along one axis
  with uniform spacing or coordinate arrays.
- `numlua.loadtxt` and `numlua.genfromtxt` accept a single-character
  `quotechar`, so quoted numeric CSV fields can contain delimiters and spaces.
  A doubled quote character escapes itself.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives, ZIP64 reading, raw binary and text I/O, and all non-compression
features work without it.

## Verification

- 397 tests passed with system zlib
- 397 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.25.0-1.rockspec`

Tag and GitHub release: `v0.25.0`.
