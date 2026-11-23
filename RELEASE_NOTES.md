# numlua 0.20.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.sort` returns sorted values along the last axis by default, with
  optional axis and descending controls matching `argsort`.
- `numlua.unique` flattens and sorts unique values and can also return
  first-occurrence indices, the inverse mapping, and counts. Indices follow
  numlua's 1-based convention.
- `numlua.nonzero` returns one 1-based coordinate tensor per axis, while
  `numlua.argwhere` returns the same coordinates as an `(N, ndim)` matrix.
- `numlua.loadtxt` gains a `converters` option for sentinel handling and custom
  field parsing, keyed by 1-based physical columns.
- `numlua.save` and `numlua.load` accept open Lua file handles, including
  reading several arrays sequentially from one handle. Handles stay open and
  are used at their current position.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it.

## Verification

- 259 tests passed with system zlib
- 259 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.20.0-1.rockspec`

Tag and GitHub release: `v0.20.0`.
