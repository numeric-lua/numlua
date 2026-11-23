# numlua 0.21.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.split`, `array_split`, `hsplit`, `vsplit`, and `dsplit` split along
  1-based axes, with NumPy's even/uneven section rules and view-returning
  boundary tables.
- `numlua.diag`, `diagflat`, `tril`, and `triu` add NumPy's diagonal
  constructors/extractors and triangular masks, including N-D and bool input.
- `numlua.isclose`, `allclose`, `array_equal`, and `array_equiv` add
  broadcasting-aware closeness/equality predicates with NumPy's tolerance and
  infinity/NaN behavior.
- `numlua.fromfile` and `numlua.tofile` add raw binary I/O for every supported
  dtype, with `dtype`/`count`/byte-`offset` controls and path or handle targets.
- `numlua.loadtxt` and `numlua.savetxt` accept open Lua file handles at their
  current position and leave caller-owned handles open.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives, raw binary I/O, and all non-compression features work without it.

## Verification

- 292 tests passed with system zlib
- 292 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.21.0-1.rockspec`

Tag and GitHub release: `v0.21.0`.
