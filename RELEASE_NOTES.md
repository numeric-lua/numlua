# numlua 0.22.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.rot90`, `numlua.fliplr`, and `numlua.flipud` add NumPy's quarter-turn
  and axis-specific flip helpers, including explicit 1-based axis planes.
- `numlua.diagonal` and `numlua.trace` extract and sum diagonals over any two
  axes, keeping batch dimensions and preserving dtypes (including `bool`).
- `numlua.zeros_like`, `numlua.ones_like`, `numlua.empty_like`, and
  `numlua.full_like` inherit shape/dtype and accept `dtype` and `shape`
  overrides.
- `.npz` I/O accepts open Lua file handles for `savez`, `savez_compressed`,
  and `loadz`; caller-owned handles stay open.
- `numlua.fromfile`/`numlua.tofile` gain NumPy-style text mode via a non-empty
  `sep`, with `tofile` format strings and float/integer/bool parsing.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives, raw binary and text I/O, and all non-compression features work
without it.

## Verification

- 315 tests passed with system zlib
- 315 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.22.0-1.rockspec`

Tag and GitHub release: `v0.22.0`.
