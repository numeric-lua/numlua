# numlua 0.19.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.loadtxt` gains five NumPy text-I/O controls:
  `skiprows`/`max_rows` for row windows, 1-based `usecols` for column
  selection, `unpack` to transpose a 2-D result, and `ndmin` to keep one-row or
  one-column input as a `(1, N)`/`(N, 1)` tensor. `usecols` supports negative
  columns and preserves duplicates, and only selected fields are converted.
- `numlua.savetxt` gains `header`, `footer`, `comments`, and `newline` for
  comment-framed text files, plus per-column `fmt` tables. Header/footer lines
  are written so `loadtxt` ignores them, and existing output is unchanged when
  the new options are omitted.
- All new options validate their types and ranges with `numlua.loadtxt:` /
  `numlua.savetxt:` errors. `skiprows` counts physical lines, `max_rows` counts
  data rows, and rows beyond `max_rows` are not parsed.
- No runtime dependency was added. The text implementation remains pure Lua
  plus luatorch.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it.

## Verification

- 237 tests passed with system zlib
- 237 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.19.0-1.rockspec`

Tag and GitHub release: `v0.19.0`.
