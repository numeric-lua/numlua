# numlua 0.9.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `savez_compressed` writes deflate-compressed (ZIP method 8) `.npz` archives
  through the optional system zlib already used for reading, with an explicit
  `{ fallback = "stored" }` mode and a clear error when zlib is unavailable;
- compressed round-trip, method-8 header, archive-size, fallback, and
  zlib-disabled tests;
- broadcast/take edge-shape coverage for 0-D inputs, zero-size axes, high-rank
  right alignment, singleton axes, and empty selections;
- zero-size broadcasting follows NumPy: a size-1 axis against a size-0 axis
  yields a size-0 result;
- `take` rejects non-integer index tensors and reports empty-axis selection
  clearly.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib is optional: stored archives and
all non-compression features work without it.

## Verification

- 128 tests passed with system zlib
- 128 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
