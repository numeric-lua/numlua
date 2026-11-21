# numlua 0.11.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.put(t, indices, values[, mode])` adds `np.put`-shaped flat assignment.
  It accepts integer-tensor or Lua-table indices and tensor or Lua-table
  values, reuses values cyclically, applies duplicate indices in order (last
  write wins), and supports `"raise"`, `"wrap"`, and `"clip"` modes under the
  1-based index convention. It mutates and returns `t`; clone first for a copy;
- `.npz` deflate reads reject a declared uncompressed size beyond DEFLATE's
  1032:1 expansion bound before allocating the output buffer, so a tiny corrupt
  archive cannot request a multi-gigabyte allocation. Wrong sizes that remain
  plausible still reach zlib and report the real size mismatch, and highly
  compressible legitimate members continue to round-trip;
- zlib-unavailable errors now name `NUMLUA_DISABLE_ZLIB` when that environment
  switch, rather than a missing library, disabled compression;
- release-consistency tests keep `numlua.VERSION`, the rockspec version/tag,
  the CHANGELOG entry, and the packaged module list from drifting apart.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib is optional: stored archives,
`put`, and all non-compression features work without it. No runtime dependency
was added.

## Verification

- 147 tests passed with system zlib
- 147 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
