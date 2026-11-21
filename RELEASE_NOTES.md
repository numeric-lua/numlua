# numlua 0.10.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- ZIP writer limits are enforced instead of silently wrapping ZIP32 fields.
  Archives with more than 65535 members, member names above 65535 bytes,
  per-entry sizes or a central-directory offset above 4294967295 bytes now fail
  with a `numlua.savez:`/`numlua.savez_compressed:` error before the output
  file is opened; ZIP64 writing is not supported;
- `numlua.load` accepts native-endian `=` descr values (`=f4`, `=i8`, ...) as
  aliases for the little-endian `<` spellings, matching NumPy on little-endian
  hosts, while unsupported `=` dtypes still fail with the original spelling in
  the error;
- `.npz` edge coverage for empty archives, keys already ending in `.npy`, and a
  writer-limit boundary that accepts an archive exactly at the limit;
- regression tests prove guard failures leave no partial archive and exercise
  stored and deflate size checks.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib is optional: stored archives and
all non-compression features work without it. No runtime dependency was added.

## Verification

- 135 tests passed with system zlib
- 135 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
