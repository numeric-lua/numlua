# Test fixtures

This directory contains two kinds of fixture data:

- **Committed NumPy-written files** (`float64-2x2.npy`, `int64-1d.npy`,
  `bool-1d.npy`, `empty-2x0.npy`, `scalar-int64.npy`, and
  `big-endian-f4.npy`). These are the interoperability reference files; they
  were written by NumPy and exercise the reader without going through
  `numlua.save`.
- **A generated corpus** produced by `generate.lua` at test time under
  `tests/.tmp/`. It covers 3-D and 2-D shapes, all supported dtypes (`float16`,
  `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, and
  `bool`), zero-size arrays, 0-D scalars, stored-ZIP and deflate-ZIP `.npz`
  archives, and a big-endian file for the rejection path. Deflate members use
  real zlib output when zlib is available and valid stored DEFLATE blocks
  otherwise; the reader still requires zlib to inflate them.

`generate.lua` writes the NPY v1.0 headers, little-endian payloads, and ZIP
containers directly. It deliberately does not call `numlua.save`/`savez`, so
the reader is tested against independently constructed bytes. Generated files
are left in the ignored `tests/.tmp/` directory; no generated `.npy`/`.npz`
binary is committed. `tests/test_npz.lua` mutates generated and saved archives
to exercise corrupt central directories, CRC mismatches, and truncation.

Run the corpus tests with the rest of the suite:

```sh
./scripts/test.sh
```
