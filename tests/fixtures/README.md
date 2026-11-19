# Test fixtures

This directory contains two kinds of fixture data:

- **Committed NumPy-written files** (`float64-2x2.npy`, `int64-1d.npy`,
  `bool-1d.npy`, `empty-2x0.npy`, `scalar-int64.npy`, and
  `big-endian-f4.npy`). These are the interoperability reference files; they
  were written by NumPy and exercise the reader without going through
  `numlua.save`.
- **A generated corpus** produced by `generate.lua` at test time under
  `tests/.tmp/`. It covers 3-D and 2-D shapes, all four supported dtypes,
  zero-size arrays, 0-D scalars, a stored-ZIP `.npz` with several members, and
  a big-endian file for the rejection path.

`generate.lua` writes the NPY v1.0 headers, little-endian payloads, and stored
ZIP container directly. It deliberately does not call `numlua.save`/`savez`, so
the reader is tested against independently constructed bytes. Generated files
are left in the ignored `tests/.tmp/` directory; no generated `.npy`/`.npz`
binary is committed.

Run the corpus tests with the rest of the suite:

```sh
./scripts/test.sh
```
