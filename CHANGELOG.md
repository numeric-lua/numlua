# Changelog

## 0.3.0 — 2026-11-21

### Added

- NumPy-style elementwise ufuncs: `add`, `subtract`, `multiply`, `divide`,
  `negative`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`, `maximum`,
  `minimum`, `power`, and `square`, with scalar/tensor operands and ATen
  broadcasting;
- `argmin(t[, axis[, keepdims]])`, the minimum counterpart of `argmax`, with
  the same 1-based flat/axis indices and `keepdims` handling;
- generated `.npy`/`.npz` fixture corpus covering 3-D/2-D shapes, all supported
  dtypes (including `float16`, `int8`, `int16`, `int32`, and `uint8`), empty
  and 0-D arrays, stored and deflate ZIP members, and big-endian rejection;
  non-contiguous write behavior is covered separately by the `.npy`/`.npz`
  round-trip tests, since a NumPy file payload is C-order by construction;
- optional deflate-compressed `.npz` reading through a system zlib loaded via
  LuaJIT FFI; stored ZIP reading/writing remains dependency-free and deflate
  fails with a clear error when zlib is unavailable;
- ZIP integrity validation for corrupt central directories, local/central CRC
  mismatches, truncated archives, encrypted/multi-disk/ZIP64 archives, and
  unsupported compression methods;
- `linalg.matrix_norm` accepts `ord=-2` (the smallest singular value), rounding
  out NumPy's supported matrix-norm orders alongside `2` and `"nuc"`;
- `bench/bench.lua` and `scripts/bench.sh` for construction, reduction,
  broadcasting, layout, and `.npy` I/O comparisons against direct ATen/raw
  Lua baselines;
- `docs/performance.md` with the dated run, machine/compiler conditions,
  method, explicit losses, and caveats;
- `swapaxes`, `moveaxis`, `atleast_1d`, `atleast_2d`, `atleast_3d`, and
  `resize` with C-order zero-fill/truncation semantics;
- `pad` modes `edge`, `reflect`, and `symmetric` in addition to `constant`;
- explicit view/copy documentation plus tests for axis permutations,
  zero-size dimensions, negative axes, invalid modes, and independent storage;
- a large layout regression test proving documented views do not silently copy
  and that non-contiguous `ravel` does copy.

### Tests

- 112 tests, passing with system zlib available and with
  `NUMLUA_DISABLE_ZLIB=1`; the `.npz` suite now covers encrypted, multi-disk,
  ZIP64, and inconsistent local-header rejection.

## 0.2.0 — 2026-11-19

### Added

- `broadcast_to`, `broadcast_arrays`, `tile`, `repeat_`, `flip`, `roll`, `pad`,
  `diff`, `cumsum`, `cumprod`, `argsort`, and `take`;
- `linalg.pinv`, `linalg.lstsq`, and `linalg.matrix_norm`;
- `promote_types`, `result_type`, and `can_cast` safe-promotion helpers;
- stored-ZIP `.npz` support via `savez`/`loadz`;
- committed `.npy` fixtures for NumPy interoperability;
- LuaRocks package `numlua-0.2.0-1` and CI against luatorch.

### Tests

- 88 tests.

## 0.1.0

- initial NumPy-shaped facade over luatorch;
- constructors, shape/reduction/predicate helpers, `.npy` and text I/O.
