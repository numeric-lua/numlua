# Changelog

## Unreleased

### Added

- generated `.npy`/`.npz` fixture corpus covering 3-D/2-D shapes, all supported
  dtypes, empty and 0-D arrays, stored ZIP members, and big-endian rejection;
- `bench/bench.lua` and `scripts/bench.sh` for construction, reduction,
  broadcasting, layout, and `.npy` I/O comparisons against direct ATen/raw
  Lua baselines;
- `docs/performance.md` with the dated run, machine/compiler conditions,
  method, explicit losses, and caveats;
- a large layout regression test proving documented views do not silently copy
  and that non-contiguous `ravel` does copy.

### Tests

- 91 tests.

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
