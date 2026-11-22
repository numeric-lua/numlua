# Changelog

## Unreleased

### Added

- `numlua.pad` gains the NumPy modes `"linear_ramp"`, `"maximum"`, `"mean"`,
  `"median"`, and `"minimum"`. `"linear_ramp"` accepts `end_values` through
  the existing `value` argument; the statistic modes accept `stat_length` and
  fill each side from the first or last values along the axis. All three
  options take a number, a `{before, after}` pair, or one entry per axis.
  Axes are processed in order, so later axes see earlier padding exactly as in
  `np.pad`'s corner rule. Integer end values and statistics round half to
  even before the input dtype cast, and bool input works with every statistic
  mode. The four non-constant modes without a fill argument (`"edge"`,
  `"reflect"`, `"symmetric"`, `"wrap"`) now reject a non-nil `value` instead
  of silently ignoring it.

### Tests

- Pad parity tests build linear ramps and axis statistics with an independent
  C-order reference walker and cover 1-D/2-D/3-D input, widths larger than the
  axis, per-axis end values, every `stat_length` form, bool and integer
  rounding, dtype/storage behavior, non-contiguous input, NumPy's corner
  order, and validation errors.

## 0.15.0 — 2026-11-22

### Added

- `numlua.pad` mode `"wrap"`: each padded axis repeats cyclically, matching
  `np.pad(..., mode="wrap")`. Widths may exceed the axis length, per-axis
  before/after widths are independent, and zero-size axes can keep a
  zero-width wrap but cannot be extended by it;
- `numlua.pad` constant values follow NumPy's `constant_values` forms in
  addition to the historical scalar: a `{before, after}` pair broadcasts over
  every axis, and a table with one entry per axis gives per-axis values (each
  a number or pair). Scalar calls and the default zero fill are unchanged, and
  fills keep the input dtype.

### Changed

- `numlua.savez` and `numlua.savez_compressed` reject keys that normalize to
  the same ZIP member name (`x` and `x.npy`, for example) and the bare key
  `.npy`, before writing payloads or creating the archive. Previously the
  duplicate names silently overwrote each other on read;
- `numlua.pad` now validates numeric and one-pair `pad_width` forms as
  non-negative integers. Negative or fractional widths no longer pass through
  silently.

### Tests

- 193 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  Dedicated pad parity tests build wrapped expectations with an independent
  index walk and cover widths larger than the axis, 1-D/2-D/3-D and
  non-contiguous inputs, dtype/storage behavior, every constant-value form,
  and validation errors; selection/IO edge tests add empty and 0-D `select`
  operands, numeric truth testing, non-contiguous defaults, Unicode/spaced and
  colliding `.npz` member names, EOCD archive comments, invalid `.npy`
  members, multi-character text delimiters, and zero-width text tensors.

## 0.14.0 — 2026-11-22

### Added

- `.npz` member-version writing: `numlua.savez` and
  `numlua.savez_compressed` accept `{ version = 1|2|3 }` (or the dotted
  `"1.0"`/`"2.0"`/`"3.0"` strings) and write every member with that NPY header
  version. The default remains v1.0, so existing archives stay byte-identical.
  `savez_compressed` may combine `version` with `{ fallback = "stored" }`.
  Version spellings are validated by the same helper as `numlua.save`, unknown
  options are rejected, and errors are raised before member payloads or the
  output file are written;
- `numlua.select(condlist, choicelist[, default])`, matching `np.select`.
  Conditions are tested in order and the first true condition wins; missing
  matches take `default` (0 when omitted). Conditions and choices accept
  tensors or Lua scalars with the same truth testing and 0-D scalar behavior as
  `numlua.where`, and ATen performs broadcasting and dtype promotion.
  Empty/mismatched lists, invalid entries, and non-broadcastable operands raise
  a `numlua.select:` error.

### Tests

- 178 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`,
  covering byte-level `.npz` member versions for stored, compressed, and
  stored-fallback archives, scalar/empty/non-contiguous members, default
  byte-identity, and option/version validation; `select` first-true-wins,
  broadcasting, scalar and dtype behavior, non-contiguous operands, and error
  paths.

## 0.13.0 — 2026-11-21

### Added

- `numlua.save(path, tensor[, options])` can write NPY v2.0 or v3.0 with
  `{ version = 2 }` / `{ version = 3 }` (the dotted strings `"2.0"`/`"3.0"` are
  accepted too). The default remains v1.0, so existing output stays
  byte-identical. v2.0/v3.0 use the 4-byte header-length field and the same
  64-byte header alignment; unknown options and invalid versions are rejected
  before the output file is opened, and a forced v1.0 header above 65535 bytes
  still raises;
- `numlua.take` accepts 0-D and N-D integer index tensors. The selected axis is
  replaced by the index shape, a 0-D index removes the axis (a scalar result
  when `axis` is omitted), and the flattened-input form returns exactly the
  index shape, matching `np.take`'s result-shape rule. Out-of-range values now
  raise a `numlua.take:` error naming the offending 1-based index, the axis or
  flattened input, and its size, instead of forwarding ATen's 0-based message;
- byte-level NPY writer conformance tests covering all three header versions,
  the 64-byte alignment rule, header-length widths, every supported dtype, and
  0-D/empty/non-contiguous tensors.

### Changed

- `numlua.put` rejects NaN and infinite indices with a clear
  `numlua.put: indices must be finite integers` error in every mode, before any
  write is attempted.

### Tests

- 169 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`,
  covering the new writer versions, scalar/N-D `take` shapes, explicit
  selection range errors, finite-index `put` validation, and release metadata.

## 0.12.0 — 2026-11-21

### Added

- `numlua.take_along_axis(t, indices[, axis])`, the `np.take_along_axis`
  companion to `take`. Indices are 1-based integer tensors with the same rank
  as the input; non-axis size-1 dimensions broadcast in either operand; the
  selection length along the axis comes from the indices; and `axis=nil`
  flattens both operands. Out-of-range indices, including `0` and negative
  values, raise a `numlua.take_along_axis:` error, and the result owns
  independent storage;
- `numlua.load` accepts NPY v2.0 and v3.0 headers (the 4-byte header-length
  field, plus v3.0's UTF-8 header encoding) in addition to v1.0. `numlua.save`
  keeps writing v1.0, and `.npz` members may use any of the three versions;
- documentation for `take_along_axis`, the v1.0/v2.0/v3.0 reader support, and
  the view/copy boundary of per-slice selection.

### Tests

- tests for `take_along_axis` across 0-D, 1-D, 2-D, and 3-D inputs, index
  broadcasting, negative and omitted axes, every integer index dtype, empty
  selections, non-contiguous inputs, and its error paths;
- v2.0/v3.0 `.npy` round-trips, crafted v2.0 headers, truncated and unsupported
  version errors, and `.npz` archives whose members use v2.0/v3.0 headers;
- 156 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`,
  including release-metadata checks for the documented selection APIs and NPY
  version support.

## 0.11.0 — 2026-11-21

### Added

- `numlua.put(t, indices, values[, mode])`, a `np.put`-shaped flat assignment
  helper. It writes into the flattened tensor, mutates and returns `t`, accepts
  integer tensors or Lua tables for indices and values, reuses values
  cyclically, applies duplicate indices in order (last write wins), and
  supports `"raise"` (default), `"wrap"`, and `"clip"` modes under the facade's
  1-based index convention. Values are cast to the target dtype, including
  bool targets and truncation toward zero for integer targets;
- tests for `put` on 0-D, 3-D, non-contiguous, bool, and integer targets,
  including extreme wrap/clip indices and all integer index dtypes.

### Fixed

- `.npz` deflate reads now reject a declared uncompressed size beyond
  DEFLATE's 1032:1 expansion bound before allocating the output buffer, so a
  tiny corrupt archive cannot request a multi-gigabyte allocation. A wrong size
  that is still plausible reaches zlib and reports the real size mismatch, and
  a legitimate high-ratio member still round-trips;
- zlib-unavailable errors now name `NUMLUA_DISABLE_ZLIB` when that environment
  switch, rather than a missing library, disabled compression.

### Tests

- 147 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`,
  including the new `put` selection cases, the deflate expansion guard, and a
  release-metadata check that keeps `VERSION`, the rockspec, CHANGELOG,
  packaged modules, and the documented API in sync.

## 0.10.0 — 2026-11-21

### Added

- `numlua.load` accepts native-endian (`=`) `descr` values (`=f4`, `=i8`, ...)
  as aliases for the little-endian `<` spellings on little-endian hosts,
  matching NumPy's dtype parser; unsupported `=` dtypes are still rejected with
  the original spelling in the error;
- `.npz` edge coverage for empty archives and array keys already ending in
  `.npy`.

### Fixed

- ZIP writer limits are now enforced instead of silently wrapping fields:
  archives with more than 65535 members, member names above 65535 bytes,
  per-entry compressed or uncompressed sizes above 4294967295 bytes, or a
  central-directory offset above 4294967295 bytes raise an explicit
  `numlua.savez:`/`numlua.savez_compressed:` error before the output file is
  opened. ZIP64 writing remains unsupported.

### Tests

- 135 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`,
  including the new limit-boundary, no-partial-file, native-endian, empty
  archive, and pre-suffixed-key cases.

## 0.9.0 — 2026-11-21

### Added

- `numlua.savez_compressed(path, arrays[, options])`, writing deflate-compressed
  `.npz` members (ZIP method 8, raw DEFLATE) through the existing optional
  system zlib FFI, with `{ fallback = "stored" }` for zlib-less environments and
  a clear error otherwise;
- tests for compressed round-trip, method-8 headers, archive-size reduction,
  zlib-disabled error/fallback behavior, and option validation;
- broadcast/take edge-shape coverage: 0-D inputs, zero-size axes, high-rank
  right alignment, singleton axes, empty index tensors, and 0-D/empty `take`.

### Fixed

- `broadcast_arrays` now broadcasts a size-1 axis against a zero-size axis to a
  zero-size result, matching NumPy, instead of failing inside ATen `expand`;
- `take` rejects non-integer index tensors with a `numlua.take:` error instead
  of silently casting them, and reports selection with non-empty indices from
  an empty tensor or empty axis clearly.

### Tests

- 128 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.

## 0.8.0 — 2026-11-21

### Added

- `numlua.searchsorted(sorted, values[, side])` with left/right semantics;
- `numlua.bincount(input[, weights[, minlength]])` facade;
- tests for search positions and weighted counts.

## 0.7.0 — 2026-11-21

### Added

- vector `q` tables for `numlua.quantile` and `numlua.percentile`;
- tests for flat and axis-aligned vector quantiles;
- 119 tests.

## 0.6.0 — 2026-11-21

### Added

- `numlua.flatnonzero(t)` returning 1-based flat indices;
- take coverage for flattened and axis-aligned selection;
- 118 tests.

## 0.5.0 — 2026-11-21

### Added

- `numlua.percentile` as a percent-scaled quantile alias;
- `numlua.compress(condition, tensor)` boolean-mask selection;
- tests for percentile and compress.

## 0.4.0 — 2026-11-21

### Added

- `numlua.median(t[, axis[, keepdims]])`;
- `numlua.quantile(t, q[, axis[, keepdims]])` with linear interpolation;
- tests for flat/axis medians and quantiles.

### Fixed

- `numlua.repeat` with a repeats table and no axis now flattens the input first,
  matching NumPy.

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
