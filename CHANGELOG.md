# Changelog

## Unreleased

### Added

- `numlua.broadcast_shapes(...)` exposes right-aligned shape arithmetic as a
  Lua-table helper without allocating a tensor.
- `numlua.diff` accepts optional `prepend`/`append` values, and
  `numlua.ediff1d` flattens, differences, and optionally adds begin/end values.
- `numlua.average(t[, axis[, weights[, returned]]])` adds weighted means with
  1-D or full-shape weights and an optional sum-of-weights result.
- `numlua.nanargmin` and `numlua.nanargmax` ignore NaNs along an axis and raise
  on all-NaN slices.
- `loadtxt`/`genfromtxt` now keep configured comment markers inside quoted
  fields; comment stripping is quote-aware when `quotechar` is set.

### Tests

- 437 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_broadcast_shapes.lua` covers right alignment, zero-size axes, and shape
  validation; `test_diff_edges.lua` covers prepend/append ranks, N-D axes,
  `ediff1d`, and errors; `test_average.lua` covers unweighted/weighted/returned
  forms and validation; `test_nan_arg.lua` covers NaN skipping, axes,
  all-NaN errors, and integer aliases; `test_quotechar.lua` adds quoted
  comment-marker coverage.

## 0.26.0 — 2026-11-24

### Added

- Elementwise ufuncs `isinf`, `isfinite`, `sign`, `floor`, `ceil`, `trunc`,
  `round`, `reciprocal`, `log1p`, `logical_and`, `logical_or`, `logical_xor`,
  and `logical_not`. Logical ufuncs accept booleans, numbers, or tensors and
  return `bool` tensors.
- NaN helpers `nan_to_num`, `nancumsum`, and `nancumprod`; cumulative variants
  treat NaN as the sum/product identity and preserve floating dtypes.
- `numlua.unwrap(t[, discont[, axis[, period]]])` removes phase jumps along a
  1-based axis with configurable period and discontinuity threshold.
- `numlua.interp(x, xp, fp[, left[, right]])` provides linear interpolation
  over a strictly increasing 1-D grid and preserves the shape of `x`.
- Boolean text I/O: `savetxt` writes `True`/`False` (or `1`/`0` through a
  custom `fmt`), while `loadtxt` and `genfromtxt` accept `dtype = "bool"`,
  boolean spellings, numeric truth values, converters, and missing/filling
  values.

### Changed

- Text dtype validation now accepts `bool` for all three text functions;
  missing bool fields default to false.
- `numlua.round` follows ATen's round-half-to-even behavior and is documented
  with the other rounding ufuncs.

### Tests

- 421 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_ufunc_extras.lua` covers predicates, rounding, reciprocal/log1p, and
  logical broadcasting; `test_nan_utils.lua` covers replacement values,
  cumulative NaN behavior, axes, and dtypes; `test_unwrap.lua` covers jumps,
  custom periods/discontinuities, axes, promotion, and errors;
  `test_interp.lua` covers scalar/vector/shape handling, out-of-range values,
  NaN coordinates, float64 output, and grid validation; `test_bool_text.lua`
  covers True/False and numeric spellings, custom formats, converters,
  `genfromtxt` fills, and invalid fields.

## 0.25.0 — 2026-11-24

### Added

- `numlua.dot` now follows NumPy's shape rules for vector, matrix, and batched
  products instead of requiring two 1-D operands.
- `numlua.inner` sums over the last axis of both operands, `numlua.outer`
  flattens both operands before forming the product, and `numlua.cross`
  computes 3-vector cross products over a 1-based axis with ATen batch
  broadcasting.
- `numlua.trapezoid` and its `numlua.trapz` alias integrate along one axis
  with uniform `dx` or coordinate arrays, matching `np.trapezoid`.
- `numlua.loadtxt` and `numlua.genfromtxt` accept a single-character
  `quotechar`; quoted fields may contain delimiters and whitespace, and a
  doubled quote character escapes itself.

### Changed

- `numlua.dot` now delegates to ATen matmul for all supported ranks, so mixed
  integer/floating operands follow ATen's matmul dtype checks.
- `numlua.cross` rejects operands with different ranks and supports only
  3-component vectors, matching the ATen kernel.

### Tests

- 397 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_dot.lua` covers vector/matrix/N-D shapes, dtypes, non-contiguous
  inputs, and errors; `test_inner_outer.lua` covers last-axis sums, empty axes,
  flattening, dtypes, and errors; `test_cross.lua` covers axes, batches,
  broadcasting, dtypes, and errors; `test_trapezoid.lua` covers uniform and
  coordinate spacing, axes, empty intervals, dtypes, and errors;
  `test_quotechar.lua` covers quoted CSV/whitespace fields, escaped quotes,
  embedded delimiters, `genfromtxt`, and malformed input.

## 0.24.0 — 2026-11-24

### Added

- `numlua.prod(t[, axis[, keepdims]])` delegates the product reduction to
  luatorch/ATen, including the empty-reduction identity of `1`, and
  `numlua.nanprod` treats NaN as the multiplicative identity so all-NaN slices
  return `1`.
- `numlua.ptp(t[, axis[, keepdims]])` returns `max - min` over the selected
  axis; bool input is cast to `int64` so the subtraction is defined.
- `numlua.logspace(start, stop[, num[, base[, opts]]])` and
  `numlua.geomspace(start, stop[, num[, opts]])` build logarithmically spaced
  sequences, with `endpoint`/`dtype` options and NumPy-compatible negative-sign
  handling for `geomspace`.
- `numlua.identity(n[, opts])`, `numlua.tri(N[, M[, k[, opts]]])`, and
  `numlua.vander(x[, N[, increasing]])` cover the remaining common NumPy
  constructors; `tri` and `vander` preserve dtype and materialize their result.
- `numlua.compress(condition, t[, axis])` now supports axis-aligned selection,
  numeric condition truth testing, and NumPy's short-condition prefix rule in
  both flattened and axis modes.
- `numlua.loadz` now reads ZIP64 archives, including ZIP64 central/local extra
  fields and the ZIP64 end-of-central-directory record and locator. Writer
  output remains ZIP32-only.

### Changed

- `numlua.compress` accepts Lua tables and numeric conditions and rejects a
  condition longer than the selected input/axis with a clear error. Axis
  conditions must be 1-D.

### Tests

- 373 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_product_ptp.lua` covers axis/keepdims, dtype, empty, and NaN cases;
  `test_logspace.lua` covers base/endpoint/dtype, signed `geomspace`, and
  validation; `test_vander_tri.lua` covers constructor overloads and errors;
  `test_compress_axis.lua` covers flat/axis selection, prefixes, empty
  conditions, and ownership; `test_zip64.lua` builds ZIP64 fixtures in Lua and
  covers extra fields, large members, and corrupt/missing records.

## 0.23.0 — 2026-11-24

### Added

- `numlua.var(t[, axis[, keepdims[, ddof]]])` and `numlua.std(...)` implement
  NumPy's population variance/standard deviation default (`ddof = 0`), accept
  any finite `ddof`, and preserve floating dtypes while promoting integer/bool
  input to `float64`.
- NaN-aware reductions `numlua.nansum`, `numlua.nanmean`, `numlua.nanmin`,
  `numlua.nanmax`, `numlua.nanvar`, and `numlua.nanstd` ignore NaNs along an
  optional 1-based axis, treat NaN as zero for `nansum`, and return NaN for
  slices with no valid values.
- `numlua.meshgrid(x1, x2, ...[, opts])` builds coordinate grids from 1-D
  tensors or Lua tables with NumPy's `indexing = "xy"`/`"ij"`, `sparse`, and
  `copy` options; each grid keeps its input dtype and materializes independent
  storage by default.
- `numlua.gradient(t[, spacing[, axis]])` computes NumPy-style first-order edge
  and central interior differences with uniform, coordinate, or per-axis
  spacing, returning one tensor per selected axis.
- `numlua.genfromtxt(path_or_file[, options])` extends the numeric text reader
  with `skip_header`/`skip_footer`, `missing_values`, `filling_values`, and
  `max_rows`, while keeping `usecols`, converters, `unpack`, `ndmin`, handle
  ownership, and the existing squeeze rules. Missing floats default to NaN and
  missing integers to zero; converters take precedence over missing markers.

### Changed

- Shared text-reader validation helpers now carry the calling function name, so
  `genfromtxt` errors consistently use a `numlua.genfromtxt:` prefix instead of
  inheriting `loadtxt` messages. Existing `loadtxt` behavior is unchanged.
- Release metadata, the README, API reference, and runnable example cover the
  new statistics, meshgrid, gradient, and missing-value text features.
- No runtime dependency was added; all new work delegates to luatorch or uses
  only the Lua standard library.

### Tests

- 346 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_statistics.lua` covers ddof, axis/keepdims, and dtype promotion;
  `test_nan_reductions.lua` covers all-NaN slices, axis reductions, and integer
  aliases; `test_meshgrid.lua` covers `xy`/`ij`, sparse/copy, three-input
  ordering, and errors; `test_gradient.lua` covers edges, uniform/coordinate
  spacing, axis tables, promotion, and errors; `test_genfromtxt.lua` covers
  headers/footers, missing/filling forms, converters, usecols, handles, empty
  input, and error paths.

## 0.22.0 — 2026-11-23

### Added

- `numlua.rot90(t[, k[, axes]])` rotates in the plane of two 1-based axes,
  reducing `k` modulo four; `numlua.fliplr` and `numlua.flipud` reverse the
  column and row axes. All three preserve dtype (`bool` included); `rot90`
  materializes its result except for `k = 0`, which returns the input.
- `numlua.diagonal(t[, offset[, axis1[, axis2]]])` extracts a diagonal over any
  two axes and appends it after the batch dimensions, and
  `numlua.trace(t[, offset[, axis1[, axis2]]])` sums that diagonal while
  keeping batch axes. Non-empty numeric diagonals are writable views; bool
  diagonals are materialized through `uint8`.
- `numlua.zeros_like`, `numlua.ones_like`, `numlua.empty_like`, and
  `numlua.full_like` inherit a prototype's shape and dtype and accept `dtype`
  and NumPy 2.x-style `shape` overrides, including 0-D results. `empty_like`
  leaves storage uninitialized; `full_like` requires a numeric fill value.
- `numlua.savez`, `numlua.savez_compressed`, and `numlua.loadz` accept an
  already-open Lua file handle as well as a path. Caller-owned handles are used
  at their current position and are never closed; `loadz` reads to EOF.
- `numlua.fromfile` and `numlua.tofile` support NumPy-style text mode through a
  non-empty `sep`. `fromfile` parses float `nan`/`inf` spellings, truncates
  numeric integer fields toward zero, and accepts `True`/`False`/`1`/`0` for
  bool; `tofile` writes values joined by `sep` with a trailing newline and
  accepts a printf-style `format`.

### Changed

- The release metadata, README, API reference, and runnable example cover the
  new rotation, diagonal, `*_like`, `.npz` handle, and raw text-mode behavior.
- `.npz` and raw text I/O now document their handle ownership and EOF rules;
  the previous `sep`-is-binary-only restriction is gone.
- No runtime dependency was added; all new work delegates to luatorch or uses
  only the Lua standard library.

### Tests

- 315 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_rotate.lua` covers quarter turns, negative/reduced `k`, explicit axes,
  dtypes, materialization, and rank errors; `test_diagonal.lua` covers offsets,
  axis pairs, batch and empty shapes, bool input, trace shapes, and errors;
  `test_like.lua` covers dtype/shape overrides, 0-D and non-contiguous
  prototypes, independence, and invalid options; `test_npz_handles.lua` covers
  stored/deflate handle round-trips, offsets, compressed fallback, and failure
  ownership; `test_raw_text.lua` covers every dtype, separators, formats,
  count/offset, parsing, handles, and error paths.

## 0.21.0 — 2026-11-23

### Added

- `numlua.split`, `numlua.array_split`, `numlua.hsplit`, `numlua.vsplit`, and
  `numlua.dsplit` split tensors along a 1-based axis. Integer sections require
  an even division for `split` and distribute the remainder for
  `array_split`; boundary tables are 1-based, accept negative and clamped
  positions, and return views. `hsplit`/`vsplit`/`dsplit` use NumPy's
  column/row/depth axes and rank requirements.
- `numlua.diag`, `numlua.diagflat`, `numlua.tril`, and `numlua.triu` cover
  NumPy's diagonal constructors/extractors and triangular masks. Offsets are
  positive above and negative below the main diagonal; N-D triangular helpers
  operate on the last two axes, and all four preserve dtype including `bool`.
- `numlua.isclose`, `numlua.allclose`, `numlua.array_equal`, and
  `numlua.array_equiv` implement NumPy's closeness and equality predicates,
  including `rtol`/`atol` defaults, broadcasting, infinity handling, and
  optional NaN equality. `allclose`, `array_equal`, and `array_equiv` return
  Lua booleans.
- `numlua.fromfile` and `numlua.tofile` provide NumPy-style raw binary I/O for
  every supported dtype. `fromfile` accepts `dtype` (default `"float64"`),
  `count`, and byte `offset`, reads paths or open handles, and `tofile` writes
  C-order little-endian payloads. Raw files carry no metadata, so reading
  requires the matching dtype.
- `numlua.loadtxt` and `numlua.savetxt` now accept an already-open Lua file
  handle as well as a path. Caller-owned handles are used at their current
  position and are never closed, including on write failure.

### Changed

- The release metadata, README, API reference, and runnable example cover the
  new split, diagonal, closeness, raw-binary, and text-handle behavior.
- `hsplit` rejects 0-D input explicitly, and `diag` handles empty bool vectors.
- No runtime dependency was added; all new work delegates to luatorch or uses
  only the Lua standard library and LuaJIT FFI for raw buffers.

### Tests

- 292 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_split.lua` covers even/uneven sections, 1-based/negative/clamped
  boundaries, empty axes, and rank errors; `test_diag.lua` covers vector
  construction, matrix extraction, offsets, N-D and bool triangular masks, and
  independence; `test_close.lua` covers tolerances, infinities, NaN,
  broadcasting, and equality shapes; `test_raw_io.lua` covers every dtype,
  count/offset windows, C-order writes, sequential handles, truncation, and
  option errors; `test_txt_handles.lua` covers handle round-trips, current
  positions, empty input, closed handles, and write-failure ownership.

## 0.20.0 — 2026-11-23

### Added

- `numlua.sort(t[, axis[, descending]])`, the value-returning counterpart to
  `argsort`. The last axis is the default, negative axes are accepted, and the
  optional `descending` flag matches `argsort`'s numlua extension.
- `numlua.unique(t[, options])`, matching `np.unique` without `axis`: the input
  is flattened in C order and sorted unique values are returned. The
  `return_index`, `return_inverse`, and `return_counts` boolean options append
  results in NumPy's order, using 1-based first-occurrence/inverse indices.
- `numlua.nonzero(t)` returns one 1-based C-order coordinate tensor per input
  axis, and `numlua.argwhere(t)` returns the same coordinates as an
  `(N, ndim)` `int64` matrix. Both reject 0-D input; `flatnonzero` remains the
  flat/0-D helper.
- `numlua.loadtxt(..., { converters = ... })` accepts a single function for all
  physical columns or a table of 1-based physical column numbers to functions.
  Converters receive the raw field string, run before the numeric parser, must
  return a Lua number, and are validated with line/column errors.
- `numlua.save` and `numlua.load` now accept an already-open Lua file handle as
  well as a path. Caller-owned handles are used at their current position and
  are never closed; `load` can read consecutive arrays from one handle.

### Changed

- The release metadata, README, API reference, and runnable example cover the
  new sorting, set, coordinate, converter, and file-handle behavior.
- No runtime dependency was added; all new work delegates to luatorch or uses
  only the Lua standard library.

### Tests

- 259 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_numpy_helpers.lua` covers sort axes/descending/independence, unique
  result ordering/0-D/empty/non-contiguous inputs, and nonzero/argwhere
  coordinate shapes and errors; `test_txt.lua` covers converter dispatch,
  physical `usecols` interaction, early stopping, and failures;
  `test_npy_handles.lua` covers handle round-trips, sequential arrays, path
  compatibility, closed handles, and write-failure ownership.

## 0.19.0 — 2026-11-22

### Added

- `numlua.loadtxt(..., { skiprows = n, max_rows = m })`, matching the
  corresponding NumPy options: `skiprows` counts physical lines including
  comments and blanks, `max_rows` counts data rows after the skip, and rows
  beyond the limit are not parsed.
- `numlua.loadtxt(..., { usecols = ..., unpack = ... })`. `usecols` accepts a
  single 1-based column or a table of columns, preserves the requested order
  and duplicates, accepts negative columns (`-1` is the last), and converts
  only the selected fields. `unpack = true` transposes a 2-D result after
  squeezing/`ndmin`, following NumPy.
- `numlua.loadtxt(..., { ndmin = 0|1|2 })`. `ndmin = 2` keeps the physical
  row/column shape, so one-row and one-column files become `(1, N)` and
  `(N, 1)` instead of 1-D vectors; empty input becomes `(0, 1)` and is
  transposed when `unpack` is also set.
- `numlua.savetxt(..., { header = ..., footer = ..., comments = ...,
  newline = ... })`, following `np.savetxt`. Header/footer lines are
  comment-prefixed (default `"# "`), multi-line values are prefixed per line,
  and `newline` applies to data and annotations. Existing output is unchanged
  when these options are omitted.
- `numlua.savetxt` accepts a `fmt` table with one format string or function per
  column, in addition to the existing single string/function form.

### Changed

- `numlua.loadtxt` now validates `skiprows`, `max_rows`, `usecols`, `unpack`,
  and `ndmin` with `numlua.loadtxt:` errors instead of forwarding Lua type or
  arithmetic failures.
- `numlua.savetxt` validates `header`, `footer`, `comments`, `newline`, and
  per-column `fmt` tables before opening the output file.

### Tests

- 237 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_txt.lua` covers physical-line skipping, data-row limits and early
  stop, 1-based/negative/duplicate `usecols`, unused malformed fields,
  `unpack` on 1-D/2-D/empty input, every `ndmin` mode, header/footer/comment
  framing, custom newlines, per-column format tables, empty/zero-column
  output, and the validation error paths.

## 0.18.0 — 2026-11-22

### Added

- `numlua.apply_over_axes(func, t, axes[, ...])`, matching
  `np.apply_over_axes` for Lua callbacks. `axes` is a 1-based axis or a table
  of axes (negative axes accepted, repeats allowed), applied in order to the
  current value; `func` receives the current tensor, the normalized 1-based
  axis, and any extra positional arguments. A callback result that keeps the
  input rank is used unchanged (so keepdims-style reductions and same-shape
  callbacks such as `cumsum` work), while a result that drops one dimension is
  re-expanded with a size-1 axis at the same position. Any other rank change
  raises a `numlua.apply_over_axes:` error, and 0-D input is rejected because
  it has no axis. Callback results may be tensors, Lua numbers, booleans, or
  Lua tables, following `apply_along_axis`; an empty axis table returns the
  input unchanged.

### Changed

- `numlua.argsort` validates its optional `descending` argument: it must be a
  boolean or `nil`, so a truthy number or string no longer silently selects
  ascending order.
- `numlua.loadtxt` and `numlua.savetxt` reject unknown option keys instead of
  ignoring them, matching `save`/`savez`.
- `numlua.loadtxt` accepts CR-only line endings in addition to LF and CRLF,
  matching Python's `splitlines`. Numeric fields never contain a carriage
  return, so normalizing before splitting cannot change values.

### Tests

- 225 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_apply_over_axes.lua` covers nested keepdims reductions on 1-D/2-D/3-D
  input, every axis and order including negative and repeated axes, same-rank
  `cumsum` and keepdims callbacks, single-number axes, empty axes, callback
  argument forwarding, scalar/boolean/Lua-table results, zero-size reductions,
  non-contiguous input, dtype preservation, and the validation and rank-error
  paths;
- selection/IO edge tests cover descending `argsort` values and flag
  validation, non-contiguous `argsort`/`take`/`take_along_axis`/`compress`/
  `flatnonzero` inputs, empty broadcast selections, CR-only and mixed line
  endings, unknown text option keys, multi-character save delimiters, int8/
  uint8 bounds, exact-range int64 and float16 text round-trips, trailing empty
  fields, and stored ZIP members that use data descriptors. `generate.lua`
  writes the descriptor fixture and documents that it is not produced through
  `numlua.savez`.

## 0.17.0 — 2026-11-22

### Added

- `numlua.apply_along_axis(func, axis, t[, ...])`, matching
  `np.apply_along_axis` for Lua callbacks. `func` receives a 1-D luatorch view
  of each slice along the 1-based `axis` (negative axes accepted) and may
  return a scalar or 1-D tensor, Lua number, boolean, or Lua table. Scalar
  results remove the axis; 1-D results append their length after the remaining
  dimensions. Extra positional arguments are forwarded, slices are visited in
  C order, and the output dtype follows the first callback result. The result
  owns independent storage. An empty selected axis is allowed; a zero-size
  non-axis dimension is rejected because no first slice exists to infer the
  result shape and dtype.

### Changed

- `numlua.argsort` rejects 0-D input with a `numlua.argsort:` error instead of
  forwarding luatorch's generic dimension message.

### Tests

- 213 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  `test_apply_along_axis.lua` covers scalar and 1-D callbacks on 1-D/2-D/3-D
  input, every axis including negative axes, C-order slice visitation, extra
  arguments, dtype propagation for tensors, numbers, booleans, and Lua tables,
  output ownership, empty axes, non-contiguous inputs, and the validation and
  error paths;
- selection/IO edge tests add `compress`/`flatnonzero` coverage, 0-D `argsort`,
  zero-size gather dimensions, non-tensor `take` indices, every numeric
  `loadtxt` dtype id, literal delimiter characters, comment-marker validation,
  `savetxt` special floats and one-row/one-column shapes, invalid format
  strings, and short or empty NPY prefixes.

## 0.16.0 — 2026-11-22

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

- 200 tests, passing with system zlib and with `NUMLUA_DISABLE_ZLIB=1`.
  Pad parity tests build linear ramps and axis statistics with an independent
  C-order reference walker and cover 1-D/2-D/3-D input, widths larger than the
  axis, per-axis end values, every `stat_length` form, bool and integer
  rounding, dtype/storage behavior, non-contiguous input, NumPy's corner
  order, and validation errors;
- IO/selection edge tests add disabled comment markers, files without a final
  newline, numeric `loadtxt` dtype ids, `savetxt` format functions and their
  error paths, double-quoted NPY header keys for v1.0 and v3.0, and
  negative-axis plus empty `take_along_axis`/`take` selections.

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
