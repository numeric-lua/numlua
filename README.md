# numlua

NumPy-shaped arrays for LuaJIT, implemented as a **thin facade over
`luatorch` tensors**.

`numlua` gives LuaJIT the parts of NumPy that are useful around a tensor
engine: array construction, shape manipulation, elementwise ufuncs, reductions,
predicates and selection, small linear algebra operations, dtype conversion,
NumPy `.npy` and raw binary file I/O, and simple whitespace/CSV text I/O. It does not implement
a tensor engine of its own.
Array-producing functions return `luatorch` tensors, and luatorch/ATen performs
the actual computation.

> **Status: 0.26.0 MVP.** The public API is small but usable: construction,
> shape/layout helpers, elementwise ufuncs, reductions, predicates, linalg,
> `.npy`/`.npz`/raw binary and text, and CSV I/O.  It is packaged on LuaRocks
> and tested against luatorch 0.34.x.

## Why a facade?

The numeric-lua stack has one tensor/autograd engine: luatorch. `numlua` is a
convenience and interoperability layer, not a second implementation of tensors,
dtypes, broadcasting, or autograd. That keeps memory management, kernels, and
correctness in one place.

Use `numlua` when you want NumPy-shaped names:

```lua
local numlua = require("numlua")

local a = numlua.arange(1, 7):reshape(2, 3)
print(a:format())
-- tensor([[1, 2, 3],
--         [4, 5, 6]])

print(numlua.sum(a, 1):format())   -- column sums
print(numlua.mean(a):item())       -- one number
print(numlua.argmax(a):item())     -- 1-based flat index
print(numlua.argmin(a):item())     -- 1-based flat index of the minimum

local values = numlua.array({ 1, 4, 9 })
print(numlua.sqrt(values):format())                -- {1, 2, 3}
print(numlua.add(numlua.multiply(values, 2), 1):format()) -- {3, 9, 19}
```

Use the returned tensors directly with the luatorch API for anything not
covered here:

```lua
local torch = require("torch")
local b = torch.ones(3, 2)
print(numlua.matmul(numlua.astype(a, "float32"), b):format())
```

Small linear algebra is available through the same facade:

```lua
local la = numlua.linalg
print(la.det(numlua.eye(2)):item())         -- 1
local q, r = la.qr(numlua.array({ { 1, 2 }, { 3, 4 } }))
print(la.norm(q:matmul(r)):item())
```

Predicates, clipping, and conditional selection are available too:

```lua
local values = numlua.array({ -2, -0.5, 0, 0.5, 2 })
print(numlua.clip(values, -1, 1):format())
print(numlua.where(values:gt(0), values, 0):format())
print(numlua.select({ values:lt(0), values:gt(0) }, { -1, 1 }, 0):format())
print(numlua.count_nonzero(values):item(), numlua.any(values:gt(0)):item())
```

Closeness predicates follow NumPy's defaults and semantics for infinities and
NaN; `array_equal` requires matching shapes while `array_equiv` broadcasts:

```lua
print(numlua.isclose(values, values + 1e-7):all():item())  -- true
print(numlua.allclose(values, values, 0, 0))               -- true
print(numlua.array_equal(numlua.array({ 1 }), numlua.array({ 1, 1 })))
-- false
print(numlua.array_equiv(1, numlua.array({ 1, 1 })))       -- true
```

Flat selection writes use the same 1-based indices; `numlua.put` mutates and
returns the input tensor:

```lua
local edited = numlua.arange(6)
numlua.put(edited, { 1, 6 }, { 10, 60 })  -- {10, 1, 2, 3, 4, 60}
numlua.put(edited, { 9 }, -1, "wrap")     -- wraps index 9 to index 3
print(edited:format())
```

Array splitting follows `np.split`: `numlua.split` requires an even integer
division, `numlua.array_split` distributes the remainder, and boundary tables
use 1-based positions. `numlua.hsplit`, `vsplit`, and `dsplit` are the
column/row/depth conveniences:

```lua
local rows = numlua.split(numlua.arange(6):reshape(3, 2), 3, 1)
print(rows[1]:format(), rows[2]:format(), rows[3]:format())
local left, right = table.unpack(numlua.hsplit(rows[1]:reshape(1, 2), 2))
```

`numlua.take` uses the same 1-based indices and accepts scalar (0-D) or N-D
integer index tensors. The index shape replaces the selected axis; a 0-D index
removes it, matching `np.take`:

```lua
local grid = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
print(numlua.take(grid, numlua.array(2, { dtype = "int64" })):item())
-- 20 (flat scalar index)
print(numlua.take(grid,
  numlua.array({ { 3, 1 }, { 1, 2 } }, { dtype = "int64" }), 2):format())
-- (2, 2, 2) tensor selected along the last axis
```

Padding follows `np.pad` for constant, edge, linear ramp, maximum, mean,
median, minimum, reflect, symmetric, and periodic wrap modes. The optional fill
argument is mode-specific: `constant_values` for constant, `end_values` for
linear ramp, and `stat_length` for the statistic modes. Each accepts a number,
a before/after pair, or one entry per axis:

```lua
local v = numlua.array({ 1, 2, 3 })
print(numlua.pad(v, { 1, 2 }, "constant", { 9, 8 }):format())
-- tensor([9., 1., 2., 3., 8., 8.])
print(numlua.pad(v, 2, "wrap"):format())
-- tensor([2., 3., 1., 2., 3., 1., 2.])
print(numlua.pad(numlua.array({ 1, 2, 3, 4, 5 }), { 2, 3 },
  "linear_ramp", { 5, -4 }):format())
-- tensor([5., 3., 1., 2., 3., 4., 5., 2., -1., -4.])
```

Per-slice selection follows `np.take_along_axis`; combining it with `argsort`
sorts along one axis without a separate sort function:

```lua
local matrix = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
print(numlua.take_along_axis(matrix, numlua.argsort(matrix, 2), 2):format())
-- tensor([[10, 20, 30],
--         [40, 50, 60]])
```

`numlua.apply_along_axis` runs a Lua callback over 1-D slices along one axis,
following `np.apply_along_axis`. A scalar callback removes the axis; a 1-D
callback result appends its length after the remaining dimensions, and extra
arguments are forwarded:

```lua
local rows = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
print(numlua.apply_along_axis(function(row)
  return numlua.flip(row, 1)
end, 2, rows):format())
-- tensor([[3., 2., 1.],
--         [6., 5., 4.]])
print(numlua.apply_along_axis(function(row)
  return numlua.sum(row)
end, 1, rows):format())
-- tensor([5., 7., 9.])
```

`numlua.apply_over_axes` applies a reduction callback over a list of axes and
restores each reduced axis as size 1, following `np.apply_over_axes` (with
1-based axes). A callback that keeps the rank is used unchanged, so
same-shape transforms such as `cumsum` also work:

```lua
local cube = numlua.arange(24):reshape(2, 3, 4)
print(numlua.apply_over_axes(numlua.sum, cube, { 1, 2 }):format())
-- tensor([[[60, 66, 72, 78]]])   (shape 1, 1, 4)
print(numlua.apply_over_axes(numlua.sum, cube, { -1 }):format())
-- tensor([[[6],
--           [22],
--           [38]],
--         [[54],
--           [70],
--           [86]]])              (shape 2, 3, 1)
-- numlua.apply_over_axes(numlua.cumsum, cube, { 3 }) keeps shape 2, 3, 4
```

Extra callback arguments are forwarded, and a callback result must either keep
the input rank or drop exactly one axis. Scalar, boolean, and Lua-table results
are materialized like `apply_along_axis` results.

Sorting and coordinate helpers follow the same 1-based convention:

```lua
local a = numlua.array({ { 3, 1, 2 }, { 3, 2, 1 } })
print(numlua.sort(a):format())              -- rows sorted ascending
local values, counts = numlua.unique(a, { return_counts = true })
print(values:format(), counts:format())     -- {1, 2, 3}, {2, 2, 2}
local rows, columns = numlua.nonzero(a:eq(1))
print(rows:format(), columns:format())      -- {1, 2}, {2, 3}
print(numlua.argwhere(a:eq(1)):format())    -- {{1, 2}, {2, 3}}
```

## Requirements

- LuaJIT 2.1 (the primary and tested target).
- A luatorch checkout with its native shim built
  (`build/libluatorch.so`). `numlua` loads luatorch through `require("torch")`.
- No external Lua rocks. The library and test harness use only the Lua standard
  library, LuaJIT FFI, and luatorch.

In the organization workspace this repository sits next to `luatorch`, so
`../luatorch/lua` is the sibling checkout. `scripts/test.sh` searches
`../luatorch`, the old staging layout, and the main checkout's sibling
`luatorch` when run from a linked worktree automatically. Set `LUATORCH_ROOT`
if your checkout is elsewhere. Set `LUATORCH_LIB` if luatorch cannot find its
shared library.

## Install and `LUA_PATH`

Install from LuaRocks (the native luatorch shim still has to be built
separately with its own `scripts/build.sh`):

```sh
luarocks --lua-version=5.1 install numlua
```

Run from a checkout with luatorch on `LUA_PATH`:

```sh
LUA_PATH="../luatorch/lua/?.lua;../luatorch/lua/?/init.lua;./?.lua;./?/init.lua;;" \
  luajit tests/run.lua
```

For your own scripts, keep the same two luatorch entries and add this
repository's entries:

```sh
export LUA_PATH="../luatorch/lua/?.lua;../luatorch/lua/?/init.lua;./?.lua;./?/init.lua;;"
luajit examples/basic.lua
```

`require("numlua")` works from the repository root because the module lives at
`numlua/init.lua`. The LuaRocks package installs the same tree and declares
`luatorch >= 0.8` as a dependency.

## Conventions (read this before porting NumPy code)

- **Axes and indices are 1-based.** `axis=1` is NumPy's `axis=0`, and the first
  `argmax`/`argmin` result is `1`, not `0`. This is the numeric-lua convention.
- **Functions return luatorch tensors.** Reductions never return bare Lua
  numbers. All-axis reductions return a 0-dimensional tensor; call `:item()`
  when you want a number.
- **`max`/`min` return values only.** Luatorch's dimensional `max` also returns
  indices; `numlua` discards them. Use `numlua.argmax`/`numlua.argmin` for
  indices.
- **Default dtypes follow luatorch.** `array({1, 2, 3})` is `float32` because
  Lua numbers do not distinguish integers. Pass `{dtype = "int64"}` for
  integers. `zeros`, `ones`, `eye`, and `linspace` default to `float32`.
  `arange` infers `int64` for integer bounds and `float32` otherwise.
  `mean` promotes integer/bool input to `float64` (ATen would otherwise reject
  it); `var`/`std` do the same and default to `ddof = 0`, as in NumPy. Floating
  input keeps its dtype.
- **Elementwise dtype promotion is ATen's.** The ufuncs accept tensor or Lua
  number operands and delegate shape broadcasting and dtype promotion to
  luatorch/ATen. This is not NumPy's promotion table: for example, `int64` plus
  `float32` produces `float32`, and integer/integer division produces
  `float32`. See [docs/api.md](docs/api.md#dtype-behavior).
- **Broadcasting is luatorch's.** `matmul` and the elementwise operations use
  ATen broadcasting. `numlua` does not add another broadcast layer.
- **`.npy` bytes are NumPy-compatible.** The Lua-facing axes stay 1-based; the
  file format does not change.

See [docs/api.md](docs/api.md) for the full surface and limits.

## API at a glance

| Group | Functions |
| --- | --- |
| Constructors | `array`, `asarray`, `zeros`, `ones`, `full`, `eye`, `identity`, `tri`, `vander`, `arange`, `linspace`, `logspace`, `geomspace`, `meshgrid`, `zeros_like`, `ones_like`, `empty_like`, `full_like` |
| Shape | `shape`, `reshape`, `transpose`, `swapaxes`, `moveaxis`, `ravel`, `squeeze`, `unsqueeze`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `resize`, `diag`, `diagflat`, `diagonal`, `trace`, `tril`, `triu` |
| Combine | `concatenate`, `stack`, `split`, `array_split`, `hsplit`, `vsplit`, `dsplit` |
| Layout | `broadcast_to`, `broadcast_arrays`, `broadcast_shapes`, `tile`, `repeat_`, `flip`, `fliplr`, `flipud`, `rot90`, `roll`, `pad`, `diff`, `ediff1d`, `gradient`, `unwrap`, `cumsum`, `cumprod`, `argsort`, `sort`, `unique`, `nonzero`, `argwhere`, `take`, `take_along_axis`, `put`, `ravel_multi_index`, `unravel_index` |
| Iteration | `apply_along_axis`, `apply_over_axes` |
| Elementwise | `add`, `subtract`, `multiply`, `divide`, `negative`, `abs`, `sqrt`, `exp`, `log`, `log1p`, `sin`, `cos`, `tanh`, `maximum`, `minimum`, `power`, `square`, `sign`, `floor`, `ceil`, `trunc`, `round`, `reciprocal`, `isinf`, `isfinite`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `isposinf`, `isneginf`, `hypot`, `remainder`, `mod` |
| Reductions | `sum`, `prod`, `mean`, `average`, `var`, `std`, `max`, `min`, `ptp`, `argmax`, `argmin`, `nanargmin`, `nanargmax`, `nansum`, `nanprod`, `nanmean`, `nanmin`, `nanmax`, `nanvar`, `nanstd`, `nancumsum`, `nancumprod`, `nan_to_num`, `cov`, `corrcoef` |
| Numerical | `trapezoid`, `trapz`, `interp`, `polyval`, `polyfit`, `histogram`, `histogram_bin_edges` |
| Predicates/selection | `where`, `select`, `clip`, `compress`, `isnan`, `isclose`, `allclose`, `array_equal`, `array_equiv`, `any`, `all`, `count_nonzero` |
| Linear algebra | `dot`, `inner`, `outer`, `cross`, `matmul`, `linalg.pinv`, `linalg.lstsq`, `linalg.matrix_norm` |
| Dtype | `dtype`, `astype`, `promote_types`, `result_type`, `can_cast` |
| NumPy I/O | `save`, `load`, `fromfile`, `tofile`, `savez`, `savez_compressed`, `loadz` |
| Text/CSV I/O | `loadtxt`, `savetxt`, `genfromtxt` |

`numlua.zeros_like`, `numlua.ones_like`, `numlua.empty_like`, and
`numlua.full_like` inherit the prototype's shape and dtype by default and
accept `{ dtype = ..., shape = ... }` overrides. `empty_like` leaves storage uninitialized; `full_like` takes a
numeric fill value.

`numlua.meshgrid(x1, x2, ...[, opts])` returns a table of coordinate grids and
accepts NumPy's `indexing = "xy"`/`"ij"`, `sparse`, and `copy` options. It
materializes independent storage by default, and each grid keeps its input
dtype.

The constructor family also includes `numlua.identity(n)`, `numlua.tri(N[, M[,
k]])`, and `numlua.vander(x[, N[, increasing]])`, plus the logarithmic
spacings `numlua.logspace(start, stop[, num[, base[, opts]]])` and
`numlua.geomspace(start, stop[, num[, opts]])`. `logspace` defaults to base 10
and `geomspace` keeps a common negative sign, as in NumPy; both accept
`{ endpoint = ..., dtype = ... }` as their trailing options table.

`numlua.var` and `numlua.std` default to population variance (`ddof = 0`), as
NumPy does, and accept any finite `ddof`. Their `nanvar`/`nanstd` counterparts
and the other `nan*` reductions ignore NaNs, returning NaN for slices with no
valid values; `nansum` treats NaN as zero and `numlua.nanprod` treats NaN as
one. `numlua.prod` delegates the product reduction to ATen, and `numlua.ptp`
returns `max - min` over the same axis (bool input becomes `int64`).
`numlua.average` computes an optional weighted mean along one axis and can
return the sum of weights; `numlua.nanargmin`/`numlua.nanargmax` ignore NaNs
and reject all-NaN slices. `numlua.cov` computes sample covariance matrices and
`numlua.corrcoef` normalizes them to correlations.
`numlua.trapezoid` (with `numlua.trapz` as its historical alias) integrates
along one axis with uniform `dx` or coordinate arrays; the default axis is the
last one. `numlua.interp` performs linear interpolation over a strictly
increasing 1-D grid and keeps the shape of its coordinate input. `numlua.unwrap`
removes phase jumps along a 1-based axis with a configurable period and
discontinuity threshold. `numlua.polyval` evaluates descending-degree
coefficients and `numlua.polyfit` performs an SVD least-squares fit.
`numlua.histogram`/`numlua.histogram_bin_edges` provide counts, explicit or
integer bin edges, weights, and density normalization.

The elementwise surface also includes `numlua.isinf`, `numlua.isfinite`,
`numlua.sign`, `numlua.floor`, `numlua.ceil`, `numlua.trunc`, `numlua.round`,
`numlua.reciprocal`, `numlua.log1p`, and the logical ufuncs
`numlua.logical_and`, `numlua.logical_or`, `numlua.logical_xor`, and
`numlua.logical_not`. The newer additions are `numlua.isposinf`,
`numlua.isneginf`, `numlua.hypot`, `numlua.remainder`, and its `numlua.mod`
alias.
`numlua.nan_to_num` replaces NaN/infinity values, while `numlua.nancumsum` and
`numlua.nancumprod` skip NaNs in cumulative reductions.

`numlua.dot` now follows NumPy's shape rules for vectors, matrices, and batched
inputs. `numlua.inner` sums over the last axis of both operands, `numlua.outer`
flattens both operands before forming the product, and `numlua.cross` computes
3-vector cross products over a 1-based axis (ATen batch broadcasting).

`transpose`, `swapaxes`, `moveaxis`, `atleast_*`, `broadcast_*`, `squeeze`,
`unsqueeze`, and contiguous `reshape`/`ravel` are views. `resize`, `diag`,
`diagflat`, `tril`, `triu`, `tri`, `vander`, `compress`, `pad`, `tile`,
`repeat_`, `flip`, `fliplr`,
`flipud`, `rot90`, `roll`, `diff`, `ediff1d`, `gradient`, `unwrap`,
`interp`, `cumsum`, `cumprod`, `argsort`,
`ravel_multi_index`, `unravel_index`,
`sort`, `unique`, `nonzero`, `argwhere`, `take`, and `take_along_axis`
allocate independent storage (`rot90(t, 0)` returns the input); see
[docs/api.md](docs/api.md#view-and-copy-boundaries) for the full boundary.
`numlua.rot90` rotates in a 1-based axis plane (with `fliplr`/`flipud` as the
axis-specific flips); `numlua.diagonal` returns a writable view for non-empty
numeric input, while `numlua.trace` sums that diagonal and returns a 0-D tensor
for 2-D input or the batch shape for higher-rank input.
`numlua.gradient` follows NumPy's first-order edge differences and central
interior differences. It accepts uniform spacing, coordinate arrays, and
per-axis spacing, and returns one tensor per requested axis (a single tensor
for a numeric `axis` or for a 1-D input with no `axis`).
`numlua.diff` accepts optional `prepend`/`append` values, and
`numlua.ediff1d` flattens its input before differencing and adding optional
begin/end values. `numlua.broadcast_shapes(...)` exposes the same right-aligned
shape arithmetic as `broadcast_arrays` without allocating tensors.

`put` is the exception: it writes into its input tensor in place and returns
it. `pad` supports `constant`, `edge`, `linear_ramp`, `maximum`, `mean`,
`median`, `minimum`, `reflect`, `symmetric`, and `wrap` modes. Its fill value
may be a number, a `{before, after}` pair broadcast over every axis, or a table
with one entry per axis (each a number or pair), matching NumPy's
`constant_values`, `end_values`, and `stat_length` forms.

`numlua.compress(condition, t[, axis])` now supports axis-aligned selection in
addition to flattened selection. A short condition is used as a prefix and a
longer one is an error, matching `np.compress`; numeric conditions are
truth-tested, and axis conditions must be 1-D.

`numlua` arrays are ordinary luatorch tensors, so luaplotlib can plot them
directly through its tensor adapter. See `examples/plot.lua`.

## `.npz` I/O

`numlua.savez(path_or_file, { arrays = tensor, ... })` writes a stored
(uncompressed) ZIP archive of `.npy` members; `numlua.loadz(path_or_file)`
returns a table keyed by array name. This matches the common `np.savez` output
and needs no compression library. `numlua.savez_compressed` writes
deflate-compressed members through the optional system zlib, matching
`np.savez_compressed`; without zlib it raises a clear error unless
`{ fallback = "stored" }` is passed, in which case it writes stored entries and
returns `true, "stored"`.

Both writers and the reader accept an open Lua file handle as well as a path.
Caller-owned handles are used at their current position and are never closed;
`loadz` reads to EOF, so seek to the start of a later archive before reading
it.

Both writers accept an options table with `version = 1`, `2`, or `3` (or
`"1.0"`, `"2.0"`, `"3.0"`), selecting the NPY header version for every member.
The default is v1.0, so output without the option stays byte-identical;
`savez_compressed` may combine `version` with `fallback`. Unknown options and
invalid versions are rejected before any file is created. `numlua.loadz`
already reads members written in all three versions.

`numlua.loadz` accepts these ZIP compression methods:

| method | written by | availability |
| --- | --- | --- |
| stored (0) | `np.savez`, `numlua.savez` | always, dependency-free |
| deflate (8) | `np.savez_compressed`, `numlua.savez_compressed` | read/write; requires a system zlib library |

Deflate support is optional. When zlib is present, it is loaded through LuaJIT
FFI (`libz`/`libz.so.1` on Linux, `libz.dylib` on macOS); there is no mandatory
LuaRocks or compression dependency. `savez_compressed` emits raw DEFLATE
streams, so its archives are ordinary ZIP files readable by NumPy and other
tools. If zlib is absent, `savez_compressed` fails before creating a file (or
falls back to stored when requested), and a deflate entry encountered by
`loadz` fails with an error naming the entry and explaining that zlib is
required. Encrypted, multi-disk, and other compression methods are rejected
explicitly. ZIP64 archives are read, including ZIP64 extra fields and the
ZIP64 end-of-central-directory record; writing remains ZIP32-only. A declared
deflate output size beyond DEFLATE's 1032:1
expansion bound is rejected before allocating a buffer, so a tiny corrupt
archive cannot request a multi-gigabyte allocation. Stored archives continue to
work without zlib.
Set `NUMLUA_DISABLE_ZLIB=1` to exercise the no-zlib path in tests; the
zlib-unavailable error names that variable when it is the cause.

Writing is limited to ZIP32 fields: at most 65535 members, member names of at
most 65535 bytes, 32-bit member sizes, and a 32-bit central-directory offset.
Archives that would overflow any of these limits raise a `numlua.savez:` or
`numlua.savez_compressed:` error before the output file is created, instead of
silently truncating ZIP fields. ZIP64 writing is not supported; split very
large arrays across multiple archives. An empty `arrays` table produces a valid
empty archive, and keys already ending in `.npy` are stored without a second
suffix (as in NumPy). Keys that normalize to the same member name (`x` and
`x.npy`, for example) are rejected before any file is created instead of
silently overwriting a member. Unicode and spaced names are preserved. Members
may use any NPY header version the reader supports (v1.0, v2.0, or v3.0),
including archives produced by NumPy.

## `.npy` I/O

`numlua.save(path_or_file, tensor)` writes a NumPy v1.0 `.npy` file in C order
and `numlua.load(path_or_file)` returns a luatorch tensor. Both accept a path
string or an already-open Lua file handle; handles are read/written at their
current position and are never closed by `numlua`. `load` accepts NPY v1.0,
v2.0, and v3.0 headers. Pass `{ version = 2 }` or `{ version = 3 }` (or `"2.0"`/`"3.0"`)
as the third argument to write v2.0/v3.0; the default remains v1.0, so existing
output is byte-identical.

Supported dtypes:

| numlua / luatorch | `.npy` descr | bytes per element |
| --- | --- | --- |
| `float16` | `<f2` | 2 |
| `float32` | `<f4` | 4 |
| `float64` | `<f8` | 8 |
| `int8` | `\|i1` | 1 |
| `int16` | `<i2` | 2 |
| `int32` | `<i4` | 4 |
| `int64` | `<i8` | 8 |
| `uint8` | `\|u1` | 1 |
| `bool` | `\|b1` | 1 |

The reader supports 1-D, 2-D, and N-D arrays (and 0-D scalars) and rejects
big-endian files, `fortran_order=True`, unsupported `descr` values, NPY versions
other than 1.0, 2.0, and 3.0, and truncated data with clear Lua errors.
Native-endian `=` descr values (`=f4`, `=i8`, ...) are accepted as aliases for
the `<` spellings
on little-endian hosts, matching NumPy's dtype parser. Non-contiguous tensors
are written in C order. Files are always written little-endian; only
little-endian hosts are supported. luatorch cannot construct `float16` tensors
directly from Lua tables, so create a tensor in another dtype and cast it with
`:to("float16")` before saving.

Example:

```lua
local numlua = require("numlua")

local a = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } })
numlua.save("arrays.npy", a)
local b = numlua.load("arrays.npy")
assert(b:equal(a))

-- Opt in to the wider v2.0/v3.0 header format when the reader supports it.
numlua.save("arrays-v3.npy", a, { version = 3 })
assert(numlua.load("arrays-v3.npy"):equal(a))

-- File handles avoid reopening the path; seek before reading a write handle.
local handle = assert(io.open("arrays-handle.npy", "w+b"))
numlua.save(handle, a)
handle:seek("set", 0)
assert(numlua.load(handle):equal(a))
handle:close()
```

Raw arrays use `numlua.tofile(path_or_file, tensor)` and
`numlua.fromfile(path_or_file, options)`. These mirror NumPy's raw
`tofile`/`fromfile` pair: there is no header, so the reader must be told the
dtype. Both accept paths or open Lua handles and leave caller-owned handles
open. With the default `sep = ""` they write/read C-order little-endian bytes;
`fromfile` supports `dtype` (default `"float64"`), `count` (`-1` for all), and a
byte `offset`.

```lua
numlua.tofile("values.bin", numlua.array({ 1, 2, 3 }, { dtype = "int32" }))
local values = numlua.fromfile("values.bin", { dtype = "int32" })
-- {1, 2, 3}
```

A non-empty `sep` selects NumPy-style numeric text mode: `tofile` writes each
C-order value separated by `sep` and ends with a newline, while `fromfile`
parses fields from the remaining text. `tofile` also accepts a printf-style
`format`; `fromfile` accepts float `nan`/`inf` spellings, integer values
truncated toward zero, and `True`/`False` or `1`/`0` for bool. Text mode has no
headers, comments, quoting, or column selection; use `loadtxt`/`savetxt` for
files that need those. It reads a caller-owned handle to EOF even when `count`
is given.

```lua
local ints = numlua.array({ 1, 2, 3 }, { dtype = "int32" })
numlua.tofile("values.txt", ints, { sep = "," })
numlua.fromfile("values.txt", { dtype = "int32", sep = "," }) -- {1, 2, 3}
```

Use `.npy` instead when the file should carry its dtype and shape.

## Text/CSV I/O

`numlua.savetxt(path_or_file, tensor[, options])` writes a 1-D or 2-D tensor as
numeric/bool text. `numlua.loadtxt(path_or_file[, options])` reads it back as a
luatorch tensor. Both accept a path or an already-open Lua file handle; handles
are used at their current position and are never closed by `numlua`. Plain
whitespace is the default separator, and `delimiter = ","` handles simple CSV.
Comment markers default to `"#"` and can be changed, supplied as a table, or
disabled.

```lua
local numlua = require("numlua")

local a = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } })
numlua.savetxt("values.csv", a, { delimiter = "," })
local b = numlua.loadtxt("values.csv", {
  delimiter = ",",
  comments = "#",
  dtype = "float32",
})
assert(b:equal(a))
```

`loadtxt` ignores blank lines and comments, requires every data row to have the
same number of fields, and reports malformed rows or invalid numbers with the
line number. A `converters` function (or a table of 1-based physical column
numbers to functions) can transform raw fields before the numeric parser, which
handles sentinels such as `"NA"`. Pass a single-character `quotechar` to treat
quoted fields as atomic: delimiters, whitespace, and comment markers inside the
quotes are kept, and a doubled quote character escapes itself. `genfromtxt`
accepts the same option. `skiprows` skips physical lines (including comments and
blanks),
while `max_rows` limits the number of data rows read after that skip; `max_rows`
does not count blank or comment lines. `usecols` selects 1-based columns (with
`-1` for the last column and a table preserving order), `unpack = true`
transposes a 2-D result, and `ndmin = 2` keeps one-row/one-column input as a
2-D `(1, N)`/`(N, 1)` tensor instead of squeezing it. One-row and one-column
files become 1-D
tensors (NumPy's default squeeze behavior), and an all-comment/empty file loads
as an empty 1-D tensor. The default load dtype is `float32`; pass `dtype` to load
integers, wider floats, or `bool` exactly. Boolean fields accept `True`/`False`
or numeric values (zero is false), and a bool `savetxt` writes `True`/`False`
by default. Text files do not store shape or dtype, so a
1xN or Nx1 matrix cannot be distinguished from a vector after loading.

`numlua.genfromtxt(path_or_file[, options])` extends the same numeric parser
with missing-value handling. `missing_values` accepts a marker string, a marker
list, or a table keyed by 1-based physical column; `filling_values` accepts one
number, a per-column list, or a per-column map. Missing floats default to NaN,
missing integers to zero, and missing bools to false; converters run before
missing-value handling, and
`skip_header`/`skip_footer` replace `loadtxt`'s `skiprows`. Numeric-only scope,
`usecols`, `unpack`, `ndmin`, and the squeeze rules are unchanged.

`savetxt` writes sufficient digits for `float16`/`float32`/`float64`
round-trips, writes `nan`/`inf` explicitly, and supports a `fmt` format string,
function, or table with one format per column. `header` and `footer` add
comment-prefixed text before and after
the data (`comments` defaults to `"# "`), and `newline` changes the line
ending. It rejects zero-dimensional and higher-than-2-D tensors. Bool tensors
write `True`/`False` by default; a custom `fmt` receives `1`/`0` for bool
values. See
[docs/api.md](docs/api.md#text-and-csv-io) for the full option list.

## Testing

From the repository root:

```sh
LUA_PATH="../luatorch/lua/?.lua;../luatorch/lua/?/init.lua;./?.lua;./?/init.lua;;" \
  luajit tests/run.lua
```

or:

```sh
./scripts/test.sh
make test
```

The suite is dependency-free and covers constructors, shape operations,
elementwise ufuncs, reductions, predicates and conditional selection, linear
algebra, dtype conversion, `.npy` round-trips for all supported dtypes and
1-D/2-D/3-D/empty/0-D shapes, file-handle `.npy` I/O, non-contiguous tensors,
whitespace/CSV text round-trips, `loadtxt` converters, and `genfromtxt`
missing-value forms, `sort`/`unique`/`nonzero`/`argwhere`, generated stored and
deflate `.npy`/`.npz` fixtures, ZIP corruption paths, ZIP32 writer limits,
native-endian `.npy` headers, NPY v2.0 and v3.0 readers and writers, byte-level
writer conformance, `.npz` members and member versions, conditional `select`,
empty and pre-suffixed `.npz` members, layout view/copy semantics,
broadcast/take/take_along_axis/put edge shapes, split/diagonal/triangular
helpers, variance/NaN reductions, product/`ptp` reductions, logarithmic and
Vandermonde/triangular constructors, dot/inner/outer/cross products,
trapezoidal integration, linear interpolation, phase unwrapping, weighted
averages, covariance/correlation, polynomial evaluation/fitting, histograms,
coordinate/flat-index conversion, NaN-aware arg reductions, additional ufuncs
and NaN helpers, difference edges, broadcast-shape arithmetic, quoted/bool text
fields, meshgrid, gradients, closeness predicates,
raw binary I/O, text file handles, deflate expansion guardrails, ZIP64 `.npz`
fixtures, release
metadata, and error paths. The pad tests compare
wrap results with an independent index walk and cover every constant-value
form; the selection and I/O tests cover empty/0-D branches, Unicode and
colliding `.npz` names, EOCD comments, invalid members, zero-width text
tensors, sequential handles, and truncation.
It currently contains 460 tests.
`examples/basic.lua` is runnable with the same `LUA_PATH`.

## Performance notes

`bench/bench.lua` (run with `./scripts/bench.sh`) compares construction,
reductions, broadcasting, layout operations, and `.npy` I/O with direct
luatorch/ATen or raw Lua/FFI baselines. A dated snapshot with machine,
compiler, and method details is in [docs/performance.md](docs/performance.md).
Timing is deliberately not a CI gate; run the benchmark on the target machine
rather than treating the recorded numbers as constants.

## Repository layout

```text
numlua/
  numlua/init.lua     -- public facade
  numlua/ufunc.lua    -- elementwise ufunc facade over luatorch/ATen
  numlua/npy.lua      -- .npy v1-v3 reader/writer (v1.0 default)
  numlua/text.lua     -- whitespace/CSV text reader/writer
  docs/api.md         -- API reference
  examples/basic.lua  -- runnable example
  bench/bench.lua     -- facade-vs-ATen micro-benchmarks
  docs/performance.md -- dated benchmark snapshot and caveats
  tests/run.lua       -- dependency-free test harness
  scripts/test.sh     -- test wrapper with LUA_PATH setup
  scripts/bench.sh    -- benchmark wrapper with LUA_PATH setup
  Makefile            -- make test, make bench
```

## Status and known limits

This is an MVP, not a complete NumPy replacement:

- reductions, including `any`, `all`, and `count_nonzero`, accept one axis at a
  time (no tuple axes);
- there is no general dtype-promotion system; `mean`, `var`, `std`, and the
  NaN statistics promote integer/bool input to `float64`, while scalar
  `where`/`select` branches are materialized as 0-D `float32` (or `bool`)
  tensors before ATen broadcasting;
- `var`/`std` and the `nan*` reductions accept one axis at a time; `ddof` must
  be finite, and all-NaN slices return NaN without NumPy's runtime warning;
- `meshgrid` requires 1-D inputs, keeps each input's dtype instead of applying
  NumPy result-type promotion, and `copy = false` returns expanded views;
- `gradient` implements edge order 1 only (no `edge_order = 2`), promotes
  integer/bool input to `float64`, and requires at least two elements on each
  selected axis;
- `dot` follows NumPy's vector/matrix/batched shape rules but delegates the
  multiply to ATen, so mixed integer/floating operands follow ATen's matmul
  dtype checks rather than NumPy's promotion;
- `cross` supports 3-component vectors only (not NumPy's 2-component form),
  requires the same rank for both operands, and broadcasts batch axes through
  ATen;
- `trapezoid`/`trapz` default to the last axis, accept a 1-D coordinate vector
  or full-shape coordinates, and use ATen's arithmetic promotion (integer
  input becomes `float32`);
- `interp` requires a strictly increasing 1-D grid, returns `float64`, and
  takes `left`/`right` positionally; `unwrap` takes `discont`/`axis`/`period`
  positionally and promotes integer input to `float64`;
- `average` accepts no weights, 1-D weights along one axis, or full-shape
  weights; weighted sums that are zero raise an error;
- `diff` prepend/append values must match the input rank and non-axis shape
  (or be numbers, which fill one slice); `ediff1d` always flattens;
- `broadcast_shapes(...)` returns a Lua shape table and does not allocate a
  tensor;
- `cov` defaults to sample covariance (`ddof = 1`) and supports 1-D/2-D inputs,
  `rowvar = false`, and optional `y` observations; `corrcoef` uses unit sample
  covariance and clips round-off to `[-1, 1]`;
- `polyval`/`polyfit` use descending-degree coefficients; `polyfit` returns
  `float64` and requires fewer degrees than data points;
- `histogram` edges are `float64`; explicit edges must be strictly increasing,
  and `range` cannot be combined with explicit edges;
- `numlua.ravel_multi_index`/`numlua.unravel_index` use 1-based coordinates and flat indices,
  support `"C"`/`"F"` order, and return `int64` tensors;
- elementwise ufuncs delegate dtype promotion to ATen, not NumPy's promotion
  rules or `numlua.promote_types`; integral input to transcendental functions
  becomes `float32`, and integer division becomes `float32`;
- ufuncs have no `out=`, `where=`, `dtype=`, or `casting=` options. Arithmetic
  ufuncs reject Lua boolean scalars (use a `bool` tensor); the logical ufuncs
  accept booleans, numbers, and tensors;
- fancy indexing is not exposed; use the returned luatorch tensors and `torch`
  for those operations;
- `sort` defaults to the last axis and accepts a numlua `descending` flag;
  0-D input is rejected, as in `argsort`;
- `unique` always flattens and sorts; it has no `axis`, `equal_nan`, or unsorted
  mode, and its first-occurrence/inverse indices are 1-based;
- `nonzero` and `argwhere` use 1-based C-order coordinates and reject 0-D
  input; use `flatnonzero` for flat indices or a 0-D scalar;
- `squeeze` delegates to luatorch; squeezing an axis whose size is not 1 leaves
  that axis unchanged rather than raising NumPy's error;
- `split` boundary tables are 1-based, and integer `split` requires an even
  division; `array_split` gives the remainder to the leading chunks. The
  `hsplit`/`vsplit`/`dsplit` helpers use 1-based column/row/depth axes.
- `fromfile`/`tofile` default to raw binary (`sep = ""`) and carry no shape or
  dtype metadata; a non-empty `sep` selects simple numeric text with an
  optional printf `format`, not the richer `loadtxt`/`savetxt` options. Text
  mode reads a caller-owned handle to EOF. `fromfile` defaults to `float64`;
  pass the matching dtype used by `tofile`. Use `.npy` for self-describing
  files;
- `.npy` reading supports v1.0, v2.0, and v3.0 headers; writing defaults to
  v1.0 and can emit v2.0/v3.0 with the `version` option. Files must be C order
  on little-endian hosts and use the dtypes above; `load` reads the complete
  file into memory before copying it into a tensor. Native-endian `=` descr
  values are accepted only as aliases for the little-endian `<` spellings.
  `save`/`load`, `savez`/`savez_compressed`/`loadz`, `fromfile`/`tofile`, and
  `loadtxt`/`savetxt` accept open Lua file handles and leave them open;
  `loadz` reads from the current position to EOF;
- `.npz` writing is stored by default and deflate through `savez_compressed`;
  reading supports stored and deflate entries plus ZIP64 archives. Deflate
  needs a system zlib
  library loaded through FFI, while stored archives remain dependency-free.
  Deflate writing without zlib raises a clear error unless the stored fallback
  is requested. Both writers choose the member NPY header version with
  `{ version = 2 }`/`{ version = 3 }`; the default remains v1.0. Writer output
  is ZIP32-only and rejects entry counts above
  65535, names above 65535 bytes, or member/archive sizes that do not fit
  32-bit fields; ZIP64 writing is not supported.
- text I/O is intentionally simple: it does not store shape/dtype metadata or
  write quoted fields, or parse header metadata (`savetxt` header/footer lines
  are comments to `loadtxt`). Bool is supported with `True`/`False`/`0`/`1`
  spellings. `loadtxt`/`genfromtxt` can read quoted fields with
  a single-character `quotechar`, but quoted text is still parsed as a
  numeric/bool value.
  `loadtxt` `usecols` and `converters` use 1-based physical
  columns (NumPy uses 0-based keys), and one-row/one-column input is squeezed to
  1-D unless `ndmin = 2`.
  LF, CRLF, and CR line endings load, unknown option keys are rejected, and
  `int64` text output can round values beyond the exact double range (use
  `.npy` for those). All three text functions accept caller-owned file handles
  at their current position and leave them open;
- `genfromtxt` stays numeric/bool-only: it has no structured dtypes, `names`, string
  columns, masked output, `usemask`, `loose`, `invalid_raise`, or `skiprows`
  alias. Missing/filling/converters use 1-based physical columns, default dtype
  is `float32`, a malformed non-missing field is an error, and `max_rows`
  cannot be combined with `skip_footer`;
- LuaRocks packaging is provided by `numlua-0.26.0-1.rockspec`; the native
  luatorch shim must still be built separately.

## License

BSD 3-Clause. See [LICENSE](LICENSE). `numlua` depends on luatorch, which links
against libtorch/PyTorch; neither dependency is redistributed here.
