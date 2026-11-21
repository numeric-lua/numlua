# numlua API

`numlua` is a thin, NumPy-shaped facade over luatorch tensors. Array-producing
functions return luatorch `Tensor`s (`shape` and `dtype` return plain Lua
values, and `save`/`savetxt` return `true`), and the numerical work is
delegated to luatorch/ATen.

Load it with:

```lua
local numlua = require("numlua")
print(numlua.VERSION) -- "0.6.0"
```

## Conventions

- **Axes and indices are 1-based.** `axis=1` is the first axis. Negative axes
  count from the end (`-1` is the last axis), as in luatorch.
- `argmax` and `argmin` results are 1-based. With no axis, the result is a flat
  C-order index into the tensor.
- Reductions return tensors. `numlua.sum(a):item()` extracts a Lua number from
  the 0-D all-axis result.
- `max` and `min` return values only. Use `argmax`/`argmin` for indices.
- `dtype` accepts luatorch dtype names (`"float32"`, `"float64"`, `"int64"`,
  `"bool"`, ...) or luatorch dtype ids (`torch.float32`, ...).
- Options tables are the luatorch option tables; `dtype` is the commonly used
  key.

## Constructors and coercion

### `numlua.array(value[, opts])`

Converts a nested rectangular Lua table, number, or boolean to a luatorch
tensor. An existing tensor is cloned. Numeric tables infer `float32` because Lua
does not distinguish integers from floats; boolean tables infer `bool`. Pass
`opts.dtype` to choose another dtype.

```lua
numlua.array({ { 1, 2 }, { 3, 4 } })                    -- float32, 2x2
numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
```

### `numlua.asarray(value[, opts])`

Like `array`, but an existing tensor is returned unchanged unless
`opts.dtype` requests a conversion. Lua values are converted as in `array`.

### `numlua.zeros(...)`

`zeros(d1, d2, ...[, opts])` or `zeros(shape_table[, opts])`. Default dtype is
`float32`.

### `numlua.ones(...)`

Same signatures as `zeros`.

### `numlua.full(shape, fill_value[, opts])`

`shape` is a table or a single dimension. `fill_value` must be numeric.

```lua
numlua.full({ 2, 3 }, 7, { dtype = "int64" })
numlua.full(4, 1.5)
```

### `numlua.eye(n[, opts])`

`n` by `n` identity matrix. Default dtype is `float32`.

### `numlua.arange([start,] stop[, step][, opts])`

Stop-exclusive, like NumPy. Integer bounds infer `int64`; non-integer bounds
infer `float32`.

```lua
numlua.arange(5)          -- int64: 0, 1, 2, 3, 4
numlua.arange(2, 10, 2)   -- int64: 2, 4, 6, 8
numlua.arange(0, 1.5, 0.5)
```

### `numlua.linspace(start, stop[, num[, opts]])`

`num` defaults to 50 and is inclusive of both endpoints. Default dtype is
`float32`. Passing an options table as the third argument is allowed.

```lua
numlua.linspace(0, 1, 5)
numlua.linspace(0, 1, { dtype = "float64" })
```

## Shape

### `numlua.shape(t)`

Returns a plain Lua array of dimensions: `{2, 3}`.

### `numlua.reshape(t, ...)`

Accepts variadic dimensions or a shape table. `-1` infers one dimension.
For a contiguous input this returns a view; a non-contiguous reshape may
materialize a copy, following luatorch/ATen.

```lua
numlua.reshape(t, 2, 3)
numlua.reshape(t, { 3, 2 })
numlua.reshape(t, 2, -1)
```

### `numlua.transpose(t[, axes])`

With no `axes`, reverses all dimensions (NumPy's default). With `axes`, applies
a 1-based permutation. This is a view: the result shares storage with the input.

```lua
numlua.transpose(t)              -- reverse axes
numlua.transpose(t, { 2, 1 })    -- 2-D transpose
numlua.transpose(t, { 2, 3, 1 }) -- 3-D permutation
```

Zero- and one-dimensional inputs are returned unchanged when `axes` is omitted.

### `numlua.ravel(t)`

Returns a C-order contiguous 1-D result with `numel(t)` elements. It is a view
when the input is already C-contiguous and a materialized copy otherwise.

### `numlua.squeeze(t[, axis])`

With no `axis`, removes all size-1 dimensions. With `axis`, delegates one
dimension to luatorch; an axis whose size is not 1 is left unchanged.

### `numlua.unsqueeze(t, axis)`

Inserts a size-1 dimension at the 1-based `axis`. Negative axes are accepted.

### `numlua.swapaxes(t, axis1, axis2)`

Interchanges two axes. Axes are 1-based and negative axes count from the end.
This is a view; the result shares storage with the input.

```lua
numlua.swapaxes(t, 1, 3)   -- first and last axes
numlua.swapaxes(t, -1, -3) -- the same interchange
```

### `numlua.moveaxis(t, source, destination)`

Moves one or more axes to new positions, leaving the order of all other axes
unchanged. `source` and `destination` may each be an axis or a table of axes.
Negative axes and negative destination positions count from the end. This is a
view: the result shares storage with the input.

```lua
numlua.moveaxis(t, 1, 3)        -- move axis 1 to position 3
numlua.moveaxis(t, -1, 1)       -- move the last axis to the front
numlua.moveaxis(t, { 1, 2 }, { 3, 1 })
```

### `numlua.atleast_1d(...)`, `atleast_2d`, `atleast_3d`

Accept one or more tensors and return one result per input. These helpers only
add size-1 dimensions, so they are views:

| Input rank | `atleast_1d` | `atleast_2d` | `atleast_3d` |
| --- | --- | --- | --- |
| 0-D | `{1}` | `{1, 1}` | `{1, 1, 1}` |
| 1-D, length `N` | unchanged | `{1, N}` | `{1, N, 1}` |
| 2-D, `M x N` | unchanged | unchanged | `{M, N, 1}` |
| 3-D or higher | unchanged | unchanged | unchanged |

```lua
local a, b = numlua.atleast_2d(scalar, vector)
```

### `numlua.resize(t, shape)`

Accepts a shape table or variadic dimensions and returns a new tensor with that
shape. Existing elements are kept in C order, any new space is zero-filled, and
surplus elements are dropped. This follows `ndarray.resize`'s zero-fill and
truncation rule; unlike `np.resize`, data is not tiled. The result has
independent storage and the input is not modified. Zero-size dimensions are
allowed.

```lua
numlua.resize(numlua.array({ 1, 2, 3 }), { 2, 3 })
-- { { 1, 2, 3 }, { 0, 0, 0 } }
numlua.resize(t, 4) -- a 1-D result with four elements
```

## Combining

### `numlua.concatenate(tensors[, axis])`

Concatenates a non-empty table of tensors along `axis` (default `1`).

### `numlua.stack(tensors[, axis])`

Stacks a non-empty table of same-shaped tensors along a new `axis` (default
`1`).

## Reductions

### `numlua.sum(t[, axis[, keepdims]])`

### `numlua.mean(t[, axis[, keepdims]])`

All-axis reductions return a 0-D tensor. With `axis`, the reduced axis is
removed unless `keepdims` is true. `keepdims` must be a boolean when provided.

`mean` returns `float32` for `float32` input and `float64` for `float64` input.
Because ATen refuses to infer a dtype for integer/bool `mean`, numlua promotes
non-floating input to `float64` first.

### `numlua.max(t[, axis[, keepdims]])`

### `numlua.min(t[, axis[, keepdims]])`

Return values only. With `axis`, luatorch also computes indices, but `numlua`
discards them.

### `numlua.argmax(t[, axis[, keepdims]])`

Returns an `int64` tensor of 1-based indices. If `axis` is omitted, the result
is the flat C-order index of the maximum (a 0-D tensor). With `axis`, the
reduced axis is removed unless `keepdims` is true.

### `numlua.median(t[, axis[, keepdims]])`

Returns the median. With no axis this is a Lua number; with an axis it returns
a tensor with the reduced axis removed unless `keepdims` is true (when
supported by the underlying luatorch median path).

### `numlua.quantile(t, q[, axis[, keepdims]])`

Returns the `q`-quantile using linear interpolation between order statistics;
`q` must be in `[0, 1]`. With no axis the input is flattened and a Lua number
is returned. With an axis the reduced axis is removed unless `keepdims` is
true.

### `numlua.percentile(t, q[, axis[, keepdims]])`

Percent-scaled alias for `numlua.quantile`; `q` is in `[0, 100]`.

### `numlua.compress(condition, t[, axis])`

Selects the flattened elements where the boolean `condition` is true. An
explicit `axis` is not implemented yet.

### `numlua.argmin(t[, axis[, keepdims]])`

The minimum counterpart of `argmax`: returns an `int64` tensor of 1-based
indices, using a flat C-order index when `axis` is omitted. The same 1-based
axis and `keepdims` rules apply. The reduction is luatorch's `argmin` kernel;
numlua only validates arguments and exposes the name.

```lua
local t = numlua.array({ { 4, 7 }, { 2, 6 } })
numlua.argmin(t):item()     -- 3, the flat index of 2
numlua.argmin(t, 1)         -- {2, 2}
numlua.argmin(t, 2)         -- {1, 1}
```

## Predicates and selection

### `numlua.where(condition, x, y)`

Selects elements from `x` where `condition` is true and from `y` elsewhere.
`condition`, `x`, and `y` may be tensors or Lua scalars (`number`/`boolean`);
scalar operands become 0-D luatorch tensors, so ATen performs all broadcasting
between the three operands. A non-boolean `condition` tensor or number is
truth-tested as `condition ~= 0`. The result is a luatorch tensor.

```lua
local condition = numlua.array({ { true }, { false } })
numlua.where(condition, numlua.array({ { 1, 2, 3 } }), -1)
-- {{1, 2, 3}, {-1, -1, -1}}
```

Scalar numeric branches are materialized as `float32` 0-D tensors; tensor
branches keep luatorch/ATen's dtype promotion.

### `numlua.clip(t, minimum, maximum)`

Clamps every element of `t` to the inclusive `[minimum, maximum]` range.
Bounds are normally numbers, but tensor bounds are also accepted and broadcast
by ATen's elementwise `maximum`/`minimum`; scalar bounds use luatorch's clamp
kernel. The result is a luatorch tensor. NaN input values are preserved, and if
`minimum > maximum` the result is `maximum`, matching luatorch/NumPy clamping.
Missing or non-numeric/non-tensor bounds raise a `numlua.clip:` error.

```lua
numlua.clip(numlua.array({ -2, -0.5, 0, 0.5, 2 }), -1, 1)
-- {-1, -0.5, 0, 0.5, 1}
```

### `numlua.isnan(t)`

Returns a `bool` tensor that is true where `t` is NaN and false elsewhere.
Infinities are not NaN. Integer and boolean inputs produce an all-false mask.
The implementation uses the IEEE rule that NaN is not equal to itself
(`t:ne(t)`), so no separate native kernel is needed.

### `numlua.any(t[, axis[, keepdims]])`

### `numlua.all(t[, axis[, keepdims]])`

Return a `bool` tensor. With no `axis`, the result is a 0-D tensor (or an
all-ones shape when `keepdims` is true). With an `axis`, the reduced axis is
removed unless `keepdims` is true. Axes are 1-based and may be negative. NaN
values are truthy, like NumPy.

### `numlua.count_nonzero(t[, axis])`

Returns an `int64` tensor with the number of elements that are not equal to
zero. With no `axis`, the result is a 0-D tensor; with an `axis`, that axis is
reduced and removed. NaN counts as nonzero. There is no `keepdims` parameter.

```lua
local t = numlua.array({ { 0, 1, 0 }, { 2, 0, 3 } })
numlua.count_nonzero(t):item()        -- 3
numlua.count_nonzero(t, 1):totable()  -- {1, 1, 1}
numlua.count_nonzero(t, 2):totable()  -- {1, 2}
```

## Elementwise ufuncs

All elementwise functions return a luatorch tensor and leave their inputs
unchanged. They delegate the arithmetic to luatorch/ATen; there is no separate
kernel or broadcast layer in numlua. Operands may be tensors or Lua numbers, and
tensor operands broadcast using ATen's normal right-aligned rules. Passing two
numbers produces a 0-D tensor.

| Function | Notes |
| --- | --- |
| `numlua.add(a, b)` | elementwise addition |
| `numlua.subtract(a, b)` | elementwise subtraction |
| `numlua.multiply(a, b)` | elementwise multiplication |
| `numlua.divide(a, b)` | elementwise true division |
| `numlua.power(a, b)` | elementwise power, equivalent to ATen `pow` |
| `numlua.maximum(a, b)` | elementwise maximum |
| `numlua.minimum(a, b)` | elementwise minimum |
| `numlua.negative(t)` | unary negation |
| `numlua.abs(t)` | absolute value |
| `numlua.sqrt(t)` | square root |
| `numlua.exp(t)` | exponential |
| `numlua.log(t)` | natural logarithm |
| `numlua.sin(t)` / `numlua.cos(t)` / `numlua.tanh(t)` | trigonometric and hyperbolic functions |
| `numlua.square(t)` | elementwise square |

```lua
local a = numlua.array({ 1, 2, 3 })
numlua.add(a, 10)             -- {11, 12, 13}
numlua.subtract(5, a)         -- {4, 3, 2}
numlua.multiply(a, 2)         -- {2, 4, 6}
numlua.divide(a, 2)           -- {0.5, 1, 1.5}
numlua.power(a, 2)            -- {1, 4, 9}
numlua.maximum(a, 2)          -- {2, 2, 3}
numlua.sqrt(numlua.array({ 1, 4, 9 })) -- {1, 2, 3}
```

Lua booleans are not accepted as scalar operands; pass a `bool` tensor if you
need boolean tensor arithmetic. Non-numeric/non-tensor operands raise a
`numlua.<function>:` error. Incompatible broadcast shapes and unsupported dtype
combinations are reported by luatorch/ATen and forwarded as-is.

### Dtype behavior

Elementwise dtype promotion is **ATen's**, not NumPy's and not the small
`numlua.promote_types` table. For example, an `int64` tensor plus a `float32`
tensor produces `float32`, even though `numlua.promote_types("float32",
"int64")` returns `"float64"`. A `float32` plus `float64` tensor produces
`float64`.

Lua scalars are untyped, so the scalar path follows ATen/luatorch's scalar
conventions:

- an integral scalar (`2`) keeps an integer input dtype for `add`, `subtract`,
  `multiply`, `power`, `maximum`, and `minimum` (bool input follows ATen's
  scalar behavior and normally becomes `int64` here);
- a fractional scalar (`2.5`) on an integral input produces `float32`;
- floating input keeps its own dtype when combined with a scalar;
- `divide` uses ATen true division, so integer/integer division produces
  `float32`, not NumPy's usual `float64`. Integer inputs are converted to a
  floating result, so division by zero yields `inf`/`nan` rather than an error;
- both-scalar calls produce a 0-D `float32` tensor, following `numlua.array`.

Unary operations split into two groups:

- `negative`, `abs`, and `square` preserve an integral input dtype. ATen does
  not implement `abs` or negation for `bool` tensors; those calls error;
- `sqrt`, `exp`, `log`, `sin`, `cos`, and `tanh` produce `float32` for
  integer/bool input and preserve `float32`/`float64` floating input. NumPy
  would normally use `float64` for integral input.

For `maximum`/`minimum` with a scalar, numlua materializes a 0-D operand with
the input's dtype for integral scalars (or `int64` for a `bool` input) and
`float32` for fractional/non-finite scalars. This only supplies the missing
scalar overload; the comparison itself is still ATen's broadcast kernel.

There are no `out=`, `where=`, `dtype=`, or `casting=` keyword arguments, and no
separate ufunc objects. Use `numlua.astype` for an explicit result dtype.

## Linear algebra

### `numlua.dot(a, b)`

1-D inner product. Both operands must be 1-D; the result is a 0-D tensor. Use
`matmul` for matrix or batched products.

### `numlua.matmul(a, b)`

Delegates to luatorch/ATen `matmul`, including matrix-vector, matrix-matrix,
batched, and broadcasted forms. Broadcasting behavior and shape errors are
ATen's.

### `numlua.linalg`

A NumPy-shaped facade for the luatorch linear-algebra methods. It is available
as both `numlua.linalg` and `require("numlua.linalg")`. Dimensions are
**1-based**, and array results are luatorch tensors.

```lua
local la = numlua.linalg
local a = numlua.array({ { 4, 7 }, { 2, 6 } }, { dtype = "float64" })

la.det(a):item()          -- 10
la.inv(a)                 -- 2x2
la.solve(a, numlua.array({ 1, 0 }, { dtype = "float64" }))
la.norm(a)                -- 0-D Frobenius norm
local u, s, vh = la.svd(a, false)
local q, r = la.qr(a)
```

| Function | Result |
| --- | --- |
| `linalg.norm(t[, p[, dim[, keepdim]]])` | 0-D norm, or a tensor when `dim` is given; `p` defaults to 2 |
| `linalg.inv(t)` | inverse of a square matrix |
| `linalg.det(t)` | 0-D determinant of a square matrix |
| `linalg.solve(a, b)` | solves `a * x = b`; `b` is a vector or matrix |
| `linalg.svd(t[, full_matrices])` | returns `U, S, Vh`; `full_matrices` defaults to `true` |
| `linalg.qr(t[, mode])` | returns `Q, R`; `mode` is `"reduced"` (default), `"complete"`, or `"r"` |

`svd` accepts rectangular matrices. `qr` returns the ATen shapes for the chosen
mode. Invalid dtypes, singular systems, and shape errors are reported by
luatorch/ATen; the facade adds clear errors for non-tensors, non-square
inverses/determinants, zero dimensions, and mismatched `solve` shapes.

Additional matrix helpers:

| Function | Result |
| --- | --- |
| `linalg.pinv(t[, rcond])` | Moore-Penrose pseudoinverse via SVD |
| `linalg.lstsq(a, b[, rcond])` | minimum-norm least-squares solution |
| `linalg.matrix_norm(t[, ord])` | `"fro"` (default; `"f"` alias), `"nuc"`, `1`, `-1`, `2`, `-2`, infinity, or `-infinity` |

`matrix_norm` returns a 0-D tensor. `ord=2` is the spectral norm (largest
singular value), `ord=-2` the smallest singular value, and `"nuc"` the nuclear
norm; all three use the SVD. `"fro"` is the default Frobenius norm, and the
remaining orders follow NumPy's max/min absolute column- or row-sum rules.
Unsupported orders raise a `numlua.linalg.matrix_norm:` error.

`lstsq` uses the SVD pseudoinverse; it is a convenience wrapper, not a LAPACK
driver.

## Array layout and selection

These functions are 1-based and return luatorch tensors:

| Function | Notes |
| --- | --- |
| `numlua.swapaxes(t, axis1, axis2)` | interchange two axes (view); negative axes accepted |
| `numlua.moveaxis(t, source, destination)` | move one or more axes (view); accepts axis tables |
| `numlua.atleast_1d(...)` / `atleast_2d` / `atleast_3d` | add size-1 dimensions (view); one result per input |
| `numlua.resize(t, shape)` | zero-fill/truncate to a new shape (materialized) |
| `numlua.broadcast_to(t, shape)` | broadcast view; NumPy-compatible right alignment |
| `numlua.broadcast_arrays(...)` | returns multiple broadcast views |
| `numlua.tile(t, reps)` | repeat whole blocks (materialized); `reps` is a number or table |
| `numlua.repeat_(t, repeats[, axis])` | repeat elements (materialized); `numlua["repeat"]` is the same function |
| `numlua.flip(t, axis)` | reverse one axis or a table of axes (materialized) |
| `numlua.roll(t, shift[, axis])` | cyclic shift; flattens when `axis` is omitted |
| `numlua.pad(t, pad_width[, mode[, value]])` | modes `"constant"`, `"edge"`, `"reflect"`, and `"symmetric"` (materialized) |
| `numlua.diff(t[, n[, axis]])` | default axis is the last |
| `numlua.cumsum(t[, axis])` / `numlua.cumprod` | flatten when `axis` is omitted |
| `numlua.argsort(t[, axis[, descending]])` | 1-based indices |
| `numlua.take(t, indices[, axis])` | flatten when `axis` is omitted |
| `numlua.flatnonzero(t)` | 1-based flat indices of non-zero entries |

`pad_width` follows NumPy's common forms: a number pads every axis equally, a
`{before, after}` pair pads every axis with that pair, and a table of per-axis
pairs pads each axis separately. `"edge"` repeats the first and last values,
`"reflect"` mirrors without repeating the edge, and `"symmetric"` mirrors with
the edge included. Zero-size axes can be extended only with `"constant"`.

### View and copy boundaries

Views share storage with the input, so mutating one is visible through the
other: `transpose`, `swapaxes`, `moveaxis`, `atleast_1d`, `atleast_2d`,
`atleast_3d`, `squeeze`, `unsqueeze`, `broadcast_to`, `broadcast_arrays`,
contiguous `reshape`, and contiguous `ravel`.

These allocate independent storage: `resize`, `pad`, `tile`, `repeat_`, `flip`,
`roll`, `diff`, `cumsum`, `cumprod`, `argsort`, and `take`. `reshape` and
`ravel` also copy when their input is not contiguous, because ATen must
materialize a C-order layout first. Tests cover the view/copy boundary,
including large arrays, so a regression cannot silently turn a view into a
full-size copy or make an allocating helper alias its input.

## Dtype

### `numlua.dtype(t)`

Returns the luatorch dtype name, such as `"float32"`, `"float64"`, `"int64"`,
or `"bool"`.

### `numlua.astype(t, dtype)`

Returns a converted tensor. The source tensor is not mutated.

```lua
local x = numlua.astype(numlua.array({ 1, 2 }), "int64")
```

### Promotion helpers

```lua
numlua.promote_types("int32", "float32") -- "float64"
numlua.result_type("bool", "int16", "float32") -- "float32"
numlua.can_cast("int8", "int64") -- true
```

This is a deliberately small safe-promotion table, documented in the README;
it is not full NumPy type-promotion parity.

## NumPy `.npy` / `.npz` I/O

### `numlua.save(path, tensor)`

Writes a NumPy v1.0 `.npy` file in C order and returns `true`. The tensor is
detached and made contiguous before writing; autograd metadata is not stored.

Supported dtypes:

| dtype | descr | size |
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

Writing an unsupported dtype raises a `numlua.save: unsupported dtype ...`
error. Only little-endian hosts are supported. luatorch cannot construct a
`float16` tensor directly from a Lua table; create it in another dtype and
cast with `:to("float16")` first.

### `numlua.load(path)`

Reads an `.npy` v1.0 file and returns a luatorch tensor. Supports 1-D, 2-D, N-D,
and 0-D arrays with the dtypes above.

Rejected with explicit errors:

- big-endian `descr` values (`>...`);
- `fortran_order=True`;
- unsupported `descr` values;
- NPY versions other than 1.0;
- malformed headers, missing trailing newline, bad magic;
- data whose byte length does not match `shape * itemsize`;
- non-file paths or unreadable files.

The header is validated before any tensor allocation. `load` reads the complete
file into memory, then copies the payload into a C-order tensor. Arrays with
shape `()` (0-D scalars), zero-sized dimensions, and non-contiguous source
tensors are supported; non-contiguous tensors are written in C order.

### `numlua.savez(path, arrays)`

Writes a ZIP archive of `.npy` members and returns `true`. `arrays` is a table
mapping non-empty array names to luatorch tensors. Entries are always written
with the stored (uncompressed) ZIP method, so writing has no compression
dependency and matches `np.savez`.

### `numlua.loadz(path)`

Reads an `.npz` archive and returns a table whose keys are member names with
the `.npy` suffix removed. Supported ZIP compression methods are:

| method | written by | availability |
| --- | --- | --- |
| stored (0) | `np.savez`, `numlua.savez` | always, dependency-free |
| deflate (8) | `np.savez_compressed` | read-only; requires system zlib |

Deflate support is optional and detected lazily through LuaJIT FFI: on Linux
the loader tries `libz`, `libz.so.1`, and `libz.so`, and on macOS it also tries
`libz.dylib`. No LuaRocks or compression package is required. Stored archives
work when zlib is absent; encountering a deflate entry then raises a
`numlua.loadz: deflate-compressed entry ... requires a system zlib library`
error. `NUMLUA_DISABLE_ZLIB=1` forces the no-zlib path for testing.

The reader validates member CRCs against the uncompressed payload, checks
local and central-directory headers and sizes before copying, and reports
corrupt central directories, CRC mismatches, truncation, encryption, multi-disk
archives, ZIP64 archives, and unsupported compression methods as Lua errors.

## Text and CSV I/O

### `numlua.loadtxt(path[, options])`

Reads a whitespace- or delimiter-separated numeric text file and returns a
luatorch tensor. Options:

| option | default | meaning |
| --- | --- | --- |
| `delimiter` | whitespace runs | field separator, such as `","` or `"\t"` |
| `comments` | `"#"` | comment marker, a table of markers, or `false`/`""` to disable |
| `dtype` | `"float32"` | result dtype (a luatorch dtype name or id) |

Blank lines and comments are ignored. Comment text starts at the first occurrence
of any configured marker on a line, so both full-line and trailing comments work.
Every data row must have the same number of fields. One-row and one-column files
load as 1-D tensors, following NumPy's default squeeze behavior; a file with no
data rows returns an empty 1-D tensor. Numeric fields accept decimal, scientific,
`nan`, `inf`, and `-inf` spellings.

```lua
-- data.csv:
--   # measured values
--   1.5, 2.5
--   3.5, 4.5
local values = numlua.loadtxt("data.csv", {
  delimiter = ",",
  comments = "#",
  dtype = "float64",
})
```

Malformed input raises a `numlua.loadtxt:` error that names the physical line
and, for an invalid value, the column:

```text
numlua.loadtxt: line 3 has 4 column(s), expected 3
numlua.loadtxt: line 2: invalid number 'oops' in column 2
```

Supported result dtypes are the numeric luatorch dtypes: `float16`, `float32`,
`float64`, `int8`, `int16`, `int32`, `int64`, and `uint8`. `bool` is rejected;
write `0`/`1` with an integral dtype instead.

### `numlua.savetxt(path, tensor[, options])`

Writes a 1-D or 2-D tensor as numeric text and returns `true`. Options:

| option | default | meaning |
| --- | --- | --- |
| `delimiter` | `" "` | field separator |
| `fmt` | dtype-aware | `printf` format string or function for each value |

The default floating formats use enough significant digits to round-trip the
tensor's dtype (`float16`, `float32`, or `float64`); integral dtypes use base 10.
`nan`, `inf`, and `-inf` are written explicitly. A custom `fmt` is applied to
every value:

```lua
numlua.savetxt("values.csv", numlua.arange(6):reshape(2, 3), {
  delimiter = ",",
  fmt = "%.3f",
})
```

Zero-dimensional and higher-than-2-D tensors are rejected. `bool` is not a
supported text dtype. Text files do not record shape or dtype, so load with the
matching `dtype` when exact integer or wider floating-point values matter; a
1xN or Nx1 matrix cannot be distinguished from a vector after loading.

## Errors

Invalid arguments raise ordinary Lua errors with a `numlua.<function>:` prefix
where the facade validates them. Errors from luatorch/ATen regarding shape,
dtype, or backend constraints are forwarded as-is.

## Differences from NumPy and known limits

- Axes/indices are 1-based; NumPy examples must shift axes by one.
- Default dtypes differ: `float32` is the luatorch default for constructors and
  Lua numeric tables; NumPy usually defaults to `float64`.
- Reductions, including `any`, `all`, and `count_nonzero`, take a single axis
  only, not a tuple of axes.
- The facade has no NumPy-style general dtype-promotion system of its own.
  `mean` promotes integer/bool input to `float64`; elementwise ufuncs delegate
  promotion to ATen. Scalar `where` branches are materialized as 0-D `float32`
  (or `bool`) tensors before ATen broadcasting.
- `dot` is vector-only; NumPy's `dot` also performs matrix products.
- Elementwise ufuncs delegate dtype promotion to ATen rather than to NumPy's
  promotion rules or `numlua.promote_types`. In particular, `int64` + `float32`
  stays `float32`, integer/integer division produces `float32` (not `float64`),
  and transcendental functions on integral input produce `float32`.
- Elementwise ufuncs have no `out=`, `where=`, `dtype=`, or `casting=` options.
  Lua scalar operands are numbers only; booleans require a `bool` tensor.
- Fancy indexing is not exposed; use the returned luatorch tensors and `torch`
  directly.
- Broadcasting is not reimplemented; it is whatever luatorch/ATen does for the
  delegated operation.
- `.npy` is limited to v1.0, C order, little-endian hosts, and the dtypes
  listed above.
- Text I/O stores only numeric fields: no shape/dtype metadata, quoted strings,
  or headers. `loadtxt` defaults to `float32`, and one-row/one-column files are
  squeezed to 1-D. `bool` is not supported by either text function.
- `squeeze` follows luatorch's permissive behavior for non-unit axes.
- `.npz` writing is always stored; reading supports stored and deflate entries.
  Deflate reading requires a system zlib library through LuaJIT FFI, while
  stored archives and all writing remain dependency-free.
- LuaRocks installs the Lua facade; the native `libluatorch.so` must be built
  separately against libtorch.
