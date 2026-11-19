# numlua API

`numlua` is a thin, NumPy-shaped facade over luatorch tensors. Array-producing
functions return luatorch `Tensor`s (`shape` and `dtype` return plain Lua
values, and `save`/`savetxt` return `true`), and the numerical work is
delegated to luatorch/ATen.

Load it with:

```lua
local numlua = require("numlua")
print(numlua.VERSION) -- "0.2.0"
```

## Conventions

- **Axes and indices are 1-based.** `axis=1` is the first axis. Negative axes
  count from the end (`-1` is the last axis), as in luatorch.
- `argmax` results are 1-based. With no axis, the result is a flat C-order
  index into the tensor.
- Reductions return tensors. `numlua.sum(a):item()` extracts a Lua number from
  the 0-D all-axis result.
- `max` and `min` return values only. Use `argmax` for indices.
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
| `linalg.matrix_norm(t[, ord])` | `"fro"`/2 (default), `1`, `-1`, infinity, `-infinity`, `2`, `"nuc"` |

`lstsq` uses the SVD pseudoinverse; it is a convenience wrapper, not a LAPACK
driver.

## Array layout and selection

These functions are 1-based and return luatorch tensors:

| Function | Notes |
| --- | --- |
| `numlua.broadcast_to(t, shape)` | broadcast view; NumPy-compatible right alignment |
| `numlua.broadcast_arrays(...)` | returns multiple broadcast views |
| `numlua.tile(t, reps)` | repeat whole blocks (materialized); `reps` is a number or table |
| `numlua.repeat_(t, repeats[, axis])` | repeat elements (materialized); `numlua["repeat"]` is the same function |
| `numlua.flip(t, axis)` | reverse one axis or a table of axes (materialized) |
| `numlua.roll(t, shift[, axis])` | cyclic shift; flattens when `axis` is omitted |
| `numlua.pad(t, pad_width[, mode[, value]])` | `mode` currently only `"constant"` |
| `numlua.diff(t[, n[, axis]])` | default axis is the last |
| `numlua.cumsum(t[, axis])` / `numlua.cumprod` | flatten when `axis` is omitted |
| `numlua.argsort(t[, axis[, descending]])` | 1-based indices |
| `numlua.take(t, indices[, axis])` | flatten when `axis` is omitted |

`broadcast_to`, `broadcast_arrays`, `transpose`, contiguous `reshape`, and
contiguous `ravel` are views; mutating the source is visible through the result.
Operations documented as materialized return independent storage. Tests cover
the view/copy boundary on large arrays so a regression cannot silently turn a
view into a full-size copy.

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
| `float32` | `<f4` | 4 |
| `float64` | `<f8` | 8 |
| `int64` | `<i8` | 8 |
| `bool` | `\|b1` | 1 |

Writing an unsupported dtype raises a `numlua.save: unsupported dtype ...`
error. Only little-endian hosts are supported.

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
file into memory, then copies the payload into a C-order tensor.

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
- There is no general dtype promotion. `mean` is the only place the facade
  promotes, and only integer/bool -> `float64`. Scalar `where` branches are
  materialized as 0-D `float32` (or `bool`) tensors before ATen broadcasting.
- `dot` is vector-only; NumPy's `dot` also performs matrix products.
- No elementwise arithmetic, ufuncs, or fancy indexing are exposed; use the
  returned luatorch tensors and `torch` directly.
- Broadcasting is not reimplemented; it is whatever luatorch/ATen does for the
  delegated operation.
- `.npy` is limited to v1.0, C order, little-endian hosts, and the four dtypes
  listed above.
- Text I/O stores only numeric fields: no shape/dtype metadata, quoted strings,
  or headers. `loadtxt` defaults to `float32`, and one-row/one-column files are
  squeezed to 1-D. `bool` is not supported by either text function.
- `squeeze` follows luatorch's permissive behavior for non-unit axes.
- `.npz` supports stored (uncompressed) ZIP entries, which is what `np.savez`
  writes; deflate-compressed `np.savez_compressed` archives are rejected.
- LuaRocks installs the Lua facade; the native `libluatorch.so` must be built
  separately against libtorch.
