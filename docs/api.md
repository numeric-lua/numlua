# numlua API

`numlua` is a thin, NumPy-shaped facade over luatorch tensors. Every public
function returns a luatorch `Tensor` (except `shape`, `dtype`, and `save`), and
the numerical work is delegated to luatorch/ATen.

Load it with:

```lua
local numlua = require("numlua")
print(numlua.VERSION) -- "0.1.0"
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

```lua
numlua.reshape(t, 2, 3)
numlua.reshape(t, { 3, 2 })
numlua.reshape(t, 2, -1)
```

### `numlua.transpose(t[, axes])`

With no `axes`, reverses all dimensions (NumPy's default). With `axes`, applies
a 1-based permutation.

```lua
numlua.transpose(t)              -- reverse axes
numlua.transpose(t, { 2, 1 })    -- 2-D transpose
numlua.transpose(t, { 2, 3, 1 }) -- 3-D permutation
```

Zero- and one-dimensional inputs are returned unchanged when `axes` is omitted.

### `numlua.ravel(t)`

Returns a C-order contiguous 1-D view/copy with `numel(t)` elements.

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

## Linear algebra

### `numlua.dot(a, b)`

1-D inner product. Both operands must be 1-D; the result is a 0-D tensor. Use
`matmul` for matrix or batched products.

### `numlua.matmul(a, b)`

Delegates to luatorch/ATen `matmul`, including matrix-vector, matrix-matrix,
batched, and broadcasted forms. Broadcasting behavior and shape errors are
ATen's.

## Dtype

### `numlua.dtype(t)`

Returns the luatorch dtype name, such as `"float32"`, `"float64"`, `"int64"`,
or `"bool"`.

### `numlua.astype(t, dtype)`

Returns a converted tensor. The source tensor is not mutated.

```lua
local x = numlua.astype(numlua.array({ 1, 2 }), "int64")
```

## NumPy `.npy` I/O

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

## Errors

Invalid arguments raise ordinary Lua errors with a `numlua.<function>:` prefix
where the facade validates them. Errors from luatorch/ATen regarding shape,
dtype, or backend constraints are forwarded as-is.

## Differences from NumPy and known limits

- Axes/indices are 1-based; NumPy examples must shift axes by one.
- Default dtypes differ: `float32` is the luatorch default for constructors and
  Lua numeric tables; NumPy usually defaults to `float64`.
- Reductions take a single axis only, not a tuple of axes.
- There is no general dtype promotion. `mean` is the only place the facade
  promotes, and only integer/bool -> `float64`.
- `dot` is vector-only; NumPy's `dot` also performs matrix products.
- No elementwise arithmetic, ufuncs, or fancy indexing are exposed; use the
  returned luatorch tensors and `torch` directly.
- Broadcasting is not reimplemented; it is whatever luatorch/ATen does for the
  delegated operation.
- `.npy` is limited to v1.0, C order, little-endian hosts, and the four dtypes
  listed above.
- `squeeze` follows luatorch's permissive behavior for non-unit axes.
- No LuaRocks package yet.
