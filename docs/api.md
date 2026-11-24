# numlua API

`numlua` is a thin, NumPy-shaped facade over luatorch tensors. Array-producing
functions return luatorch `Tensor`s (`shape` and `dtype` return plain Lua
values, and `save`/`tofile`/`savetxt` return `true`), and the numerical work
is delegated to luatorch/ATen.

Load it with:

```lua
local numlua = require("numlua")
print(numlua.VERSION) -- "0.26.0"
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

### `numlua.identity(n[, opts])`

NumPy's square identity alias for `eye`. `n` must be a non-negative integer.

### `numlua.tri(N[, M[, k[, opts]]])`

Builds an `N` by `M` float array with ones on and below the `k`-th diagonal and
zeros above it, following `np.tri`. `M` defaults to `N`, `k` defaults to `0`,
and `opts` accepts `dtype`. Zero dimensions are allowed.

```lua
numlua.tri(3)
-- {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}}
numlua.tri(2, 4, 1, { dtype = "int64" })
-- {{1, 1, 0, 0}, {1, 1, 1, 0}}
```

### `numlua.vander(x[, N[, increasing]])`

Builds the Vandermonde matrix from a 1-D tensor or Lua numeric table. `N`
defaults to `length(x)` and must be a non-negative integer; `increasing` is an
optional boolean. The result keeps the input dtype and is materialized.

```lua
numlua.vander(numlua.array({ 1, 2, 3 }), 4)
-- columns are x^3, x^2, x, 1
```

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

### `numlua.logspace(start, stop[, num[, base[, opts]]])`

Builds `num` values spaced evenly on a log scale, with `num = 50` and
`base = 10` by default. The result defaults to `float32`; `opts` accepts
`dtype` and `endpoint` (default `true`). Exponents are computed in `float64`
before the requested dtype cast. `num = 0` returns an empty 1-D tensor.

```lua
numlua.logspace(0, 2, 3)                    -- {1, 10, 100}
numlua.logspace(0, 3, 4, 2)                 -- {1, 2, 4, 8}
numlua.logspace(0, 2, 3, 10, { endpoint = false })
```

### `numlua.geomspace(start, stop[, num[, opts]])`

Builds `num` values spaced evenly in log space, with `num = 50`. Both endpoints
must be finite and non-zero and must share a sign; negative endpoints produce a
negative geometric sequence, as in NumPy. `opts` accepts `dtype` and
`endpoint`.

```lua
numlua.geomspace(1, 1000, 4)  -- {1, 10, 100, 1000}
numlua.geomspace(-1, -100, 3) -- {-1, -10, -100}
```

### `numlua.meshgrid(x1, x2, ...[, opts])`

Builds coordinate grids from 1-D tensors or Lua numeric tables and returns a
table with one tensor per input. With the default `indexing = "xy"` the first
two output axes are swapped; use `indexing = "ij"` to keep input order. The
optional trailing options table accepts:

| option | default | meaning |
| --- | --- | --- |
| `indexing` | `"xy"` | `"xy"` swaps the first two output axes; `"ij"` keeps input order |
| `sparse` | `false` | keep size-1 axes away from each input's own axis |
| `copy` | `true` | allocate independent storage; `false` returns expanded views |

Each output keeps its input dtype. A single input returns a one-element table
with that input's shape. At least one 1-D input is required, and
`indexing`/`sparse`/`copy` must have their documented types.

```lua
local x, y = numlua.array({ 1, 2, 3 }), numlua.array({ 10, 20 })
local gx, gy = unpack(numlua.meshgrid(x, y, { indexing = "ij" }))
-- gx: 3x2, gy: 3x2
```

### `numlua.zeros_like(t[, opts])`, `numlua.ones_like`, `numlua.empty_like`, `numlua.full_like`

The `*_like` constructors use `t` for shape and dtype unless `opts` overrides
them, matching NumPy's `np.zeros_like` family. `opts.dtype` is a dtype name or
luatorch dtype id; `opts.shape` is a dimension or a shape table, where `{}`
requests a 0-D result and `0` requests a length-0 vector.
`full_like(t, fill_value[, opts])`
requires a numeric fill value. `empty_like` leaves the new storage
uninitialized and allocates it in C order; the other three initialize every
element and do not alias the prototype. 0-D prototypes are supported.

```lua
local a = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
numlua.zeros_like(a)                       -- int64 2x2 of zeros
numlua.ones_like(a, { dtype = "float64" }) -- float64 2x2 of ones
numlua.full_like(a, 7, { shape = { 4 } })  -- int64 {7, 7, 7, 7}
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

### `numlua.diag(v[, k])`

With a 1-D input, constructs a square matrix with `v` on the `k`-th diagonal
and zeros elsewhere. `k = 0` is the main diagonal, positive values are above
it, and negative values are below it. The result keeps the input dtype and has
shape `{ n + abs(k), n + abs(k) }`.

With a 2-D input, extracts the `k`-th diagonal as a 1-D tensor; other ranks are
rejected. `bool` input is supported on both paths. The result owns independent
storage.

```lua
numlua.diag(numlua.array({ 1, 2, 3 }))
-- {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}
numlua.diag(numlua.array({ { 1, 2 }, { 3, 4 } }), -1)
-- {3}
```

### `numlua.diagflat(t[, k])`

Flattens any-rank input in C order and then constructs the diagonal matrix,
matching `np.diagflat`. A scalar produces a `1 x 1` matrix.

```lua
numlua.diagflat(numlua.array({ { 1, 2 }, { 3, 4 } }))
-- a 4 x 4 matrix with 1, 2, 3, 4 on the main diagonal
```

### `numlua.diagonal(t[, offset[, axis1[, axis2]]])`

Extracts the `offset`-th diagonal from the plane spanned by `axis1` and
`axis2`, appending the diagonal as the last dimension after any batch
dimensions. Axes are 1-based, default to `1` and `2`, may be negative, and must
be different; the input must be at least 2-D. The offset follows `diag`:
positive is above the main diagonal and negative is below it. Non-empty numeric
diagonals alias the input, matching NumPy's view (NumPy marks the view
read-only; luatorch does not). `bool` input is supported by casting through
`uint8`, so bool diagonals are materialized. An out-of-range offset returns an
empty last dimension.

```lua
local cube = numlua.arange(24):reshape(2, 3, 4)
numlua.diagonal(cube, 0, 1, 3) -- shape {3, 2}: {{0, 13}, {4, 17}, {8, 21}}
```

### `numlua.trace(t[, offset[, axis1[, axis2]]])`

Sums the diagonal selected by `diagonal` and returns a tensor. A 2-D input
produces a 0-D tensor; higher-rank input keeps the batch dimensions, so the
above 3-D example returns shape `{3}`. The dtype follows the delegated sum, so
`bool` and narrow integer diagonals use luatorch/ATen's reduction dtype rather
than NumPy's default integer promotion.

```lua
numlua.trace(numlua.array({ { 1, 2 }, { 3, 4 } })) -- 0-D tensor containing 5
```

### `numlua.tril(t[, k])`, `numlua.triu(t[, k])`

Return a copy with the elements above (`tril`) or below (`triu`) the `k`-th
diagonal zeroed. For N-D input the operation applies to the last two
dimensions, and the result has the input shape and dtype. `k = 0` includes the
main diagonal, positive values select a diagonal above it, and negative values
select one below it.

```lua
local m = numlua.array({ { 1, 2 }, { 3, 4 } })
numlua.tril(m)      -- {{1, 0}, {3, 4}}
numlua.triu(m)      -- {{1, 2}, {0, 4}}
numlua.tril(m, 1)   -- {{1, 2}, {3, 4}} (nothing below the superdiagonal is zeroed)
```

## Combining

### `numlua.concatenate(tensors[, axis])`

Concatenates a non-empty table of tensors along `axis` (default `1`).

### `numlua.stack(tensors[, axis])`

Stacks a non-empty table of same-shaped tensors along a new `axis` (default
`1`).

### `numlua.split(t, sections[, axis])`
### `numlua.array_split(t, sections[, axis])`

Split a tensor along `axis` (default `1`) and return the resulting table of
tensors. An integer `sections` divides the axis: `split` requires an even
division and raises a `numlua.split:` error otherwise, while `array_split`
distributes the remainder over the leading chunks, exactly like NumPy.

A table of `sections` contains split boundaries. Boundaries are **1-based**:
with `axis = 1`, `{ 2, 4 }` returns elements `1..1`, `2..3`, and `4..end`.
Negative boundaries count from the end (`-1` splits before the last element),
and out-of-range or non-monotonic boundaries are clamped or produce empty
chunks instead of errors, following NumPy. An empty boundary table returns the
whole input as one chunk. The chunks are views into the original tensor.

```lua
local v = numlua.array({ 1, 2, 3, 4, 5 })
local a, b, c = table.unpack(numlua.split(v, { 2, 4 }))
-- a = {1}, b = {2, 3}, c = {4, 5}
local first, second, third = table.unpack(numlua.array_split(v, 3))
-- shapes 2, 2, 1
```

### `numlua.hsplit(t, sections)`, `vsplit`, `dsplit`

Convenience splitters with NumPy's axis choices translated to 1-based axes:

| function | input rank | axis |
| --- | --- | --- |
| `hsplit` | 1-D | `1` |
| `hsplit` | 2-D or higher | `2` (columns) |
| `vsplit` | 2-D or higher | `1` (rows) |
| `dsplit` | 3-D or higher | `3` (depth) |

`vsplit` and `dsplit` reject input below their required rank with a
`numlua.vsplit:`/`numlua.dsplit:` error. All three use `split`'s even-division
rule for an integer section count and its boundary-table rule otherwise.

## Reductions

### `numlua.sum(t[, axis[, keepdims]])`

### `numlua.mean(t[, axis[, keepdims]])`

All-axis reductions return a 0-D tensor. With `axis`, the reduced axis is
removed unless `keepdims` is true. `keepdims` must be a boolean when provided.

`mean` returns `float32` for `float32` input and `float64` for `float64` input.
Because ATen refuses to infer a dtype for integer/bool `mean`, numlua promotes
non-floating input to `float64` first.

### `numlua.prod(t[, axis[, keepdims]])`

Multiplies elements over the selected axis, with the same `axis`/`keepdims`
rules as `sum`. An empty reduction returns the multiplicative identity `1`.
The dtype follows ATen's product reduction; bool input produces `int64`.

### `numlua.var(t[, axis[, keepdims[, ddof]]])`, `numlua.std`

Return the variance or standard deviation as a tensor, including a 0-D tensor
for an all-axis reduction. `ddof` defaults to `0` (population variance), unlike
luatorch's unbiased default; any finite number is accepted, and denominators
`N - ddof` are computed from the reduced axis. Integer and bool input promotes
to `float64`, while floating input keeps its dtype. `keepdims` follows the
other reductions. `std` is the square root of `var`.

### `numlua.nansum`, `numlua.nanprod`, `numlua.nanmean`, `numlua.nanmin`, `numlua.nanmax`, `numlua.nanvar`, `numlua.nanstd`

NaN-aware counterparts of `sum`, `prod`, `mean`, `min`, `max`, `var`, and
`std`. They accept the same `axis`/`keepdims` arguments; the variance pair also
accepts `ddof`. `nansum` treats NaN as zero and `nanprod` treats NaN as one, so
an all-NaN product reduces to `1`; the other reductions ignore NaN values and
return NaN when a slice has no valid values. Integer input follows
the ordinary reduction aliases (no promotion for `nansum`/`nanprod`/`nanmin`/
`nanmax`, `float64` for `nanmean`/`nanvar`/`nanstd`); floating input keeps its
dtype.

### `numlua.nan_to_num(t[, nan[, posinf[, neginf]]])`

Replaces NaN and infinities with finite values. With no replacement arguments,
ATen uses the dtype's maximum finite value for positive and negative infinity
and `0` for NaN. Replacement values must be numbers or nil; integer and bool
input is returned unchanged.

```lua
numlua.nan_to_num(numlua.array({ 0 / 0, math.huge }), 7, 8, 9)
-- {7, 8}
```

### `numlua.nancumsum(t[, axis])`, `numlua.nancumprod`

NaN-aware cumulative reductions. NaN is treated as zero for `nancumsum` and as
one for `nancumprod`; an omitted axis flattens the input, while a 1-based axis
accumulates along that dimension. Integer input follows the ordinary cumulative
reduction and floating input keeps its dtype.

### `numlua.max(t[, axis[, keepdims]])`

### `numlua.min(t[, axis[, keepdims]])`

Return values only. With `axis`, luatorch also computes indices, but `numlua`
discards them.

### `numlua.ptp(t[, axis[, keepdims]])`

Returns the peak-to-peak range `max - min` over the selected axis, using the
same `axis`/`keepdims` rules as `max`/`min`. The dtype follows the subtraction,
so it matches the input for numeric dtypes; bool input is cast to `int64`.
NaN values propagate through the extrema.

### `numlua.trapezoid(t[, x[, dx[, axis]]])`, `numlua.trapz`

Integrates `t` along `axis` with the trapezoidal rule, following
`np.trapezoid` (and its historical `np.trapz` spelling). The axis defaults to
the last one and is 1-based. With `x` omitted, `dx` is the uniform spacing
(default `1`). With `x`, the coordinate array may be 1-D and match the selected
axis (it is expanded across the other dimensions) or have the input's full
shape. Empty and single-point intervals return zero. Integer input follows
ATen's arithmetic promotion (`float32`), while floating input keeps its dtype.

```lua
numlua.trapezoid(numlua.array({ 1, 2, 3 }))          -- 4
numlua.trapezoid(numlua.array({ 1, 2, 3 }), nil, 2)  -- 8
numlua.trapezoid(numlua.array({ 1, 2, 3 }), { 0, 1, 3 }) -- 6.5
```

### `numlua.interp(x, xp, fp[, left[, right]])`

Linear interpolation of the 1-D strictly increasing grid `xp`/`fp` at `x`.
`x` may be a number, tensor, or table and keeps its shape; the result is
`float64`. Values outside the grid use `left`/`right`, defaulting to `fp[1]`
and `fp[-1]`. NaN coordinates produce NaN.

```lua
numlua.interp(2, numlua.array({ 1, 3, 5 }), numlua.array({ 10, 30, 50 }))
-- 20 (0-D float64 tensor)
```

### `numlua.unwrap(t[, discont[, axis[, period]]])`

Removes phase jumps larger than `discont` along a 1-based axis by adding
multiples of `period`, following `np.unwrap`. `axis` defaults to the last
dimension, `period` defaults to `2*pi`, and `discont` defaults to
`period/2`. Integer input is promoted to `float64`; the result owns independent
storage. `period` must be positive and finite, and `discont` must be finite
and non-negative.

```lua
numlua.unwrap(numlua.array({ 0, 3.5, 7 })) -- {0, -2.7832, -5.5664}
```

### `numlua.argmax(t[, axis[, keepdims]])`

Returns an `int64` tensor of 1-based indices. If `axis` is omitted, the result
is the flat C-order index of the maximum (a 0-D tensor). With `axis`, the
reduced axis is removed unless `keepdims` is true.

### `numlua.median(t[, axis[, keepdims]])`

Returns the median. With no axis this is a Lua number; with an axis it returns
a tensor with the reduced axis removed unless `keepdims` is true (when
supported by the underlying luatorch median path).

### `numlua.quantile(t, q[, axis[, keepdims]])`

Returns the `q`-quantile using linear interpolation between order statistics.
`q` may be a number in `[0, 1]` or a non-empty table of probabilities; a table
returns a tensor with the quantile axis first. With no axis the input is flattened and a Lua number
is returned. With an axis the reduced axis is removed unless `keepdims` is
true.

### `numlua.searchsorted(sorted, values[, side])`

Binary-search insertion positions in a 1-D sorted tensor. `side` is `"left"`
(default) or `"right"`. Scalar values return a number; tables/tensors return an
`int64` tensor.

### `numlua.bincount(input[, weights[, minlength]])`

Count non-negative integer values; delegates to luatorch's `bincount`.

### `numlua.percentile(t, q[, axis[, keepdims]])`

Percent-scaled alias for `numlua.quantile`; `q` is in `[0, 100]` and may also
be a table of percentages.

### `numlua.compress(condition, t[, axis])`

Selects elements where `condition` is true, following `np.compress`.
`condition` may be a boolean tensor/table or a numeric tensor/table, in which
case nonzero values are true. With no `axis`, the input is flattened in C order
and a short condition selects a prefix of it. With a 1-based `axis`, the
condition must be 1-D and selects slices along that axis; a condition shorter
than the axis is used as a prefix, while a longer condition raises a
`numlua.compress:` error. The result owns independent storage and keeps the
input dtype.

```lua
local t = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
numlua.compress({ true, false }, t)          -- {10} (flat prefix)
numlua.compress({ true, false, true }, t, 2) -- {{10, 30}, {40, 60}}
```

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

### `numlua.select(condlist, choicelist[, default])`

NumPy's `np.select`: `condlist` and `choicelist` are non-empty Lua tables of the
same length. Each condition follows `where`'s truth testing (a `bool` tensor is
used directly, a numeric tensor or number is tested as `~= 0`, and a boolean
scalar is accepted), while each choice may be a tensor or a Lua
`number`/`boolean`. Conditions are tested in order, the first true condition
wins, and elements where every condition is false take `default`, which
defaults to `0`. All operands broadcast through ATen; tensor choices keep their
own dtype under ATen's promotion, while scalar choices and a scalar default are
materialized as 0-D `float32` (or `bool`) tensors, exactly like `where` scalar
branches. Wrap a scalar in a typed array when a specific dtype is required.
Empty condition/choice tensors are valid when they broadcast, and 0-D
conditions, choices, or defaults behave like `where` scalars. Empty or
length-mismatched lists, non-tensor/non-scalar entries, and choices that do not
broadcast raise a `numlua.select:` error.

```lua
local values = numlua.array({ -2, -1, 0, 1, 2 })
local negative = values:lt(0)
local positive = values:gt(0)
numlua.select({ negative, positive }, { -1, 1 }, 0)
-- {-1, -1, 0, 1, 1}
```

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

### `numlua.isclose(a, b[, rtol[, atol[, equal_nan]]])`

Returns a `bool` tensor using NumPy's closeness rule
`|a - b| <= atol + rtol * |b|`, with `rtol = 1e-5` and `atol = 1e-8` by
default. Operands may be tensors or Lua numbers and broadcast through ATen.
Equal infinities are close, opposite or finite/infinite pairs are not, and a
NaN is close only when `equal_nan = true`. Mixed or integral dtypes are
compared through `float64`; equal floating dtypes keep their precision.
`rtol` and `atol` must be finite and non-negative. The result shape follows
ATen broadcasting.

```lua
numlua.isclose(numlua.array({ 1, 2, math.huge }),
  numlua.array({ 1 + 1e-6, 2.5, math.huge }))
-- {true, false, true}
```

### `numlua.allclose(a, b[, rtol[, atol[, equal_nan]]])`

Returns a Lua boolean that is true when `numlua.isclose` is true for every
broadcast element, matching `np.allclose`. It is a scalar predicate, not a
tensor reduction; use `:item()` on `isclose` when a tensor is wanted.

```lua
assert(numlua.allclose(numlua.array({ 1, 2 }), numlua.array({ 1, 2 })))
```

### `numlua.array_equal(a, b[, equal_nan])`

Returns a Lua boolean. Shapes must match exactly, every element must compare
equal, and NaN is unequal unless `equal_nan = true`. Dtype differences are
allowed when the values compare equal, as in NumPy.

### `numlua.array_equiv(a, b)`

Returns a Lua boolean. Unlike `array_equal`, shapes only need to be
broadcast-compatible, so a scalar is equivalent to a matching array. Values
must still all compare equal. Incompatible shapes return `false` rather than
raising, matching `np.array_equiv`; NaN never compares equal.

```lua
assert(numlua.array_equiv(1, numlua.array({ 1, 1, 1 })))
assert(not numlua.array_equiv(numlua.array({ 1, 2 }),
  numlua.array({ 1, 2, 3 })))
```

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
| `numlua.log1p(t)` | log(1 + x) |
| `numlua.sin(t)` / `numlua.cos(t)` / `numlua.tanh(t)` | trigonometric and hyperbolic functions |
| `numlua.square(t)` | elementwise square |
| `numlua.sign(t)` | sign (`-1`, `0`, `1`) |
| `numlua.floor(t)` / `numlua.ceil(t)` / `numlua.trunc(t)` | rounding toward -infinity, +infinity, and zero |
| `numlua.round(t)` | round half to even, following ATen |
| `numlua.reciprocal(t)` | multiplicative inverse |
| `numlua.isinf(t)` / `numlua.isfinite(t)` | boolean infinity/finiteness masks |
| `numlua.logical_not(t)` | boolean negation |
| `numlua.logical_and(a, b)` / `numlua.logical_or(a, b)` / `numlua.logical_xor(a, b)` | boolean logic over truth-tested operands |

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

Lua booleans are not accepted as scalar operands by the arithmetic and
transcendental ufuncs; pass a `bool` tensor if you need boolean tensor
arithmetic. The logical ufuncs are the exception: they accept booleans,
numbers, and tensors and return `bool` tensors. Non-numeric/non-tensor operands
raise a `numlua.<function>:` error. Incompatible broadcast shapes and unsupported
dtype combinations are reported by luatorch/ATen and forwarded as-is.

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

Unary operations split into four groups:

- `negative`, `abs`, and `square` preserve an integral input dtype. ATen does
  not implement `abs` or negation for `bool` tensors; those calls error;
- `sign`, `floor`, `ceil`, `trunc`, and `round` preserve numeric input dtypes;
- `sqrt`, `exp`, `log`, `log1p`, `sin`, `cos`, `tanh`, and `reciprocal`
  produce `float32` for integer/bool input and preserve `float32`/`float64`
  floating input. NumPy would normally use `float64` for integral input;
- `isinf` and `isfinite` always return `bool` tensors. The logical ufuncs
  truth-test their operands and return `bool` tensors.

For `maximum`/`minimum` with a scalar, numlua materializes a 0-D operand with
the input's dtype for integral scalars (or `int64` for a `bool` input) and
`float32` for fractional/non-finite scalars. This only supplies the missing
scalar overload; the comparison itself is still ATen's broadcast kernel.

There are no `out=`, `where=`, `dtype=`, or `casting=` keyword arguments, and no
separate ufunc objects. Use `numlua.astype` for an explicit result dtype.

## Linear algebra

### `numlua.dot(a, b)`

Follows NumPy's `dot` shape rules: two vectors produce the 0-D inner product,
matrix/vector and matrix/matrix products behave like `matmul`, and N-D
operands sum over the last axis of `a` and the second-to-last axis of `b`.
ATen's `matmul` performs the multiplication, so mixed integer/floating dtypes
follow ATen's matmul checks rather than NumPy promotion.

### `numlua.inner(a, b)`

Sums the product over the last axis of both operands and returns the leading
dimensions concatenated, matching `np.inner`. A 1-D pair produces a 0-D tensor;
an empty last axis produces zeros with the concatenated leading shape.

```lua
numlua.inner(numlua.array({ 1, 2 }), numlua.array({ 3, 4 })) -- 11
```

### `numlua.outer(a, b)`

Flattens both operands in C order and returns their outer product, matching
`np.outer`. Scalars become length-1 vectors, so `outer(2, 3)` is a 1x1 matrix.

### `numlua.cross(a, b[, axis])`

Computes the 3-vector cross product over a 1-based `axis`, defaulting to the
last axis. Both operands must have the same non-zero rank and length 3 on the
selected axis; batch dimensions broadcast through ATen. The result keeps the
input dtype. NumPy's 2-component cross product is not supported.

```lua
numlua.cross(numlua.array({ 1, 2, 3 }), numlua.array({ 4, 5, 6 }))
-- {-3, 6, -3}
```

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
| `numlua.diag(v[, k])` | construct from a vector or extract a diagonal (materialized) |
| `numlua.diagflat(t[, k])` | flatten and construct a diagonal matrix (materialized) |
| `numlua.diagonal(t[, offset[, axis1[, axis2]]])` | extract a diagonal over two axes (view for non-empty numeric input; bool is materialized) |
| `numlua.trace(t[, offset[, axis1[, axis2]]])` | sum the selected diagonal, keeping batch axes |
| `numlua.tril(t[, k])` / `numlua.triu(t[, k])` | zero one triangle of the last two axes (materialized) |
| `numlua.split(t, sections[, axis])` | split along an axis; 1-based boundary tables (views) |
| `numlua.array_split(t, sections[, axis])` | split into unequal sections (views) |
| `numlua.hsplit` / `vsplit` / `dsplit` | NumPy column/row/depth splitters (views) |
| `numlua.broadcast_to(t, shape)` | broadcast view; NumPy-compatible right alignment |
| `numlua.broadcast_arrays(...)` | returns multiple broadcast views |
| `numlua.tile(t, reps)` | repeat whole blocks (materialized); `reps` is a number or table |
| `numlua.repeat_(t, repeats[, axis])` | repeat elements (materialized); `numlua["repeat"]` is the same function |
| `numlua.flip(t, axis)` | reverse one axis or a table of axes (materialized) |
| `numlua.fliplr(t)` | reverse axis 2; requires at least 2-D (materialized) |
| `numlua.flipud(t)` | reverse axis 1; requires at least 1-D (materialized) |
| `numlua.rot90(t[, k[, axes]])` | quarter-turn rotation; `k` is reduced modulo 4 and `axes` is a pair of 1-based axes (materialized; `k = 0` returns the input) |
| `numlua.roll(t, shift[, axis])` | cyclic shift; flattens when `axis` is omitted |
| `numlua.pad(t, pad_width[, mode[, value]])` | modes `"constant"`, `"edge"`, `"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`, and `"wrap"`; `value` is mode-specific (materialized) |
| `numlua.diff(t[, n[, axis]])` | default axis is the last |
| `numlua.gradient(t[, spacing[, axis]])` | first-order edge and central interior differences |
| `numlua.cumsum(t[, axis])` / `numlua.cumprod` | flatten when `axis` is omitted |
| `numlua.argsort(t[, axis[, descending]])` | 1-based indices; at least 1-D input |
| `numlua.sort(t[, axis[, descending]])` | sorted values; last axis by default; at least 1-D input |
| `numlua.unique(t[, options])` | sorted unique flattened values with optional index/inverse/counts |
| `numlua.nonzero(t)` | one 1-based coordinate tensor per input axis |
| `numlua.argwhere(t)` | `(N, ndim)` int64 matrix of 1-based non-zero coordinates |
| `numlua.take(t, indices[, axis])` | flatten when `axis` is omitted; scalar and N-D index tensors |
| `numlua.take_along_axis(t, indices[, axis])` | per-slice 1-based selection; flattens when `axis` is omitted |
| `numlua.apply_along_axis(func, axis, t[, ...])` | call `func` on each 1-D slice along `axis` (materialized) |
| `numlua.apply_over_axes(func, t, axes[, ...])` | apply `func` over axes, restoring each reduced axis |
| `numlua.put(t, indices, values[, mode])` | flat in-place writes; modes `"raise"`, `"wrap"`, `"clip"` |
| `numlua.flatnonzero(t)` | 1-based flat indices of non-zero entries |

`numlua.gradient(t[, spacing[, axis]])` computes first-order edge differences
and second-order central interior differences along every axis, or along a
1-based `axis` number/table. `spacing` may be a uniform number, a 1-D table or
tensor of coordinates for one selected axis, or a table/tensor with one
spacing per selected axis; coordinate tables may also be supplied per axis.
Coordinate values must be 1-D, match the axis size, and not repeat a position;
uniform spacings must be finite and non-zero. Floating input keeps its dtype,
integer/bool input promotes to `float64`. With `axis = nil`, a 1-D input returns
one tensor and higher-rank input returns a table in axis order; an integer
`axis` returns one tensor and an axis table returns a table in the requested
order. Every selected axis needs at least two elements.

```lua
local t = numlua.array({ 1, 4, 9, 16 }, { dtype = "float64" })
numlua.gradient(t)               -- {3, 4, 6, 7}
numlua.gradient(t, 2)            -- {1.5, 2, 3, 3.5}
numlua.gradient(t, {0, 1, 3, 6}) -- coordinate spacing
```

`broadcast_to` and `broadcast_arrays` follow NumPy's right-aligned rules,
including 0-D inputs and zero-size axes: a size-1 axis broadcast against a
zero-size axis produces a zero-size result, while mismatched non-1 axes raise
`numlua.broadcast_arrays: incompatible shapes ...`. `take` requires an integer
index tensor (`int64`, `int32`, `int16`, `int8`, or `uint8`). The index tensor
may have any rank, following NumPy's result-shape rule: the selected axis is
replaced by the index shape, a 0-D index removes that axis (a scalar result
when no axis is given), and an N-D index contributes all of its dimensions.
With no axis, the input is flattened in C order first and the result has
exactly the index shape. Empty index tensors are allowed, including empty N-D
shapes. Non-integer indices, index values outside `1 .. size`, and selecting
with non-empty indices from an empty tensor or empty axis raise a
`numlua.take:` error; range errors name the offending 1-based index, the axis
or flattened input, and its size. 0-D and non-contiguous inputs are flattened
in C order before selection. All selections are 1-based.

`numlua.argsort(t[, axis[, descending]])` returns 1-based indices along `axis`
(the last axis by default). `descending` is a numlua extension; when provided it
must be a boolean or `nil`, so a typo such as `1` or `"true"` raises an error
instead of silently sorting ascending.

`numlua.sort(t[, axis[, descending]])` returns the sorted values, using the
last axis by default like `np.sort`. `axis` may be negative. Passing
`descending = true` is a numlua extension that matches
`numlua.argsort(t, axis, true)`, and the result owns independent storage.
Inputs below 1-D are rejected, matching `argsort`; use `numlua.ravel` first if a
flat sort is wanted.

```lua
local a = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })
numlua.sort(a)          -- {{1, 2, 3}, {4, 5, 6}}
numlua.sort(a, 2, true) -- {{3, 2, 1}, {6, 5, 4}}
```

`numlua.unique(t[, options])` flattens the input in C order and returns the
sorted unique values, following `np.unique` without an `axis` argument. The
input may be 0-D; the result is always 1-D. The optional `options` table
accepts `return_index`, `return_inverse`, and `return_counts`, all booleans.
Requested results are returned after the values in NumPy's fixed order:
`values[, index[, inverse[, counts]]]`. `index` contains the 1-based flat
position of the first occurrence of each value, `inverse` has the input's
shape (a 1-D shape for 0-D input) and maps each position to its 1-based slot
in `values`, and `counts` gives the multiplicity of each value. All index
results are `int64`; `values` keeps the input dtype. Unknown options and
non-boolean flags raise a `numlua.unique:` error. There is no `axis`,
`equal_nan`, or unsorted mode.

```lua
local a = numlua.array({ { 3, 1, 2 }, { 3, 2, 1 } })
local values, index, inverse, counts = numlua.unique(a, {
  return_index = true,
  return_inverse = true,
  return_counts = true,
})
-- values  = {1, 2, 3}
-- index   = {2, 3, 1}
-- inverse = {{3, 1, 2}, {3, 2, 1}}
-- counts  = {2, 2, 2}
```

`numlua.nonzero(t)` returns one `int64` coordinate tensor per input axis,
matching `np.nonzero` with 1-based coordinates. The input must be at least 1-D;
columns are enumerated in C order. Because the number of return values depends
on the input rank, assign one variable per axis (for example
`local rows, columns = numlua.nonzero(a)`). `numlua.argwhere(t)` returns the
same coordinates as a single `(N, ndim)` `int64` matrix, including `(0, ndim)`
for empty input. Both functions treat NaN and any non-zero value as non-zero.
Use `numlua.flatnonzero` for a flat index vector or for 0-D input.

```lua
local a = numlua.array({ { 0, 2, 0 }, { 3, 0, 4 } })
local rows, columns = numlua.nonzero(a)
-- rows    = {1, 2, 2}
-- columns = {2, 1, 3}
numlua.argwhere(a) -- {{1, 2}, {2, 1}, {2, 3}}
```

`numlua.take_along_axis(t, indices[, axis])` is the per-slice companion to
`take`, matching `np.take_along_axis`. `indices` must be an integer tensor with
the same rank as `t` and the usual integer index dtypes (`int64`, `int32`,
`int16`, `int8`, or `uint8`). `axis` selects the axis to index; when it is
omitted or `nil`, both operands are flattened in C order and `indices` must be
1-D. Non-axis dimensions may match or be 1 in either operand, following
NumPy's broadcasting rule, and the length along `axis` comes from `indices`; it
need not equal the input's length. Indices are 1-based, so `0` and negative
values are out of range and raise a `numlua.take_along_axis:` error rather than
NumPy's negative-index wrapping. The result has the broadcast shape and owns
independent storage.

```lua
local a = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
numlua.take_along_axis(a, numlua.argsort(a, 2), 2)
-- {{10, 20, 30}, {40, 50, 60}}
```

`numlua.apply_along_axis(func, axis, t[, ...])` mirrors
`np.apply_along_axis` for Lua callbacks. `axis` is required and 1-based; it may
be negative. `func` receives a 1-D view of each slice along `axis`, followed by
any extra arguments, and must return a scalar or a 1-D result as a luatorch
tensor, Lua number, boolean, or Lua table. Returning a tensor with rank greater
than one, mixing scalar and 1-D results, or changing the result length between
calls raises a `numlua.apply_along_axis:` error. Callback errors propagate
unchanged.

The result follows NumPy's shape rule: the non-axis input dimensions come
first, and a 1-D callback result appends its length after them. A scalar
callback therefore removes the axis, a 1-D input with a scalar callback
produces a 0-D tensor, and a 1-D callback on a 1-D input produces a 1-D result.
Slices are visited in C order, with the last non-axis dimension varying
fastest. The output dtype is taken from the first callback result; later
results are cast to it. The output owns independent storage, so callbacks may
return views of the input without aliasing the source.

An empty selected axis is allowed and calls `func` on an empty 1-D slice. A
zero-size non-axis dimension has no slices, so there is no first result from
which to infer a result shape or dtype; that case raises a
`numlua.apply_along_axis:` error instead of guessing.

```lua
local rows = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
numlua.apply_along_axis(function(row) return numlua.flip(row, 1) end, 2, rows)
-- {{3, 2, 1}, {6, 5, 4}}
numlua.apply_along_axis(function(row) return numlua.sum(row) end, 1, rows)
-- {5, 7, 9}
```

`numlua.apply_over_axes(func, t, axes[, ...])` mirrors `np.apply_over_axes`
for Lua callbacks. `axes` is a 1-based axis or a table of axes; entries may be
negative and may repeat. The axes are applied in order to the current value,
and `func` is called as `func(value, axis, ...)` with the normalized 1-based
axis and any extra positional arguments (the extra arguments are a numlua
extension; NumPy forwards none).

The callback follows `apply_along_axis`'s return spellings: a luatorch tensor,
Lua number, boolean, or Lua table. The rank rule matches NumPy. A result with
the same rank as the current value is used unchanged, which supports
keepdims-style reductions and same-shape callbacks such as `numlua.cumsum`. A
result with one fewer dimension is re-expanded with a size-1 axis at the same
1-based position, so ordinary reductions such as `numlua.sum` behave like a
`keepdims` reduction. Any other rank difference raises a
`numlua.apply_over_axes:` error; 0-D input is rejected because there is no axis
to apply. An empty `axes` table returns the input unchanged.

Because every reduced axis is restored, the result has the same rank as the
input (before any same-rank callback changes sizes). Reductions over a zero-size
axis therefore produce a size-1 axis, not an error. Callback errors propagate
unchanged. The helper delegates each call to the callback, so result storage and
dtype follow whatever the callback returns; a callback that returns a view can
keep the result aliased to the input.

```lua
local cube = numlua.arange(24):reshape(2, 3, 4)
numlua.apply_over_axes(numlua.sum, cube, { 1, 2 })
-- shape {1, 1, 4}, a keepdims-style sum over the first two axes
numlua.apply_over_axes(numlua.sum, cube, { -1 })
-- shape {2, 3, 1}
numlua.apply_over_axes(numlua.cumsum, cube, { 3 })
-- shape {2, 3, 4}, equal to numlua.cumsum(cube, 3)
```

`numlua.put(t, indices, values[, mode])` is the assignment counterpart to
`take`. It writes into the flattened elements selected by 1-based `indices`,
mutates `t` in place, and returns `t`; clone first if a copy is wanted.
Indices may be an integer tensor (`int64`, `int32`, `int16`, `int8`, or
`uint8`), a number, or a nested Lua table of integers. Index values must be
finite integers: NaN and infinities are rejected in every mode. Values may be a
tensor, number, boolean, or nested Lua table; they are flattened in C order and
reused cyclically when there are fewer values than indices, matching `np.put`.
Duplicate indices are applied in order, so the last write wins.

`mode` controls indices outside `1 .. t:numel()`:

| mode | behavior |
| --- | --- |
| `"raise"` (default) | error naming the offending index |
| `"wrap"` | wrap cyclically, so `0` selects the last element and `numel + 1` selects the first |
| `"clip"` | clamp to the first or last element |

Because the facade is 1-based, `0` and negative indices are out of range in
`"raise"` mode. Values are cast to the target dtype: booleans become `1`/`0`
for numeric targets, numeric values truth-test for a `bool` target, and
fractional values truncate toward zero for integer targets. Putting into an
empty tensor with non-empty indices, using empty values with non-empty indices,
or passing a non-finite index raises a `numlua.put:` error.

```lua
local t = numlua.arange(6)                      -- {0, 1, 2, 3, 4, 5}
numlua.put(t, { 1, 6 }, { 10, 60 })            -- {10, 1, 2, 3, 4, 60}
numlua.put(t, { 8 }, -1, "wrap")               -- writes index 2
numlua.put(t, { 99 }, 0, "clip")               -- writes index 6
```

`pad_width` follows NumPy's common forms: a number pads every axis equally, a
`{before, after}` pair pads every axis with that pair, and a table of per-axis
pairs pads each axis separately. `"edge"` repeats the first and last values,
`"reflect"` mirrors without repeating the edge, `"symmetric"` mirrors with the
edge included, and `"wrap"` repeats each axis cyclically, accepting widths
larger than the axis. `"linear_ramp"` fills each side with a linear ramp from
`value` to the adjacent edge value, excluding the edge itself, so it matches
`np.pad(..., mode="linear_ramp")`. `"maximum"`, `"mean"`, `"median"`, and
`"minimum"` fill each side with that statistic of the axis. Statistic modes
apply axes in order, so a later axis sees the padding added to earlier axes
(the NumPy corner rule). Zero-size axes can be extended only with
`"constant"`; `"wrap"` may leave an empty axis at zero width but cannot extend
it.

The optional `value` argument is mode-specific and follows NumPy's keyword
forms with 1-based axes:

| mode | `value` |
| --- | --- |
| `"constant"` (default) | `constant_values`; defaults to `0` |
| `"linear_ramp"` | `end_values`; defaults to `0` |
| `"maximum"`, `"mean"`, `"median"`, `"minimum"` | `stat_length`; defaults to the whole axis |
| `"edge"`, `"reflect"`, `"symmetric"`, `"wrap"` | not accepted |

For `constant_values` and `end_values`, a number fills every side, a
`{before, after}` pair broadcasts over every axis, and a table with one entry
per axis gives each axis its own number or pair. `stat_length` uses the same
forms with positive integers; a length larger than the axis is clamped to the
axis, and the statistic is computed from the first or last `stat_length`
values on that side. Integer end values and statistics round half to even
before the result is cast back to the input dtype, matching NumPy; bool input
supports every statistic mode.

```lua
local v = numlua.array({ 1, 2, 3 })
numlua.pad(v, { 1, 2 }, "constant", { 9, 8 })
-- {9, 1, 2, 3, 8, 8}
numlua.pad(v, 2, "wrap")
-- {2, 3, 1, 2, 3, 1, 2}
numlua.pad(v, 2, "maximum")
-- {3, 3, 1, 2, 3, 3, 3}
numlua.pad(numlua.array({ 1, 2, 3, 4, 5 }), { 2, 3 }, "linear_ramp", { 5, -4 })
-- {5, 3, 1, 2, 3, 4, 5, 2, -1, -4}

local m = numlua.array({ { 1, 2 }, { 3, 4 } })
numlua.pad(m, { { 1, 0 }, { 0, 1 } }, "constant", { { 10, 11 }, { 20, 21 } })
-- {{10, 10, 10}, {1, 2, 21}, {3, 4, 21}}
```

### View and copy boundaries

Views share storage with the input, so mutating one is visible through the
other: `transpose`, `swapaxes`, `moveaxis`, `atleast_1d`, `atleast_2d`,
`atleast_3d`, `squeeze`, `unsqueeze`, `broadcast_to`, `broadcast_arrays`,
contiguous `reshape`, and contiguous `ravel`.

`diagonal` returns a writable view for non-empty numeric input (a copy for
`bool` input or an empty diagonal); `trace` only allocates its result.

`split`, `array_split`, `hsplit`, `vsplit`, and `dsplit` return views created
with `narrow`. These allocate independent storage: `resize`, `diag`,
`diagflat`, `tril`, `triu`, `tri`, `vander`, `compress`, `pad`, `tile`,
`repeat_`, `flip`, `fliplr`,
`flipud`, `rot90`, `roll`, `diff`, `gradient`, `unwrap`, `interp`,
`nan_to_num`, `nancumsum`, `nancumprod`, `cumsum`, `cumprod`, `argsort`,
`sort`,
`unique`, `nonzero`, `argwhere`, `take`, `take_along_axis`, and
`apply_along_axis`. `rot90(t, 0)` is the one exception and returns its input
unchanged. `reshape` and
`ravel` also copy when their input is not contiguous, because ATen must
materialize a C-order layout first. Tests cover
the view/copy boundary, including large arrays, so a regression cannot silently
turn a view into a full-size copy or make an allocating helper alias its input.
`take` and `take_along_axis` allocate independent results; `put` instead
mutates its input in place and returns it.

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

### `numlua.save(path_or_file, tensor[, options])`

Writes a NumPy `.npy` file in C order to a path or an already-open Lua file
handle and returns `true`. The default format is
v1.0. Pass `options.version = 2` or `3` (the strings `"2.0"` and `"3.0"` are
accepted too) to write v2.0/v3.0 instead. Those versions replace v1.0's 2-byte
header length with a 4-byte one; v3.0 additionally permits UTF-8 header text,
though this writer only emits ASCII keys. Unknown options and invalid version
values raise a `numlua.save:` error before any output is written. A v1.0
header above 65535 bytes is rejected; v2.0/v3.0 headers are bounded by their
32-bit length field. A caller-owned handle is written at its current position
and is never closed by `save`; close or reposition it yourself. A closed handle
is rejected. Readers that only understand v1.0 cannot open v2.0/v3.0
output, so request a newer version only when the consumer supports it.

The tensor is detached and made contiguous before writing; autograd metadata is
not stored. Supported dtypes:

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

### `numlua.load(path_or_file)`

Reads a NumPy `.npy` v1.0, v2.0, or v3.0 file from a path or an already-open
Lua file handle and returns a luatorch tensor.
v2.0/v3.0 differ from v1.0 only in the 4-byte header-length field (and v3.0's
UTF-8 header encoding), which `load` parses directly; `save` writes v1.0 by
default and can emit v2.0/v3.0 when asked. Supports 1-D, 2-D, N-D, and 0-D
arrays with the dtypes above.

Native-endian `descr` values (`=f2`, `=f4`, `=f8`, `=i2`, `=i4`, `=i8`, and
their byte-sized spellings) select the host byte order in NumPy. On the
supported little-endian hosts, `load` accepts them as aliases for the `<`
spellings; the data is still copied as little-endian bytes.

Rejected with explicit errors:

- big-endian `descr` values (`>...`);
- `fortran_order=True`;
- unsupported `descr` values;
- NPY versions other than 1.0, 2.0, and 3.0 (including non-zero minor
  versions);
- malformed headers, missing trailing newline, bad magic;
- data whose byte length does not match `shape * itemsize`;
- invalid targets, unreadable files, and closed file handles.

The header is validated before any tensor allocation. For a path, `load` reads
the complete file into memory; for an open handle, it reads from the current
position and leaves the handle open. If another NPY array follows in the same
handle, `load` positions the handle at the next array so subsequent calls can
read it. `save` and `load` never close caller-owned handles. Arrays with shape
`()` (0-D scalars), zero-sized dimensions, and non-contiguous source tensors
are supported; non-contiguous tensors are written in C order.

```lua
local handle = assert(io.open("arrays.npy", "w+b"))
numlua.save(handle, numlua.array({ 1, 2, 3 }))
handle:seek("set", 0)
local values = numlua.load(handle)
handle:close()
```

### `numlua.fromfile(path_or_file[, options])`

Reads elements from a path or an already-open Lua file handle and returns a
1-D tensor. Unlike `load`, there is no `.npy` header and therefore no dtype or
shape metadata: pass the matching `dtype` option. Options:

| option | default | meaning |
| --- | --- | --- |
| `dtype` | `"float64"` | element dtype, a luatorch name/id or supported `descr` |
| `count` | `-1` | number of elements to read; `-1` reads everything |
| `offset` | `0` | bytes to skip before reading |
| `sep` | `""` | `""` reads raw little-endian bytes; a non-empty string reads text |

The supported dtypes are the same as `save`: `float16`, `float32`, `float64`,
`int8`, `int16`, `int32`, `int64`, `uint8`, and `bool`.

With the default binary `sep = ""`, `fromfile` consumes exactly
`offset + count * itemsize` bytes from a caller-owned handle (or to the end
when `count = -1`), leaves the handle open and positioned after the data, and
never reads past the requested count. A path is opened and closed by the call.
Truncated input, a final byte count that is not a multiple of the dtype size
when reading to the end, unsupported dtypes, negative counts/offsets, and
closed handles raise a `numlua.fromfile:` error.

With a non-empty `sep`, the remaining text is parsed as numeric fields. A
whitespace-only separator such as `" "` or `"\t"` splits on runs of whitespace;
any other separator is matched literally, with surrounding whitespace trimmed
from each field. `count` limits how many fields are parsed (extra data is
ignored, and fewer fields than `count` is not an error). Text mode reads to EOF
even when `count` is given, so a caller-owned handle ends at EOF. Floating
dtypes accept `nan`, `inf`, `-inf`, and `infinity` spellings; integers
truncate toward zero and reject non-finite values; `bool` accepts
`True`/`False` (case-insensitive) and `1`/`0`. Empty interior fields, invalid
tokens, and unsupported dtypes raise a `numlua.fromfile:` error. Text files
still carry no shape or dtype metadata.

```lua
numlua.tofile("values.bin", numlua.array({ 1, 2, 3 },
  { dtype = "int32" }))
local values = numlua.fromfile("values.bin", { dtype = "int32" })
-- {1, 2, 3}, dtype int32
local middle = numlua.fromfile("values.bin", {
  dtype = "int32", offset = 4, count = 1,
})
-- {2}

numlua.tofile("values.txt", numlua.array({ 1, 2, 3 },
  { dtype = "int32" }), { sep = "," })
numlua.fromfile("values.txt", { dtype = "int32", sep = "," }) -- {1, 2, 3}
```

### `numlua.tofile(path_or_file, tensor[, options])`

Writes the tensor's C-order data to a path or an open Lua file handle and
returns `true`. Autograd metadata is not written, and a non-contiguous tensor
is materialized in C order first. Caller-owned handles are written at their
current position and are never closed.

| option | default | meaning |
| --- | --- | --- |
| `sep` | `""` | `""` writes raw little-endian bytes; a non-empty string writes text |
| `format` | dtype-specific | printf-style format for text mode; requires a non-empty `sep` |

In binary mode (`sep = ""`) the output is exactly the raw little-endian
payload. In text mode each C-order value is formatted, joined with `sep`, and
followed by one trailing newline (an empty tensor writes nothing). The default
formats are `%.5g`/`%.9g`/`%.17g` for `float16`/`float32`/`float64`, `%d` for
integer dtypes, and `True`/`False` for `bool`; a custom `format` receives the
numeric value (`1`/`0` for `bool`). `format` must be a non-empty string and
requires a non-empty `sep`. Unsupported dtypes, invalid formats, invalid
targets, and write failures raise a `numlua.tofile:` error.

Raw and text files carry no shape or dtype, so `fromfile` must use the same
dtype that `tofile` wrote. Use `.npy` when self-describing files matter.

```lua
local handle = assert(io.open("stream.bin", "w+b"))
numlua.tofile(handle, numlua.array({ 4, 5 }, { dtype = "float32" }))
handle:seek("set", 0)
local values = numlua.fromfile(handle, { dtype = "float32" })
handle:close()
```

### `numlua.savez(path_or_file, arrays[, options])`

Writes a ZIP archive of `.npy` members using the stored (uncompressed) method
and returns `true`. `path_or_file` is a non-empty path or an open Lua file
handle; caller-owned handles are written at their current position and are
never closed. `arrays` is a table mapping non-empty array names to
luatorch tensors. This matches `np.savez` and has no compression dependency.
An empty `arrays` table writes a valid empty archive. Keys already ending in
`.npy` are stored unchanged; other keys get a `.npy` suffix, as in NumPy.
Because both `x` and `x.npy` therefore name the member `x.npy`, the writers
reject keys that normalize to the same member name (and the bare key `.npy`,
which would leave an empty array name) before writing any payload or creating
the output file. Unicode and spaced array names are preserved verbatim in the
ZIP directory.

`options.version` selects the NPY header version used for every member. It
accepts the same values as `numlua.save`: `1`, `2`, `3`, `"1.0"`, `"2.0"`, or
`"3.0"`. The default is v1.0, so archives without the option remain
byte-identical. `savez` rejects `fallback` (compression is not requested) and
any other unknown option, and validates the version before writing member
payloads or opening the output file.

Writing is ZIP32-only and validates all field widths before opening the output
file:

- at most 65535 members;
- each member name (including the `.npy` suffix) at most 65535 bytes;
- each uncompressed and compressed member at most 4294967295 bytes;
- the central-directory offset at most 4294967295 bytes.

An archive that would overflow any of these limits raises a
`numlua.savez:`/`numlua.savez_compressed:` error naming the limit instead of
silently wrapping a ZIP field. No output file is created on such an error.
ZIP64 writing is not supported.

### `numlua.savez_compressed(path_or_file, arrays[, options])`

Writes the same archive with every member deflated (ZIP method 8), matching
`np.savez_compressed`, and returns `true`. `path_or_file` accepts an open Lua
file handle exactly like `savez`. Compression uses the optional system
zlib loaded through LuaJIT FFI; no package beyond the zlib already used for
reading deflate archives is required.

By default the call fails with a
`numlua.savez_compressed: deflate compression requires a system zlib library`
error when zlib cannot be loaded, before any file is created. Pass
`{ fallback = "stored" }` to write stored entries instead; the call then returns
`true, "stored"` so the fallback is observable. Other `fallback` values and
non-table options are rejected. `options.version` is also accepted, with the
same values and default as `savez`, and may be combined with `fallback`; it
chooses the NPY member header version written by either path. An invalid
version fails before any file is created. `NUMLUA_DISABLE_ZLIB=1` forces the
no-zlib path for testing.

### `numlua.loadz(path_or_file)`

Reads an `.npz` archive from a path or an open Lua file handle and returns a
table whose keys are member names with
the `.npy` suffix removed. Members are ordinary `.npy` files, so v1.0, v2.0,
and v3.0 member headers are all accepted; a ZIP end-of-central-directory
comment is ignored. An open handle is read from its current position to EOF and
is left open, so seek to the start of a later archive before reading it;
`loadz` cannot delimit concatenated archives on its own. Supported ZIP
compression methods are:

| method | written by | availability |
| --- | --- | --- |
| stored (0) | `np.savez`, `numlua.savez` | always, dependency-free |
| deflate (8) | `np.savez_compressed`, `numlua.savez_compressed` | read/write; requires system zlib |

Deflate support is optional and detected lazily through LuaJIT FFI: on Linux
the loader tries `libz`, `libz.so.1`, and `libz.so`, and on macOS it also tries
`libz.dylib`. No LuaRocks or compression package is required. `savez_compressed`
emits raw DEFLATE streams, so the result is an ordinary ZIP archive readable by
NumPy and other tools. Stored archives work when zlib is absent; encountering a
deflate entry then raises a
`numlua.loadz: deflate-compressed entry ... requires a system zlib library`
error. `NUMLUA_DISABLE_ZLIB=1` forces the no-zlib path for testing, and the
zlib-unavailable error names that variable when it is the cause.

The reader validates member CRCs against the uncompressed payload, checks
local and central-directory headers and sizes before copying, and reports
corrupt central directories, CRC mismatches, truncation, encryption, multi-disk
archives, and unsupported compression methods as Lua errors. ZIP64 archives are
read through the ZIP64 end-of-central-directory record, locator, and
central/local extra fields; ZIP64 writing remains unsupported. ZIP64 values
above Lua's exact 2^53 integer range are rejected rather than truncated.
Before allocating a deflate output buffer, it also rejects a declared
uncompressed size beyond DEFLATE's 1032:1 expansion bound (plus a small framing
allowance), so a tiny corrupt archive cannot force a multi-gigabyte
allocation; a wrong size that is still plausible reaches zlib and fails with
the real size mismatch instead.

## Text and CSV I/O

### `numlua.loadtxt(path_or_file[, options])`

Reads a whitespace- or delimiter-separated numeric text file and returns a
luatorch tensor. The source may be a path string or an already-open Lua file
handle; a caller-owned handle is read from its current position and is never
closed. Text has no record boundaries beyond rows, so `loadtxt` consumes the
remaining data from a handle (use `max_rows` when only a prefix is wanted).
Options:

| option | default | meaning |
| --- | --- | --- |
| `delimiter` | whitespace runs | field separator, such as `","` or `"\t"` |
| `comments` | `"#"` | comment marker, a table of markers, or `false`/`""` to disable |
| `dtype` | `"float32"` | result dtype (a luatorch dtype name or id) |
| `skiprows` | `0` | number of physical lines to skip before parsing |
| `max_rows` | all | maximum number of data rows to read after skipping |
| `usecols` | all | 1-based column number or table of column numbers to read |
| `converters` | none | function or 1-based column-to-function table applied to raw fields |
| `unpack` | `false` | transpose a 2-D result so columns become rows |
| `ndmin` | `0` | minimum result rank: `0`, `1`, or `2` |
| `quotechar` | none | single-character quote for fields containing delimiters or spaces |

Blank lines and comments are ignored. Comment text starts at the first occurrence
of any configured marker on a line, so both full-line and trailing comments work.
LF, CRLF, and lone-CR line endings are accepted, matching Python's `splitlines`.
Every data row must have the same number of fields. One-row and one-column files
load as 1-D tensors, following NumPy's default squeeze behavior; a file with no
data rows returns an empty 1-D tensor. Numeric fields accept decimal, scientific,
`nan`, `inf`, and `-inf` spellings. Unknown option keys raise a
`numlua.loadtxt: unknown option ...` error instead of being ignored.

With `quotechar` set to one character, a field beginning with that character is
read atomically: delimiters and whitespace inside the quotes are preserved, and
a doubled quote character represents one literal quote. The quotes are removed
before the numeric parser or a configured converter sees the field. Unterminated
quotes and extra characters after a closing quote raise a `numlua.loadtxt:`
error. Combining `quotechar` with a converter is useful for fields containing
the delimiter.

`skiprows` counts physical lines, including comments and blank lines, exactly
like NumPy. `max_rows` is applied after that skip and counts data rows only:
blank and comment lines do not consume the limit, and `max_rows = 0` returns an
empty 1-D tensor without parsing data rows. Both options must be non-negative
integers. Rows beyond `max_rows` are not parsed, so an error there is not
reported.

`usecols` selects columns with **1-based** indices. A positive index counts
from the left, `-1` is the last column, and a table preserves the requested
order and duplicates. Only selected fields are converted, so a malformed field
in an unselected column is ignored. The total row width must still be
rectangular and agree with the first data row. `usecols = 0` raises a clear
1-based error, and out-of-range indices raise a `numlua.loadtxt:` error once the
first data row establishes the width. An empty file has no width to check and
loads as an empty 1-D tensor.

`converters` provides per-column parsing hooks. A single function applies to
every physical input column; a table maps **1-based** physical column numbers
to functions. Each function receives the raw field string and must return one
Lua number. Converters run before the built-in numeric parser, so a sentinel
such as `"NA"` can be mapped to a value; only columns that are actually read
are converted. Converter keys refer to the file's physical columns, not to
`usecols` output positions, and are bounds-checked after the first row
establishes the width. A callback error or a non-number result raises a
`numlua.loadtxt:` error naming the line and column. Empty input has no fields
to convert and loads empty.

```lua
local values = numlua.loadtxt("data.csv", {
  delimiter = ",",
  dtype = "int64",
  converters = {
    [2] = function(field)
      if field == "NA" then return -1 end
      return tonumber(field)
    end,
  },
})
```

`ndmin` controls the result rank before unpacking. With the default `0` (and
with `1`), one-row and one-column input is squeezed to 1-D. With `2`, the
physical row/column shape is kept: a one-row file is `(1, N)` and a one-column
file is `(N, 1)`. Empty input becomes shape `(0,)` for `ndmin` 0/1 and `(0, 1)`
for `ndmin` 2. Only `0`, `1`, and `2` are legal, following NumPy.

`unpack = true` transposes a 2-D result after the normal squeeze/`ndmin`
handling, matching NumPy: a 2x3 file becomes 3x2, while an already 1-D
one-row/one-column result keeps its shape. Use it to unpack columns with:

```lua
local x, y = numlua.loadtxt("data.csv", {
  delimiter = ",",
  usecols = { 1, 2 },
  unpack = true,
})
-- With ndmin = 2, x and y are (1, N) rows instead of 1-D vectors.
```

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

Supported result dtypes are `float16`, `float32`, `float64`, `int8`, `int16`,
`int32`, `int64`, `uint8`, and `bool`. Boolean fields accept `True`/`False`
(case-insensitive) and numeric values, where zero is false; a configured
converter's numeric result is truth-tested.

### `numlua.genfromtxt(path_or_file[, options])`

Reads the same numeric/bool text shape as `loadtxt` but treats configured missing
fields as values instead of errors. The source may be a path or an
already-open Lua file handle; a caller-owned handle is read from its current
position and is never closed. Options:

| option | default | meaning |
| --- | --- | --- |
| `delimiter` | whitespace runs | field separator |
| `comments` | `"#"` | comment marker, a table of markers, or `false`/`""` to disable |
| `dtype` | `"float32"` | result dtype, including `bool` |
| `skip_header` | `0` | number of physical lines to skip at the start |
| `skip_footer` | `0` | number of physical lines to skip at the end |
| `max_rows` | all | maximum data rows to read; cannot be combined with `skip_footer` |
| `usecols` | all | 1-based column number or table of columns to read |
| `converters` | none | function or 1-based physical-column-to-function table |
| `missing_values` | empty | string/number, list of markers, or per-column map |
| `filling_values` | NaN/0 | number, per-column list, or per-column map |
| `unpack` | `false` | transpose a 2-D result |
| `ndmin` | `0` | minimum result rank: `0`, `1`, or `2` |
| `quotechar` | none | single-character quote for fields containing delimiters or spaces |

`missing_values` may be a string (commas split multiple markers), a number, a
list of markers applied to every column, or a table mapping **1-based physical
column numbers** to a marker or marker list. `filling_values` may be one number
for every column, a list with one value per column (remaining columns use the
default), or a per-column map. Missing floating fields default to NaN and
missing integer fields default to zero. Integer filling values must be finite.
Blank lines, comments, row-width checks, `usecols` order and negative indices,
`unpack`, `ndmin`, and the one-row/one-column squeeze follow `loadtxt`.
`quotechar` also follows `loadtxt`, with quoting applied before converters and
missing-value handling.

Converters run before missing-value handling, matching NumPy: a converter can
map a sentinel itself, and a configured missing marker is only consulted when
no converter handles that physical column. `usecols` is applied for output,
but `converters`, `missing_values`, and `filling_values` refer to physical input
columns. A field that is neither converted nor missing must still parse as a
number; `genfromtxt` does not silently replace unrelated malformed fields.
Structured dtypes, names, string columns, and masked-array output are not
supported; text I/O remains numeric/bool-only. `max_rows = 0` returns an empty
tensor, and `max_rows` together with `skip_footer` is rejected.

```lua
-- data.csv:
--   1,NA
--   2,3
local values = numlua.genfromtxt("data.csv", {
  delimiter = ",",
  missing_values = "NA",
  filling_values = -1,
})
-- values = {{1, -1}, {2, 3}}
```

Per-column filling uses physical 1-based columns, so it remains aligned with
`usecols`:

```lua
local values = numlua.genfromtxt("data.csv", {
  delimiter = ",",
  usecols = { 1, 3 },
  missing_values = { [3] = "N/A" },
  filling_values = { [3] = 0 },
})
```

### `numlua.savetxt(path_or_file, tensor[, options])`

Writes a 1-D or 2-D tensor as numeric/bool text and returns `true`. The target may
be a path string or an already-open Lua file handle; a caller-owned handle is
written at its current position and is never closed, including when a write
fails. Closed handles and invalid targets raise a `numlua.savetxt:` error.
Options:

| option | default | meaning |
| --- | --- | --- |
| `delimiter` | `" "` | field separator |
| `fmt` | dtype-aware | `printf` format string, function, or table of per-column formats |
| `header` | `""` | text written before the data, one comment-prefixed line per input line |
| `footer` | `""` | text written after the data, one comment-prefixed line per input line |
| `comments` | `"# "` | prefix applied to each header/footer line |
| `newline` | `"\n"` | line terminator for data, header, and footer |

The default floating formats use enough significant digits to round-trip the
tensor's dtype (`float16`, `float32`, or `float64`); integral dtypes use base 10.
`nan`, `inf`, and `-inf` are written explicitly. Unknown option keys raise a
`numlua.savetxt: unknown option ...` error.

`header` and `footer` are written before and after the data, with `comments`
prepended to each line (including every line of a multi-line string). The
default `comments = "# "` produces lines that `loadtxt` ignores, so header and
footer text can document a file without affecting a round trip. `newline`
applies to data rows and to the header/footer lines; it defaults to `"\n"`.
Pass `comments = ""` to write bare header/footer text. Header and footer text
are written even for empty vectors and zero-column matrices.

A custom `fmt` is applied to every value. A `fmt` table supplies one format
per column, in left-to-right order; each entry may be a `printf` string or a
function, so strings and functions can be mixed. The table length must equal
the number of output columns, and a 1-D vector has one column:

```lua
numlua.savetxt("values.csv", numlua.arange(6):reshape(2, 3), {
  delimiter = ",",
  fmt = { "%.3f", "%d", "%.1e" },
})
```

Zero-dimensional and higher-than-2-D tensors are rejected. Bool tensors are
written as `True`/`False` by default; a custom `fmt` receives `1`/`0`. Text
files do not record shape or dtype, so load with the
matching `dtype` when exact integer or wider floating-point values matter; a
1xN or Nx1 matrix cannot be distinguished from a vector after loading. Integral
values are formatted through Lua numbers, so `int64` values outside the exactly
representable double range (about ±9.0e15) may round; use `.npy` for exact
large 64-bit integers.

## Errors

Invalid arguments raise ordinary Lua errors with a `numlua.<function>:` prefix
where the facade validates them. Errors from luatorch/ATen regarding shape,
dtype, or backend constraints are forwarded as-is.

## Differences from NumPy and known limits

- Axes/indices are 1-based; NumPy examples must shift axes by one.
- Default dtypes differ: `float32` is the luatorch default for constructors and
  Lua numeric tables; NumPy usually defaults to `float64`.
- `logspace` and `geomspace` are 1-D only (NumPy's `axis` keyword is not
  accepted), require finite endpoints, and default to `float32`. `geomspace`
  rejects zero endpoints and requires both endpoints to share a sign.
- `tri`, `vander`, and `identity` keep the facade's `float32` default unless a
  `dtype` is supplied; `vander` accepts a 1-D tensor or Lua table.
- `ptp` is `max - min`; bool input is cast to `int64` so the subtraction is
  defined, and NaN values propagate.
- `compress` uses 1-based axes; with an axis its condition must be 1-D, and a
  condition longer than the selected axis is an error.
- Reductions, including `any`, `all`, and `count_nonzero`, take a single axis
  only, not a tuple of axes.
- The facade has no NumPy-style general dtype-promotion system of its own.
  `mean` promotes integer/bool input to `float64`; elementwise ufuncs delegate
  promotion to ATen. Scalar `where` branches are materialized as 0-D `float32`
  (or `bool`) tensors before ATen broadcasting.
- `dot` follows NumPy's vector/matrix/batched shape rules but delegates the
  multiplication to ATen, so mixed integer/floating operands follow ATen's
  matmul checks rather than NumPy promotion.
- `inner`, `outer`, and `cross` delegate their arithmetic to ATen. `cross`
  supports 3-component vectors only (not NumPy's 2-component form), requires
  the same rank for both operands, and broadcasts batch axes through ATen.
- `trapezoid`/`trapz` integrate along one 1-based axis. A coordinate array may
  be 1-D for the selected axis or have the full input shape; integer input
  follows ATen's promotion to `float32`.
- `interp` requires a strictly increasing 1-D `xp` grid and returns `float64`;
  `left`/`right` are positional scalars and default to the first/last `fp`
  value.
- `unwrap` takes `discont`, `axis`, and `period` positionally (NumPy exposes
  `discont`/`axis`/`period` as keywords), promotes integer input to `float64`,
  and returns independent storage.
- `nan_to_num` uses ATen's dtype-maximum defaults when replacement values are
  omitted; `nancumsum`/`nancumprod` skip NaN rather than returning NaN.
- Elementwise ufuncs delegate dtype promotion to ATen rather than to NumPy's
  promotion rules or `numlua.promote_types`. In particular, `int64` + `float32`
  stays `float32`, integer/integer division produces `float32` (not `float64`),
  and transcendental functions on integral input produce `float32`.
- Elementwise ufuncs have no `out=`, `where=`, `dtype=`, or `casting=` options.
  Arithmetic scalar operands are numbers only; the logical ufuncs additionally
  accept Lua booleans.
- Fancy indexing is not exposed; use the returned luatorch tensors and `torch`
  directly.
- Broadcasting is not reimplemented; it is whatever luatorch/ATen does for the
  delegated operation.
- `.npy` reading supports v1.0, v2.0, and v3.0 headers; writing defaults to
  v1.0 and can emit v2.0/v3.0 with the `version` option. Files must be C order
  on little-endian hosts and use the dtypes listed above. Native-endian `=`
  descr values are only aliases for the little-endian `<` spellings. `save`,
  `load`, `savez`, `savez_compressed`, and `loadz` accept open Lua file handles
  in addition to paths; `loadz` reads from the current handle position to EOF.
- `take` follows NumPy's scalar/N-D index result shapes but uses 1-based
  indices and rejects negative index values; NumPy resolves negatives relative
  to the end of the axis.
- `take_along_axis` uses 1-based indices and rejects negative index values;
  NumPy's version resolves them relative to the end of the axis.
- `split` boundary tables are 1-based; `split` with an integer section count
  requires an even division, while `array_split` distributes the remainder.
  `hsplit`, `vsplit`, and `dsplit` require at least 1-D, 2-D, and 3-D input.
- `diag`, `diagflat`, `tril`, and `triu` support all facade dtypes including
  `bool`; `tril`/`triu` apply to the last two axes and reject input below 2-D.
- `isclose` compares mixed or integral dtypes through `float64`; `allclose`,
  `array_equal`, and `array_equiv` return Lua booleans rather than tensors.
- `fromfile`/`tofile` default to binary (`sep = ""`) and carry no shape/dtype
  metadata. A non-empty `sep` selects simple numeric text mode, not
  `loadtxt`/`savetxt`: text mode has no headers, comments, quoting, or column
  selection, and it reads a caller-owned handle to EOF. `fromfile` defaults to
  `float64`, so pass the dtype written by `tofile`. Use `.npy` for
  self-describing files.
- `apply_along_axis` takes a Lua callback rather than a NumPy ufunc and has no
  keyword arguments; extra positional arguments are forwarded. Its callback
  must return a scalar or 1-D result, matching NumPy's restriction, and input
  with a zero-size non-axis dimension is rejected because there is no first
  slice to infer the output dtype and shape.
- `apply_over_axes` uses 1-based axes and takes a Lua callback rather than a
  NumPy ufunc; extra positional arguments are a numlua extension. A callback
  result must keep the input rank or drop exactly one dimension (which is
  restored at the same position); NumPy's scalar and table return spellings
  are not accepted by `np.apply_over_axes`, which expects an ndarray.
- `var`/`std` and the `nan*` reductions accept one axis at a time, like the
  other reductions. `ddof` must be finite (NumPy permits any value); variance
  denominators are not rounded, and NaN slices return NaN without emitting
  NumPy's runtime warning. Integer/bool `var`/`std` and the NaN statistics
  promote to `float64`, while floating input keeps its dtype.
- `meshgrid` requires 1-D inputs and does not apply NumPy's mixed-dtype result
  promotion: each output keeps its own input dtype. `copy = false` returns
  expanded views instead of independent writable arrays, and the options table
  is a Lua extension rather than NumPy keyword arguments.
- `gradient` implements NumPy's default `edge_order = 1` only; there is no
  `edge_order = 2` option. Spacing may be uniform, a coordinate vector for one
  selected axis, or one entry per selected axis. Integer/bool input promotes
  to `float64`, and selected axes require at least two elements.
- Text I/O stores numeric and bool fields: no shape/dtype metadata, quoted
  string columns, or parsed header metadata. `quotechar` makes quoted fields
  atomic during reading, but `savetxt` never writes quotes. `loadtxt`/`savetxt`
  accept caller-owned file
  handles at their current position and never close them. `savetxt`
  header/footer text is written as comments, so `loadtxt` ignores it rather
  than interpreting it. `loadtxt`
  defaults to `float32`, and one-row/one-column files are squeezed to 1-D
  unless `ndmin = 2`. `usecols` uses **1-based** column numbers (NumPy uses
  0-based columns), though negative columns count from the end in both. Bool
  fields use `True`/`False` or numeric zero/nonzero spellings. Unknown option
  keys are rejected, and `int64` text output can round values beyond the
  exactly representable double range; use `.npy` when exact large integers
  matter.
- `genfromtxt` keeps the numeric/bool-only text scope: no structured dtypes,
  `names`, string columns, masked-array output, `usemask`, `loose`,
  `invalid_raise`, or `skiprows` alias. Missing/filling/converters use 1-based
  physical column numbers (NumPy uses 0-based keys), default dtype is `float32`,
  and a malformed field that is neither converted nor missing is an error
  rather than a default-filled value. `max_rows` and `skip_footer` cannot be
  combined.
- `argsort` and `sort` accept a `descending` flag that is a numlua extension;
  NumPy has no descending argument. The flag must be a boolean or `nil`.
  `sort` defaults to the last axis and rejects 0-D input, like `argsort`.
- `unique` always flattens and sorts; it has no `axis`, `equal_nan`, or unsorted
  mode. Its first-occurrence and inverse indices are 1-based.
- `nonzero` and `argwhere` return 1-based coordinates in C order and reject
  0-D input; use `flatnonzero` for flat indices or a 0-D scalar.
- `squeeze` follows luatorch's permissive behavior for non-unit axes.
- `.npz` writing is stored by default and deflate through
  `savez_compressed`; reading supports stored and deflate entries plus ZIP64
  archives. Deflate
  needs a system zlib library through LuaJIT FFI, while stored archives and
  writing remain dependency-free. Writer output is ZIP32-only; archives above
  the 16-bit entry/name limits or 32-bit size/offset limits fail explicitly,
  and ZIP64 writing is not supported.
- LuaRocks installs the Lua facade; the native `libluatorch.so` must be built
  separately against libtorch.
