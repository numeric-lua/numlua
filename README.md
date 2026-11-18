# numlua

NumPy-shaped arrays for LuaJIT, implemented as a **thin facade over
`luatorch` tensors**.

`numlua` gives LuaJIT the parts of NumPy that are useful around a tensor
engine: array construction, shape manipulation, reductions, predicates and
selection, small linear algebra operations, dtype conversion, NumPy `.npy` file
I/O, and simple whitespace/CSV text I/O. It does not implement a tensor engine
of its own.
Array-producing functions return `luatorch` tensors, and luatorch/ATen performs
the actual computation.

> **Status: MVP scaffold, unreleased.** The public API is intentionally small
> and is expected to grow. There is no LuaRocks package yet.

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
print(numlua.count_nonzero(values):item(), numlua.any(values:gt(0)):item())
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

LuaRocks packaging is pending. Until then, run from the repository root with
luatorch on `LUA_PATH`:

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
`numlua/init.lua`. A future LuaRocks release will install the module and its
luatorch dependency normally.

## Conventions (read this before porting NumPy code)

- **Axes and indices are 1-based.** `axis=1` is NumPy's `axis=0`, and the first
  `argmax` result is `1`, not `0`. This is the numeric-lua convention.
- **Functions return luatorch tensors.** Reductions never return bare Lua
  numbers. All-axis reductions return a 0-dimensional tensor; call `:item()`
  when you want a number.
- **`max`/`min` return values only.** Luatorch's dimensional `max` also returns
  indices; `numlua` discards them. Use `numlua.argmax` for indices.
- **Default dtypes follow luatorch.** `array({1, 2, 3})` is `float32` because
  Lua numbers do not distinguish integers. Pass `{dtype = "int64"}` for
  integers. `zeros`, `ones`, `eye`, and `linspace` default to `float32`.
  `arange` infers `int64` for integer bounds and `float32` otherwise.
  `mean` promotes integer/bool input to `float64` (ATen would otherwise reject
  it); floating input keeps its dtype.
- **Broadcasting is luatorch's.** `matmul` and luatorch's own elementwise
  operations use ATen broadcasting. `numlua` does not add another broadcast
  layer.
- **`.npy` bytes are NumPy-compatible.** The Lua-facing axes stay 1-based; the
  file format does not change.

See [docs/api.md](docs/api.md) for the full surface and limits.

## API at a glance

| Group | Functions |
| --- | --- |
| Constructors | `array`, `asarray`, `zeros`, `ones`, `full`, `eye`, `arange`, `linspace` |
| Shape | `shape`, `reshape`, `transpose`, `ravel`, `squeeze`, `unsqueeze` |
| Combine | `concatenate`, `stack` |
| Reductions | `sum`, `mean`, `max`, `min`, `argmax` |
| Predicates/selection | `where`, `clip`, `isnan`, `any`, `all`, `count_nonzero` |
| Linear algebra | `dot`, `matmul` |
| Dtype | `dtype`, `astype` |
| NumPy I/O | `save`, `load` |
| Text/CSV I/O | `loadtxt`, `savetxt` |

## `.npy` I/O

`numlua.save(path, tensor)` writes a NumPy v1.0 `.npy` file in C order.
`numlua.load(path)` returns a luatorch tensor.

Supported dtypes:

| numlua / luatorch | `.npy` descr | bytes per element |
| --- | --- | --- |
| `float32` | `<f4` | 4 |
| `float64` | `<f8` | 8 |
| `int64` | `<i8` | 8 |
| `bool` | `\|b1` | 1 |

The reader supports 1-D, 2-D, and N-D arrays (and 0-D scalars) and rejects
big-endian files, `fortran_order=True`, unsupported `descr` values, NPY versions
other than 1.0, and truncated data with clear Lua errors. Files are always
written little-endian; only little-endian hosts are supported.

Example:

```lua
local numlua = require("numlua")

local a = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } })
numlua.save("arrays.npy", a)
local b = numlua.load("arrays.npy")
assert(b:equal(a))
```

## Text/CSV I/O

`numlua.savetxt(path, tensor[, options])` writes a 1-D or 2-D tensor as numeric
text. `numlua.loadtxt(path[, options])` reads it back as a luatorch tensor.
Plain whitespace is the default separator, and `delimiter = ","` handles simple
CSV. Comment markers default to `"#"` and can be changed, supplied as a table,
or disabled.

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
line number. One-row and one-column files become 1-D tensors (NumPy's default
squeeze behavior), and an all-comment/empty file loads as an empty 1-D tensor.
The default load dtype is `float32`; pass `dtype` to load integers or wider
floats exactly. Text files do not store shape or dtype, so a 1xN or Nx1 matrix
cannot be distinguished from a vector after loading.

`savetxt` writes sufficient digits for `float16`/`float32`/`float64`
round-trips, writes `nan`/`inf` explicitly, and supports a `fmt` format string
or function. It rejects zero-dimensional and higher-than-2-D tensors and the
`bool` dtype (use `0`/`1` values with an integral dtype instead). See
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
reductions, predicates and conditional selection, linear algebra, dtype
conversion, `.npy` round-trips for all supported dtypes and 1-D/2-D/3-D shapes,
whitespace/CSV text round-trips, and error paths. It currently contains 72
tests. `examples/basic.lua` is runnable with the same `LUA_PATH`.

## Repository layout

```text
numlua/
  numlua/init.lua    -- public facade
  numlua/npy.lua     -- .npy v1.0 reader/writer
  numlua/text.lua    -- whitespace/CSV text reader/writer
  docs/api.md        -- API reference
  examples/basic.lua -- runnable example
  tests/run.lua      -- dependency-free test harness
  scripts/test.sh    -- test wrapper with LUA_PATH setup
  Makefile           -- make test
```

## Status and known limits

This is an MVP, not a complete NumPy replacement:

- reductions, including `any`, `all`, and `count_nonzero`, accept one axis at a
  time (no tuple axes);
- there is no general dtype-promotion system; only `mean` promotes integer/bool
  input, and it always promotes to `float64`; scalar `where` branches are
  materialized as 0-D `float32` (or `bool`) tensors before ATen broadcasting;
- `dot` is the 1-D inner product only; use `matmul` for matrix and batched
  products;
- there is no elementwise arithmetic or fancy-indexing API; use the returned
  luatorch tensors and `torch` for those operations;
- `squeeze` delegates to luatorch; squeezing an axis whose size is not 1 leaves
  that axis unchanged rather than raising NumPy's error;
- `.npy` support is limited to v1.0, C order, little-endian hosts, and the four
  dtypes above; `load` reads the complete file into memory before copying it
  into a tensor;
- text I/O is intentionally simple: it does not store shape/dtype metadata,
  quote strings, or parse headers, and it rejects `bool` rather than guessing a
  textual boolean spelling;
- there is no LuaRocks rockspec yet. Packaging is pending.

## License

BSD 3-Clause. See [LICENSE](LICENSE). `numlua` depends on luatorch, which links
against libtorch/PyTorch; neither dependency is redistributed here.
