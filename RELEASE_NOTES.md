# numlua 0.2.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- layout and selection helpers: broadcast, tile, repeat, flip, roll, pad, diff,
  cumulative sums/products, argsort, take;
- pseudoinverse, least squares, and matrix norms;
- safe dtype-promotion helpers;
- stored `.npz` archives in addition to `.npy`;
- committed NumPy-generated fixture coverage;
- LuaRocks packaging and CI against luatorch 0.8.

## Requirements

- LuaJIT 2.1
- luatorch 0.8 with its native `libluatorch.so` built against libtorch 2.5.1
- no additional runtime dependencies

## Install

```sh
luarocks --lua-version=5.1 install numlua
```

The native luatorch shim must be built separately; see luatorch's README.

## Verification

- 87 tests passed
- LuaRocks install smoke test
- CI checkout of luatorch and libtorch
