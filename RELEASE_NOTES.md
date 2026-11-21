# numlua 0.3.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- NumPy-style elementwise ufuncs: add, subtract, multiply, divide, negative,
  abs, sqrt, exp, log, sin, cos, tanh, maximum, minimum, power, and square,
  with scalar/tensor operands and ATen broadcasting;
- `argmin` alongside `argmax`, with the same 1-based flat/axis indices and
  `keepdims` handling;
- `linalg.matrix_norm` now supports `ord=-2` (smallest singular value), in
  addition to Frobenius, 1, -1, 2, infinity, -infinity, and nuclear norms;
- generated `.npy`/`.npz` fixture corpus across all supported dtypes, empty and
  0-D arrays, stored and deflate ZIP members, and big-endian rejection;
- optional deflate-compressed `.npz` reading through a system zlib loaded via
  LuaJIT FFI; stored archives stay dependency-free;
- ZIP integrity rejection for encrypted, multi-disk, and ZIP64 archives plus
  corrupt central directories, CRC mismatch, truncation, and local-header
  inconsistencies;
- `swapaxes`, `moveaxis`, `atleast_*`, and `resize` layout helpers, and `pad`
  modes `edge`, `reflect`, and `symmetric`;
- benchmark harness and dated performance notes with explicit caveats.

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

- 112 tests passed with system zlib and with `NUMLUA_DISABLE_ZLIB=1`
- LuaRocks install smoke test
- CI checkout of luatorch and libtorch
