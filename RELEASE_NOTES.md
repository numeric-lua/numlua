# numlua 0.24.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.prod` and `numlua.nanprod` add product reductions with the usual
  axis/`keepdims` rules; `nanprod` treats NaN as the multiplicative identity.
- `numlua.ptp` returns the peak-to-peak `max - min` range, with bool input
  promoted to `int64`.
- `numlua.logspace` and `numlua.geomspace` build logarithmically spaced
  sequences, including `base`, `endpoint`, `dtype`, and signed `geomspace`
  support.
- `numlua.identity`, `numlua.tri`, and `numlua.vander` round out the common
  NumPy constructors.
- `numlua.compress` supports axis-aligned selection and NumPy's
  short-condition prefix rule in both flat and axis modes.
- `numlua.loadz` now reads ZIP64 archives, including ZIP64 extra fields and
  the ZIP64 end-of-central-directory record; writing remains ZIP32-only.
- All new features are documented in `README.md` and `docs/api.md`; no runtime
  dependency was added.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives, ZIP64 reading, raw binary and text I/O, and all non-compression
features work without it.

## Verification

- 373 tests passed with system zlib
- 373 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs
- LuaRocks builtin install smoke-tested from `numlua-0.24.0-1.rockspec`

Tag and GitHub release: `v0.24.0`.
