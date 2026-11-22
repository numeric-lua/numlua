# numlua 0.16.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.pad` gains NumPy's `"linear_ramp"`, `"maximum"`, `"mean"`,
  `"median"`, and `"minimum"` modes. `"linear_ramp"` takes `end_values`
  through the existing `value` argument; the statistic modes take
  `stat_length`. Both accept a number, a `{before, after}` pair, or one entry
  per axis. Axes are padded in order, so corners follow NumPy's rule that
  later axes see earlier padding;
- integer end values and statistics round half to even before the input dtype
  cast, bool input works with every statistic mode, and zero-size axes remain
  supported only by `"constant"` (as before). Modes without a fill argument
  now reject a non-nil `value` instead of silently ignoring it;
- the test suite adds an independent C-order reference walker for linear ramps
  and axis statistics, plus IO/selection edge coverage: disabled comment
  markers, missing final newlines, numeric `loadtxt` dtype ids, `savetxt`
  format functions and error paths, double-quoted NPY header keys for v1.0
  and v3.0, and negative-axis/empty selection forms. Docs, CHANGELOG, and
  README describe the new modes.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored
archives and all non-compression features work without it. No runtime
dependency was added.

## Verification

- 200 tests passed with system zlib
- 200 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
