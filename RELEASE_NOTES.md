# numlua 0.14.0

NumPy-shaped arrays for LuaJIT, backed by luatorch.

## Highlights

- `numlua.savez` and `numlua.savez_compressed` accept
  `{ version = 1|2|3 }` (or `"1.0"`/`"2.0"`/`"3.0"`) and write every `.npz`
  member with that NPY header version. The default remains v1.0, so archives
  without the option are byte-identical to 0.13.0. `savez_compressed` may
  combine `version` with `{ fallback = "stored" }`. Version spellings use the
  same parser as `numlua.save`; invalid versions and unknown options fail
  before member payloads or the output file are written;
- `numlua.select(condlist, choicelist[, default])` adds NumPy's `np.select`
  semantics: conditions are tested in order, the first true condition wins, and
  unmatched elements take `default` (0 when omitted). Conditions and choices
  accept tensors or Lua scalars with the same truth testing and 0-D scalar
  behavior as `numlua.where`; broadcasting and dtype promotion stay in
  luatorch/ATen. Empty or mismatched lists, invalid entries, and
  non-broadcastable operands raise clear `numlua.select:` errors;
- docs/API reference cover both features, including the `.npz` member-version
  option, the compressed fallback combination, and the first-match selection
  rule. Release tests keep the new symbols and options documented.

## Requirements

LuaJIT 2.1 and a built luatorch checkout. zlib remains optional: stored archives
and all non-compression features work without it. No runtime dependency was
added.

## Verification

- 178 tests passed with system zlib
- 178 tests passed with `NUMLUA_DISABLE_ZLIB=1`
- `examples/basic.lua` runs

No tag or release is created by this preparation.
