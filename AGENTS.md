# AGENTS.md

Local instructions for `numlua` agents. The organization-wide rules in
`../AGENTS.md` (when present) take precedence. This file covers this repository
only.

## Scope

`numlua` is a NumPy-shaped LuaJIT facade over **luatorch tensors**:

- array construction and coercion;
- shape manipulation;
- reductions and small linear-algebra conveniences;
- dtype conversion;
- NumPy `.npy` v1.0 I/O.

It is not a tensor engine, autograd implementation, dtype-promotion system, or
copy of NumPy. Array-producing functions return `luatorch` tensors and delegate
the numerical work. Do not add a second FFI engine or native dependency.

Dependency direction is one-way:

```text
luatorch  ->  numlua
```

Do not modify the sibling luatorch checkout from this repository. A change that
requires a new luatorch public API is a cross-repo task; document the required
core change and land it in luatorch first.

## Required checks

Run from the repository root:

```sh
LUA_PATH="../../lua/?.lua;../../lua/?/init.lua;./?.lua;./?/init.lua;;" \
  luajit tests/run.lua
```

Equivalent wrappers:

```sh
./scripts/test.sh
make test
```

Use `LUATORCH_ROOT` if the luatorch checkout is not in the default staging
location. Tests must pass without luarocks, network access, or external Lua
rocks. The test harness is dependency-free; keep it that way. Run
`luajit examples/basic.lua` with the same `LUA_PATH` after changing the public
APIs.

## Safety and release rules

- **No pushes to `main`, no merges, no tags, no GitHub releases, and no
  LuaRocks publishes** unless a human coordinator explicitly asks for that
  exact action.
- Work on a branch or worktree; leave draft PRs for review.
- Never add secrets, tokens, keys, or private paths to source, tests, docs,
  examples, or commit messages.
- Do not commit generated `.npy` files, binaries, caches, logs, or scratch
  outputs. `tests/.tmp/` and example artifacts are ignored on purpose.
- Do not create the GitHub organization, transfer repositories, or change
  visibility. Those are coordinator actions.

## API and compatibility rules

- Keep the Lua surface **1-based** for dimensions, axes, indices, and
  `argmax`. This is the numeric-lua convention; do not "fix" it to NumPy's
  0-based axes.
- Keep `.npy` bytes NumPy-compatible: little-endian, C-order, NPY v1.0, dtypes
  `float32` (`<f4`), `float64` (`<f8`), `int64` (`<i8`), and `bool` (`|b1`).
- Do not silently upcast unsupported `.npy` dtypes, accept Fortran-order files,
  or write big-endian data. Return a clear Lua error instead.
- Differences from NumPy (default dtypes, 1-based axes, integer `mean`
  promotion, `max` returning values only) must be documented in `README.md`
  and `docs/api.md`.
- Public API changes need: implementation, tests (including an error path),
  `docs/api.md` updates, and a README/example update when user-visible.
- Reductions return tensors, including 0-D tensors for all-axis reductions.
  Do not turn them into bare Lua numbers.
- Broadcasting belongs to luatorch. Do not grow a separate broadcast layer in
  `numlua`.

## Code style

- LuaJIT-compatible Lua 5.1 syntax. No external modules beyond `torch`/FFI and
  the Lua standard library.
- Match the existing style: small modules, explicit validation, `numlua.<fn>:`
  error prefixes, no trailing whitespace.
- Use `local` everywhere; avoid globals except the deliberate test harness
  globals (`test`, assertions, `TMPDIR`).
- Keep FFI use confined to `.npy` raw-buffer handling and luatorch's pointer
  helper. Do not turn `numlua` into an FFI tensor layer.
- Prefer clear Lua errors over assertions that abort the interpreter.

## Handoff

Report the files changed, the exact commands and test results, known
limitations, and anything left uncommitted. Do not commit or push as part of a
handoff unless the task explicitly says to.
