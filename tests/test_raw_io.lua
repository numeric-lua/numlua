--
-- Raw binary NumPy I/O: fromfile/tofile.
--

local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".bin"
end

local function cleanup(path)
  os.remove(path)
end

test("tofile and fromfile round-trip every supported dtype", function()
  local cases = {
    { dtype = "float32", values = { 1.5, -2.25, 0 } },
    { dtype = "float64", values = { 1.5, -2.25, 0 } },
    { dtype = "int64", values = { -3, 0, 7 } },
    { dtype = "int32", values = { -3, 0, 7 } },
    { dtype = "int16", values = { -3, 0, 7 } },
    { dtype = "int8", values = { -3, 0, 7 } },
    { dtype = "uint8", values = { 3, 0, 7 } },
    { dtype = "bool", values = { true, false, true } },
  }
  for _, case in ipairs(cases) do
    local path = tmp_path("raw-" .. case.dtype)
    local input = numlua.array(case.values, { dtype = case.dtype })
    assert_true(numlua.tofile(path, input))
    local loaded = numlua.fromfile(path, { dtype = case.dtype })
    assert_dtype(loaded, case.dtype)
    assert_shape(loaded, { #case.values })
    assert_tensor(loaded, case.values)
    cleanup(path)
  end

  -- float16 cannot be built directly from a Lua table, so cast into it.
  local path = tmp_path("raw-float16")
  local input = numlua.array({ 1.5, -2, 0 }):to("float16")
  assert_true(numlua.tofile(path, input))
  local loaded = numlua.fromfile(path, { dtype = "float16" })
  assert_dtype(loaded, "float16")
  assert_tensor(loaded, { 1.5, -2, 0 }, 0)
  cleanup(path)
end)

test("fromfile defaults to float64 and supports count and offset", function()
  local path = tmp_path("raw-window")
  local input = numlua.array({ 10, 20, 30, 40, 50 }, { dtype = "int32" })
  assert_true(numlua.tofile(path, input))

  local loaded = numlua.fromfile(path, { dtype = "int32" })
  assert_tensor(loaded, { 10, 20, 30, 40, 50 })

  local window = numlua.fromfile(path, {
    dtype = "int32",
    count = 2,
    offset = 4,
  })
  assert_tensor(window, { 20, 30 })

  local none = numlua.fromfile(path, { dtype = "int32", count = 0 })
  assert_shape(none, { 0 })
  assert_eq(none:numel(), 0)

  local at_end = numlua.fromfile(path, {
    dtype = "int32",
    offset = 20,
  })
  assert_shape(at_end, { 0 })

  -- A float64 default reads the bytes as doubles.
  local doubles = tmp_path("raw-default")
  assert_true(numlua.tofile(doubles,
    numlua.array({ 1.5, 2.5 }, { dtype = "float64" })))
  local default = numlua.fromfile(doubles)
  assert_dtype(default, "float64")
  assert_tensor(default, { 1.5, 2.5 })
  cleanup(path)
  cleanup(doubles)
end)

test("tofile writes C order for non-contiguous inputs", function()
  local path = tmp_path("raw-transposed")
  local input = numlua.transpose(numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }))
  assert_false(input:is_contiguous())
  assert_true(numlua.tofile(path, input))
  local loaded = numlua.fromfile(path, { dtype = "float32" })
  assert_tensor(loaded, { 1, 4, 2, 5, 3, 6 })
  cleanup(path)
end)

test("fromfile and tofile use caller-owned handles at the current position", function()
  local path = tmp_path("raw-handle")
  local handle = assert(io.open(path, "w+b"))
  local first = numlua.array({ 1, 2, 3 }, { dtype = "int16" })
  local second = numlua.array({ 4, 5 }, { dtype = "int16" })
  assert_true(numlua.tofile(handle, first))
  assert_true(numlua.tofile(handle, second))
  assert_eq(io.type(handle), "file", "tofile must not close a caller handle")

  assert_eq(handle:seek("set", 0), 0)
  local loaded_first = numlua.fromfile(handle,
    { dtype = "int16", count = 3 })
  assert_tensor(loaded_first, { 1, 2, 3 })
  assert_eq(io.type(handle), "file", "fromfile must not close a caller handle")
  local loaded_second = numlua.fromfile(handle, { dtype = "int16" })
  assert_tensor(loaded_second, { 4, 5 })

  -- An offset moves the handle forward by that many bytes.
  assert_eq(handle:seek("set", 0), 0)
  local skipped = numlua.fromfile(handle, {
    dtype = "int16",
    offset = 2,
    count = 1,
  })
  assert_tensor(skipped, { 2 })
  assert_eq(handle:seek("cur", 0), 4)
  assert_true(handle:close())
  cleanup(path)
end)

test("fromfile rejects truncation and non-integral byte counts", function()
  local path = tmp_path("raw-truncated")
  assert_true(numlua.tofile(path,
    numlua.array({ 1, 2, 3 }, { dtype = "int32" })))
  assert_error(function()
    numlua.fromfile(path, { dtype = "int32", count = 4 })
  end, "truncated")
  assert_error(function()
    numlua.fromfile(path, { dtype = "int64" })
  end, "not a multiple")
  cleanup(path)
end)

test("raw I/O validates options, dtypes, and targets", function()
  local path = tmp_path("raw-errors")
  local values = numlua.array({ 1, 2 })
  assert_true(numlua.tofile(path, values))

  assert_error(function()
    numlua.fromfile(path, { dtype = "complex64" })
  end, "unsupported dtype")
  assert_error(function() numlua.fromfile(path, { count = -2 }) end,
    "count must be an integer >= -1")
  assert_error(function() numlua.fromfile(path, { count = 1.5 }) end,
    "count must be an integer >= -1")
  assert_error(function() numlua.fromfile(path, { offset = -1 }) end,
    "offset must be a non-negative integer")
  assert_error(function() numlua.fromfile(path, { sep = "," }) end,
    "text-mode sep is not supported")
  assert_error(function() numlua.fromfile(path, { sep = 1 }) end,
    "sep must be a string")
  assert_error(function() numlua.fromfile(path, { nope = true }) end,
    "unknown option")
  assert_error(function() numlua.fromfile("") end,
    "non-empty file path or open file handle")
  assert_error(function() numlua.fromfile(nil) end,
    "non-empty file path or open file handle")

  assert_error(function() numlua.tofile(path, { 1, 2 }) end,
    "expects a tensor")
  assert_error(function() numlua.tofile(path, values, { sep = "," }) end,
    "text-mode sep is not supported")
  assert_error(function() numlua.tofile(path, values, { nope = true }) end,
    "unknown option")
  assert_error(function() numlua.tofile("", values) end,
    "non-empty file path or open file handle")

  local closed = assert(io.open(path, "rb"))
  assert_true(closed:close())
  assert_error(function() numlua.fromfile(closed) end,
    "closed file handle")
  assert_error(function() numlua.tofile(closed, values) end,
    "closed file handle")
  cleanup(path)
end)

test("tofile reports write failures without closing a caller handle", function()
  local path = tmp_path("raw-readonly")
  assert_true(numlua.tofile(path, numlua.zeros(2)))
  local handle = assert(io.open(path, "rb"))
  assert_error(function()
    numlua.tofile(handle, numlua.zeros(2))
  end, "failed writing file handle")
  assert_eq(io.type(handle), "file")
  assert_true(handle:close())
  cleanup(path)
end)
