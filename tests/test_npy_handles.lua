--
-- .npy file-object I/O parity: save/load accept caller-owned Lua file handles.
--

local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".npy"
end

test("save and load accept an open file handle", function()
  local path = tmp_path("handle-round-trip")
  local handle = assert(io.open(path, "w+b"))
  local input = numlua.transpose(numlua.array({
    { 1.5, 2.5 },
    { 3.5, 4.5 },
    { 5.5, 6.5 },
  }))
  assert_false(input:is_contiguous())

  assert_true(numlua.save(handle, input, { version = 3 }))
  assert_eq(io.type(handle), "file", "save must not close a caller handle")
  assert_eq(handle:seek("set", 0), 0)

  local loaded = numlua.load(handle)
  assert_eq(io.type(handle), "file", "load must not close a caller handle")
  assert_dtype(loaded, "float32")
  assert_true(loaded:equal(input))
  assert_true(handle:close())
  os.remove(path)
end)

test("one handle can hold several arrays at sequential positions", function()
  local path = tmp_path("handle-sequence")
  local first = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  local second = numlua.zeros(2, 0, { dtype = "float64" })
  local handle = assert(io.open(path, "w+b"))

  assert_true(numlua.save(handle, first))
  assert_true(numlua.save(handle, second))
  assert_eq(handle:seek("set", 0), 0)

  local loaded_first = numlua.load(handle)
  local loaded_second = numlua.load(handle)
  assert_dtype(loaded_first, "int64")
  assert_true(loaded_first:equal(first))
  assert_dtype(loaded_second, "float64")
  assert_shape(loaded_second, { 2, 0 })
  assert_eq(loaded_second:numel(), 0)
  assert_true(handle:close())
  os.remove(path)
end)

test("save still writes paths and load still reads paths", function()
  local path = tmp_path("path-round-trip")
  local input = numlua.array({ 1, 2, 3 }, { dtype = "int16" })
  assert_true(numlua.save(path, input))
  local loaded = numlua.load(path)
  assert_dtype(loaded, "int16")
  assert_true(loaded:equal(input))
  os.remove(path)
end)

test("save and load reject closed handles and invalid targets", function()
  local path = tmp_path("closed-handle")
  local handle = assert(io.open(path, "wb"))
  assert_true(handle:close())

  assert_error(function()
    numlua.save(handle, numlua.zeros(2))
  end, "closed file handle")
  assert_error(function()
    numlua.load(handle)
  end, "closed file handle")

  assert_error(function()
    numlua.save(nil, numlua.zeros(2))
  end, "non-empty file path or open file handle")
  assert_error(function()
    numlua.load({})
  end, "non-empty file path or open file handle")
  assert_error(function()
    numlua.save("", numlua.zeros(2))
  end, "non-empty file path or open file handle")
  assert_error(function()
    numlua.load("")
  end, "non-empty file path or open file handle")
  os.remove(path)
end)

test("a write failure does not close the caller's handle", function()
  local path = tmp_path("read-only-handle")
  assert_true(numlua.save(path, numlua.zeros(2)))
  local handle = assert(io.open(path, "rb"))
  assert_error(function()
    numlua.save(handle, numlua.zeros(2))
  end, "failed writing file handle")
  assert_eq(io.type(handle), "file")
  assert_true(handle:close())
  os.remove(path)
end)
