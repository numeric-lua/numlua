--
-- Text I/O parity: loadtxt/savetxt accept caller-owned Lua file handles.
--

local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".txt"
end

test("savetxt and loadtxt round-trip through an open handle", function()
  local path = tmp_path("text-handle-round-trip")
  local handle = assert(io.open(path, "w+b"))
  local input = numlua.array({
    { 1.5, 2.5 },
    { 3.5, 4.5 },
  })
  assert_true(numlua.savetxt(handle, input, {
    delimiter = ",",
    header = "values",
    comments = "# ",
  }))
  assert_eq(io.type(handle), "file", "savetxt must not close a caller handle")
  assert_eq(handle:seek("set", 0), 0)

  local loaded = numlua.loadtxt(handle, {
    delimiter = ",",
    dtype = "float32",
  })
  assert_eq(io.type(handle), "file", "loadtxt must not close a caller handle")
  assert_dtype(loaded, "float32")
  assert_true(loaded:equal(input))
  assert_true(handle:close())
  os.remove(path)
end)

test("loadtxt starts at a handle's current position", function()
  local path = tmp_path("text-handle-sequence")
  local first = numlua.array({ { 1, 2 }, { 3, 4 } })
  local second = numlua.array({ 10, 20, 30 })
  local handle = assert(io.open(path, "w+b"))

  assert_true(numlua.savetxt(handle, first, { delimiter = "," }))
  local second_start = handle:seek("cur")
  assert_true(second_start > 0)
  assert_true(numlua.savetxt(handle, second, { delimiter = "," }))

  assert_eq(handle:seek("set", 0), 0)
  local loaded_first = numlua.loadtxt(handle, {
    delimiter = ",",
    dtype = "int64",
    max_rows = 2,
  })
  assert_tensor(loaded_first, { { 1, 2 }, { 3, 4 } })

  assert_eq(handle:seek("set", second_start), second_start)
  local loaded_second = numlua.loadtxt(handle, {
    delimiter = ",",
    dtype = "int64",
  })
  assert_tensor(loaded_second, { 10, 20, 30 })
  assert_true(handle:close())
  os.remove(path)
end)

test("loadtxt reads a handle from its current position and keeps options", function()
  local path = tmp_path("text-handle-position")
  local handle = assert(io.open(path, "w+b"))
  assert_true(handle:write("# skip me\n", "1,2,3\n", "4,5,6\n"))
  assert_eq(handle:seek("set", 0), 0)

  local loaded = numlua.loadtxt(handle, {
    delimiter = ",",
    comments = "#",
    dtype = "int64",
    usecols = { 3, 1 },
    ndmin = 2,
  })
  assert_tensor(loaded, { { 3, 1 }, { 6, 4 } })
  assert_true(handle:close())

  -- Starting after the first data row reads only the remaining row.
  handle = assert(io.open(path, "r"))
  assert_true(handle:read("*l")) -- comment
  assert_true(handle:read("*l")) -- first data row
  local rest = numlua.loadtxt(handle, { delimiter = ",", dtype = "int64" })
  assert_tensor(rest, { 4, 5, 6 })
  assert_true(handle:close())
  os.remove(path)
end)

test("empty and path-based text I/O still behave as before", function()
  local path = tmp_path("text-handle-empty")
  local handle = assert(io.open(path, "w+b"))
  local empty = numlua.loadtxt(handle, { ndmin = 2 })
  assert_shape(empty, { 0, 1 })
  assert_true(handle:close())

  local input = numlua.array({ { 1.25, 2.5 }, { 3.75, 4 } })
  assert_true(numlua.savetxt(path, input, { delimiter = "," }))
  local loaded = numlua.loadtxt(path, { delimiter = "," })
  assert_true(loaded:equal(input))
  os.remove(path)
end)

test("text handle I/O rejects closed handles and invalid targets", function()
  local path = tmp_path("text-handle-errors")
  local handle = assert(io.open(path, "wb"))
  assert_true(handle:close())

  assert_error(function() numlua.loadtxt(handle) end, "closed file handle")
  assert_error(function() numlua.savetxt(handle, numlua.zeros(2)) end,
    "closed file handle")
  assert_error(function() numlua.loadtxt(nil) end,
    "non-empty file path or open file handle")
  assert_error(function() numlua.loadtxt({}) end,
    "non-empty file path or open file handle")
  assert_error(function() numlua.loadtxt("") end,
    "non-empty file path or open file handle")
  assert_error(function() numlua.savetxt(nil, numlua.zeros(2)) end,
    "non-empty file path or open file handle")
  assert_error(function() numlua.savetxt({}, numlua.zeros(2)) end,
    "non-empty file path or open file handle")
  assert_error(function() numlua.savetxt("", numlua.zeros(2)) end,
    "non-empty file path or open file handle")
  os.remove(path)
end)

test("a text write failure does not close the caller's handle", function()
  local path = tmp_path("text-handle-readonly")
  assert_true(numlua.savetxt(path, numlua.zeros(2)))
  local handle = assert(io.open(path, "rb"))
  assert_error(function()
    numlua.savetxt(handle, numlua.zeros(2))
  end, "failed writing file handle")
  assert_eq(io.type(handle), "file")
  assert_true(handle:close())
  os.remove(path)
end)
