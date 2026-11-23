--
-- NumPy text-mode raw I/O: fromfile/tofile with a non-empty sep.
--

local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".txt"
end

local function read_text(path)
  local handle = assert(io.open(path, "rb"))
  local data = handle:read("*a")
  handle:close()
  return data
end

local function write_text(path, data)
  local handle = assert(io.open(path, "wb"))
  assert(handle:write(data))
  handle:close()
end

test("tofile and fromfile round-trip text for every dtype", function()
  local cases = {
    { dtype = "float32", values = { 1.5, -2.25, 0 }, text = "1.5,-2.25,0\n" },
    { dtype = "float64", values = { 1.5, -2.25, 0 }, text = "1.5,-2.25,0\n" },
    { dtype = "float16", values = { 1.5, -2, 0 }, text = "1.5,-2,0\n" },
    { dtype = "int64", values = { -3, 0, 7 }, text = "-3,0,7\n" },
    { dtype = "int32", values = { -3, 0, 7 }, text = "-3,0,7\n" },
    { dtype = "int16", values = { -3, 0, 7 }, text = "-3,0,7\n" },
    { dtype = "int8", values = { -3, 0, 7 }, text = "-3,0,7\n" },
    { dtype = "uint8", values = { 3, 0, 7 }, text = "3,0,7\n" },
    { dtype = "bool", values = { true, false, true },
      text = "True,False,True\n" },
  }
  for _, case in ipairs(cases) do
    local path = tmp_path("text-" .. case.dtype)
    local input = numlua.array(case.values, { dtype = case.dtype })
    assert_true(numlua.tofile(path, input, { sep = "," }))
    assert_eq(read_text(path), case.text)

    local loaded = numlua.fromfile(path, { dtype = case.dtype, sep = "," })
    assert_dtype(loaded, case.dtype)
    assert_shape(loaded, { #case.values })
    assert_tensor(loaded, case.values)
    os.remove(path)
  end
end)

test("text mode supports separators, whitespace, and custom formats", function()
  local path = tmp_path("text-separators")
  local values = numlua.array({ 1, 2, 3 }, { dtype = "int32" })

  assert_true(numlua.tofile(path, values, { sep = " " }))
  assert_eq(read_text(path), "1 2 3\n")
  assert_tensor(numlua.fromfile(path, { dtype = "int32", sep = " " }),
    { 1, 2, 3 })
  -- A whitespace separator accepts any run of whitespace when reading.
  assert_tensor(numlua.fromfile(path, { dtype = "int32", sep = "\t" }),
    { 1, 2, 3 })

  assert_true(numlua.tofile(path, values, { sep = " | " }))
  assert_eq(read_text(path), "1 | 2 | 3\n")
  assert_tensor(numlua.fromfile(path, { dtype = "int32", sep = " | " }),
    { 1, 2, 3 })

  local floats = numlua.array({ 1.25, 2.5 }, { dtype = "float64" })
  assert_true(numlua.tofile(path, floats, { sep = ";", format = "%.2f" }))
  assert_eq(read_text(path), "1.25;2.50\n")
  assert_tensor(numlua.fromfile(path, { dtype = "float64", sep = ";" }),
    { 1.25, 2.5 })

  local bools = numlua.array({ true, false })
  assert_true(numlua.tofile(path, bools, { sep = ",", format = "%d" }))
  assert_eq(read_text(path), "1,0\n")
  assert_tensor(numlua.fromfile(path, { dtype = "bool", sep = "," }),
    { true, false })

  local empty = numlua.zeros(0)
  assert_true(numlua.tofile(path, empty, { sep = "," }))
  assert_eq(read_text(path), "")
  assert_shape(numlua.fromfile(path, { sep = "," }), { 0 })
  os.remove(path)
end)

test("text mode honors count and offset", function()
  local path = tmp_path("text-window")
  write_text(path, "10,20,30,40\n")

  assert_tensor(numlua.fromfile(path, { dtype = "int32", sep = "," }),
    { 10, 20, 30, 40 })
  assert_tensor(numlua.fromfile(path, {
    dtype = "int32", sep = ",", count = 2,
  }), { 10, 20 })
  assert_tensor(numlua.fromfile(path, {
    dtype = "int32", sep = ",", count = 2, offset = 3,
  }), { 20, 30 })
  assert_tensor(numlua.fromfile(path, {
    dtype = "int32", sep = ",", count = 99,
  }), { 10, 20, 30, 40 })
  assert_shape(numlua.fromfile(path, {
    dtype = "int32", sep = ",", count = 0,
  }), { 0 })
  assert_shape(numlua.fromfile(path, {
    dtype = "int32", sep = ",", offset = #read_text(path),
  }), { 0 })
  os.remove(path)
end)

test("text mode parses floats, integers, bools, and NaN/infinity", function()
  local path = tmp_path("text-parsing")
  write_text(path, "nan,inf,-inf,1.5\n")
  local floats = numlua.fromfile(path, { dtype = "float64", sep = "," })
  assert_eq(floats:numel(), 4)
  assert_true(floats:get(1) ~= floats:get(1))
  assert_eq(floats:get(2), math.huge)
  assert_eq(floats:get(3), -math.huge)
  assert_eq(floats:get(4), 1.5)

  write_text(path, "1.9,-2.9\n")
  assert_tensor(numlua.fromfile(path, { dtype = "int64", sep = "," }),
    { 1, -2 })

  write_text(path, "true,FALSE,1,0\n")
  assert_tensor(numlua.fromfile(path, { dtype = "bool", sep = "," }),
    { true, false, true, false })

  -- Whitespace-only input is empty, not an error.
  write_text(path, "  \n\t\n")
  assert_shape(numlua.fromfile(path, { sep = "," }), { 0 })

  -- Empty interior fields are malformed, while trailing ones are ignored.
  write_text(path, "1,,2\n")
  assert_error(function()
    numlua.fromfile(path, { dtype = "int32", sep = "," })
  end, "empty text field")
  write_text(path, ",1\n")
  assert_error(function()
    numlua.fromfile(path, { dtype = "int32", sep = "," })
  end, "empty text field")
  write_text(path, "1,2,\n")
  assert_tensor(numlua.fromfile(path, { dtype = "int32", sep = "," }),
    { 1, 2 })

  write_text(path, "1.5,abc\n")
  assert_error(function()
    numlua.fromfile(path, { dtype = "float64", sep = "," })
  end, "cannot parse text value")
  write_text(path, "nan,2\n")
  assert_error(function()
    numlua.fromfile(path, { dtype = "int64", sep = "," })
  end, "non-finite text value")
  os.remove(path)
end)

test("text mode uses caller-owned handles at their current position", function()
  local path = tmp_path("text-handle")
  local handle = assert(io.open(path, "w+b"))
  assert(handle:write("prefix:"))
  assert_true(numlua.tofile(handle,
    numlua.array({ 4, 5 }, { dtype = "int32" }), { sep = "," }))
  assert_eq(io.type(handle), "file", "tofile must not close a caller handle")
  assert_eq(handle:seek("set", 0), 0)
  assert_eq(handle:read(7), "prefix:")

  local loaded = numlua.fromfile(handle, { dtype = "int32", sep = "," })
  assert_tensor(loaded, { 4, 5 })
  assert_eq(io.type(handle), "file", "fromfile must not close a caller handle")
  assert_true(handle:close())
  os.remove(path)
end)

test("text mode validates options, formats, and write failures", function()
  local path = tmp_path("text-errors")
  local values = numlua.array({ 1, 2 })
  assert_true(numlua.tofile(path, values, { sep = "," }))

  assert_error(function()
    numlua.tofile(path, values, { sep = 1 })
  end, "sep must be a string")
  assert_error(function()
    numlua.tofile(path, values, { sep = ",", format = 1 })
  end, "format must be a non-empty string")
  assert_error(function()
    numlua.tofile(path, values, { sep = ",", format = "%d %d" })
  end, "invalid text format")
  assert_error(function()
    numlua.tofile(path, values, { nope = true })
  end, "unknown option")
  assert_error(function()
    numlua.fromfile(path, { sep = ",", dtype = "complex64" })
  end, "unsupported dtype")

  local readonly = assert(io.open(path, "rb"))
  assert_error(function()
    numlua.tofile(readonly, values, { sep = "," })
  end, "failed writing file handle")
  assert_eq(io.type(readonly), "file")
  assert_true(readonly:close())

  local closed = assert(io.open(path, "rb"))
  assert_true(closed:close())
  assert_error(function()
    numlua.fromfile(closed, { sep = "," })
  end, "closed file handle")
  assert_error(function()
    numlua.tofile(closed, values, { sep = "," })
  end, "closed file handle")
  os.remove(path)
end)
