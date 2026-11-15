local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".txt"
end

local function read_text(path)
  local handle = assert(io.open(path, "r"))
  local data = handle:read("*a")
  handle:close()
  return data
end

local function write_text(path, data)
  local handle = assert(io.open(path, "w"))
  assert(handle:write(data))
  handle:close()
  return path
end

test("text round-trips 1-D and 2-D arrays with whitespace", function()
  local vector = numlua.array({ 1.5, -2.25, 0, 3.5 })
  local vector_path = tmp_path("vector")
  assert_true(numlua.savetxt(vector_path, vector))
  local loaded_vector = numlua.loadtxt(vector_path)
  assert_dtype(loaded_vector, "float32")
  assert_shape(loaded_vector, { 4 })
  assert_tensor(loaded_vector, vector:totable(), 0)
  os.remove(vector_path)

  local matrix = numlua.array({
    { 1.5, 2.5, 3.5 },
    { 4.5, -5.5, 6.5 },
  })
  local matrix_path = tmp_path("matrix")
  assert_true(numlua.savetxt(matrix_path, matrix))
  local loaded_matrix = numlua.loadtxt(matrix_path)
  assert_dtype(loaded_matrix, "float32")
  assert_shape(loaded_matrix, { 2, 3 })
  assert_tensor(loaded_matrix, matrix:totable(), 0)
  assert_true(loaded_matrix:equal(matrix))
  os.remove(matrix_path)
end)

test("CSV round-trips float64 and int64 values", function()
  local floats = numlua.array({
    { 1.25, -2.5, math.pi },
    { 3.75, 4.125, -1e-7 },
  }, { dtype = "float64" })
  local float_path = tmp_path("float64")
  assert_true(numlua.savetxt(float_path, floats, { delimiter = "," }))
  local loaded_floats = numlua.loadtxt(float_path, {
    delimiter = ",",
    dtype = "float64",
  })
  assert_dtype(loaded_floats, "float64")
  assert_shape(loaded_floats, { 2, 3 })
  assert_true(loaded_floats:equal(floats))
  os.remove(float_path)

  local ints = numlua.array({
    { -3, 0, 7 },
    { 42, -100, 9 },
  }, { dtype = "int64" })
  local int_path = tmp_path("int64")
  assert_true(numlua.savetxt(int_path, ints, { delimiter = "," }))
  local loaded_ints = numlua.loadtxt(int_path, {
    delimiter = ",",
    dtype = "int64",
  })
  assert_dtype(loaded_ints, "int64")
  assert_tensor(loaded_ints, ints:totable(), 0)
  assert_true(loaded_ints:equal(ints))
  os.remove(int_path)
end)

test("loadtxt handles comments, blank lines, and CRLF input", function()
  local path = write_text(tmp_path("comments"),
    "# generated file\r\n" ..
    "\r\n" ..
    "1.25, -2.5\r\n" ..
    "# a comment between rows\r\n" ..
    "3.75, 4.125\r\n")
  local t = numlua.loadtxt(path, { delimiter = ",", comments = "#" })
  assert_dtype(t, "float32")
  assert_shape(t, { 2, 2 })
  assert_tensor(t, { { 1.25, -2.5 }, { 3.75, 4.125 } }, 0)
  os.remove(path)
end)

test("loadtxt accepts multiple comment markers and a custom marker", function()
  local path = write_text(tmp_path("markers"),
    "% header\n" ..
    "// another header\n" ..
    "1 2\n" ..
    "% between\n" ..
    "3 4\n" ..
    "1 2 # trailing comment\n")
  local t = numlua.loadtxt(path, { comments = { "%", "//", "#" } })
  assert_shape(t, { 3, 2 })
  assert_tensor(t, { { 1, 2 }, { 3, 4 }, { 1, 2 } })
  os.remove(path)

  local custom = write_text(tmp_path("semicolon-comment"), "; ignore\n5\n6\n")
  assert_tensor(numlua.loadtxt(custom, { comments = ";" }), { 5, 6 })
  os.remove(custom)
end)

test("loadtxt returns a 1-D tensor for one-column files", function()
  local path = write_text(tmp_path("one-column"), "1.5\n2.5\n-3.5\n")
  local t = numlua.loadtxt(path, { dtype = "float64" })
  assert_dtype(t, "float64")
  assert_shape(t, { 3 })
  assert_tensor(t, { 1.5, 2.5, -3.5 }, 0)
  os.remove(path)
end)

test("loadtxt handles empty files and special floating values", function()
  local empty_path = write_text(tmp_path("empty"), "# only comments\n\n")
  local empty = numlua.loadtxt(empty_path)
  assert_dtype(empty, "float32")
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)
  os.remove(empty_path)

  local special_path = write_text(tmp_path("special"),
    "nan inf -inf\n")
  local special = numlua.loadtxt(special_path)
  assert_shape(special, { 3 })
  local values = special:totable()
  assert_true(values[1] ~= values[1], "expected NaN")
  assert_eq(values[2], math.huge)
  assert_eq(values[3], -math.huge)
  os.remove(special_path)
end)

test("savetxt supports fmt and writes non-contiguous tensors in C order", function()
  local path = tmp_path("fmt")
  local t = numlua.array({ { 1.23456, 2.34567 } }, { dtype = "float64" })
  assert_true(numlua.savetxt(path, t, { delimiter = ",", fmt = "%.2f" }))
  assert_eq(read_text(path), "1.23,2.35\n")
  os.remove(path)

  local transposed = numlua.transpose(numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }))
  assert_false(transposed:is_contiguous())
  local transposed_path = tmp_path("non-contiguous")
  assert_true(numlua.savetxt(transposed_path, transposed))
  assert_eq(read_text(transposed_path), "1 4\n2 5\n3 6\n")
  local loaded = numlua.loadtxt(transposed_path, { dtype = "int64" })
  assert_tensor(loaded, { { 1, 4 }, { 2, 5 }, { 3, 6 } })
  os.remove(transposed_path)
end)

test("loadtxt reports malformed rows and invalid numbers", function()
  local bad_columns = write_text(tmp_path("bad-columns"), "1 2\n3\n")
  assert_error(function() numlua.loadtxt(bad_columns) end,
    "line 2 has 1 column(s), expected 2")
  os.remove(bad_columns)

  local bad_number = write_text(tmp_path("bad-number"), "1 2\n3 nope\n")
  assert_error(function() numlua.loadtxt(bad_number) end,
    "line 2: invalid number 'nope' in column 2")
  os.remove(bad_number)

  local bad_csv = write_text(tmp_path("bad-csv"), "1,2\n3,4,5\n")
  assert_error(function()
    numlua.loadtxt(bad_csv, { delimiter = "," })
  end, "line 2 has 3 column(s), expected 2")
  os.remove(bad_csv)
end)

test("text I/O rejects unsupported dtypes and invalid arguments", function()
  local bool_path = tmp_path("bool")
  local bools = numlua.array({ true, false })
  assert_error(function() numlua.savetxt(bool_path, bools) end,
    "unsupported dtype 'bool'")
  assert_error(function()
    numlua.loadtxt(bool_path, { dtype = "bool" })
  end, "unsupported dtype 'bool'")

  local path = write_text(tmp_path("dtype-errors"), "1 2\n")
  assert_error(function()
    numlua.loadtxt(path, { dtype = "complex64" })
  end, "unsupported dtype 'complex64'")
  os.remove(path)

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    numlua.savetxt(tmp_path("scalar"), numlua.array(1))
  end, "supports 1-D and 2-D arrays, got 0-D")
  assert_error(function()
    numlua.savetxt(tmp_path("3d"), numlua.zeros(1, 2, 3))
  end, "supports 1-D and 2-D arrays, got 3-D")
  assert_error(function() numlua.savetxt(tmp_path("table"), { 1, 2 }) end,
    "expects a tensor")

  assert_error(function() numlua.loadtxt(nil) end,
    "expects a non-empty file path")
  assert_error(function() numlua.savetxt("", matrix) end,
    "expects a non-empty file path")
  assert_error(function()
    numlua.loadtxt(TMPDIR .. "/definitely-missing.txt")
  end, "cannot open")
  assert_error(function()
    numlua.loadtxt(TMPDIR .. "/definitely-missing.txt", 5)
  end, "options must be a table")
  assert_error(function()
    numlua.savetxt(tmp_path("delimiter"), matrix, { delimiter = "" })
  end, "delimiter must be a non-empty string")
  assert_error(function()
    numlua.savetxt(tmp_path("fmt"), matrix, { fmt = 5 })
  end, "fmt must be a format string")
  assert_error(function()
    numlua.loadtxt(TMPDIR .. "/definitely-missing.txt", { comments = 5 })
  end, "comments must be a string")
end)
