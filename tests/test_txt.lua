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

test("loadtxt squeezes one-row files and supports multi-character delimiters",
    function()
  local row_path = write_text(tmp_path("one-row"), "1,2,3\n")
  local row = numlua.loadtxt(row_path, { delimiter = "," })
  assert_shape(row, { 3 })
  assert_tensor(row, { 1, 2, 3 })

  local delimited_path = write_text(tmp_path("multi-delimiter"),
    "1::2::3\n4::5::6\n")
  local delimited = numlua.loadtxt(delimited_path,
    { delimiter = "::", dtype = "int64" })
  assert_shape(delimited, { 2, 3 })
  assert_dtype(delimited, "int64")
  assert_tensor(delimited, { { 1, 2, 3 }, { 4, 5, 6 } })
  os.remove(row_path)
  os.remove(delimited_path)
end)

test("text I/O handles zero-width and empty tensors", function()
  local empty_vector = tmp_path("empty-vector")
  assert_true(numlua.savetxt(empty_vector, numlua.zeros(0)))
  assert_eq(read_text(empty_vector), "")
  local loaded_empty = numlua.loadtxt(empty_vector)
  assert_shape(loaded_empty, { 0 })
  os.remove(empty_vector)

  -- A 2x0 array writes one newline per row; the text format cannot represent
  -- the zero-width column count, so it loads back as an empty vector.
  local zero_columns = tmp_path("zero-columns")
  assert_true(numlua.savetxt(zero_columns, numlua.zeros(2, 0)))
  assert_eq(read_text(zero_columns), "\n\n")
  local loaded_zero_columns = numlua.loadtxt(zero_columns)
  assert_shape(loaded_zero_columns, { 0 })
  os.remove(zero_columns)

  -- Zero rows and columns write nothing at all.
  local zero_rows = tmp_path("zero-rows")
  assert_true(numlua.savetxt(zero_rows, numlua.zeros(0, 3)))
  assert_eq(read_text(zero_rows), "")
  os.remove(zero_rows)
end)

test("text I/O disables comments and accepts fmt functions", function()
  local disabled = write_text(tmp_path("comments-off"), "1 2 # data\n")
  assert_error(function()
    numlua.loadtxt(disabled, { comments = false })
  end, "invalid number '#'")
  assert_error(function()
    numlua.loadtxt(disabled, { comments = {} })
  end, "invalid number '#'")
  assert_tensor(numlua.loadtxt(disabled), { 1, 2 })
  os.remove(disabled)

  -- A missing final newline is accepted, and numeric dtype ids work.
  local no_final_newline = write_text(tmp_path("no-final-newline"),
    "1 2\n3 4")
  local loaded = numlua.loadtxt(no_final_newline, { dtype = "int64" })
  assert_dtype(loaded, "int64")
  assert_tensor(loaded, { { 1, 2 }, { 3, 4 } })
  assert_dtype(numlua.loadtxt(no_final_newline, { dtype = 1 }), "float64")
  os.remove(no_final_newline)

  local formatted = tmp_path("fmt-function")
  assert_true(numlua.savetxt(formatted, numlua.array({ 1.5, 2.5 }), {
    fmt = function(value) return string.format("<%.1f>", value) end,
  }))
  assert_eq(read_text(formatted), "<1.5> <2.5>\n")
  assert_error(function()
    numlua.savetxt(formatted, numlua.array({ 1 }), {
      fmt = function() return 5 end,
    })
  end, "must produce a string")
  assert_error(function()
    numlua.savetxt(formatted, numlua.array({ 1 }), {
      fmt = function() error("boom") end,
    })
  end, "fmt function failed")
  os.remove(formatted)
end)

test("savetxt writes special values, vector shapes, and bad formats",
    function()
  local special = tmp_path("special-values")
  assert_true(numlua.savetxt(special,
    numlua.array({ 0 / 0, math.huge, -math.huge }, { dtype = "float64" })))
  assert_eq(read_text(special), "nan inf -inf\n")
  local loaded_special = numlua.loadtxt(special, { dtype = "float64" })
  assert_shape(loaded_special, { 3 })
  local special_values = loaded_special:totable()
  assert_true(special_values[1] ~= special_values[1], "expected NaN")
  assert_eq(special_values[2], math.huge)
  assert_eq(special_values[3], -math.huge)
  os.remove(special)

  -- One-row and one-column matrices write normal text but load back as 1-D,
  -- because the text format does not record the squeezed dimension.
  local row = tmp_path("one-row-save")
  assert_true(numlua.savetxt(row,
    numlua.array({ { 1.5, 2.5, -3.5 } }, { dtype = "float64" })))
  assert_eq(read_text(row), "1.5 2.5 -3.5\n")
  local loaded_row = numlua.loadtxt(row, { dtype = "float64" })
  assert_shape(loaded_row, { 3 })
  assert_tensor(loaded_row, { 1.5, 2.5, -3.5 }, 0)
  os.remove(row)

  local column = tmp_path("one-column-save")
  assert_true(numlua.savetxt(column,
    numlua.array({ { 1 }, { 2 }, { 3 } }, { dtype = "int64" })))
  assert_eq(read_text(column), "1\n2\n3\n")
  local loaded_column = numlua.loadtxt(column, { dtype = "int64" })
  assert_shape(loaded_column, { 3 })
  assert_tensor(loaded_column, { 1, 2, 3 })
  os.remove(column)

  -- Invalid printf strings and argument counts must fail with the facade's
  -- message instead of leaking a raw Lua string.format error.
  local invalid = tmp_path("invalid-format")
  assert_error(function()
    numlua.savetxt(invalid, numlua.array({ 1 }), { fmt = "%" })
  end, "invalid fmt")
  assert_error(function()
    numlua.savetxt(invalid, numlua.array({ 1 }), { fmt = "%d %d" })
  end, "invalid fmt")
  os.remove(invalid)
end)

test("loadtxt accepts every numeric dtype id and validates comments",
    function()
  local path = write_text(tmp_path("dtype-ids"), "1.5 2.5\n")
  local dtypes = {
    { 0, "float32" },
    { 1, "float64" },
    { 2, "int64" },
    { 3, "int32" },
    { 4, "int16" },
    { 5, "int8" },
    { 6, "uint8" },
    { 8, "float16" },
  }
  for _, case in ipairs(dtypes) do
    local loaded = numlua.loadtxt(path, { dtype = case[1] })
    assert_dtype(loaded, case[2])
    assert_shape(loaded, { 2 })
  end
  assert_error(function()
    numlua.loadtxt(path, { dtype = 7 })
  end, "unsupported dtype 'bool'")
  os.remove(path)

  -- Comment marker tables reject empty entries and non-strings, and scalar
  -- `true` is not a valid spelling for "enabled comments".
  local markers = write_text(tmp_path("bad-markers"), "1 2\n")
  assert_error(function()
    numlua.loadtxt(markers, { comments = { "#", "" } })
  end, "non-empty strings")
  assert_error(function()
    numlua.loadtxt(markers, { comments = { 5 } })
  end, "non-empty strings")
  assert_error(function()
    numlua.loadtxt(markers, { comments = true })
  end, "comments must be")
  os.remove(markers)

  -- Delimiters are matched literally, not as Lua patterns, and can be the
  -- same character as a comment marker when comment stripping is disabled.
  local pipe = write_text(tmp_path("pipe-delimiter"), "1|2\n3|4\n")
  local piped = numlua.loadtxt(pipe, { delimiter = "|" })
  assert_shape(piped, { 2, 2 })
  assert_tensor(piped, { { 1, 2 }, { 3, 4 } })
  os.remove(pipe)

  local hash = write_text(tmp_path("hash-delimiter"), "1#2\n3#4\n")
  local hashed = numlua.loadtxt(hash, { delimiter = "#", comments = false })
  assert_shape(hashed, { 2, 2 })
  assert_tensor(hashed, { { 1, 2 }, { 3, 4 } })
  os.remove(hash)
end)

test("loadtxt accepts CR-only and mixed line endings", function()
  local cr = write_text(tmp_path("cr-only"), "1 2\r3 4\r")
  local cr_loaded = numlua.loadtxt(cr)
  assert_shape(cr_loaded, { 2, 2 })
  assert_tensor(cr_loaded, { { 1, 2 }, { 3, 4 } })
  os.remove(cr)

  local mixed = write_text(tmp_path("mixed-endings"),
    "1,2\r\n3,4\r5,6\n")
  local mixed_loaded = numlua.loadtxt(mixed, { delimiter = "," })
  assert_shape(mixed_loaded, { 3, 2 })
  assert_tensor(mixed_loaded, { { 1, 2 }, { 3, 4 }, { 5, 6 } })
  os.remove(mixed)

  -- A single CR-terminated row still squeezes to 1-D like an LF row.
  local one_row = write_text(tmp_path("cr-one-row"), "7 8 9\r")
  assert_shape(numlua.loadtxt(one_row), { 3 })
  assert_tensor(numlua.loadtxt(one_row), { 7, 8, 9 })
  os.remove(one_row)
end)

test("text options reject unknown keys before touching the file",
    function()
  local path = write_text(tmp_path("known-options"), "1 2\n")
  assert_error(function()
    numlua.loadtxt(path, { delimeter = "," })
  end, "unknown option 'delimeter'")
  assert_error(function()
    numlua.loadtxt(path, { comments = "#", formatter = tostring })
  end, "unknown option 'formatter'")
  assert_error(function()
    numlua.savetxt(tmp_path("unknown-save"), numlua.zeros(2),
      { delimeter = "," })
  end, "unknown option 'delimeter'")
  assert_error(function()
    numlua.savetxt(tmp_path("unknown-save"), numlua.zeros(2),
      { fmt = "%.2f", header = "# data" })
  end, "unknown option 'header'")
  os.remove(path)

  -- Unknown-key validation happens before open/read errors and dtype checks.
  assert_error(function()
    numlua.loadtxt(TMPDIR .. "/definitely-missing-options.txt",
      { formatter = tostring })
  end, "unknown option 'formatter'")
  assert_error(function()
    numlua.loadtxt(path, { dtype = 99 })
  end, "unsupported dtype 99")
end)

test("text formatting covers delimiters, integer bounds, and float16",
    function()
  -- Multi-character save delimiters match the loader's literal matching.
  local delimited = tmp_path("multi-save-delimiter")
  assert_true(numlua.savetxt(delimited,
    numlua.arange(4):reshape(2, 2), { delimiter = "::" }))
  assert_eq(read_text(delimited), "0::1\n2::3\n")
  local loaded = numlua.loadtxt(delimited,
    { delimiter = "::", dtype = "int64" })
  assert_shape(loaded, { 2, 2 })
  assert_tensor(loaded, { { 0, 1 }, { 2, 3 } })
  os.remove(delimited)

  -- Integer text uses base 10, including the signed/unsigned bounds.
  local bounds = tmp_path("integer-bounds")
  assert_true(numlua.savetxt(bounds,
    numlua.array({ -128, 127 }, { dtype = "int8" })))
  assert_eq(read_text(bounds), "-128 127\n")
  local unsigned = tmp_path("unsigned-bounds")
  assert_true(numlua.savetxt(unsigned,
    numlua.array({ 0, 255 }, { dtype = "uint8" })))
  assert_eq(read_text(unsigned), "0 255\n")
  os.remove(bounds)
  os.remove(unsigned)

  -- Values inside the exact double range round-trip through int64 text.
  local exact = tmp_path("int64-exact")
  local safe = { -9007199254740991, 9007199254740991 }
  assert_true(numlua.savetxt(exact,
    numlua.array(safe, { dtype = "int64" })))
  local exact_loaded = numlua.loadtxt(exact, { dtype = "int64" })
  assert_dtype(exact_loaded, "int64")
  assert_tensor(exact_loaded, safe)
  os.remove(exact)

  -- float16 keeps enough digits for a lossless round-trip.
  local half = tmp_path("float16")
  local values = { 0.5, -1.25, 65504 }
  assert_true(numlua.savetxt(half,
    numlua.array(values, { dtype = "float16" })))
  local half_loaded = numlua.loadtxt(half, { dtype = "float16" })
  assert_dtype(half_loaded, "float16")
  assert_tensor(half_loaded, values)
  os.remove(half)

  -- A trailing delimiter is an empty field, not a silently ignored column.
  local trailing = write_text(tmp_path("trailing-delimiter"), "1,2,\n3,4,\n")
  assert_error(function()
    numlua.loadtxt(trailing, { delimiter = "," })
  end, "invalid number '' in column 3")
  os.remove(trailing)
end)

test("loadtxt skiprows counts physical lines and max_rows data rows",
    function()
  local path = write_text(tmp_path("row-window"),
    "# header\n" ..
    "\n" ..
    "10 11\n" ..
    "# between\n" ..
    "20 21\n" ..
    "30 31\n")

  -- skiprows counts comments and blank lines, not data rows.
  assert_tensor(numlua.loadtxt(path, { skiprows = 3 }),
    { { 20, 21 }, { 30, 31 } })
  assert_tensor(numlua.loadtxt(path, { skiprows = 4 }),
    { { 20, 21 }, { 30, 31 } })
  assert_tensor(numlua.loadtxt(path, { skiprows = 5 }), { 30, 31 })

  -- max_rows counts data rows only; comment/blank lines do not consume it.
  assert_tensor(numlua.loadtxt(path, { max_rows = 1 }), { 10, 11 })
  assert_tensor(numlua.loadtxt(path, { max_rows = 2 }),
    { { 10, 11 }, { 20, 21 } })

  -- The window is applied after skipping, and max_rows=0 reads no data.
  assert_tensor(numlua.loadtxt(path, { skiprows = 3, max_rows = 1 }),
    { 20, 21 })
  local none = numlua.loadtxt(path, { max_rows = 0 })
  assert_shape(none, { 0 })
  assert_eq(none:numel(), 0)

  -- Skipping every line is an empty result rather than an error.
  local empty = numlua.loadtxt(path, { skiprows = 99 })
  assert_shape(empty, { 0 })
  os.remove(path)

  local malformed = write_text(tmp_path("skipped-malformed"),
    "not numbers\n1 2\n3 4\n")
  assert_tensor(numlua.loadtxt(malformed, { skiprows = 1 }),
    { { 1, 2 }, { 3, 4 } })
  assert_error(function()
    numlua.loadtxt(malformed, { skiprows = 1, max_rows = 1, delimiter = "," })
  end, "line 2: invalid number '1 2' in column 1")
  os.remove(malformed)
end)

test("loadtxt validates skiprows and max_rows", function()
  local path = write_text(tmp_path("window-errors"), "1 2\n")
  for _, option in ipairs({ "skiprows", "max_rows" }) do
    for _, value in ipairs({ -1, 1.5, "1", true }) do
      assert_error(function()
        numlua.loadtxt(path, { [option] = value })
      end, option .. " must be a non-negative integer")
    end
  end
  os.remove(path)
end)
