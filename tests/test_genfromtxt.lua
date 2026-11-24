--
-- genfromtxt: numeric text loading with missing-value handling.
--

local numlua = require("numlua")

local counter = 0
local function write_text(name, content)
  counter = counter + 1
  local path = TMPDIR .. "/genfromtxt-" .. name .. "-" .. counter .. ".csv"
  local handle = assert(io.open(path, "w"))
  assert_true(handle:write(content))
  assert_true(handle:close())
  return path
end

test("genfromtxt skips headers and footers", function()
  local path = write_text("skip", "# title\nskip\n1,2\n3,4\n# footer\n")
  local loaded = numlua.genfromtxt(path, {
    delimiter = ",",
    skip_header = 2,
    skip_footer = 1,
    dtype = "float64",
  })
  assert_dtype(loaded, "float64")
  assert_shape(loaded, { 2, 2 })
  assert_tensor(loaded, { { 1, 2 }, { 3, 4 } })
  os.remove(path)
end)

test("genfromtxt fills missing values with NaN or explicit values", function()
  local path = write_text("missing", "1,NA\n2,3\n")
  local default = numlua.genfromtxt(path, {
    delimiter = ",",
    missing_values = "NA",
  })
  assert_dtype(default, "float32")
  assert_shape(default, { 2, 2 })
  assert_true(default:get(1, 2) ~= default:get(1, 2),
    "a missing float should default to NaN")
  assert_eq(default:get(2, 2), 3)

  local filled = numlua.genfromtxt(path, {
    delimiter = ",",
    missing_values = "NA",
    filling_values = 9,
  })
  assert_tensor(filled, { { 1, 9 }, { 2, 3 } })

  local integer = numlua.genfromtxt(path, {
    delimiter = ",",
    missing_values = "NA",
    dtype = "int64",
  })
  assert_dtype(integer, "int64")
  assert_eq(integer:get(1, 2), 0, "missing integer default should be zero")
  os.remove(path)
end)

test("genfromtxt accepts missing-value lists and per-column maps", function()
  local path = write_text("maps", "1,NA,3\n4,5,N/A\n")
  local global = numlua.genfromtxt(path, {
    delimiter = ",",
    missing_values = { "NA", "N/A" },
    filling_values = -1,
  })
  assert_tensor(global, { { 1, -1, 3 }, { 4, 5, -1 } })

  local per_column = numlua.genfromtxt(path, {
    delimiter = ",",
    missing_values = { [2] = "NA", [3] = "N/A" },
    filling_values = { [2] = 20, [3] = 30 },
  })
  assert_tensor(per_column, { { 1, 20, 3 }, { 4, 5, 30 } })
  os.remove(path)
end)

test("genfromtxt applies usecols before mapping missing columns", function()
  local path = write_text("usecols", "1,NA,3\n4,5,N/A\n")
  local selected = numlua.genfromtxt(path, {
    delimiter = ",",
    usecols = { 1, 3 },
    missing_values = { [3] = "N/A" },
    filling_values = { [3] = -1 },
  })
  assert_shape(selected, { 2, 2 })
  assert_tensor(selected, { { 1, 3 }, { 4, -1 } })
  os.remove(path)
end)

test("genfromtxt converters take precedence over missing values", function()
  local path = write_text("converters", "1,NA\n2,3\n")
  local converted = numlua.genfromtxt(path, {
    delimiter = ",",
    converters = {
      [2] = function(field)
        if field == "NA" then return 42 end
        return tonumber(field)
      end,
    },
  })
  assert_tensor(converted, { { 1, 42 }, { 2, 3 } })

  local all = numlua.genfromtxt(path, {
    delimiter = ",",
    converters = function(field)
      if field == "NA" then return -5 end
      return tonumber(field)
    end,
  })
  assert_tensor(all, { { 1, -5 }, { 2, 3 } })
  os.remove(path)
end)

test("genfromtxt handles max_rows, ndmin, unpack, and empty files", function()
  local path = write_text("shape", "# comment\n1 2\n\n3 4\n")
  local first = numlua.genfromtxt(path, {
    skip_header = 1,
    max_rows = 1,
    ndmin = 2,
  })
  assert_shape(first, { 1, 2 })
  assert_tensor(first, { { 1, 2 } })

  local unpacked = numlua.genfromtxt(path, {
    skip_header = 1,
    unpack = true,
  })
  assert_shape(unpacked, { 2, 2 })
  assert_tensor(unpacked, { { 1, 3 }, { 2, 4 } })

  local empty_path = write_text("empty", "")
  local empty = numlua.genfromtxt(empty_path)
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)
  os.remove(path)
  os.remove(empty_path)
end)

test("genfromtxt accepts caller-owned file handles and leaves them open", function()
  local path = write_text("handle", "1,NA\n2,3\n")
  local handle = assert(io.open(path, "r"))
  local loaded = numlua.genfromtxt(handle, {
    delimiter = ",",
    missing_values = "NA",
    filling_values = 0,
  })
  assert_eq(io.type(handle), "file", "genfromtxt must not close a caller handle")
  assert_tensor(loaded, { { 1, 0 }, { 2, 3 } })
  assert_true(handle:close())
  os.remove(path)
end)

test("genfromtxt validates options and data", function()
  local path = write_text("errors", "1,NA\n2,3\n")
  assert_error(function()
    numlua.genfromtxt(path, { unknown = true })
  end, "unknown option")
  assert_error(function()
    numlua.genfromtxt(path, { delimiter = "," })
  end, "invalid number 'NA'")
  assert_error(function()
    numlua.genfromtxt(path, { missing_values = true })
  end, "missing_values must be a string")
  assert_error(function()
    numlua.genfromtxt(path, { filling_values = "zero" })
  end, "filling_values must be a number")
  assert_error(function()
    numlua.genfromtxt(path, {
      delimiter = ",",
      missing_values = { [9] = "NA" },
    })
  end, "out of range")
  assert_error(function()
    numlua.genfromtxt(path, {
      delimiter = ",",
      filling_values = { [9] = 0 },
    })
  end, "out of range")
  assert_error(function()
    numlua.genfromtxt(path, { skip_header = -1 })
  end, "non-negative integer")
  assert_error(function()
    numlua.genfromtxt(path, {
      delimiter = ",",
      max_rows = 1,
      skip_footer = 1,
    })
  end, "cannot be combined")
  assert_error(function()
    numlua.genfromtxt(path, {
      delimiter = ",",
      usecols = { 9 },
    })
  end, "out of range")

  local ragged = write_text("ragged", "1,2\n3\n")
  assert_error(function()
    numlua.genfromtxt(ragged, { delimiter = "," })
  end, "has 1 column(s), expected 2")
  os.remove(ragged)
  os.remove(path)
end)
