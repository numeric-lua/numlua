--
-- Boolean support in loadtxt, savetxt, and genfromtxt.
--

local numlua = require("numlua")

local function path(name)
  return TMPDIR .. "/bool-text-" .. name
end

local function write_text(file, text)
  local handle = assert(io.open(file, "wb"))
  assert(handle:write(text))
  assert(handle:close())
end

local function read_text(file)
  local handle = assert(io.open(file, "rb"))
  local text = assert(handle:read("*a"))
  assert(handle:close())
  return text
end

test("savetxt writes bool tensors as True/False", function()
  local file = path("roundtrip.txt")
  local values = numlua.array({ { true, false }, { false, true } })
  assert_true(numlua.savetxt(file, values))
  assert_eq(read_text(file), "True False\nFalse True\n")
  local loaded = numlua.loadtxt(file, { dtype = "bool" })
  assert_dtype(loaded, "bool")
  assert_tensor(loaded, values:totable())
  os.remove(file)
end)

test("loadtxt parses boolean spellings and numbers", function()
  local file = path("spellings.txt")
  write_text(file, "True,false,1,0,2,nan\n")
  local values = numlua.loadtxt(file, { delimiter = ",", dtype = "bool" })
  assert_dtype(values, "bool")
  assert_tensor(values, { true, false, true, false, true, true })
  os.remove(file)
end)

test("savetxt bool supports custom fmt values", function()
  local file = path("fmt.txt")
  local values = numlua.array({ true, false, true })
  assert_true(numlua.savetxt(file, values, { fmt = "%d" }))
  assert_eq(read_text(file), "1 0 1\n")
  assert_true(numlua.savetxt(file, values, { fmt = "%s" }))
  assert_eq(read_text(file), "1 0 1\n")
  os.remove(file)

  local matrix = numlua.array({ { true, false }, { false, true } })
  assert_true(numlua.savetxt(file, matrix,
    { delimiter = ",", fmt = { "%d", "%d" } }))
  assert_eq(read_text(file), "1,0\n0,1\n")
  os.remove(file)
end)

test("genfromtxt handles bool missing and filling values", function()
  local file = path("missing.csv")
  write_text(file, "True,NA\n0,False\n")
  local defaults = numlua.genfromtxt(file, {
    delimiter = ",",
    dtype = "bool",
    missing_values = "NA",
  })
  assert_dtype(defaults, "bool")
  assert_tensor(defaults, { { true, false }, { false, false } })

  local filled = numlua.genfromtxt(file, {
    delimiter = ",",
    dtype = "bool",
    missing_values = "NA",
    filling_values = 1,
  })
  assert_tensor(filled, { { true, true }, { false, false } })
  os.remove(file)
end)

test("bool text accepts converters and reports invalid fields", function()
  local file = path("converters.txt")
  write_text(file, "yes,no\nno,yes\n")
  local converted = numlua.loadtxt(file, {
    delimiter = ",",
    dtype = "bool",
    converters = function(field)
      return field == "yes" and 1 or 0
    end,
  })
  assert_tensor(converted, { { true, false }, { false, true } })
  os.remove(file)

  local invalid = path("invalid.txt")
  write_text(invalid, "maybe\n")
  assert_error(function()
    numlua.loadtxt(invalid, { dtype = "bool" })
  end, "invalid number 'maybe'")
  os.remove(invalid)
end)
