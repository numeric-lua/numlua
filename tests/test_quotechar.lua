--
-- quotechar support in the numeric text readers.
--

local numlua = require("numlua")

local function path(name)
  return TMPDIR .. "/quotechar-" .. name
end

local function write_text(file, text)
  local handle = assert(io.open(file, "wb"))
  assert(handle:write(text))
  assert(handle:close())
end

test("loadtxt parses quoted CSV fields", function()
  local file = path("basic.csv")
  write_text(file, "\"1.5\",\"2.5\"\n\"3.5\",\"4.5\"\n")
  local values = numlua.loadtxt(file, { delimiter = ",", quotechar = "\"" })
  assert_shape(values, { 2, 2 })
  assert_tensor(values, { { 1.5, 2.5 }, { 3.5, 4.5 } })
  os.remove(file)
end)

test("loadtxt parses quotes with whitespace delimiters", function()
  local file = path("whitespace.txt")
  write_text(file, "\"1.5\" \"2.5\"\n")
  local values = numlua.loadtxt(file, { quotechar = "\"" })
  assert_shape(values, { 2 })
  assert_tensor(values, { 1.5, 2.5 })
  os.remove(file)
end)

test("loadtxt handles doubled quotes and delimiters inside quotes", function()
  local escaped = path("escaped.csv")
  write_text(escaped, "\"1\"\"5\",2\n")
  local values = numlua.loadtxt(escaped, {
    delimiter = ",",
    quotechar = "\"",
    converters = {
      [1] = function(field)
        if field == "1\"5" then return 1.5 end
        return tonumber(field)
      end,
    },
  })
  assert_tensor(values, { 1.5, 2 })
  os.remove(escaped)

  local embedded = path("embedded.csv")
  write_text(embedded, "\"1,5\",2\n")
  local embedded_values = numlua.loadtxt(embedded, {
    delimiter = ",",
    quotechar = "\"",
    converters = {
      [1] = function(field)
        if field == "1,5" then return 1.5 end
        return tonumber(field)
      end,
    },
  })
  assert_tensor(embedded_values, { 1.5, 2 })
  os.remove(embedded)
end)

test("genfromtxt supports quotechar and missing values", function()
  local file = path("missing.csv")
  write_text(file, "\"1\",NA\n\"2\",\"3\"\n")
  local values = numlua.genfromtxt(file, {
    delimiter = ",",
    quotechar = "\"",
    missing_values = "NA",
  })
  assert_shape(values, { 2, 2 })
  assert_eq(values:get(1, 1), 1)
  assert_true(values:get(1, 2) ~= values:get(1, 2))
  assert_eq(values:get(2, 1), 2)
  assert_eq(values:get(2, 2), 3)
  os.remove(file)
end)

test("quotechar rejects malformed quotes and options", function()
  local unterminated = path("unterminated.csv")
  write_text(unterminated, "\"1.5,2.5\n")
  assert_error(function()
    numlua.loadtxt(unterminated, { delimiter = ",", quotechar = "\"" })
  end, "unterminated quoted field")
  os.remove(unterminated)

  local trailing = path("trailing.csv")
  write_text(trailing, "\"1\"x,2\n")
  assert_error(function()
    numlua.loadtxt(trailing, { delimiter = ",", quotechar = "\"" })
  end, "unexpected characters after a closing quote")
  os.remove(trailing)

  local file = path("options.csv")
  write_text(file, "1,2\n")
  assert_error(function()
    numlua.loadtxt(file, { quotechar = "ab" })
  end, "single character")
  assert_error(function()
    numlua.loadtxt(file, { quotechar = "" })
  end, "single character")
  assert_error(function()
    numlua.genfromtxt(file, { quotechar = "ab" })
  end, "single character")
  assert_error(function()
    numlua.savetxt(file, numlua.array({ 1 }), { quotechar = "\"" })
  end, "unknown option")
  os.remove(file)
end)
