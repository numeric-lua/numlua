--
-- fromregex Lua-pattern extraction.
--

local numlua = require("numlua")

test("fromregex extracts numeric and bool matches", function()
  local file = TMPDIR .. "/fromregex.txt"
  local handle = assert(io.open(file, "w"))
  handle:write("x=1; y=2; z=30\n")
  handle:close()
  assert_tensor(numlua.fromregex(file, "(%d+)", "int64"), { 1, 2, 30 })
  assert_tensor(numlua.fromregex(file, "(%d+)", "int64", 2), { 1, 2 })
  os.remove(file)

  file = TMPDIR .. "/fromregex-bool.txt"
  handle = assert(io.open(file, "w"))
  handle:write("true false true\n")
  handle:close()
  assert_tensor(numlua.fromregex(file, "(%a+)", "bool"),
    { true, false, true })
  os.remove(file)
end)

test("fromregex validates pattern, dtype, count, and matches", function()
  local file = TMPDIR .. "/fromregex-errors.txt"
  local handle = assert(io.open(file, "w"))
  handle:write("x=1\n")
  handle:close()
  assert_error(function() numlua.fromregex(file, "") end, "pattern")
  assert_error(function() numlua.fromregex(file, "(%d+)", "nope") end,
    "dtype")
  assert_error(function() numlua.fromregex(file, "(%d+)", "int64", -1.5)
    end, "count")
  assert_error(function() numlua.fromregex(file, "(%a+)", "int64") end,
    "cannot parse")
  os.remove(file)
end)
