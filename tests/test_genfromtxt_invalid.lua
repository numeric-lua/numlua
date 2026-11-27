--
-- genfromtxt invalid_raise skipping.
--

local numlua = require("numlua")

test("genfromtxt skips invalid rows when invalid_raise is false", function()
  local file = TMPDIR .. "/invalid-raise.csv"
  local handle = assert(io.open(file, "w"))
  handle:write("1,2\n3,4\nbad,row\n5,6\n")
  handle:close()
  local values = numlua.genfromtxt(file, {
    delimiter = ",",
    invalid_raise = false,
  })
  assert_tensor(values, { { 1, 2 }, { 3, 4 }, { 5, 6 } })
  assert_error(function()
    numlua.genfromtxt(file, { delimiter = "," })
  end, "invalid number")
  os.remove(file)
end)

test("genfromtxt invalid_raise skips wrong column counts", function()
  local file = TMPDIR .. "/invalid-columns.csv"
  local handle = assert(io.open(file, "w"))
  handle:write("1,2\n3,4,5\n6,7\n")
  handle:close()
  assert_tensor(numlua.genfromtxt(file, {
    delimiter = ",",
    invalid_raise = false,
  }), { { 1, 2 }, { 6, 7 } })
  assert_error(function()
    numlua.genfromtxt(file, { delimiter = ",", invalid_raise = true })
  end, "column")
  os.remove(file)
end)

test("genfromtxt validates invalid_raise", function()
  local file = TMPDIR .. "/invalid-option.csv"
  local handle = assert(io.open(file, "w"))
  handle:write("1\n")
  handle:close()
  assert_error(function()
    numlua.genfromtxt(file, { invalid_raise = 1 })
  end, "invalid_raise")
  os.remove(file)
end)
