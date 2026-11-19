local numlua = require("numlua")

local function fixture(name)
  return "tests/fixtures/" .. name
end

test("loads committed NumPy .npy fixtures", function()
  local f = numlua.load(fixture("float64-2x2.npy"))
  assert_dtype(f, "float64")
  assert_shape(f, { 2, 2 })
  assert_near(f:get(1, 1), 1.5, 0)
  assert_near(f:get(1, 2), -2.25, 0)

  local i = numlua.load(fixture("int64-1d.npy"))
  assert_dtype(i, "int64")
  assert_tensor(i, { -3, 0, 7 })

  local b = numlua.load(fixture("bool-1d.npy"))
  assert_dtype(b, "bool")
  assert_tensor(b, { true, false, true, true })

  local empty = numlua.load(fixture("empty-2x0.npy"))
  assert_shape(empty, { 2, 0 })
  assert_eq(empty:numel(), 0)

  local scalar = numlua.load(fixture("scalar-int64.npy"))
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 42)
end)

test("rejects a committed big-endian fixture", function()
  assert_error(function() numlua.load(fixture("big-endian-f4.npy")) end,
    "big-endian")
end)
