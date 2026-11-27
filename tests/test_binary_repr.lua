--
-- binary_repr formatting.
--

local numlua = require("numlua")

test("binary_repr formats positive and negative integers", function()
  assert_eq(numlua.binary_repr(0), "0")
  assert_eq(numlua.binary_repr(5), "101")
  assert_eq(numlua.binary_repr(5, 8), "00000101")
  assert_eq(numlua.binary_repr(-5), "-101")
  assert_eq(numlua.binary_repr(-1, 4), "1111")
  assert_eq(numlua.binary_repr(-5, 4), "1011")
end)

test("binary_repr validates value and width", function()
  assert_error(function() numlua.binary_repr(1.5) end, "integer")
  assert_error(function() numlua.binary_repr(1, 0) end, "width")
  assert_error(function() numlua.binary_repr(1, 54) end, "width")
  assert_error(function() numlua.binary_repr(-9, 3) end, "does not fit")
end)
