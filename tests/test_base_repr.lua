--
-- base_repr formatting.
--

local numlua = require("numlua")

test("base_repr formats integers in bases 2..36", function()
  assert_eq(numlua.base_repr(5), "101")
  assert_eq(numlua.base_repr(255, 16), "FF")
  assert_eq(numlua.base_repr(35, 36), "Z")
  assert_eq(numlua.base_repr(5, 2, 8), "00000101")
  assert_eq(numlua.base_repr(-10, 2), "-1010")
  assert_eq(numlua.base_repr(10, 10), "10")
end)

test("base_repr validates value, base, and padding", function()
  assert_error(function() numlua.base_repr(1.5) end, "integer")
  assert_error(function() numlua.base_repr(1, 1) end, "2..36")
  assert_error(function() numlua.base_repr(1, 37) end, "2..36")
  assert_error(function() numlua.base_repr(1, 2, -1) end, "padding")
end)
