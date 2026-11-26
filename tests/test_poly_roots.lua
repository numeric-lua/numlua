--
-- poly from roots.
--

local numlua = require("numlua")

test("poly builds descending coefficients from roots", function()
  assert_tensor(numlua.poly({ 1, 2 }), { 1, -3, 2 })
  assert_tensor(numlua.poly({ 0 }), { 1, 0 })
  assert_tensor(numlua.poly({}), { 1 })
  assert_tensor(numlua.poly(numlua.array({ 1, 2, 3 })),
    { 1, -6, 11, -6 })
  assert_dtype(numlua.poly({ 1 }), "float64")
end)

test("poly validates roots", function()
  assert_error(function() numlua.poly(numlua.zeros(2, 2)) end, "1-D")
  assert_error(function() numlua.poly(1) end, "tensor or table")
end)
