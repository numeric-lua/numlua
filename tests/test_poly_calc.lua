--
-- Polynomial derivatives and antiderivatives.
--

local numlua = require("numlua")

test("polyder differentiates descending coefficients", function()
  assert_tensor(numlua.polyder({ 1, 2, 3 }), { 2, 2 })
  assert_tensor(numlua.polyder({ 1, 2, 3 }, 2), { 2 })
  assert_tensor(numlua.polyder({ 1, 2, 3 }, 0), { 1, 2, 3 })
  assert_tensor(numlua.polyder({ 5 }), { 0 })
  assert_tensor(numlua.polyder({ 2, 0, 1 }, 2), { 4 })
end)

test("polyint integrates and applies the constant", function()
  assert_tensor(numlua.polyint({ 2, 1 }), { 1, 1, 0 })
  assert_tensor(numlua.polyint({ 2, 1 }, 5), { 1, 1, 5 })
  assert_tensor(numlua.polyint({ 1, 1, 1 }), { 1 / 3, 1 / 2, 1, 0 })
  assert_dtype(numlua.polyint({ 2 }), "float64")
end)

test("polyder/polyint validate order and constants", function()
  assert_error(function() numlua.polyder({ 1 }, -1) end, "order")
  assert_error(function() numlua.polyder({ 1 }, 1.5) end, "order")
  assert_error(function() numlua.polyint({ 1 }, "x") end, "constant")
  assert_error(function() numlua.polyint({}) end, "coefficient")
end)
