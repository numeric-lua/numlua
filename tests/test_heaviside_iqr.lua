--
-- Heaviside step and interquartile range.
--

local numlua = require("numlua")

test("heaviside follows the three branches", function()
  local x = numlua.array({ -2, 0, 2, 0 / 0 })
  assert_tensor(numlua.heaviside(x, 0.5), { 0, 0.5, 1, 0.5 })
  assert_tensor(numlua.heaviside(numlua.array({ -1, 0, 1 }),
    numlua.array({ 4, 5, 6 })), { 0, 5, 1 })
  assert_error(function() numlua.heaviside({ 1 }, 0) end,
    "expects a tensor")
  assert_error(function() numlua.heaviside(numlua.array({ 1 }), true)
    end, "x2")
end)

test("iqr returns the interquartile range", function()
  assert_near(numlua.iqr(numlua.array({ 1, 2, 3, 4 })):item(), 1.5, 1e-6)
  local matrix = numlua.array({ { 1, 2, 3, 4 }, { 10, 20, 30, 40 } })
  assert_tensor(numlua.iqr(matrix, 2), { 1.5, 15 })
  assert_shape(numlua.iqr(matrix, 2, true), { 2, 1 })
  assert_dtype(numlua.iqr(matrix, 2), "float32")
end)

test("iqr validates input, axis, and keepdims", function()
  assert_error(function() numlua.iqr({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.iqr(numlua.zeros(2, 2), 0) end, "1-based")
  assert_error(function() numlua.iqr(numlua.zeros(2, 2), 1, 1) end,
    "keepdims")
end)
