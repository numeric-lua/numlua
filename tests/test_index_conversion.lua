--
-- ravel_multi_index and unravel_index conversions.
--

local numlua = require("numlua")

test("ravel_multi_index converts 1-based coordinates in C order", function()
  assert_tensor(numlua.ravel_multi_index({ { 1, 2 }, { 2, 3 } }, { 2, 3 }),
    { 2, 6 })
  assert_tensor(numlua.ravel_multi_index(
    { numlua.array({ 1, 2 }, { dtype = "int64" }), { 1, 2 } }, { 2, 2 }),
    { 1, 4 })
  assert_tensor(numlua.ravel_multi_index(
    { { 1, 2 }, { 1, 2 }, { 1, 2 } }, { 2, 2, 2 }), { 1, 8 })
  assert_shape(numlua.ravel_multi_index({ {}, {} }, { 2, 3 }), { 0 })
end)

test("ravel_multi_index supports Fortran order", function()
  assert_tensor(numlua.ravel_multi_index(
    { { 1, 2 }, { 2, 3 } }, { 2, 3 }, "F"), { 3, 6 })
  assert_tensor(numlua.ravel_multi_index(
    { { 1, 2 }, { 1, 2 }, { 1, 2 } }, { 2, 2, 2 }, "F"), { 1, 8 })
end)

test("unravel_index returns coordinates in C order", function()
  local first, second = numlua.unravel_index(
    numlua.array({ 2, 6 }, { dtype = "int64" }), { 2, 3 })
  assert_tensor(first, { 1, 2 })
  assert_tensor(second, { 2, 3 })
  assert_dtype(first, "int64")

  local a, b, c = numlua.unravel_index({ 1, 8 }, { 2, 2, 2 })
  assert_tensor(a, { 1, 2 })
  assert_tensor(b, { 1, 2 })
  assert_tensor(c, { 1, 2 })
end)

test("unravel_index supports Fortran order and validation", function()
  local first, second = numlua.unravel_index({ 3, 6 }, { 2, 3 }, "F")
  assert_tensor(first, { 1, 2 })
  assert_tensor(second, { 2, 3 })

  assert_error(function() numlua.ravel_multi_index({ { 1 } }, { 2, 3 }) end,
    "coordinate count")
  assert_error(function() numlua.ravel_multi_index({ { 0 }, { 1 } }, { 2, 3 }) end,
    "out of range")
  assert_error(function() numlua.ravel_multi_index({ { 1.5 }, { 1 } }, { 2, 3 }) end,
    "integers")
  assert_error(function() numlua.ravel_multi_index({ { 1 }, { 1 } }, { 2, 3 }, "X")
    end, "order")
  assert_error(function() numlua.ravel_multi_index({ { 1 }, { 1 } }, { 0, 3 }) end,
    "positive integers")
  assert_error(function() numlua.unravel_index({ 7 }, { 2, 3 }) end,
    "out of range")
  assert_error(function() numlua.unravel_index({ 0 }, { 2, 3 }) end,
    "out of range")
  assert_error(function() numlua.unravel_index({ 1.5 }, { 2, 3 }) end,
    "integers")
  assert_error(function() numlua.unravel_index({ 1 }, { 2, 3 }, "X") end,
    "order")
end)
