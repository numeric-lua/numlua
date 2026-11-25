--
-- NaN-aware argmin/argmax reductions.
--

local numlua = require("numlua")

test("nanargmin and nanargmax ignore NaNs", function()
  local values = numlua.array({ 1, 0 / 0, -2, 3 })
  assert_eq(numlua.nanargmin(values):item(), 3)
  assert_eq(numlua.nanargmax(values):item(), 4)
  assert_dtype(numlua.nanargmin(values), "int64")

  local matrix = numlua.array({
    { 1, 0 / 0, 3 },
    { 0 / 0, 5, 2 },
  })
  assert_tensor(numlua.nanargmin(matrix, 1), { 1, 2, 2 })
  assert_tensor(numlua.nanargmax(matrix, 1), { 1, 2, 1 })
  assert_tensor(numlua.nanargmin(matrix, 2), { 1, 3 })
  assert_tensor(numlua.nanargmax(matrix, 2), { 3, 2 })
  assert_shape(numlua.nanargmin(matrix, 2, true), { 2, 1 })
  assert_tensor(numlua.nanargmin(matrix, -1), { 1, 3 })
end)

test("nanargmin and nanargmax alias integer reductions", function()
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_tensor(numlua.nanargmin(ints, 1), { 1, 1 })
  assert_tensor(numlua.nanargmax(ints, 1), { 2, 2 })
  assert_eq(numlua.nanargmax(ints):item(), 4)
  assert_dtype(numlua.nanargmax(ints), "int64")
end)

test("nanargmin and nanargmax reject all-NaN slices and bad arguments", function()
  assert_error(function()
    numlua.nanargmin(numlua.array({ 0 / 0, 0 / 0 }))
  end, "all-NaN slice")
  assert_error(function()
    numlua.nanargmax(numlua.array({ { 0 / 0, 0 / 0 }, { 1, 2 } }), 2)
  end, "all-NaN slice")
  assert_error(function() numlua.nanargmin({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.nanargmin(numlua.zeros(2), 0) end,
    "1-based")
  assert_error(function() numlua.nanargmax(numlua.zeros(2), 2) end,
    "out of range")
  assert_error(function() numlua.nanargmin(numlua.zeros(2), 1, 1) end,
    "keepdims")
end)
