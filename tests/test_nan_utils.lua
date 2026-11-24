--
-- NaN replacement and NaN-aware cumulative reductions.
--

local numlua = require("numlua")

test("nan_to_num replaces NaN and infinities", function()
  local values = numlua.array({ 1, 0 / 0, math.huge, -math.huge })
  local replaced = numlua.nan_to_num(values, 7, 8, 9)
  assert_dtype(replaced, "float32")
  assert_tensor(replaced, { 1, 7, 8, 9 })

  local default = numlua.nan_to_num(values)
  assert_eq(default:get(1), 1)
  assert_eq(default:get(2), 0)
  assert_true(default:get(3) > 3e38)
  assert_true(default:get(4) < -3e38)
end)

test("nan_to_num leaves integer and bool values unchanged", function()
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_dtype(numlua.nan_to_num(ints), "int64")
  assert_tensor(numlua.nan_to_num(ints), { { 1, 2 }, { 3, 4 } })
  local bools = numlua.array({ true, false })
  assert_dtype(numlua.nan_to_num(bools), "bool")
  assert_tensor(numlua.nan_to_num(bools), { true, false })
end)

test("nancumsum and nancumprod ignore NaNs", function()
  local values = numlua.array({ 1, 0 / 0, 3 })
  assert_tensor(numlua.nancumsum(values), { 1, 1, 4 })
  assert_tensor(numlua.nancumprod(values), { 1, 1, 3 })
  assert_tensor(numlua.nancumsum(numlua.array({ 0 / 0, 0 / 0 })),
    { 0, 0 })
  assert_tensor(numlua.nancumprod(numlua.array({ 0 / 0, 0 / 0 })),
    { 1, 1 })
  assert_dtype(numlua.nancumsum(values), "float32")

  local matrix = numlua.array({
    { 1, 0 / 0, 3 },
    { 0 / 0, 5, 6 },
  })
  assert_tensor(numlua.nancumsum(matrix, 1), {
    { 1, 0, 3 },
    { 1, 5, 9 },
  })
  assert_tensor(numlua.nancumsum(matrix, 2), {
    { 1, 1, 4 },
    { 0, 5, 11 },
  })
  assert_tensor(numlua.nancumprod(matrix, 2), {
    { 1, 1, 3 },
    { 1, 5, 30 },
  })
  assert_tensor(numlua.nancumsum(matrix, -2), {
    { 1, 0, 3 },
    { 1, 5, 9 },
  })
end)

test("nancumsum and nancumprod alias integer cumulative reductions", function()
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_dtype(numlua.nancumsum(ints, 1), "int64")
  assert_tensor(numlua.nancumsum(ints, 1), { { 1, 2 }, { 4, 6 } })
  assert_tensor(numlua.nancumprod(ints, 2), { { 1, 2 }, { 3, 12 } })
end)

test("nan helpers validate their arguments", function()
  assert_error(function() numlua.nan_to_num({ 1 }) end, "expects a tensor")
  assert_error(function() numlua.nan_to_num(numlua.zeros(1), "x") end,
    "nan must be a number")
  assert_error(function() numlua.nan_to_num(numlua.zeros(1), 0, "x") end,
    "posinf must be a number")
  assert_error(function() numlua.nancumsum(numlua.zeros(2), 0) end,
    "1-based")
  assert_error(function() numlua.nancumprod(numlua.zeros(2), 2) end,
    "out of range")
  assert_error(function() numlua.nancumsum({ 1, 2 }) end, "expects a tensor")
end)
