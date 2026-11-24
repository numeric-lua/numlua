--
-- Product, peak-to-peak, and NaN-aware product reductions.
--

local numlua = require("numlua")

test("prod reduces every axis, one axis, and keeps dimensions", function()
  local t = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_eq(numlua.prod(t):item(), 720)
  assert_tensor(numlua.prod(t, 1), { 4, 10, 18 })
  assert_tensor(numlua.prod(t, 2), { 6, 120 })
  assert_shape(numlua.prod(t, 1, true), { 1, 3 })
  assert_tensor(numlua.prod(t, 1, true), { { 4, 10, 18 } })
  assert_shape(numlua.prod(t, -1, true), { 2, 1 })
  assert_eq(numlua.prod(t, -1, true):numel(), 2)
end)

test("prod preserves numeric dtypes and uses the empty identity", function()
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_dtype(numlua.prod(ints, 1), "int64")
  assert_tensor(numlua.prod(ints, 1), { 3, 8 })
  local bools = numlua.array({ true, true, false }, { dtype = "bool" })
  assert_dtype(numlua.prod(bools), "int64")
  assert_eq(numlua.prod(bools):item(), 0)
  assert_eq(numlua.prod(numlua.zeros(0)):item(), 1)
  assert_tensor(numlua.prod(numlua.zeros(2, 0), 2),
    { 1, 1 })
end)

test("nanprod ignores NaNs and all-NaN slices become one", function()
  local nan = 0 / 0
  local t = numlua.array({ { 1, nan, 3 }, { nan, nan, nan } })
  assert_eq(numlua.nanprod(t):item(), 3)
  assert_tensor(numlua.nanprod(t, 2), { 3, 1 })
  assert_shape(numlua.nanprod(t, 1, true), { 1, 3 })
  assert_tensor(numlua.nanprod(t, 1, true), { { 1, 1, 3 } })
  assert_eq(numlua.nanprod(numlua.array({ nan, 2, nan })):item(), 2)
  assert_eq(numlua.nanprod(numlua.array({ nan, nan })):item(), 1)
  -- Integer input has no NaN path and follows prod.
  local ints = numlua.array({ { 2, 3 }, { 4, 5 } }, { dtype = "int64" })
  assert_tensor(numlua.nanprod(ints, 1), { 8, 15 })
end)

test("ptp measures the maximum minus minimum", function()
  local t = numlua.array({ { 1, 7, 3 }, { 4, 5, 6 } })
  assert_eq(numlua.ptp(t):item(), 6)
  assert_tensor(numlua.ptp(t, 1), { 3, 2, 3 })
  assert_tensor(numlua.ptp(t, 2), { 6, 2 })
  assert_shape(numlua.ptp(t, 2, true), { 2, 1 })
  assert_tensor(numlua.ptp(t, 2, true), { { 6 }, { 2 } })
  assert_tensor(numlua.ptp(t, -1), { 6, 2 })
end)

test("ptp preserves dtypes, promotes bool, and propagates NaN", function()
  local ints = numlua.array({ { 1, 5 }, { 9, 2 } }, { dtype = "int32" })
  assert_dtype(numlua.ptp(ints), "int32")
  assert_eq(numlua.ptp(ints):item(), 8)
  local bools = numlua.array({ true, false, true }, { dtype = "bool" })
  assert_dtype(numlua.ptp(bools), "int64")
  assert_eq(numlua.ptp(bools):item(), 1)
  local nan = 0 / 0
  local range = numlua.ptp(numlua.array({ 1, nan, 3 })):item()
  assert_true(range ~= range, "ptp should propagate NaN")
end)

test("product and ptp validate their arguments", function()
  assert_error(function() numlua.prod({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.prod(numlua.zeros(2), 0) end,
    "1-based")
  assert_error(function() numlua.prod(numlua.zeros(2), 3) end, "out of range")
  assert_error(function() numlua.prod(numlua.zeros(2), 1, 1) end,
    "keepdims")
  assert_error(function() numlua.nanprod({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.nanprod(numlua.zeros(2), 0) end,
    "1-based")
  assert_error(function() numlua.ptp({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.ptp(numlua.zeros(2), -3) end,
    "out of range")
  assert_error(function() numlua.ptp(numlua.zeros(2), 1, "yes") end,
    "keepdims")
end)
