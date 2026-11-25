--
-- NaN-aware median.
--

local numlua = require("numlua")

test("nanmedian ignores NaNs in flat and 1-D input", function()
  local values = numlua.array({ 1, 0 / 0, 3 })
  assert_near(numlua.nanmedian(values):item(), 2, 1e-6)
  assert_eq(numlua.nanmedian(values):dim(), 0)
  assert_near(numlua.nanmedian(numlua.array({ 1, 2, 3 })):item(), 2, 1e-6)
end)

test("nanmedian reduces along axes and keeps dimensions", function()
  local matrix = numlua.array({
    { 1, 0 / 0, 3 },
    { 4, 5, 0 / 0 },
  })
  assert_tensor(numlua.nanmedian(matrix, 2), { 2, 4.5 })
  assert_tensor(numlua.nanmedian(matrix, 1), { 2.5, 5, 3 })
  assert_shape(numlua.nanmedian(matrix, 2, true), { 2, 1 })
  assert_tensor(numlua.nanmedian(matrix, 2, true),
    { { 2 }, { 4.5 } })
  assert_near(numlua.nanmedian(matrix):item(), 3.5, 1e-6)
end)

test("nanmedian handles all-NaN slices and integer aliases", function()
  local all_nan = numlua.nanmedian(numlua.array({ 0 / 0, 0 / 0 }))
  assert_true(all_nan:item() ~= all_nan:item())
  local mixed = numlua.nanmedian(numlua.array({
    { 0 / 0, 0 / 0 },
    { 1, 2 },
  }), 2)
  assert_true(mixed:get(1) ~= mixed:get(1))
  assert_eq(mixed:get(2), 1.5)

  local ints = numlua.nanmedian(
    numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" }), 1)
  assert_tensor(ints, { 2, 3 })
  assert_dtype(ints, "float32")
end)

test("nanmedian validates axes and empty reductions", function()
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function() numlua.nanmedian({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.nanmedian(matrix, 0) end, "1-based")
  assert_error(function() numlua.nanmedian(matrix, 3) end, "out of range")
  assert_error(function() numlua.nanmedian(matrix, 1, 1) end, "keepdims")
end)
