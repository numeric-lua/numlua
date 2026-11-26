--
-- NaN-aware quantile/percentile.
--

local numlua = require("numlua")

test("nanquantile ignores NaNs and returns 0-D scalars", function()
  local values = numlua.array({ 1, 0 / 0, 3, 4 })
  assert_near(numlua.nanquantile(values, 0.5):item(), 3, 1e-6)
  assert_eq(numlua.nanquantile(values, 0.5):dim(), 0)
  assert_near(numlua.nanquantile(values, 0.25):item(), 2, 1e-6)
  assert_near(numlua.nanquantile(values, 0.75):item(), 3.5, 1e-6)
  assert_near(numlua.nanpercentile(values, 50):item(), 3, 1e-6)
end)

test("nanquantile reduces along axes and supports q tables", function()
  local matrix = numlua.array({
    { 1, 0 / 0, 3 },
    { 4, 5, 0 / 0 },
  })
  assert_tensor(numlua.nanquantile(matrix, 0.5, 2), { 2, 4.5 })
  assert_tensor(numlua.nanquantile(matrix, 0.5, 1), { 2.5, 5, 3 })
  assert_shape(numlua.nanquantile(matrix, 0.5, 2, true), { 2, 1 })
  local parts = numlua.nanquantile(matrix, { 0.5, 1.0 }, 2)
  assert_shape(parts, { 2, 2 })
  assert_tensor(parts, { { 2, 4.5 }, { 3, 5 } })
  assert_tensor(numlua.nanpercentile(matrix, { 50, 100 }, 2),
    { { 2, 4.5 }, { 3, 5 } })
end)

test("nanquantile handles all-NaN slices and no-NaN input", function()
  local values = numlua.array({ 0 / 0, 0 / 0, 2, 3 })
  assert_near(numlua.nanquantile(values, 0.5):item(), 2.5, 1e-6)
  local all_nan = numlua.nanquantile(numlua.array({ 0 / 0, 0 / 0 }), 0.5)
  assert_true(all_nan:item() ~= all_nan:item())
  assert_near(numlua.nanquantile(numlua.array({ 1, 2, 3 }), 0.5):item(),
    2, 1e-6)
end)

test("nanquantile validates inputs", function()
  local values = numlua.array({ 1, 2, 3 })
  assert_error(function() numlua.nanquantile({ 1, 2 }, 0.5) end,
    "expects a tensor")
  assert_error(function() numlua.nanquantile(values, -0.1) end, "q")
  assert_error(function() numlua.nanquantile(values, { 0.5, 2 }) end, "q")
  assert_error(function() numlua.nanquantile(values, {}) end, "non-empty")
  assert_error(function() numlua.nanquantile(values, 0.5, 2) end,
    "out of range")
  assert_error(function() numlua.nanpercentile(values, 101) end, "[0, 100]")
end)
