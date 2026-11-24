--
-- NaN-aware reductions: ignore NaNs, and return NaN for all-NaN slices.
--

local numlua = require("numlua")

local nan = 0 / 0

test("nansum treats NaN as zero and preserves floating dtypes", function()
  local t = numlua.array({ 1, nan, 3 })
  local total = numlua.nansum(t)
  assert_dtype(total, "float32")
  assert_eq(total:dim(), 0)
  assert_near(total:item(), 4)

  local matrix = numlua.array({
    { 1, nan },
    { nan, 4 },
  })
  assert_near(numlua.nansum(matrix, 1):totable(), { 1, 4 })
  assert_near(numlua.nansum(matrix, 2):totable(), { 1, 4 })
  assert_shape(numlua.nansum(matrix, 1, true), { 1, 2 })
  assert_near(numlua.nansum(numlua.array({ nan, nan })):item(), 0)
end)

test("nanmean ignores NaNs and returns NaN for all-NaN slices", function()
  local t = numlua.array({ 1, nan, 3 })
  local mean = numlua.nanmean(t)
  assert_dtype(mean, "float32")
  assert_near(mean:item(), 2)

  local matrix = numlua.array({
    { 1, nan, 3 },
    { 4, 5, nan },
  })
  assert_near(numlua.nanmean(matrix, 2):totable(), { 2, 4.5 })
  assert_shape(numlua.nanmean(matrix, 2, true), { 2, 1 })

  local all_nan = numlua.nanmean(numlua.array({ nan, nan }))
  assert_true(all_nan:item() ~= all_nan:item(), "all-NaN mean must be NaN")
end)

test("nanmin and nanmax ignore NaNs and return NaN for all-NaN slices", function()
  local t = numlua.array({ nan, 3, 1 })
  assert_eq(numlua.nanmin(t):item(), 1)
  assert_eq(numlua.nanmax(t):item(), 3)

  local matrix = numlua.array({
    { nan, 2, 5 },
    { 4, nan, 1 },
  })
  assert_near(numlua.nanmin(matrix, 1):totable(), { 4, 2, 1 })
  assert_near(numlua.nanmax(matrix, 2):totable(), { 5, 4 })
  assert_shape(numlua.nanmin(matrix, 1, true), { 1, 3 })

  local all_nan = numlua.array({ { nan, nan }, { nan, nan } })
  local minimum = numlua.nanmin(all_nan, 1)
  assert_true(minimum:get(1) ~= minimum:get(1))
  assert_true(minimum:get(2) ~= minimum:get(2))
  local maximum = numlua.nanmax(all_nan)
  assert_true(maximum:item() ~= maximum:item())
end)

test("nanvar and nanstd count only non-NaN values", function()
  local t = numlua.array({ 1, nan, 3 })
  local variance = numlua.nanvar(t)
  assert_dtype(variance, "float32")
  assert_near(variance:item(), 1)
  assert_near(numlua.nanstd(t):item(), 1)
  assert_near(numlua.nanvar(t, nil, false, 1):item(), 2)

  local matrix = numlua.array({
    { 1, nan, 3 },
    { 4, 5, nan },
  })
  assert_near(numlua.nanvar(matrix, 2):totable(), { 1, 0.25 })
  assert_near(numlua.nanstd(matrix, 2):totable(), { 1, 0.5 })
  assert_shape(numlua.nanvar(matrix, 2, true), { 2, 1 })

  local all_nan = numlua.nanvar(numlua.array({ nan, nan }))
  assert_true(all_nan:item() ~= all_nan:item(), "all-NaN variance must be NaN")
end)

test("NaN reductions alias ordinary reductions for integer input", function()
  local ints = numlua.array({ 1, 2, 3 }, { dtype = "int64" })
  local total = numlua.nansum(ints)
  assert_dtype(total, "int64")
  assert_eq(total:item(), 6)

  local mean = numlua.nanmean(ints)
  assert_dtype(mean, "float64")
  assert_near(mean:item(), 2)

  local minimum = numlua.nanmin(ints)
  assert_dtype(minimum, "int64")
  assert_eq(minimum:item(), 1)

  local variance = numlua.nanvar(ints)
  assert_dtype(variance, "float64")
  assert_near(variance:item(), 2 / 3)
end)

test("NaN reductions validate arguments", function()
  for _, name in ipairs({ "nansum", "nanmean", "nanmin", "nanmax",
      "nanvar", "nanstd" }) do
    assert_error(function() numlua[name](1) end, "expects a tensor")
  end
  assert_error(function() numlua.nansum(numlua.array({ 1 }), 0) end,
    "axis 0 is invalid")
  assert_error(function()
    numlua.nanmean(numlua.array({ 1 }), 1, 1)
  end, "keepdims must be a boolean")
  assert_error(function()
    numlua.nanvar(numlua.array({ 1 }), 1, false, "one")
  end, "ddof must be a finite number")
end)
