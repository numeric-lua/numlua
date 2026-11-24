--
-- One-dimensional linear interpolation.
--

local numlua = require("numlua")

test("interp evaluates scalar and vector inputs", function()
  local xp = numlua.array({ 1, 3, 5 })
  local fp = numlua.array({ 10, 30, 50 })
  local scalar = numlua.interp(2, xp, fp)
  assert_eq(scalar:dim(), 0)
  assert_dtype(scalar, "float64")
  assert_near(scalar:item(), 20, 0)

  local values = numlua.interp(
    numlua.array({ 0, 1, 2, 3, 4, 5, 6 }), xp, fp)
  assert_shape(values, { 7 })
  assert_dtype(values, "float64")
  assert_near(values:totable(), { 10, 10, 20, 30, 40, 50, 50 }, 0)

  assert_near(numlua.interp(1, xp, fp):item(), 10, 0)
end)

test("interp preserves the input shape and accepts table inputs", function()
  local values = numlua.interp(
    numlua.array({ { 1, 2 }, { 3, 4 } }),
    { 1, 3, 5 }, { 10, 30, 50 })
  assert_shape(values, { 2, 2 })
  assert_near(values:totable(), { { 10, 20 }, { 30, 40 } }, 0)

  assert_near(numlua.interp({ 1, 2, 4 }, { 1, 3, 5 }, { 10, 30, 50 })
    :totable(), { 10, 20, 40 }, 0)
end)

test("interp honors left, right, and NaN coordinates", function()
  local xp = numlua.array({ 1, 3, 5 })
  local fp = numlua.array({ 10, 30, 50 })
  assert_near(numlua.interp(
    numlua.array({ 0, 6 }), xp, fp, -1, -2):totable(), { -1, -2 }, 0)
  assert_near(numlua.interp(
    numlua.array({ 0, 6 }), xp, fp):totable(), { 10, 50 }, 0)
  local nan = numlua.interp(numlua.array({ 0 / 0 }), xp, fp):get(1)
  assert_true(nan ~= nan)
end)

test("interp handles empty inputs and float64 output", function()
  local empty = numlua.interp(numlua.array({}, { dtype = "float64" }),
    numlua.array({ 1, 3 }), numlua.array({ 10, 30 }))
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)

  local from_int = numlua.interp(2,
    numlua.array({ 1, 3 }, { dtype = "int64" }),
    numlua.array({ 10, 30 }, { dtype = "int64" }))
  assert_dtype(from_int, "float64")
  assert_near(from_int:item(), 20, 0)
end)

test("interp validates grids and coordinates", function()
  local xp = numlua.array({ 1, 3, 5 })
  local fp = numlua.array({ 10, 30, 50 })
  assert_error(function() numlua.interp(1, numlua.array({ 1 }), fp) end,
    "at least two")
  assert_error(function() numlua.interp(1, xp, numlua.array({ 1 })) end,
    "fp has 1")
  assert_error(function() numlua.interp(1,
    numlua.array({ 1, 3, 3 }), fp) end, "strictly increasing")
  assert_error(function() numlua.interp(1,
    numlua.array({ { 1, 2 } }), fp) end, "xp must be 1-D")
  assert_error(function() numlua.interp("x", xp, fp) end, "x must be")
  assert_error(function() numlua.interp(1, xp, fp, "left") end,
    "left must be")
  assert_error(function() numlua.interp(1, xp, fp, 0, "right") end,
    "right must be")
end)
