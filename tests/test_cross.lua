--
-- NumPy-shaped 3-vector cross products.
--

local numlua = require("numlua")

test("cross computes the vector cross product", function()
  local result = numlua.cross(
    numlua.array({ 1, 2, 3 }), numlua.array({ 4, 5, 6 }))
  assert_shape(result, { 3 })
  assert_tensor(result, { -3, 6, -3 })

  local batch = numlua.cross(
    numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } }),
    numlua.array({ { 1, 0, 0 } }))
  assert_shape(batch, { 2, 3 })
  assert_tensor(batch, { { 0, 3, -2 }, { 0, 6, -5 } })
end)

test("cross accepts an explicit or negative axis", function()
  local a = numlua.array({ { 1, 4 }, { 2, 5 }, { 3, 6 } })
  local b = numlua.array({ { 1, 0 }, { 0, 1 }, { 0, 0 } })
  local expected = { { 0, -6 }, { 3, 0 }, { -2, 4 } }
  assert_tensor(numlua.cross(a, b, 1), expected)
  assert_tensor(numlua.cross(a, b, -2), expected)

  local cube_a = numlua.array({
    { { 1, 2, 3 }, { 4, 5, 6 } },
    { { 1, 0, 0 }, { 0, 1, 0 } },
  })
  local cube_b = numlua.array({
    { { 1, 0, 0 }, { 0, 1, 0 } },
    { { 0, 0, 1 }, { 1, 1, 0 } },
  })
  assert_shape(numlua.cross(cube_a, cube_b, -1), { 2, 2, 3 })
end)

test("cross preserves floating dtypes and broadcasts batch axes", function()
  local a = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } },
    { dtype = "float64" })
  local b = numlua.array({ { 1, 1, 0 } }, { dtype = "float64" })
  local result = numlua.cross(a, b)
  assert_dtype(result, "float64")
  assert_tensor(result, { { -3, 3, -1 }, { -6, 6, -1 } })
end)

test("cross validates ranks, axis length, and the axis argument", function()
  assert_error(function() numlua.cross({ 1, 2, 3 }, numlua.zeros(3)) end,
    "expects a tensor")
  assert_error(function() numlua.cross(numlua.zeros(3), numlua.zeros(2, 3))
    end, "same non-zero rank")
  assert_error(function() numlua.cross(numlua.zeros(2), numlua.zeros(2)) end,
    "length 3")
  assert_error(function() numlua.cross(
    numlua.zeros(2, 3), numlua.zeros(2, 2)) end, "length 3")
  assert_error(function() numlua.cross(
    numlua.zeros(2, 3), numlua.zeros(2, 3), 0) end, "1-based")
  assert_error(function() numlua.cross(
    numlua.zeros(2, 3), numlua.zeros(2, 3), 3) end, "out of range")
  assert_error(function() numlua.cross(
    numlua.zeros(2, 3), numlua.zeros(2, 3), "last") end, "axis")
end)
