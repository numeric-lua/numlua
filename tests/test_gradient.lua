--
-- NumPy-parity numerical gradients with first-order edge differences.
--

local numlua = require("numlua")

test("gradient uses unit spacing by default", function()
  local t = numlua.array({ 1, 4, 9, 16 }, { dtype = "float64" })
  local result = numlua.gradient(t)
  assert_dtype(result, "float64")
  assert_shape(result, { 4 })
  assert_near(result:totable(), { 3, 4, 6, 7 })

  local two = numlua.gradient(numlua.array({ 1, 3 }))
  assert_tensor(two, { 2, 2 })
end)

test("gradient accepts uniform spacing and coordinate arrays", function()
  local t = numlua.array({ 1, 4, 9, 16 }, { dtype = "float64" })
  assert_near(numlua.gradient(t, 2):totable(), { 1.5, 2, 3, 3.5 })

  local linear = numlua.array({ 0, 1, 2, 3 }, { dtype = "float64" })
  local coordinates = { 0, 1, 3, 6 }
  local expected = { 1, 2 / 3, 2 / 5, 1 / 3 }
  assert_near(numlua.gradient(linear, coordinates):totable(), expected)
  assert_near(numlua.gradient(linear,
    numlua.array(coordinates, { dtype = "float64" })):totable(), expected)
end)

test("gradient defaults to every axis and accepts axis selection", function()
  local matrix = numlua.array({
    { 1, 2, 4 },
    { 2, 4, 8 },
    { 3, 6, 12 },
  }, { dtype = "float64" })

  local gradients = numlua.gradient(matrix)
  assert_eq(#gradients, 2)
  assert_near(gradients[1]:totable(), {
    { 1, 2, 4 },
    { 1, 2, 4 },
    { 1, 2, 4 },
  })
  assert_near(gradients[2]:totable(), {
    { 1, 1.5, 2 },
    { 2, 3, 4 },
    { 3, 4.5, 6 },
  })

  assert_near(numlua.gradient(matrix, 1, 2):totable(), {
    { 1, 1.5, 2 },
    { 2, 3, 4 },
    { 3, 4.5, 6 },
  })

  local reversed = numlua.gradient(matrix, 1, { 2, 1 })
  assert_eq(#reversed, 2)
  assert_near(reversed[1]:totable(), gradients[2]:totable())
  assert_near(reversed[2]:totable(), gradients[1]:totable())
end)

test("gradient accepts per-axis scalar and coordinate spacing", function()
  local matrix = numlua.array({
    { 1, 2, 4 },
    { 2, 4, 8 },
    { 3, 6, 12 },
  }, { dtype = "float64" })

  local scaled = numlua.gradient(matrix, { 1, 2 })
  assert_near(scaled[1]:totable(), {
    { 1, 2, 4 },
    { 1, 2, 4 },
    { 1, 2, 4 },
  })
  assert_near(scaled[2]:totable(), {
    { 0.5, 0.75, 1 },
    { 1, 1.5, 2 },
    { 1.5, 2.25, 3 },
  })

  local tensor_spacing = numlua.gradient(matrix,
    numlua.array({ 1, 2 }, { dtype = "float64" }))
  assert_near(tensor_spacing[2]:totable(), scaled[2]:totable())

  local coordinates = numlua.gradient(matrix, {
    { 1, 2, 3 },
    { 0, 2, 6 },
  })
  assert_near(coordinates[1]:totable(), {
    { 1, 2, 4 },
    { 1, 2, 4 },
    { 1, 2, 4 },
  })
  assert_near(coordinates[2]:totable(), {
    { 0.5, 0.5, 0.5 },
    { 1, 1, 1 },
    { 1.5, 1.5, 1.5 },
  })
end)

test("gradient promotes integer input to float64", function()
  local ints = numlua.array({ 1, 4, 9, 16 }, { dtype = "int64" })
  local result = numlua.gradient(ints)
  assert_dtype(result, "float64")
  assert_near(result:totable(), { 3, 4, 6, 7 })
end)

test("gradient validates inputs, axes, and spacing", function()
  assert_error(function() numlua.gradient(1) end, "expects a tensor")
  assert_error(function()
    numlua.gradient(numlua.array(1))
  end, "at least a 1-D tensor")
  assert_error(function() numlua.gradient(numlua.array({ 1 })) end,
    "needs at least 2 elements")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), 1, 0)
  end, "axis must be a non-zero integer")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), 1, 2)
  end, "out of range")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), 1, {})
  end, "axis table must not be empty")
  assert_error(function()
    numlua.gradient(numlua.array({ { 1, 2 } }), 1, { 1, 1 })
  end, "axes must be unique")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), 0)
  end, "finite and non-zero")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), 0 / 0)
  end, "finite and non-zero")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), math.huge)
  end, "finite and non-zero")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), "one")
  end, "spacing must be a number")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), { 1, 2, 3 })
  end, "spacing length")
  assert_error(function()
    numlua.gradient(numlua.array({ 1, 2 }), { 1, 1 })
  end, "must not repeat")
  assert_error(function()
    numlua.gradient(numlua.array({ { 1, 2 } }), 1, "first")
  end, "axis must be an integer, table, or nil")
end)
