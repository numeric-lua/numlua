--
-- NumPy inner and outer products.
--

local numlua = require("numlua")

test("inner sums over the last axis", function()
  local scalar = numlua.inner(numlua.array({ 1, 2, 3 }),
    numlua.array({ 4, 5, 6 }))
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 32)

  local matrix = numlua.inner(
    numlua.array({ { 1, 2 }, { 3, 4 } }),
    numlua.array({ { 5, 6 }, { 7, 8 } }))
  assert_shape(matrix, { 2, 2 })
  assert_tensor(matrix, { { 17, 23 }, { 39, 53 } })

  local batched = numlua.inner(
    numlua.arange(12):reshape(2, 2, 3),
    numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } }))
  assert_shape(batched, { 2, 2, 2 })
  assert_tensor(batched, {
    { { 8, 17 }, { 26, 62 } },
    { { 44, 107 }, { 62, 152 } },
  })
end)

test("inner handles empty last axes and dtypes", function()
  local empty = numlua.inner(numlua.zeros(2, 0), numlua.zeros(3, 0))
  assert_shape(empty, { 2, 3 })
  assert_eq(empty:numel(), 6)
  assert_eq(empty:sum():item(), 0)
  local empty_vector = numlua.inner(numlua.zeros(0), numlua.zeros(0))
  assert_eq(empty_vector:dim(), 0)
  assert_eq(empty_vector:item(), 0)

  local ints = numlua.inner(
    numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" }),
    numlua.array({ { 1, 1 }, { 1, 1 } }, { dtype = "int64" }))
  assert_dtype(ints, "int64")
  assert_tensor(ints, { { 3, 3 }, { 7, 7 } })

  local transposed = numlua.transpose(numlua.array({ { 1, 2 }, { 3, 4 } }))
  assert_tensor(numlua.inner(transposed, numlua.array({ 1, 1 })),
    { 4, 6 })
end)

test("outer flattens both operands in C order", function()
  local result = numlua.outer(
    numlua.array({ { 1, 2 }, { 3, 4 } }),
    numlua.array({ 5, 6 }))
  assert_shape(result, { 4, 2 })
  assert_tensor(result, { { 5, 6 }, { 10, 12 }, { 15, 18 }, { 20, 24 } })

  local scalar = numlua.outer(numlua.array(2), numlua.array(3))
  assert_shape(scalar, { 1, 1 })
  assert_eq(scalar:get(1, 1), 6)

  local empty = numlua.outer(numlua.zeros(0), numlua.array({ 1, 2 }))
  assert_shape(empty, { 0, 2 })
  assert_eq(empty:numel(), 0)

  local floats = numlua.outer(
    numlua.array({ 1.5, 2.5 }, { dtype = "float64" }),
    numlua.array({ 2, 4 }, { dtype = "float64" }))
  assert_dtype(floats, "float64")
  assert_near(floats:totable(), { { 3, 6 }, { 5, 10 } }, 0)

  local transposed = numlua.transpose(numlua.array({ { 1, 2 }, { 3, 4 } }))
  assert_tensor(numlua.outer(transposed, numlua.array({ 1, 1 })),
    { { 1, 1 }, { 3, 3 }, { 2, 2 }, { 4, 4 } })
end)

test("inner and outer validate their operands", function()
  assert_error(function() numlua.inner({ 1, 2 }, numlua.zeros(2)) end,
    "expects a tensor")
  assert_error(function() numlua.inner(numlua.array(1), numlua.array(1)) end,
    "at least 1-D")
  assert_error(function() numlua.inner(numlua.zeros(2, 3),
    numlua.zeros(2)) end, "last-axis sizes differ")
  assert_error(function() numlua.outer({ 1, 2 }, numlua.zeros(2)) end,
    "expects a tensor")
end)
