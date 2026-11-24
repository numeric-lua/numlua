--
-- NumPy-style dot products across vector, matrix, and batched shapes.
--

local numlua = require("numlua")

test("dot covers vector and matrix shapes", function()
  local vector = numlua.dot(numlua.array({ 1, 2, 3 }),
    numlua.array({ 4, 5, 6 }))
  assert_eq(vector:dim(), 0)
  assert_eq(vector:item(), 32)

  local matrix = numlua.dot(
    numlua.array({ { 1, 2 }, { 3, 4 } }),
    numlua.array({ { 5, 6 }, { 7, 8 } }))
  assert_shape(matrix, { 2, 2 })
  assert_tensor(matrix, { { 19, 22 }, { 43, 50 } })

  assert_tensor(numlua.dot(
    numlua.array({ { 1, 2 }, { 3, 4 } }), numlua.array({ 5, 6 })),
    { 17, 39 })
  assert_tensor(numlua.dot(
    numlua.array({ 5, 6 }), numlua.array({ { 1, 2 }, { 3, 4 } })),
    { 23, 34 })
end)

test("dot follows NumPy's N-D shape rule", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local matrix = numlua.arange(20):reshape(4, 5)
  local result = numlua.dot(cube, matrix)
  assert_shape(result, { 2, 3, 5 })
  assert_eq(result:get(1, 1, 1), 70)
  assert_eq(result:get(1, 1, 5), 94)
  assert_eq(result:get(2, 3, 5), 1014)

  local vector = numlua.dot(cube,
    numlua.array({ 1, 1, 1, 1 }, { dtype = "int64" }))
  assert_shape(vector, { 2, 3 })
  assert_tensor(vector, { { 6, 22, 38 }, { 54, 70, 86 } })
end)

test("dot preserves dtypes and accepts non-contiguous inputs", function()
  local ints = numlua.dot(
    numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" }),
    numlua.array({ { 1, 0 }, { 0, 1 } }, { dtype = "int64" }))
  assert_dtype(ints, "int64")
  assert_tensor(ints, { { 1, 2 }, { 3, 4 } })

  local floats = numlua.dot(
    numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } }, { dtype = "float64" }),
    numlua.array({ 1, 1 }, { dtype = "float64" }))
  assert_dtype(floats, "float64")
  assert_near(floats:totable(), { 4, 8 }, 0)

  local transposed = numlua.transpose(
    numlua.array({ { 1, 2 }, { 3, 4 } }))
  assert_tensor(numlua.dot(transposed, numlua.array({ 1, 1 })),
    { 4, 6 })
end)

test("dot validates operands and inner dimensions", function()
  assert_error(function() numlua.dot(numlua.array(1), numlua.array(1)) end,
    "at least 1-D")
  assert_error(function() numlua.dot({ 1, 2 }, numlua.array({ 1, 2 })) end,
    "expects a tensor")
  assert_error(function() numlua.dot(
    numlua.zeros(2, 3), numlua.zeros(2, 3)) end)
  assert_error(function() numlua.dot(
    numlua.zeros(3), numlua.zeros(2, 3)) end)
end)
