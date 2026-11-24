local numlua = require("numlua")

test("dot computes a 1-D inner product", function()
  local a = numlua.array({ 1, 2, 3 })
  local b = numlua.array({ 4, 5, 6 })
  local result = numlua.dot(a, b)
  assert_eq(result:dim(), 0)
  assert_dtype(result, "float32")
  assert_eq(result:item(), 32)

  local ints = numlua.dot(
    numlua.array({ 1, 2, 3 }, { dtype = "int64" }),
    numlua.array({ 4, 5, 6 }, { dtype = "int64" }))
  assert_dtype(ints, "int64")
  assert_eq(ints:item(), 32)

  local floats = numlua.dot(
    numlua.array({ 1.5, 2.5 }, { dtype = "float64" }),
    numlua.array({ 2, 4 }, { dtype = "float64" }))
  assert_near(floats:item(), 13)
end)

test("matmul multiplies matrices and vectors", function()
  local a = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  })
  local b = numlua.array({
    { 7, 8 },
    { 9, 10 },
    { 11, 12 },
  })
  assert_shape(numlua.matmul(a, b), { 2, 2 })
  assert_tensor(numlua.matmul(a, b), {
    { 58, 64 },
    { 139, 154 },
  })
  assert_tensor(numlua.matmul(a, numlua.array({ 1, 1, 1 })), { 6, 15 })
  assert_tensor(numlua.matmul(numlua.array({ 1, 1, 1 }), b), { 27, 30 })
end)

test("matmul supports batched and broadcasted tensors", function()
  local batch_a = numlua.ones(2, 2, 3)
  local batch_b = numlua.ones(2, 3, 2)
  local batched = numlua.matmul(batch_a, batch_b)
  assert_shape(batched, { 2, 2, 2 })
  assert_tensor(batched, {
    { { 3, 3 }, { 3, 3 } },
    { { 3, 3 }, { 3, 3 } },
  })

  local broadcast = numlua.matmul(numlua.ones(2, 3), numlua.ones(1, 3, 2))
  assert_shape(broadcast, { 1, 2, 2 })
  assert_tensor(broadcast, { { { 3, 3 }, { 3, 3 } } })
end)

test("matmul preserves floating dtypes", function()
  local a = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } }, { dtype = "float64" })
  local b = numlua.eye(2, { dtype = "float64" })
  local product = numlua.matmul(a, b)
  assert_dtype(product, "float64")
  assert_near(product:totable(), a:totable(), 0)
end)

test("linalg rejects invalid shapes", function()
  assert_error(function() numlua.dot(numlua.array(1), numlua.array(1)) end,
    "at least 1-D")
  assert_error(function() numlua.dot(numlua.zeros(2), numlua.zeros(3)) end)
  assert_error(function() numlua.matmul(numlua.zeros(2, 3), numlua.zeros(2, 3)) end)
  assert_error(function() numlua.matmul(nil, numlua.zeros(2)) end,
    "numlua.matmul expects a tensor")
end)
