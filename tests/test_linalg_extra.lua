--
-- Additional small linear-algebra helpers.
--

local numlua = require("numlua")

test("vdot flattens and computes an inner product", function()
  assert_near(numlua.linalg.vdot(
    numlua.array({ 1, 2, 3 }), numlua.array({ 4, 5, 6 })):item(),
    32, 1e-6)
  assert_near(numlua.linalg.vdot(
    numlua.array({ { 1, 2 }, { 3, 4 } }), numlua.array({ 1, 1, 1, 1 })):item(),
    10, 1e-6)
  assert_error(function()
    numlua.linalg.vdot(numlua.zeros(2), numlua.zeros(3))
  end, "same number")
  assert_error(function() numlua.linalg.vdot({ 1 }, { 1 }) end,
    "expects a tensor")
end)

test("multi_dot chains matrices", function()
  local a = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local b = numlua.array({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local c = numlua.array({ { 1, 0 }, { 0, 1 } })
  local result = numlua.linalg.multi_dot({ a, b, c })
  assert_shape(result, { 2, 2 })
  assert_true(result:equal(a:matmul(b)))
  assert_tensor(numlua.linalg.multi_dot({ a, b }), a:matmul(b):totable())
end)

test("multi_dot validates the chain", function()
  local a = numlua.array({ { 1, 2 } })
  assert_error(function() numlua.linalg.multi_dot({}) end, "non-empty")
  assert_error(function() numlua.linalg.multi_dot({ a, a }) end,
    "incompatible")
  assert_error(function() numlua.linalg.multi_dot({ numlua.zeros(2) }) end,
    "matrix")
end)

test("matrix_rank counts singular values", function()
  assert_eq(numlua.linalg.matrix_rank(
    numlua.array({ { 1, 0 }, { 0, 1 } })):item(), 2)
  assert_eq(numlua.linalg.matrix_rank(
    numlua.array({ { 1, 2 }, { 2, 4 } })):item(), 1)
  assert_eq(numlua.linalg.matrix_rank(
    numlua.array({ { 1, 0, 0 }, { 0, 1, 0 } })):item(), 2)
  assert_eq(numlua.linalg.matrix_rank(
    numlua.array({ { 1, 2 }, { 2, 4 } }), 10):item(), 0)
  assert_error(function()
    numlua.linalg.matrix_rank(numlua.array({ { 1 } }), -1)
  end, "tolerance")
end)

test("slogdet returns sign and log absolute determinant", function()
  local sign, logdet = numlua.linalg.slogdet(
    numlua.array({ { 1, 0 }, { 0, 2 } }))
  assert_eq(sign, 1)
  assert_near(logdet, math.log(2), 1e-6)
  sign, logdet = numlua.linalg.slogdet(
    numlua.array({ { -1, 0 }, { 0, 2 } }))
  assert_eq(sign, -1)
  assert_near(logdet, math.log(2), 1e-6)
  sign, logdet = numlua.linalg.slogdet(numlua.zeros(2, 2))
  assert_eq(sign, 0)
  assert_eq(logdet, -math.huge)
end)

test("kron handles 1-D and 2-D operands", function()
  assert_tensor(numlua.linalg.kron(
    numlua.array({ 1, 2 }), numlua.array({ 3, 4 })), { 3, 4, 6, 8 })
  assert_tensor(numlua.linalg.kron(
    numlua.array({ { 1, 2 }, { 3, 4 } }),
    numlua.array({ { 0, 5 }, { 6, 7 } })), {
    { 0, 5, 0, 10 },
    { 6, 7, 12, 14 },
    { 0, 15, 0, 20 },
    { 18, 21, 24, 28 },
  })
  assert_error(function()
    numlua.linalg.kron(numlua.zeros(2), numlua.zeros(2, 2))
  end, "same rank")
  assert_error(function()
    numlua.linalg.kron(numlua.zeros(2, 2, 2), numlua.zeros(2, 2, 2))
  end, "1-D and 2-D")
end)
