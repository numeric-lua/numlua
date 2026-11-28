--
-- Batched square-matrix linear algebra.
--

local numlua = require("numlua")

local function batch()
  return numlua.array({
    { { 2, 0 }, { 0, 4 } },
    { { 3, 0 }, { 0, 5 } },
  }, { dtype = "float64" })
end

test("batched inv/det/slogdet work on stacked matrices", function()
  local a = batch()
  assert_tensor(numlua.linalg.inv(a), {
    { { 0.5, 0 }, { 0, 0.25 } },
    { { 1 / 3, 0 }, { 0, 0.2 } },
  }, 1e-6)
  assert_tensor(numlua.linalg.det(a), { 8, 15 }, 1e-6)
  local sign, logdet = numlua.linalg.slogdet(a)
  assert_tensor(sign, { 1, 1 })
  assert_tensor(logdet, { math.log(8), math.log(15) }, 1e-6)
end)

test("batched cholesky/eigh/eigvalsh work on stacks", function()
  local a = batch()
  assert_tensor(numlua.linalg.cholesky(a), {
    { { math.sqrt(2), 0 }, { 0, 2 } },
    { { math.sqrt(3), 0 }, { 0, math.sqrt(5) } },
  }, 1e-6)
  local values, vectors = numlua.linalg.eigh(a)
  assert_tensor(values, { { 2, 4 }, { 3, 5 } }, 1e-6)
  assert_shape(vectors, { 2, 2, 2 })
  assert_tensor(numlua.linalg.eigvalsh(a), { { 2, 4 }, { 3, 5 } }, 1e-6)
end)

test("batched svd/svdvals preserve leading dimensions", function()
  local a = batch()
  local u, s, vh = numlua.linalg.svd(a, false)
  assert_shape(u, { 2, 2, 2 })
  assert_shape(s, { 2, 2 })
  assert_shape(vh, { 2, 2, 2 })
  assert_tensor(numlua.linalg.svdvals(a), { { 4, 2 }, { 5, 3 } }, 1e-6)
end)

test("batched matrix_power and lu work on stacks", function()
  local a = batch()
  assert_tensor(numlua.linalg.matrix_power(a, 2), {
    { { 4, 0 }, { 0, 16 } },
    { { 9, 0 }, { 0, 25 } },
  }, 1e-6)
  assert_tensor(numlua.linalg.matrix_power(a, 0), {
    { { 1, 0 }, { 0, 1 } },
    { { 1, 0 }, { 0, 1 } },
  }, 1e-6)
  local p, l, u = numlua.linalg.lu(a)
  assert_shape(p, { 2, 2, 2 })
  assert_shape(l, { 2, 2, 2 })
  assert_shape(u, { 2, 2, 2 })
  assert_tensor(p:matmul(a), l:matmul(u):totable(), 1e-6)
end)

test("batched linalg validates ranks and square shapes", function()
  assert_error(function() numlua.linalg.inv(numlua.zeros(2)) end,
    "square")
  assert_error(function() numlua.linalg.det(numlua.zeros(2, 3)) end,
    "square")
  assert_error(function() numlua.linalg.svd(numlua.zeros(2)) end,
    "matrix")
  assert_error(function() numlua.linalg.cholesky(numlua.zeros(2, 2, 3))
    end, "square")
end)
