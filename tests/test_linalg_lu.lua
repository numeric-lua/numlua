--
-- LU factorization, singular values, and vector_norm.
--

local numlua = require("numlua")

test("lu returns P, L, U with P*A = L*U", function()
  local a = numlua.array({ { 4, 3 }, { 6, 3 } }, { dtype = "float64" })
  local p, l, u = numlua.linalg.lu(a)
  assert_shape(p, { 2, 2 })
  assert_shape(l, { 2, 2 })
  assert_shape(u, { 2, 2 })
  assert_tensor(p:matmul(a), l:matmul(u):totable(), 1e-6)
  assert_error(function() numlua.linalg.lu(numlua.zeros(2, 3)) end,
    "square")
end)

test("svdvals returns the singular values", function()
  local a = numlua.array({ { 3, 0 }, { 0, 2 } }, { dtype = "float64" })
  assert_tensor(numlua.linalg.svdvals(a), { 3, 2 }, 1e-6)
  local _, singular = numlua.linalg.svd(a, false)
  assert_tensor(numlua.linalg.svdvals(a), singular:reshape(-1):totable(),
    1e-6)
  assert_error(function() numlua.linalg.svdvals(numlua.zeros(2)) end,
    "matrix")
end)

test("vector_norm mirrors norm along an axis", function()
  local a = numlua.array({ { 3, 4 }, { 0, 12 } }, { dtype = "float64" })
  assert_tensor(numlua.linalg.vector_norm(a, 2, 2), { 5, 12 }, 1e-6)
  assert_tensor(numlua.linalg.vector_norm(a, 1, 1), { 3, 16 }, 1e-6)
  assert_near(numlua.linalg.vector_norm(a):item(),
    numlua.linalg.norm(a):item(), 1e-6)
end)
