--
-- Cholesky, symmetric eigenvalues, powers, and condition numbers.
--

local numlua = require("numlua")

test("cholesky factors a positive-definite matrix", function()
  local factor = numlua.linalg.cholesky(numlua.array({
    { 4, 2 }, { 2, 3 },
  }, { dtype = "float64" }))
  assert_tensor(factor, { { 2, 0 }, { 1, math.sqrt(2) } }, 1e-6)
  assert_error(function()
    numlua.linalg.cholesky(numlua.zeros(2, 3))
  end, "square")
end)

test("eigh and eigvalsh return symmetric eigenvalues", function()
  local values, vectors = numlua.linalg.eigh(numlua.array({
    { 1, 0 }, { 0, 3 },
  }, { dtype = "float64" }))
  assert_tensor(values, { 1, 3 }, 1e-6)
  assert_shape(vectors, { 2, 2 })
  assert_tensor(numlua.linalg.eigvalsh(numlua.array({
    { 1, 0 }, { 0, 3 },
  }, { dtype = "float64" })), { 1, 3 }, 1e-6)
  assert_error(function()
    numlua.linalg.eigh(numlua.zeros(2, 3))
  end, "square")
end)

test("matrix_power handles positive, zero, and negative exponents", function()
  local a = numlua.array({ { 1, 1 }, { 0, 1 } }, { dtype = "float64" })
  assert_tensor(numlua.linalg.matrix_power(a, 3), { { 1, 3 }, { 0, 1 } })
  assert_tensor(numlua.linalg.matrix_power(a, 0), { { 1, 0 }, { 0, 1 } })
  assert_tensor(numlua.linalg.matrix_power(a, -1), { { 1, -1 }, { 0, 1 } })
  assert_error(function() numlua.linalg.matrix_power(a, 1.5) end,
    "integer")
end)

test("cond computes common condition numbers", function()
  local identity = numlua.array({ { 1, 0 }, { 0, 1 } }, { dtype = "float64" })
  assert_near(numlua.linalg.cond(identity):item(), 1, 1e-6)
  assert_near(numlua.linalg.cond(identity, 1):item(), 1, 1e-6)
  assert_near(numlua.linalg.cond(identity, math.huge):item(), 1, 1e-6)
  assert_near(numlua.linalg.cond(identity, "fro"):item(), 2, 1e-6)
  local singular = numlua.linalg.cond(numlua.array({
    { 1, 2 }, { 2, 4 },
  }, { dtype = "float64" }))
  assert_true(singular:item() > 1e12)
  assert_error(function() numlua.linalg.cond(identity, 3) end,
    "unsupported")
end)
