--
-- Batched solve, QR, condition numbers, rank, and LU solve.
--

local numlua = require("numlua")

local function batch()
  return numlua.array({
    { { 2, 0 }, { 0, 4 } },
    { { 3, 0 }, { 0, 5 } },
  }, { dtype = "float64" })
end

test("batched solve handles stacked right-hand sides", function()
  local a = batch()
  local b = numlua.array({ { 2, 8 }, { 3, 10 } }, { dtype = "float64" })
  assert_tensor(numlua.linalg.solve(a, b), { { 1, 2 }, { 1, 2 } }, 1e-6)
  assert_error(function()
    numlua.linalg.solve(a, numlua.zeros(3, 2))
  end, "incompatible")
end)

test("batched qr reconstructs stacks", function()
  local a = batch()
  local q, r = numlua.linalg.qr(a)
  assert_shape(q, { 2, 2, 2 })
  assert_shape(r, { 2, 2, 2 })
  assert_tensor(q:matmul(r), a:totable(), 1e-6)
  assert_error(function() numlua.linalg.qr(numlua.zeros(2)) end, "matrix")
end)

test("batched cond and matrix_rank work per matrix", function()
  local a = batch()
  assert_tensor(numlua.linalg.cond(a), { 2, 5 / 3 }, 1e-6)
  assert_tensor(numlua.linalg.cond(a, "fro"), {
    math.sqrt(20) * math.sqrt(1 / 4 + 1 / 16),
    math.sqrt(34) * math.sqrt(1 / 9 + 1 / 25),
  }, 1e-6)
  local ranks = numlua.array({
    { { 1, 2 }, { 2, 4 } },
    { { 1, 0 }, { 0, 1 } },
  }, { dtype = "float64" })
  assert_tensor(numlua.linalg.matrix_rank(ranks), { 1, 2 })
  assert_error(function()
    numlua.linalg.cond(numlua.zeros(2))
  end, "square")
  assert_error(function()
    numlua.linalg.matrix_rank(numlua.zeros(2))
  end, "matrix")
end)

test("batched lu_solve solves from stacked factors", function()
  local a = batch()
  local p, l, u = numlua.linalg.lu(a)
  local b = numlua.array({ { 2, 8 }, { 3, 10 } }, { dtype = "float64" })
  assert_tensor(numlua.linalg.lu_solve(p, l, u, b),
    a:solve(b):totable(), 1e-6)
  assert_error(function()
    numlua.linalg.lu_solve(p, l, u, numlua.zeros(3, 2))
  end, "incompatible")
end)
