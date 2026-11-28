--
-- Solve with existing LU factors.
--

local numlua = require("numlua")

test("lu_solve uses P, L, U factors", function()
  local a = numlua.array({ { 4, 3 }, { 6, 3 } }, { dtype = "float64" })
  local p, l, u = numlua.linalg.lu(a)
  local b = numlua.array({ 1, 2 }, { dtype = "float64" })
  assert_tensor(numlua.linalg.lu_solve(p, l, u, b),
    a:solve(b):totable(), 1e-6)
  local matrix_b = numlua.array({ { 1, 2 }, { 3, 4 } },
    { dtype = "float64" })
  assert_tensor(numlua.linalg.lu_solve(p, l, u, matrix_b),
    a:solve(matrix_b):totable(), 1e-6)
end)

test("lu_solve validates factor and right-hand shapes", function()
  local a = numlua.array({ { 1, 0 }, { 0, 1 } }, { dtype = "float64" })
  local p, l, u = numlua.linalg.lu(a)
  assert_error(function()
    numlua.linalg.lu_solve(p, l, u, numlua.zeros(3))
  end, "first dimension")
  assert_error(function()
    numlua.linalg.lu_solve(numlua.zeros(2, 3), l, u, numlua.zeros(2))
  end, "n by n")
end)
