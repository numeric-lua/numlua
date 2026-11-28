--
-- tensorsolve reshaping and solve.
--

local numlua = require("numlua")

test("tensorsolve solves flattened tensor systems", function()
  local identity = numlua.eye(4, { dtype = "float64" }):reshape(2, 2, 2, 2)
  local b = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "float64" })
  assert_tensor(numlua.linalg.tensorsolve(identity, b), b:totable(), 1e-6)

  local matrix = numlua.array({ { 2, 0 }, { 0, 4 } },
    { dtype = "float64" })
  assert_tensor(numlua.linalg.tensorsolve(matrix,
    numlua.array({ 2, 8 }, { dtype = "float64" })), { 1, 2 }, 1e-6)
end)

test("tensorsolve validates rank and shapes", function()
  local identity = numlua.eye(4, { dtype = "float64" }):reshape(2, 2, 2, 2)
  assert_error(function()
    numlua.linalg.tensorsolve(numlua.zeros(2, 2, 2), numlua.zeros(2))
  end, "even-rank")
  assert_error(function()
    numlua.linalg.tensorsolve(identity, numlua.zeros(2))
  end, "rank")
  assert_error(function()
    numlua.linalg.tensorsolve(identity, numlua.zeros(2, 2), { 1 })
  end, "axes")
end)
