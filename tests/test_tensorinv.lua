--
-- tensorinv reshaping and pseudoinverse.
--

local numlua = require("numlua")

test("tensorinv inverts a reshaped identity mapping", function()
  local identity = numlua.eye(4, { dtype = "float64" }):reshape(2, 2, 2, 2)
  local inverse = numlua.linalg.tensorinv(identity)
  assert_shape(inverse, { 2, 2, 2, 2 })
  assert_tensor(inverse, identity:totable(), 1e-6)

  local matrix = numlua.array({ { 2, 0 }, { 0, 4 } },
    { dtype = "float64" })
  local inverse_matrix = numlua.linalg.tensorinv(matrix, 1)
  assert_tensor(inverse_matrix, { { 0.5, 0 }, { 0, 0.25 } }, 1e-6)
end)

test("tensorinv validates ind and products", function()
  local identity = numlua.eye(4, { dtype = "float64" }):reshape(2, 2, 2, 2)
  assert_error(function() numlua.linalg.tensorinv(identity, 0) end, "ind")
  assert_error(function() numlua.linalg.tensorinv(identity, 4) end, "ind")
  assert_error(function()
    numlua.linalg.tensorinv(numlua.zeros(2, 3, 4, 2), 1)
  end, "products")
end)
