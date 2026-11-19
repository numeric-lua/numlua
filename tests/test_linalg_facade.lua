local torch = require("torch")
local numlua = require("numlua")
local linalg = require("numlua.linalg")

local function f64(values)
  return numlua.array(values, { dtype = "float64" })
end

test("linalg is exposed as a submodule", function()
  assert_true(numlua.linalg == linalg)
  assert_true(type(linalg.norm) == "function")
end)

test("linalg.norm supports vector, matrix, dim, and keepdim", function()
  local matrix = f64({ { 3, 0 }, { 0, 4 } })
  assert_near(linalg.norm(matrix):item(), 5, 1e-12)
  assert_shape(linalg.norm(matrix), {})

  assert_shape(linalg.norm(matrix, 1, 1), { 2 })
  assert_tensor(linalg.norm(matrix, 1, 1), { 3, 4 })
  assert_shape(linalg.norm(matrix, 2, 2, true), { 2, 1 })
  assert_tensor(linalg.norm(matrix, 2, 2, true), { { 3 }, { 4 } })
end)

test("linalg.inv and det", function()
  local matrix = f64({ { 4, 7 }, { 2, 6 } })
  assert_near(linalg.det(matrix):item(), 10, 1e-12)
  assert_shape(linalg.det(matrix), {})
  assert_shape(linalg.inv(matrix), { 2, 2 })
  assert_tensor(linalg.inv(matrix), { { 0.6, -0.7 }, { -0.2, 0.4 } }, 1e-12)
end)

test("linalg.solve handles vector and matrix right-hand sides", function()
  local matrix = f64({ { 4, 7 }, { 2, 6 } })
  assert_shape(linalg.solve(matrix, f64({ 1, 0 })), { 2 })
  assert_tensor(linalg.solve(matrix, f64({ 1, 0 })), { 0.6, -0.2 }, 1e-12)
  assert_shape(linalg.solve(matrix, f64({ { 1 }, { 0 } })), { 2, 1 })
end)

test("linalg.svd reconstructs a rectangular matrix", function()
  local matrix = f64({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local u, s, vh = linalg.svd(matrix, false)
  assert_shape(u, { 3, 2 })
  assert_shape(s, { 2 })
  assert_shape(vh, { 2, 2 })
  local diagonal = torch.zeros(2, 2, { dtype = torch.float64 })
  diagonal:set(1, 1, s:get(1))
  diagonal:set(2, 2, s:get(2))
  assert_near(u:matmul(diagonal):matmul(vh):totable(), matrix:totable(),
    1e-9)
end)

test("linalg.qr reconstructs and validates modes", function()
  local matrix = f64({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  local q, r = linalg.qr(matrix)
  assert_shape(q, { 3, 2 })
  assert_shape(r, { 2, 2 })
  assert_near(q:matmul(r):totable(), matrix:totable(), 1e-9)

  local complete_q, complete_r = linalg.qr(matrix, "complete")
  assert_shape(complete_q, { 3, 3 })
  assert_shape(complete_r, { 3, 2 })
end)

test("linalg rejects invalid inputs", function()
  assert_error(function() linalg.norm(1) end, "expects a tensor")
  assert_error(function() linalg.norm(f64({ 1, 2 }), 2, 0) end,
    "1-based")
  assert_error(function() linalg.norm(f64({ 1, 2 }), "two") end,
    "p must be a number")
  assert_error(function() linalg.inv(f64({ { 1, 2, 3 }, { 4, 5, 6 } })) end,
    "square matrix")
  assert_error(function() linalg.det(f64({ 1, 2 })) end, "square matrix")
  assert_error(function()
    linalg.solve(f64({ { 1, 0 }, { 0, 1 } }), f64({ 1, 2, 3 }))
  end, "first dimension")
  assert_error(function() linalg.svd(f64({ 1, 2 })) end, "expects a matrix")
  assert_error(function()
    linalg.qr(f64({ { 1, 0 }, { 0, 1 } }), 1)
  end, "mode must be a string")
end)

test("pinv and lstsq solve a least-squares line", function()
  local a = numlua.array({
    { 1, 1 }, { 1, 2 }, { 1, 3 }, { 1, 4 },
  }, { dtype = "float64" })
  local b = numlua.array({ 3, 5, 7, 9 }, { dtype = "float64" })
  local x = numlua.linalg.lstsq(a, b)
  assert_shape(x, { 2 })
  assert_near(x:get(1), 1, 1e-8)
  assert_near(x:get(2), 2, 1e-8)

  local p = numlua.linalg.pinv(numlua.array({
    { 2, 0 }, { 0, 4 },
  }, { dtype = "float64" }))
  assert_near(p:get(1, 1), 0.5, 1e-8)
  assert_near(p:get(2, 2), 0.25, 1e-8)
end)

test("matrix_norm covers common orders", function()
  local t = numlua.array({ { 1, -2 }, { 3, 4 } }, { dtype = "float64" })
  assert_near(numlua.linalg.matrix_norm(t):item(), math.sqrt(30), 1e-8)
  assert_near(numlua.linalg.matrix_norm(t, 1):item(), 6, 1e-8)
  assert_near(numlua.linalg.matrix_norm(t, math.huge):item(), 7, 1e-8)
  local nuclear = numlua.linalg.matrix_norm(t, "nuc"):item()
  assert_true(nuclear > 0)
end)
