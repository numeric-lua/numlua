--
-- Infinity predicates, hypot, and remainder.
--

local numlua = require("numlua")

test("isposinf and isneginf classify infinities", function()
  local values = numlua.array({ 1, math.huge, -math.huge, 0 / 0 })
  assert_dtype(numlua.isposinf(values), "bool")
  assert_tensor(numlua.isposinf(values), { false, true, false, false })
  assert_tensor(numlua.isneginf(values), { false, false, true, false })
  assert_tensor(numlua.isposinf(numlua.array({ 1, 2 })), { false, false })
  assert_tensor(numlua.isneginf(numlua.array({ true, false })),
    { false, false })
end)

test("hypot accepts tensors and scalar operands", function()
  assert_tensor(numlua.hypot(
    numlua.array({ 3, 5 }), numlua.array({ 4, 12 })), { 5, 13 })
  assert_tensor(numlua.hypot(numlua.array({ 3, 5 }), 4), { 5, math.sqrt(41) })
  assert_tensor(numlua.hypot(3, numlua.array({ 4, 5 })), { 5, math.sqrt(34) })
  local scalar = numlua.hypot(3, 4)
  assert_eq(scalar:dim(), 0)
  assert_near(scalar:item(), 5, 1e-6)
  assert_dtype(numlua.hypot(
    numlua.array({ 3 }, { dtype = "float64" }), 4), "float64")
end)

test("remainder and mod follow Python-style modulo", function()
  assert_tensor(numlua.remainder(
    numlua.array({ 5, -5, 7 }), numlua.array({ 2, 2, -3 })),
    { 1, 1, -2 })
  assert_tensor(numlua.remainder(numlua.array({ 5, -5 }), 2), { 1, 1 })
  assert_tensor(numlua.remainder(5, numlua.array({ 2, 3 })), { 1, 2 })
  local scalar = numlua.remainder(5, 2)
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 1)
  assert_tensor(numlua.mod(numlua.array({ 5, -5 }), 2), { 1, 1 })
  assert_dtype(numlua.remainder(
    numlua.array({ 5 }, { dtype = "float64" }), 2), "float64")
end)

test("new math ufuncs validate operands", function()
  assert_error(function() numlua.isposinf("x") end, "expects a tensor or number")
  assert_error(function() numlua.isneginf({ 1 }) end, "expects a tensor or number")
  assert_error(function() numlua.hypot("x", 1) end, "hypot")
  assert_error(function() numlua.hypot(1, "x") end, "hypot")
  assert_error(function() numlua.remainder("x", 1) end, "remainder")
end)
