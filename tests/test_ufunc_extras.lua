--
-- Additional floating predicates, rounding, and logical ufuncs.
--

local numlua = require("numlua")

test("isinf and isfinite classify floating values", function()
  local values = numlua.array({ 1, math.huge, -math.huge, 0 / 0 })
  local infinite = numlua.isinf(values)
  assert_dtype(infinite, "bool")
  assert_tensor(infinite, { false, true, true, false })
  local finite = numlua.isfinite(values)
  assert_dtype(finite, "bool")
  assert_tensor(finite, { true, false, false, false })

  local ints = numlua.array({ 1, 0, -2 }, { dtype = "int64" })
  assert_tensor(numlua.isinf(ints), { false, false, false })
  assert_tensor(numlua.isfinite(ints), { true, true, true })
  assert_tensor(numlua.isfinite(numlua.array({ true, false })), { true, true })
end)

test("sign, floor, ceil, trunc, and round cover numeric inputs", function()
  local values = numlua.array({ 1.5, -1.5, 0, 2.5, -2.5 })
  assert_tensor(numlua.sign(values), { 1, -1, 0, 1, -1 })
  assert_dtype(numlua.sign(values), "float32")
  assert_tensor(numlua.floor(values), { 1, -2, 0, 2, -3 })
  assert_tensor(numlua.ceil(values), { 2, -1, 0, 3, -2 })
  assert_tensor(numlua.trunc(values), { 1, -1, 0, 2, -2 })
  -- ATen's round uses round-half-to-even.
  assert_tensor(numlua.round(values), { 2, -2, 0, 2, -2 })

  local ints = numlua.array({ 3, -3, 0 }, { dtype = "int64" })
  assert_dtype(numlua.sign(ints), "int64")
  assert_tensor(numlua.sign(ints), { 1, -1, 0 })
  local bools = numlua.array({ true, false })
  assert_tensor(numlua.sign(bools), { true, false })
end)

test("reciprocal and log1p delegate to ATen", function()
  assert_tensor(numlua.reciprocal(numlua.array({ 2, 4 })), { 0.5, 0.25 })
  assert_dtype(numlua.reciprocal(numlua.array({ 2, 4 })), "float32")
  local ints = numlua.reciprocal(numlua.array({ 1, 2 }, { dtype = "int64" }))
  assert_dtype(ints, "float32")
  assert_tensor(ints, { 1, 0.5 })

  assert_near(numlua.log1p(numlua.array({ 0, 1 })):totable(),
    { 0, math.log(2) }, 1e-6)
  assert_near(numlua.log1p(
    numlua.array({ 0, 1 }, { dtype = "float64" })):totable(),
    { 0, math.log(2) }, 1e-12)
end)

test("logical ufuncs truth-test tensors, numbers, and booleans", function()
  local left = numlua.array({ 1, 0, 2 })
  assert_dtype(numlua.logical_and(left, true), "bool")
  assert_tensor(numlua.logical_and(left, true), { true, false, true })
  assert_tensor(numlua.logical_and(left, 0), { false, false, false })
  assert_tensor(numlua.logical_or(false, left), { true, false, true })
  assert_tensor(numlua.logical_xor(
    numlua.array({ true, false, true }), numlua.array({ true, true, false })),
    { false, true, true })
  assert_tensor(numlua.logical_not(left), { false, true, false })
  assert_eq(numlua.logical_not(false):item(), true)

  local broadcast = numlua.logical_and(
    numlua.array({ { true, false }, { true, true } }, { dtype = "bool" }),
    numlua.array({ true, false }))
  assert_shape(broadcast, { 2, 2 })
  assert_tensor(broadcast, { { true, false }, { true, false } })
end)

test("new ufuncs validate their operands", function()
  assert_error(function() numlua.isinf("x") end, "expects a tensor or number")
  assert_error(function() numlua.floor({ 1 }) end, "expects a tensor or number")
  assert_error(function() numlua.logical_and("x", true) end,
    "logical_and")
  assert_error(function() numlua.logical_or(true, "x") end,
    "logical_or")
  assert_error(function() numlua.logical_not({}) end,
    "logical_not")
end)
