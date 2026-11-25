--
-- np.average-shaped weighted means.
--

local numlua = require("numlua")

test("average without weights matches mean", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_near(numlua.average(values):item(), 2.5, 0)
  assert_tensor(numlua.average(values, 1), { 2, 3 })
  assert_tensor(numlua.average(values, 2), { 1.5, 3.5 })
  assert_tensor(numlua.average(values, -1), { 1.5, 3.5 })
  assert_dtype(numlua.average(values, 1), "float32")
end)

test("average supports 1-D weights along an axis", function()
  local values = numlua.array({ 1, 2, 3 })
  assert_near(numlua.average(values, nil, { 1, 2, 3 }):item(),
    (1 + 4 + 9) / 6, 1e-6)
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.average(matrix, 1, { 1, 3 }), { 2.5, 3.5 })
  assert_tensor(numlua.average(matrix, 2, { 1, 3 }), { 1.75, 3.75 })
  assert_tensor(numlua.average(matrix, -2, { 1, 3 }), { 2.5, 3.5 })
end)

test("average supports full-shape weights and returned weights", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  local weights = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_near(numlua.average(values, nil, weights):item(),
    (1 + 4 + 9 + 16) / 10, 1e-6)
  assert_tensor(numlua.average(values, 1, weights), { 2.5, 20 / 6 })
  local value, total = numlua.average(values, 1, { 1, 3 }, true)
  assert_tensor(value, { 2.5, 3.5 })
  assert_tensor(total, { 4, 4 })
  local _, count = numlua.average(values, 2, nil, true)
  assert_tensor(count, { 2, 2 })
  assert_dtype(count, "float64")
end)

test("average preserves ATen promotion and validates weights", function()
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_dtype(numlua.average(ints, 1), "float64")
  assert_dtype(numlua.average(ints, 1, { 1, 1 }), "float32")

  assert_error(function() numlua.average({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.average(numlua.zeros(2), 0) end, "1-based")
  assert_error(function() numlua.average(numlua.zeros(2), 2) end,
    "out of range")
  assert_error(function() numlua.average(numlua.zeros(2), nil, 1) end,
    "weights to be")
  assert_error(function() numlua.average(numlua.zeros(2), nil, { 1 }) end,
    "weights must match")
  assert_error(function() numlua.average(numlua.zeros(2, 3), 1, { 1 }) end,
    "weights have 1 elements")
  assert_error(function() numlua.average(numlua.zeros(2, 3), 1,
    numlua.zeros(2, 2)) end, "weights shape does not match")
  assert_error(function() numlua.average(numlua.zeros(2), nil,
    { 1, -1 }) end, "sum of weights")
  assert_error(function() numlua.average(numlua.zeros(2), nil,
    { 1, 2 }, "yes") end, "returned")
end)
