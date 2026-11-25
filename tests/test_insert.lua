--
-- insert with scalar positions.
--

local numlua = require("numlua")

test("insert adds values to flattened and 1-D input", function()
  local values = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.insert(values, 1, 0), { 0, 1, 2, 3 })
  assert_tensor(numlua.insert(values, 4, 4), { 1, 2, 3, 4 })
  assert_tensor(numlua.insert(values, 2, { 8, 9 }), { 1, 8, 9, 2, 3 })
  assert_tensor(numlua.insert(values, -1, 9), { 1, 2, 9, 3 })
  assert_tensor(numlua.insert(numlua.zeros(0), 1, { 1, 2 }), { 1, 2 })
  assert_tensor(numlua.insert(numlua.array({ { 1, 2 }, { 3, 4 } }), 2, 0),
    { 1, 0, 2, 3, 4 })
end)

test("insert works along an axis and fills scalar values", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.insert(values, 1, 0, 1),
    { { 0, 0 }, { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.insert(values, 2, 9, 2),
    { { 1, 9, 2 }, { 3, 9, 4 } })
  assert_tensor(numlua.insert(values, 3,
    numlua.array({ { 5, 6 } }), 1),
    { { 1, 2 }, { 3, 4 }, { 5, 6 } })
  assert_dtype(numlua.insert(
    numlua.array({ 1 }, { dtype = "int64" }), 1, 2.9), "int64")
end)

test("insert validates positions, ranks, shapes, and axes", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function() numlua.insert(values, 0, 1) end, "out of range")
  assert_error(function() numlua.insert(values, 6, 1) end, "out of range")
  assert_error(function() numlua.insert(values, 1.5, 1) end, "integer")
  assert_error(function() numlua.insert(values, { 1, 2 }, 1) end,
    "scalar position")
  assert_error(function() numlua.insert(values, 1, 1, 0) end, "axis")
  assert_error(function() numlua.insert(values, 1, { 1, 2 }, 1) end,
    "rank")
  assert_error(function() numlua.insert(values, 1,
    numlua.array({ { 1, 2, 3 } }), 1) end, "shape does not match")
  assert_error(function() numlua.insert({ 1, 2 }, 1, 0) end,
    "expects a tensor")
end)
