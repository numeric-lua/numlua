--
-- np.delete-shaped index removal.
--

local numlua = require("numlua")

test("delete removes flat and axis-aligned indices", function()
  local values = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  assert_tensor(numlua.delete(values, 2), { 10, 30, 40, 50, 60 })
  assert_tensor(numlua.delete(values, { 1, 6 }), { 20, 30, 40, 50 })
  assert_tensor(numlua.delete(values, -1), { 10, 20, 30, 40, 50 })
  assert_tensor(numlua.delete(values, 1, 1), { { 40, 50, 60 } })
  assert_tensor(numlua.delete(values, { 1, 3 }, 2),
    { { 20 }, { 50 } })
  assert_tensor(numlua.delete(values, {}, 1), values:totable())
  assert_shape(numlua.delete(values, { 1, 2 }, 1), { 0, 3 })
end)

test("delete resolves negative axes and duplicates", function()
  local values = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(numlua.delete(values, -1, -1), { { 1, 2 }, { 4, 5 } })
  assert_tensor(numlua.delete(values, { 1, 1 }, 1), { { 4, 5, 6 } })
  local copy = numlua.delete(values, {}, nil)
  assert_tensor(copy, { 1, 2, 3, 4, 5, 6 })
  copy:set(1, 99)
  assert_eq(values:get(1, 1), 1)
end)

test("delete validates indices, axes, and input", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function() numlua.delete({ 1, 2 }, 1) end,
    "expects a tensor")
  assert_error(function() numlua.delete(values, 0) end, "out of range")
  assert_error(function() numlua.delete(values, 5) end, "out of range")
  assert_error(function() numlua.delete(values, 1.5) end, "integers")
  assert_error(function() numlua.delete(values, true) end, "number, table")
  assert_error(function() numlua.delete(values, 1, 0) end, "axis")
  assert_error(function() numlua.delete(values, 3, 1) end, "out of range")
end)
