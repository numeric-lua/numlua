--
-- np.compress-shaped selection, including axis-aligned slices.
--

local numlua = require("numlua")

test("compress selects slices along an axis", function()
  local t = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  assert_shape(numlua.compress({ true, false }, t, 1), { 1, 3 })
  assert_tensor(numlua.compress({ true, false }, t, 1), { { 10, 20, 30 } })
  assert_tensor(numlua.compress({ true, false, true }, t, 2),
    { { 10, 30 }, { 40, 60 } })
  assert_tensor(numlua.compress({ false, true }, t, -2),
    { { 40, 50, 60 } })
  assert_tensor(numlua.compress({ false, true, true }, t, -1),
    { { 20, 30 }, { 50, 60 } })
end)

test("compress treats short conditions as a prefix and empty ones as empty", function()
  local t = numlua.array({ { 10, 20 }, { 30, 40 } })
  assert_tensor(numlua.compress({ true }, t, 1), { { 10, 20 } })
  assert_tensor(numlua.compress({ true }, t, 2), { { 10 }, { 30 } })
  local empty = numlua.compress({}, t, 1)
  assert_shape(empty, { 0, 2 })
  assert_eq(empty:numel(), 0)
  local empty_columns = numlua.compress({}, t, 2)
  assert_shape(empty_columns, { 2, 0 })
  assert_eq(empty_columns:numel(), 0)
end)

test("compress truth-tests numeric conditions and accepts tables", function()
  local t = numlua.array({ 10, 20, 30, 40 })
  assert_tensor(numlua.compress({ 1, 0, 1 }, t), { 10, 30 })
  assert_tensor(numlua.compress(
    numlua.array({ 0, 2, 0, 0 }), t), { 20 })
  local mask = numlua.array({ true, false, true, false }, { dtype = "bool" })
  local selected = numlua.compress(mask, t)
  assert_dtype(selected, "float32")
  -- The result owns independent storage.
  selected:set(1, 99)
  assert_eq(t:get(1), 10)
end)

test("compress works on 3-D tensors and preserves dtype", function()
  local cube = numlua.arange(24, { dtype = "int64" }):reshape(2, 3, 4)
  local selected = numlua.compress({ true, false, true }, cube, 2)
  assert_shape(selected, { 2, 2, 4 })
  assert_dtype(selected, "int64")
  assert_eq(selected:get(1, 1, 1), 0)
  assert_eq(selected:get(1, 2, 1), 8)
  assert_eq(selected:get(2, 1, 1), 12)
  assert_eq(selected:get(2, 2, 1), 20)
end)

test("compress validates its operands", function()
  local t = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function() numlua.compress("mask", t) end,
    "boolean tensor or table")
  assert_error(function() numlua.compress({ true }, { 1, 2 }) end,
    "expects a tensor")
  assert_error(function() numlua.compress(
    { true, false, true, false, true }, t) end,
    "condition has 5 elements but the input has only 4")
  assert_error(function() numlua.compress({ true, false, true }, t, 1) end,
    "condition has 3 elements but axis 1 has size 2")
  assert_error(function() numlua.compress({ true, false }, t, 3) end,
    "out of range")
  assert_error(function() numlua.compress({ true, false }, t, 0) end,
    "1-based")
  assert_error(function() numlua.compress(
    numlua.array({ { true, false }, { false, true } }), t, 1) end,
    "1-D condition")
end)
