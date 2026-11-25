--
-- unique(axis=...) support.
--

local numlua = require("numlua")

test("unique along axis 1 deduplicates rows", function()
  local values = numlua.array({
    { 1, 2 },
    { 3, 4 },
    { 1, 2 },
  })
  local unique, index, inverse, counts = numlua.unique(values, {
    axis = 1,
    return_index = true,
    return_inverse = true,
    return_counts = true,
  })
  assert_tensor(unique, { { 1, 2 }, { 3, 4 } })
  assert_tensor(index, { 1, 2 })
  assert_tensor(inverse, { 1, 2, 1 })
  assert_tensor(counts, { 2, 1 })
end)

test("unique along axis 2 deduplicates columns", function()
  local values = numlua.array({
    { 1, 3, 1 },
    { 2, 4, 2 },
  })
  local unique, index, inverse, counts = numlua.unique(values, {
    axis = -1,
    return_index = true,
    return_inverse = true,
    return_counts = true,
  })
  assert_tensor(unique, { { 1, 3 }, { 2, 4 } })
  assert_tensor(index, { 1, 2 })
  assert_tensor(inverse, { 1, 2, 1 })
  assert_tensor(counts, { 2, 1 })
end)

test("unique along an axis works for N-D input", function()
  local cube = numlua.array({
    { { 1, 2 }, { 1, 2 } },
    { { 3, 4 }, { 3, 4 } },
  })
  local unique = numlua.unique(cube, { axis = 1 })
  assert_shape(unique, { 2, 2, 2 })
  assert_tensor(unique, { { { 1, 2 }, { 1, 2 } },
    { { 3, 4 }, { 3, 4 } } })
end)

test("unique axis handles empty axes and validates arguments", function()
  local empty = numlua.unique(numlua.zeros(0, 2), { axis = 1 })
  assert_shape(empty, { 0, 2 })
  assert_eq(empty:numel(), 0)

  local values = numlua.array({ { 1, 2 }, { 1, 2 } })
  assert_error(function() numlua.unique(values, { axis = 0 }) end, "axis")
  assert_error(function() numlua.unique(values, { axis = 3 }) end,
    "out of range")
  assert_error(function() numlua.unique(numlua.array(1), { axis = 1 }) end,
    "at least a 1-D")
  assert_error(function() numlua.unique(values, { axis = "1" }) end, "axis")
end)
