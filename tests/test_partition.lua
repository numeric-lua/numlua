--
-- NumPy partition/argpartition contracts.
--

local numlua = require("numlua")

test("partition returns sorted values satisfying the kth contract", function()
  local values = numlua.array({ 3, 1, 4, 2 })
  local partitioned = numlua.partition(values, 2)
  assert_tensor(partitioned, { 1, 2, 3, 4 })
  assert_eq(partitioned:get(2), 2)
  assert_dtype(partitioned, "float32")

  local matrix = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })
  assert_tensor(numlua.partition(matrix, 1, 2),
    { { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(numlua.partition(matrix, 2, 1),
    { { 3, 1, 2 }, { 6, 4, 5 } })
  assert_tensor(numlua.partition(matrix, { 1, 3 }, -1),
    { { 1, 2, 3 }, { 4, 5, 6 } })
end)

test("argpartition returns indices consistent with partition", function()
  local values = numlua.array({ 30, 10, 40, 20 })
  local indices = numlua.argpartition(values, 2)
  assert_dtype(indices, "int64")
  assert_tensor(indices, { 2, 4, 1, 3 })
  local gathered = values:index_select(1, indices)
  assert_tensor(gathered, { 10, 20, 30, 40 })

  local matrix = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })
  local matrix_indices = numlua.argpartition(matrix, 1, 2)
  assert_shape(matrix_indices, { 2, 3 })
end)

test("partition returns independent storage and validates arguments", function()
  local values = numlua.array({ 3, 1, 2 })
  local result = numlua.partition(values, 1)
  result:set(1, 99)
  assert_eq(values:get(1), 3)

  assert_error(function() numlua.partition(numlua.array(1), 1) end,
    "at least a 1-D")
  assert_error(function() numlua.partition(values, 0) end, "out of range")
  assert_error(function() numlua.partition(values, 4) end, "out of range")
  assert_error(function() numlua.partition(values, 1.5) end, "out of range")
  assert_error(function() numlua.partition(values, {}) end,
    "at least one kth")
  assert_error(function() numlua.partition(values, 1, 0) end, "axis")
  assert_error(function() numlua.partition(values, 1, 2) end, "out of range")
  assert_error(function() numlua.argpartition(values, 0) end, "out of range")
end)
