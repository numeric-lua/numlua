--
-- NumPy parity helpers added in 0.20.0: sort, unique, nonzero, argwhere.
--

local numlua = require("numlua")

test("sort returns values along the last axis by default", function()
  local matrix = numlua.array({ { 6, 1, 2 }, { 3, 4, 5 } })
  local sorted = numlua.sort(matrix)
  assert_dtype(sorted, "float32")
  assert_shape(sorted, { 2, 3 })
  assert_tensor(sorted, { { 1, 2, 6 }, { 3, 4, 5 } })

  -- The default, an explicit last axis, and a negative last axis agree.
  assert_true(sorted:equal(numlua.sort(matrix, 2)))
  assert_true(sorted:equal(numlua.sort(matrix, -1)))

  -- Axis 1 is a per-column sort, unlike the default per-row sort.
  assert_tensor(numlua.sort(matrix, 1), { { 3, 1, 2 }, { 6, 4, 5 } })

  -- Sorting is the value counterpart of argsort: gather the argsort order.
  for axis = 1, 2 do
    local order = numlua.argsort(matrix, axis)
    local gathered = numlua.take_along_axis(matrix, order, axis)
    assert_true(gathered:equal(numlua.sort(matrix, axis)),
      "sort disagrees with take_along_axis(argsort) on axis " .. axis)
  end
end)

test("sort handles 1-D, non-contiguous, and integer inputs", function()
  local vector = numlua.array({ 3, -1, 2, -1 })
  assert_tensor(numlua.sort(vector), { -1, -1, 2, 3 })

  local transposed = numlua.transpose(numlua.array({
    { 9, 1, 5 },
    { 2, 8, 3 },
  }))
  assert_false(transposed:is_contiguous())
  assert_tensor(numlua.sort(transposed),
    { { 2, 9 }, { 1, 8 }, { 3, 5 } })

  local ints = numlua.array({ { 4, 2 }, { 9, 1 } }, { dtype = "int64" })
  local sorted_ints = numlua.sort(ints, 2)
  assert_dtype(sorted_ints, "int64")
  assert_tensor(sorted_ints, { { 2, 4 }, { 1, 9 } })
end)

test("sort supports descending order and empty axes", function()
  local matrix = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })
  local descending = numlua.sort(matrix, 2, true)
  assert_tensor(descending, { { 3, 2, 1 }, { 6, 5, 4 } })
  assert_true(descending:equal(
    numlua.take_along_axis(matrix, numlua.argsort(matrix, 2, true), 2)))

  local empty = numlua.sort(numlua.zeros(2, 0))
  assert_shape(empty, { 2, 0 })
  assert_eq(empty:numel(), 0)
end)

test("sort validates rank, axes, and the descending flag", function()
  assert_error(function() numlua.sort(numlua.array(7)) end,
    "at least a 1-D tensor")
  assert_error(function() numlua.sort(numlua.array({ 1, 2 }), 0) end,
    "axis must be a non-zero integer")
  assert_error(function() numlua.sort(numlua.array({ 1, 2 }), 2) end,
    "out of range")
  assert_error(function()
    numlua.sort(numlua.array({ 1, 2 }), 1, 1)
  end, "descending must be a boolean")
  assert_error(function() numlua.sort({ 1, 2 }) end, "expects a tensor")
end)

test("sort returns independent storage", function()
  local input = numlua.array({ 3, 1, 2 })
  local sorted = numlua.sort(input)
  sorted:fill_(0)
  assert_tensor(input, { 3, 1, 2 })
end)
