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

test("unique returns sorted flattened values by default", function()
  local matrix = numlua.array({ { 3, 1, 2 }, { 3, 2, 1 } })
  local values, extra = numlua.unique(matrix)
  assert_dtype(values, "float32")
  assert_shape(values, { 3 })
  assert_tensor(values, { 1, 2, 3 })
  assert_eq(extra, nil)

  local ints = numlua.unique(
    numlua.array({ 5, -1, 5, 0, -1 }, { dtype = "int64" }))
  assert_dtype(ints, "int64")
  assert_tensor(ints, { -1, 0, 5 })

  local bools = numlua.unique(numlua.array({ true, false, true }))
  assert_dtype(bools, "bool")
  assert_tensor(bools, { false, true })
end)

test("unique returns first indices, inverse, and counts in NumPy order",
    function()
  local matrix = numlua.array({ { 3, 1, 2 }, { 3, 2, 1 } })
  local values, index, inverse, counts = numlua.unique(matrix, {
    return_index = true,
    return_inverse = true,
    return_counts = true,
  })
  assert_tensor(values, { 1, 2, 3 })
  -- First occurrences in C order are 1 at position 2, 2 at 3, 3 at 1.
  assert_dtype(index, "int64")
  assert_tensor(index, { 2, 3, 1 })
  assert_dtype(inverse, "int64")
  assert_shape(inverse, { 2, 3 })
  assert_tensor(inverse, { { 3, 1, 2 }, { 3, 2, 1 } })
  assert_dtype(counts, "int64")
  assert_tensor(counts, { 2, 2, 2 })

  -- Requested results are appended in NumPy's fixed order, independent of
  -- the order of the option keys.
  local only_index = numlua.unique(matrix, { return_index = true })
  assert_tensor(only_index, numlua.unique(matrix):totable())
  local _, selected_index = numlua.unique(matrix, { return_index = true })
  assert_tensor(selected_index, { 2, 3, 1 })

  local _, selected_inverse = numlua.unique(matrix, { return_inverse = true })
  assert_tensor(selected_inverse, { { 3, 1, 2 }, { 3, 2, 1 } })

  local _, selected_counts = numlua.unique(matrix, { return_counts = true })
  assert_tensor(selected_counts, { 2, 2, 2 })
end)

test("unique handles 0-D, empty, and non-contiguous inputs", function()
  local scalar_values, scalar_index, scalar_inverse, scalar_counts =
    numlua.unique(numlua.array(7), {
      return_index = true,
      return_inverse = true,
      return_counts = true,
    })
  assert_shape(scalar_values, { 1 })
  assert_tensor(scalar_values, { 7 })
  assert_tensor(scalar_index, { 1 })
  assert_shape(scalar_inverse, { 1 })
  assert_tensor(scalar_inverse, { 1 })
  assert_tensor(scalar_counts, { 1 })

  local empty_values, empty_inverse, empty_counts = numlua.unique(
    numlua.zeros(0, 2), { return_inverse = true, return_counts = true })
  assert_shape(empty_values, { 0 })
  assert_shape(empty_inverse, { 0, 2 })
  assert_shape(empty_counts, { 0 })

  local transposed = numlua.transpose(numlua.array({
    { 3, 1 },
    { 2, 3 },
  }))
  local values, inverse = numlua.unique(transposed, { return_inverse = true })
  assert_tensor(values, { 1, 2, 3 })
  assert_tensor(inverse, { { 3, 2 }, { 1, 3 } })
end)

test("unique validates options and the input", function()
  local t = numlua.array({ 1, 1, 2 })
  assert_error(function() numlua.unique({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.unique(t, 1) end,
    "options must be a table")
  assert_error(function()
    numlua.unique(t, { axis = 1 })
  end, "unknown option 'axis'")
  assert_error(function()
    numlua.unique(t, { return_counts = 1 })
  end, "return_counts must be a boolean")
  assert_error(function()
    numlua.unique(t, { return_index = "yes" })
  end, "return_index must be a boolean")
end)
