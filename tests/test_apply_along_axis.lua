--
-- Focused parity checks for numlua.apply_along_axis.
--
-- The callback is a Lua function, so these tests pin the slice shape and
-- C-order visitation, the NumPy result-shape rule (scalar removes the axis,
-- 1-D results append their length), dtype propagation, extra-argument
-- forwarding, and the empty/error edges that a straightforward sum-based test
-- would miss.
--
local numlua = require("numlua")

local function sum_slice(slice)
  return numlua.sum(slice)
end

test("apply_along_axis reduces a scalar callback on every axis", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  for axis = 1, 3 do
    local result = numlua.apply_along_axis(sum_slice, axis, cube)
    local expected = numlua.sum(cube, axis)
    assert_shape(result, expected:sizes())
    assert_true(result:equal(expected),
      "apply_along_axis(sum) disagrees with sum on axis " .. axis)
  end

  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(numlua.apply_along_axis(sum_slice, 1, matrix), { 5, 7, 9 })
  assert_tensor(numlua.apply_along_axis(sum_slice, 2, matrix), { 6, 15 })
  assert_dtype(numlua.apply_along_axis(sum_slice, 2, matrix), "float32")

  -- A 1-D input has no iteration axes left, so a scalar callback produces a
  -- 0-D tensor, matching NumPy.
  local scalar = numlua.apply_along_axis(sum_slice, 1,
    numlua.array({ 1, 2, 3 }))
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 6)

  -- Negative axes count from the end.
  local negative = numlua.apply_along_axis(sum_slice, -1, cube)
  assert_true(negative:equal(numlua.sum(cube, 3)))
end)

test("apply_along_axis visits slices in C order", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local first_values = {}
  local result = numlua.apply_along_axis(function(slice)
    assert_eq(slice:dim(), 1, "callback slice must be 1-D")
    assert_eq(slice:size(1), 3, "callback slice length must match the axis")
    first_values[#first_values + 1] = slice:get(1)
    return slice:get(1)
  end, 2, cube)

  -- The non-axis dimensions are (axis 1, axis 3), so slices are visited with
  -- the last outer dimension varying fastest.
  assert_eq(first_values, { 0, 1, 2, 3, 12, 13, 14, 15 })
  assert_shape(result, { 2, 4 })
  assert_tensor(result, { { 0, 1, 2, 3 }, { 12, 13, 14, 15 } })

  local columns = {}
  numlua.apply_along_axis(function(slice)
    columns[#columns + 1] = slice:get(1)
    return slice:get(1)
  end, 1, numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } }))
  assert_eq(columns, { 1, 2, 3 })
end)

test("apply_along_axis appends 1-D callback results", function()
  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })

  local reversed_rows = numlua.apply_along_axis(function(slice)
    return numlua.flip(slice, 1)
  end, 2, matrix)
  assert_shape(reversed_rows, { 2, 3 })
  assert_tensor(reversed_rows, { { 3, 2, 1 }, { 6, 5, 4 } })

  local reversed_columns = numlua.apply_along_axis(function(slice)
    return numlua.flip(slice, 1)
  end, 1, matrix)
  assert_shape(reversed_columns, { 3, 2 })
  assert_tensor(reversed_columns, {
    { 4, 1 },
    { 5, 2 },
    { 6, 3 },
  })

  -- A 3-D input keeps NumPy's "non-axis dimensions, then result length" order
  -- rather than inserting the result back at the reduced axis.
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local rows = numlua.apply_along_axis(function(slice)
    return slice:narrow(1, 1, 2)
  end, 2, cube)
  assert_shape(rows, { 2, 4, 2 })
  assert_tensor(rows:select(1, 1), {
    { 0, 4 },
    { 1, 5 },
    { 2, 6 },
    { 3, 7 },
  })

  -- Lua tables and numbers are materialized, so callbacks can stay in plain
  -- Lua; table results follow the same 1-D result rule.
  local pair = numlua.apply_along_axis(function(slice)
    return { slice:get(1), slice:get(2) }
  end, 2, matrix)
  assert_dtype(pair, "float32")
  assert_shape(pair, { 2, 2 })
  assert_tensor(pair, { { 1, 2 }, { 4, 5 } })

  local lengths = numlua.apply_along_axis(function(slice)
    return slice:size(1)
  end, 1, matrix)
  assert_dtype(lengths, "float32")
  assert_tensor(lengths, { 2, 2, 2 })
end)

test("apply_along_axis preserves tensor dtypes and owns its output", function()
  local integers = numlua.array({ { 1, 2 }, { 3, 4 } },
    { dtype = "int64" })
  local sums = numlua.apply_along_axis(sum_slice, 2, integers)
  assert_dtype(sums, "int64")
  assert_tensor(sums, { 3, 7 })

  local flags = numlua.apply_along_axis(function(slice)
    return numlua.sum(slice):item() > 2
  end, 2, integers)
  assert_dtype(flags, "bool")
  assert_tensor(flags, { true, true })

  -- The callback may return a view of the input; the output must still copy
  -- rather than alias the source storage.
  local copy = numlua.apply_along_axis(function(slice)
    return slice
  end, 2, integers)
  assert_shape(copy, { 2, 2 })
  assert_dtype(copy, "int64")
  copy:set(1, 1, 99)
  assert_tensor(integers, { { 1, 2 }, { 3, 4 } })

  local transposed = numlua.transpose(integers)
  local copied = numlua.apply_along_axis(function(slice)
    return slice
  end, 1, transposed)
  copied:set(1, 1, 77)
  assert_eq(transposed:get(1, 1), 1)
end)

test("apply_along_axis forwards extra callback arguments", function()
  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local result = numlua.apply_along_axis(function(slice, offset, scale)
    return numlua.sum(slice):item() * scale + offset
  end, 2, matrix, 10, 2)
  assert_tensor(result, { 22, 40 })

  local seen = 0
  local count = numlua.apply_along_axis(function(slice, a, b, c)
    seen = seen + 1
    assert_eq(a, "a", "first extra argument")
    assert_eq(b, "b", "second extra argument")
    assert_eq(c, nil, "unexpected extra argument")
    return slice:get(1)
  end, 1, matrix, "a", "b")
  assert_eq(seen, 3)
  assert_shape(count, { 3 })
end)

test("apply_along_axis handles empty axes and empty iteration spaces",
    function()
  -- The slices are empty but there are still two of them, so a scalar
  -- callback can produce a normal 0-filled result.
  local empty_rows = numlua.apply_along_axis(sum_slice, 2,
    numlua.zeros(2, 0))
  assert_shape(empty_rows, { 2 })
  assert_tensor(empty_rows, { 0, 0 })

  local empty_vector = numlua.apply_along_axis(sum_slice, 1,
    numlua.zeros(0))
  assert_eq(empty_vector:dim(), 0)
  assert_eq(empty_vector:item(), 0)

  -- A zero non-axis dimension still leaves one slice per remaining position.
  local empty_columns = numlua.apply_along_axis(sum_slice, 1,
    numlua.zeros(0, 3))
  assert_shape(empty_columns, { 3 })
  assert_tensor(empty_columns, { 0, 0, 0 })

  -- A 1-D callback over an empty axis appends the zero result length.
  local empty_results = numlua.apply_along_axis(function(slice)
    return slice
  end, 2, numlua.zeros(2, 0, 3))
  assert_shape(empty_results, { 2, 3, 0 })
  assert_eq(empty_results:numel(), 0)

  -- With no slices at all there is no first result to infer a dtype or
  -- result length from, matching NumPy's failure mode.
  assert_error(function()
    numlua.apply_along_axis(sum_slice, 2, numlua.zeros(0, 3))
  end, "no slices along axis 2")
  assert_error(function()
    numlua.apply_along_axis(sum_slice, 1, numlua.zeros(2, 0, 3))
  end, "no slices along axis 1")
end)

test("apply_along_axis supports non-contiguous inputs", function()
  local source = numlua.arange(6):reshape(2, 3)
  local transposed = numlua.transpose(source)
  assert_false(transposed:is_contiguous())

  local result = numlua.apply_along_axis(sum_slice, 1, transposed)
  assert_true(result:equal(numlua.sum(transposed, 1)))

  local cube = numlua.arange(24):reshape(2, 3, 4)
  local permuted = numlua.transpose(cube, { 3, 1, 2 })
  assert_false(permuted:is_contiguous())
  for axis = 1, 3 do
    local picked = numlua.apply_along_axis(sum_slice, axis, permuted)
    assert_true(picked:equal(numlua.sum(permuted, axis)),
      "non-contiguous sum disagrees on axis " .. axis)
  end
end)

test("apply_along_axis validates the callback, input, axis, and results",
    function()
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function()
    numlua.apply_along_axis(nil, 1, matrix)
  end, "expects a function")
  assert_error(function()
    numlua.apply_along_axis({}, 1, matrix)
  end, "expects a function")
  assert_error(function()
    numlua.apply_along_axis(sum_slice, 1, { 1, 2 })
  end, "expects a tensor")
  assert_error(function()
    numlua.apply_along_axis(sum_slice, 1, numlua.array(3))
  end, "at least a 1-D tensor")
  assert_error(function()
    numlua.apply_along_axis(sum_slice, 0, matrix)
  end, "non-zero integer")
  assert_error(function()
    numlua.apply_along_axis(sum_slice, nil, matrix)
  end, "non-zero integer")
  assert_error(function()
    numlua.apply_along_axis(sum_slice, 3, matrix)
  end, "out of range")

  assert_error(function()
    numlua.apply_along_axis(function(slice)
      return numlua.stack({ slice, slice })
    end, 2, matrix)
  end, "scalar or 1-D array")
  assert_error(function()
    numlua.apply_along_axis(function(slice)
      return { { slice:get(1) }, { slice:get(2) } }
    end, 2, matrix)
  end, "scalar or 1-D array")
  assert_error(function()
    numlua.apply_along_axis(function(slice)
      if slice:get(1) == 1 then return slice end
      return numlua.sum(slice)
    end, 2, matrix)
  end, "after a 1-D result")
  assert_error(function()
    numlua.apply_along_axis(function(slice)
      if slice:get(1) == 1 then return slice end
      return slice:narrow(1, 1, 1)
    end, 2, matrix)
  end, "length")
  assert_error(function()
    numlua.apply_along_axis(function() return "nope" end, 2, matrix)
  end, "must return a scalar")
  assert_error(function()
    numlua.apply_along_axis(function() error("callback boom") end, 2, matrix)
  end, "callback boom")
end)
