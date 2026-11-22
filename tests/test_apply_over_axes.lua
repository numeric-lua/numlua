--
-- Focused parity checks for numlua.apply_over_axes.
--
-- The helper applies a Lua callback repeatedly, so these tests pin NumPy's
-- rank rule: a callback that removes the axis is re-expanded at the same
-- position, a callback that keeps the rank is used unchanged, and any other
-- rank change is an error.  The reference results are built with ordinary
-- keepdims reductions instead of hard-coded literal tables.
--
local numlua = require("numlua")

local function sum_scalar(value, dim)
  return numlua.sum(value, dim)
end

local function nested_keepdims(value, axes)
  local expected = value
  for _, dim in ipairs(axes) do
    expected = numlua.sum(expected, dim, true)
  end
  return expected
end

test("apply_over_axes matches nested keepdims reductions", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local cases = {
    { 1 }, { 2 }, { 3 }, { -1 }, { -2 }, { -3 },
    { 1, 2 }, { 2, 1 }, { 1, 3 }, { 3, 1 }, { 2, 3 }, { 3, 2 },
    { 1, 2, 3 }, { 3, 2, 1 },
    -- Repeating an axis is legal in NumPy; after the first reduction the
    -- re-expanded axis has size 1, so the second reduction is a no-op.
    { 1, 1 }, { 2, 2 },
  }
  for _, axes in ipairs(cases) do
    local expected = nested_keepdims(cube, axes)
    local actual = numlua.apply_over_axes(sum_scalar, cube, axes)
    assert_shape(actual, expected:sizes(),
      "wrong shape for axes {" .. table.concat(axes, ", ") .. "}")
    assert_true(actual:equal(expected),
      "apply_over_axes(sum) disagrees with nested keepdims sum for axes {" ..
      table.concat(axes, ", ") .. "}")
    assert_dtype(actual, "int64")
  end

  -- A vector becomes a length-1 vector, never a 0-D tensor.
  local vector = numlua.array({ 1, 2, 3 })
  local reduced = numlua.apply_over_axes(sum_scalar, vector, { 1 })
  assert_shape(reduced, { 1 })
  assert_eq(reduced:get(1), 6)

  -- A single axis may be passed as a number instead of a table.
  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_true(numlua.apply_over_axes(sum_scalar, matrix, 1):equal(
    numlua.sum(matrix, 1, true)))

  -- An empty axis list returns the input untouched, matching NumPy.
  local identity = numlua.apply_over_axes(sum_scalar, matrix, {})
  assert_true(rawequal(identity, matrix))
end)

test("apply_over_axes keeps same-rank callback results", function()
  local matrix = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })

  -- cumsum keeps the rank, so it exercises the no-expand branch.  Applying it
  -- along axes 1 then 2 must match calling it twice by hand.
  local cumulative = numlua.apply_over_axes(numlua.cumsum, matrix, { 1, 2 })
  local expected = numlua.cumsum(numlua.cumsum(matrix, 1), 2)
  assert_shape(cumulative, { 2, 3 })
  assert_true(cumulative:equal(expected))

  -- A keepdims-style callback also keeps the rank, including the size-1
  -- reduced axis.
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local keepdims = numlua.apply_over_axes(function(value, dim)
    return numlua.sum(value, dim, true)
  end, cube, { 1, 2 })
  local nested = numlua.sum(numlua.sum(cube, 1, true), 2, true)
  assert_shape(keepdims, { 1, 1, 4 })
  assert_true(keepdims:equal(nested))

  -- A nontrivial non-axis shape is preserved for same-rank callbacks.
  local scaled = numlua.apply_over_axes(function(value, dim)
    return numlua.cumsum(value, dim):mul(2)
  end, matrix, { 2 })
  assert_true(scaled:equal(numlua.cumsum(matrix, 2):mul(2)))
end)

test("apply_over_axes normalizes axes and forwards callback arguments",
    function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local seen = {}
  local result = numlua.apply_over_axes(function(value, dim, offset)
    seen[#seen + 1] = { value:sizes(), dim }
    -- A scalar callback result: the rank rule re-expands it.
    return numlua.sum(value, dim):add(offset)
  end, cube, { -1, -3 }, 100)

  -- Negative axes are normalized against the current rank and the callback
  -- sees the current (re-expanded) value, not the original input.
  assert_eq(seen[1][2], 3)
  assert_eq(seen[2][2], 1)
  assert_eq(seen[1][1], { 2, 3, 4 })
  assert_eq(seen[2][1], { 2, 3, 1 })

  -- The extra argument reached every call: each reduction adds 100 once.
  local expected = numlua.sum(cube, 3, true)
  expected = numlua.sum(expected:add(100), 1, true):add(100)
  assert_shape(result, { 1, 3, 1 })
  assert_true(result:equal(expected))

  -- Lua tables and booleans are materialized like apply_along_axis results.
  local table_result = numlua.apply_over_axes(function(value, dim)
    return { value:size(dim), value:size(dim) + 1 }
  end, numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } }), { 1 })
  assert_shape(table_result, { 1, 2 })
  assert_dtype(table_result, "float32")
  assert_tensor(table_result, { { 2, 3 } })

  local boolean_result = numlua.apply_over_axes(function()
    return true
  end, numlua.array({ 1, 2, 3 }), { 1 })
  assert_shape(boolean_result, { 1 })
  assert_dtype(boolean_result, "bool")
  assert_true(boolean_result:get(1))
end)

test("apply_over_axes handles empty and non-contiguous inputs", function()
  -- Reductions over a zero-size selected axis still produce a size-1
  -- re-expanded axis, exactly like keepdims reductions.
  local cases = {
    { numlua.zeros(2, 0), { 2 } },
    { numlua.zeros(0, 3), { 1 } },
    { numlua.zeros(2, 0, 3), { 1, 2 } },
    { numlua.zeros(2, 0, 3), { 2, 3 } },
    { numlua.zeros(0), { 1 } },
  }
  for _, case in ipairs(cases) do
    local expected = nested_keepdims(case[1], case[2])
    local actual = numlua.apply_over_axes(sum_scalar, case[1], case[2])
    assert_shape(actual, expected:sizes())
    assert_true(actual:equal(expected))
    assert_eq(actual:numel(), expected:numel())
  end

  -- Non-contiguous views go through the same reductions and expansion.
  local transposed = numlua.transpose(
    numlua.arange(6):reshape(2, 3))
  assert_false(transposed:is_contiguous())
  local expected = nested_keepdims(transposed, { 2, 1 })
  local actual = numlua.apply_over_axes(sum_scalar, transposed, { 2, 1 })
  assert_shape(actual, { 1, 1 })
  assert_true(actual:equal(expected))
end)

test("apply_over_axes preserves reduction dtypes", function()
  local integers = numlua.array({ { 1, 2 }, { 3, 4 } },
    { dtype = "int64" })
  local sums = numlua.apply_over_axes(sum_scalar, integers, { 1, 2 })
  assert_dtype(sums, "int64")
  assert_shape(sums, { 1, 1 })
  assert_eq(sums:get(1, 1), 10)

  -- numlua.mean promotes integer input to float64, and the helper keeps that
  -- result dtype when it re-expands the axis.
  local means = numlua.apply_over_axes(numlua.mean, integers, { 2 })
  assert_dtype(means, "float64")
  assert_shape(means, { 2, 1 })
  assert_near(means:get(1, 1), 1.5)
  assert_near(means:get(2, 1), 3.5)
end)

test("apply_over_axes validates arguments and callback results", function()
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  local vector = numlua.array({ 1, 2, 3 })

  assert_error(function()
    numlua.apply_over_axes(sum_scalar, { 1, 2 }, { 1 })
  end, "expects a tensor")
  assert_error(function()
    numlua.apply_over_axes(nil, matrix, { 1 })
  end, "expects a function")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, numlua.array(7), { 1 })
  end, "not applied to a scalar")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, matrix, nil)
  end, "axis or a table of axes")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, matrix, "1")
  end, "axis or a table of axes")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, matrix, { 0 })
  end, "non-zero integer")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, matrix, { 1.5 })
  end, "non-zero integer")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, matrix, { "1" })
  end, "non-zero integer")
  assert_error(function()
    numlua.apply_over_axes(sum_scalar, matrix, { 3 })
  end, "out of range")

  -- Callback return types follow apply_along_axis, with a clear error for
  -- unsupported values and ragged tables.
  assert_error(function()
    numlua.apply_over_axes(function() return "nope" end, matrix, { 1 })
  end, "must return a tensor")
  assert_error(function()
    numlua.apply_over_axes(function() return nil end, matrix, { 1 })
  end, "must return a tensor")
  assert_error(function()
    numlua.apply_over_axes(function()
      return { { 1 }, { 1, 2 } }
    end, matrix, { 1 })
  end, "invalid table")

  -- A 2-D input must produce a 2-D or 1-D result.  Returning a 0-D scalar or
  -- a 2-D result from a 1-D input leaves the rank rule unsatisfied.
  assert_error(function()
    numlua.apply_over_axes(function()
      return numlua.array(1)
    end, matrix, { 1 })
  end, "correct shape")
  assert_error(function()
    numlua.apply_over_axes(function()
      return numlua.zeros(2, 2)
    end, vector, { 1 })
  end, "correct shape")

  -- Callback errors propagate unchanged.
  assert_error(function()
    numlua.apply_over_axes(function()
      error("callback boom")
    end, matrix, { 1 })
  end, "callback boom")
end)
