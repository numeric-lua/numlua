--
-- Focused parity checks for 1-based per-slice selection.
--
-- take_along_axis is a facade over ATen gather, so these tests compare it
-- with independent selection paths (sort/argsort, concatenated reductions,
-- take over explicitly computed flat C-order indices) instead of only
-- re-checking literal outputs.
--
local numlua = require("numlua")

local function int64(values)
  return numlua.array(values, { dtype = "int64" })
end

local function count_of(shape)
  local count = 1
  for dimension = 1, #shape do count = count * shape[dimension] end
  return count
end

-- Build a nested table of `shape` whose values cycle through 1 .. limit.
local function cyclic_indices(shape, limit)
  local counter = 0
  local function build(depth)
    if depth > #shape then
      counter = counter + 1
      return ((counter - 1) % limit) + 1
    end
    local branch = {}
    for position = 1, shape[depth] do
      branch[position] = build(depth + 1)
    end
    return branch
  end
  return build(1)
end

-- Convert nested 1-based coordinates selected along `axis` into the flat
-- C-order index that numlua.take expects.  `sizes` is the input shape and
-- `values` has the output (here, input) shape.
local function flatten_indices(values, sizes, axis)
  local flat = {}
  local coords = {}
  local function walk(value, depth)
    if depth > #sizes then
      local offset = 1
      for dimension = 1, #sizes do
        local coordinate = dimension == axis and value or coords[dimension]
        offset = (offset - 1) * sizes[dimension] + coordinate
      end
      flat[#flat + 1] = offset
      return
    end
    for position = 1, #value do
      coords[depth] = position
      walk(value[position], depth + 1)
    end
  end
  walk(values, 1)
  return int64(flat)
end

test("take_along_axis reproduces argsort values on every axis", function()
  local t = numlua.arange(24):reshape(2, 3, 4)
  assert_dtype(numlua.argsort(t, 1), "int64")
  for axis = 1, 3 do
    local picked = numlua.take_along_axis(t, numlua.argsort(t, axis), axis)
    local sorted_values = t:sort(axis)
    assert_shape(picked, { 2, 3, 4 })
    assert_true(picked:equal(sorted_values),
      "take_along_axis(argsort(t)) is not sorted on axis " .. axis)
  end
end)

test("take_along_axis supports NumPy's min/max recipe", function()
  local t = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
  local both = numlua.concatenate({
    numlua.argmin(t, 2, true),
    numlua.argmax(t, 2, true),
  }, 2)
  assert_tensor(both, { { 1, 2 }, { 2, 1 } })
  assert_tensor(numlua.take_along_axis(t, both, 2),
    { { 10, 30 }, { 40, 60 } })
end)

test("take_along_axis matches take over flat C-order indices", function()
  local cases = {
    { shape = { 12 }, axis = 1 },
    { shape = { 3, 4 }, axis = 1 },
    { shape = { 3, 4 }, axis = 2 },
    { shape = { 2, 3, 4 }, axis = 1 },
    { shape = { 2, 3, 4 }, axis = 2 },
    { shape = { 2, 3, 4 }, axis = 3 },
    { shape = { 2, 2, 2, 3 }, axis = 4 },
  }
  for _, case in ipairs(cases) do
    local total = count_of(case.shape)
    local t = numlua.arange(1, total + 1):reshape(case.shape)
    local values = cyclic_indices(case.shape, case.shape[case.axis])
    local indices = int64(values)
    local picked = numlua.take_along_axis(t, indices, case.axis)
    local expected = numlua.take(t,
      flatten_indices(values, case.shape, case.axis)):reshape(case.shape)
    assert_shape(picked, case.shape)
    assert_true(picked:equal(expected),
      "take_along_axis disagrees with take for axis " .. case.axis ..
      " of a " .. #case.shape .. "-D input")
  end
end)

test("take_along_axis broadcasts size-1 non-axis dimensions", function()
  -- The input's first axis is size 1 and grows to the index length.
  local t = numlua.arange(1, 7):reshape(1, 3, 2)
  local values = { { { 2, 1 }, { 1, 2 }, { 2, 2 } },
    { { 1, 1 }, { 2, 1 }, { 1, 2 } } }
  local picked = numlua.take_along_axis(t, int64(values), 3)
  assert_shape(picked, { 2, 3, 2 })
  assert_tensor(picked, {
    { { 2, 1 }, { 3, 4 }, { 6, 6 } },
    { { 1, 1 }, { 4, 3 }, { 5, 6 } },
  })

  -- A size-1 index dimension broadcasts against a larger input axis.
  local matrix = numlua.arange(1, 13):reshape(3, 4)
  local column = numlua.take_along_axis(matrix, int64({ { 4 } }), 2)
  assert_shape(column, { 3, 1 })
  assert_tensor(column, { { 4 }, { 8 }, { 12 } })

  -- axis=nil is the flat form of take and keeps the index dtype contract.
  local flat = int64({ 24, 1, 13, 5 })
  assert_true(numlua.take_along_axis(numlua.arange(24), flat):equal(
    numlua.take(numlua.arange(24), flat)))
end)

test("take_along_axis rejects rank, axis, and dtype mismatches", function()
  local t = numlua.arange(24):reshape(2, 3, 4)
  assert_error(function()
    numlua.take_along_axis(t, int64({ 1, 2 }), 2)
  end, "same rank")
  assert_error(function()
    numlua.take_along_axis(t,
      int64({ { { 1 }, { 2 } }, { { 1 }, { 2 } }, { { 1 }, { 2 } } }), 2)
  end, "broadcast")
  assert_error(function()
    numlua.take_along_axis(t, int64({ { { 5 } } }), 3)
  end, "out of range")
  assert_error(function()
    numlua.take_along_axis(t, int64({ { { 1 } } }), 4)
  end, "out of range")
  assert_error(function()
    numlua.take_along_axis(t, numlua.array({ { { 1.5 } } }), 3)
  end, "integer Tensor")
end)

test("take_along_axis normalizes negative axes and empty selections",
    function()
  local matrix = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
  local order = numlua.argsort(matrix, -1)
  local sorted_last = matrix:sort(-1)
  local sorted_second = matrix:sort(2)
  assert_tensor(numlua.take_along_axis(matrix, order, -1),
    sorted_last:totable())
  assert_tensor(numlua.take_along_axis(matrix, order, 2),
    sorted_second:totable())

  assert_tensor(numlua.take(matrix,
    numlua.array({ 1, 3 }, { dtype = "int64" }), -1),
    { { 10, 20 }, { 60, 50 } })

  local empty = numlua.take_along_axis(matrix,
    numlua.zeros(2, 0, { dtype = "int64" }), -1)
  assert_shape(empty, { 2, 0 })
  assert_eq(empty:numel(), 0)
end)

test("selection edges cover compress, flatnonzero, and 0-D argsort",
    function()
  -- compress selects in C order, preserves the input dtype, and accepts an
  -- empty/zero-size selection.
  local values = numlua.array({ { 10, 20 }, { 30, 40 } })
  local mask = numlua.array({ true, false, true, false }, { dtype = "bool" })
  assert_tensor(numlua.compress(mask, values), { 10, 30 })
  assert_dtype(numlua.compress(mask, values), "float32")
  local empty_mask = numlua.array({}, { dtype = "bool" })
  local empty = numlua.compress(empty_mask, numlua.zeros(0))
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)

  -- Numeric conditions are truth-tested, and a shorter condition is a prefix,
  -- following np.compress.
  assert_tensor(numlua.compress(numlua.array({ 1, 0, 1 }), values),
    { 10, 30 })
  assert_tensor(numlua.compress(numlua.array({ true }), values), { 10 })
  assert_tensor(numlua.compress(numlua.array({ true }), values, 2),
    { { 10 }, { 30 } })
  assert_error(function()
    numlua.compress(numlua.array({ true, false, true }), values, 1)
  end, "condition has 3 elements but axis 1 has size 2")

  -- flatnonzero accepts 0-D input: a nonzero scalar has flat index 1, and a
  -- zero scalar has no indices at all.
  assert_tensor(numlua.flatnonzero(numlua.array(3)), { 1 })
  assert_shape(numlua.flatnonzero(numlua.array(0)), { 0 })
  assert_shape(numlua.flatnonzero(numlua.zeros(0)), { 0 })
  -- NaN is non-zero for this predicate, as in NumPy's boolean truth test.
  assert_tensor(numlua.flatnonzero(numlua.array({ 0, 0 / 0, 3 })),
    { 2, 3 })

  -- argsort's default axis is the last axis, and a 0-D input has no axis to
  -- sort; the facade reports that as a numlua error instead of leaking the
  -- luatorch dimension message.
  local matrix = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })
  assert_tensor(numlua.argsort(matrix), numlua.argsort(matrix, 2):totable())
  assert_error(function() numlua.argsort(numlua.array(7)) end,
    "at least a 1-D tensor")
  assert_error(function() numlua.argsort(numlua.array(7), 1) end,
    "at least a 1-D tensor")
end)

test("selection edges cover zero-size gather dimensions and bad indices",
    function()
  -- Zero-size non-axis dimensions do not require a first slice, so the
  -- broadcast shape is still well-defined.
  assert_shape(numlua.take_along_axis(numlua.zeros(0, 3),
    numlua.zeros(0, 1, { dtype = "int64" }), 1), { 0, 3 })
  assert_shape(numlua.take_along_axis(numlua.zeros(0, 3),
    numlua.zeros(0, 3, { dtype = "int64" }), 2), { 0, 3 })
  assert_shape(numlua.take_along_axis(numlua.zeros(2, 0, 3),
    numlua.zeros(2, 0, 3, { dtype = "int64" }), 2), { 2, 0, 3 })

  assert_error(function()
    numlua.take_along_axis(numlua.array(7),
      numlua.array({ 1 }, { dtype = "int64" }), 1)
  end, "out of range for 0-D input")

  -- take and take_along_axis reject non-tensor and nil indices before ATen
  -- can see them, including for 0-D inputs.
  assert_error(function() numlua.take(numlua.arange(3), { 1 }) end,
    "indices must be an integer Tensor")
  assert_error(function() numlua.take(numlua.arange(3), nil) end,
    "indices must be an integer Tensor")
  assert_error(function()
    numlua.take(numlua.array(7),
      numlua.array({ 1 }, { dtype = "int64" }), 1)
  end, "out of range for 0-D input")
end)

test("argsort descending orders values and validates the flag", function()
  local matrix = numlua.array({ { 3, 1, 2 }, { 6, 4, 5 } })
  local descending = numlua.argsort(matrix, 2, true)
  assert_dtype(descending, "int64")
  assert_tensor(descending, { { 1, 3, 2 }, { 1, 3, 2 } })

  -- The index result really is descending when used to reorder the values.
  assert_tensor(numlua.take_along_axis(matrix, descending, 2),
    { { 3, 2, 1 }, { 6, 5, 4 } })

  -- The default axis is the last one, and false matches the default order.
  assert_true(numlua.argsort(matrix, nil, true):equal(descending))
  assert_true(numlua.argsort(matrix, 2, false):equal(
    numlua.argsort(matrix, 2)))
  assert_true(numlua.argsort(matrix, -1, true):equal(descending))
  assert_tensor(numlua.argsort(numlua.array({ 3, 1, 2 }), 1, true),
    { 1, 3, 2 })

  -- Descending must be a boolean; truthy numbers and strings are rejected
  -- instead of silently selecting ascending order.
  assert_error(function()
    numlua.argsort(matrix, 2, 1)
  end, "descending must be a boolean")
  assert_error(function()
    numlua.argsort(matrix, 2, "true")
  end, "descending must be a boolean")
  assert_error(function()
    numlua.argsort(matrix, 2, {})
  end, "descending must be a boolean")
end)

test("selection edges cover non-contiguous inputs and empty broadcasts",
    function()
  -- A transposed 3x2 view indexes along either axis without a copy.
  local transposed = numlua.transpose(numlua.arange(6):reshape(2, 3))
  assert_false(transposed:is_contiguous())
  assert_tensor(numlua.take(transposed,
    numlua.array({ 3, 1 }, { dtype = "int64" }), 1),
    { { 2, 5 }, { 0, 3 } })
  assert_tensor(numlua.take(transposed,
    numlua.array({ 2, 1 }, { dtype = "int64" }), 2),
    { { 3, 0 }, { 4, 1 }, { 5, 2 } })
  assert_tensor(numlua.take_along_axis(transposed,
    numlua.array({ { 2, 1 }, { 1, 2 }, { 1, 1 } },
      { dtype = "int64" }), 2),
    { { 3, 0 }, { 1, 4 }, { 2, 2 } })

  -- Empty index tensors still broadcast non-axis dimensions from the input.
  local empty = numlua.take_along_axis(numlua.zeros(1, 0, 3),
    numlua.zeros(2, 0, 3, { dtype = "int64" }), 1)
  assert_shape(empty, { 2, 0, 3 })
  assert_eq(empty:numel(), 0)
  assert_shape(numlua.take(numlua.zeros(2, 3),
    numlua.zeros(0, { dtype = "int64" }), 1), { 0, 3 })

  -- compress and flatnonzero flatten non-contiguous operands in C order.
  local mask = numlua.transpose(numlua.array({
    { true, false, false },
    { false, true, false },
  }))
  assert_false(mask:is_contiguous())
  assert_tensor(numlua.compress(mask, numlua.arange(6)), { 0, 3 })
  assert_tensor(numlua.flatnonzero(mask), { 1, 4 })
  assert_tensor(numlua.flatnonzero(numlua.array({ false, true, false })),
    { 2 })
end)
