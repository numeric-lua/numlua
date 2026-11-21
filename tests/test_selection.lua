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
