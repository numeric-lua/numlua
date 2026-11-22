--
-- Focused np.pad parity checks for the wrap mode and constant_values forms.
--
-- Expected values are built by an independent index walk, so these tests do
-- not just re-run the implementation's modular arithmetic.
--
local numlua = require("numlua")

local function wrap_index(position, size, before)
  local index = position - before
  while index < 1 do index = index + size end
  while index > size do index = index - size end
  return index
end

-- Build the wrapped result for an arbitrary nested array and per-axis
-- {before, after} pairs.
local function wrap_expected(values, pads)
  local function build(value, dim)
    if dim > #pads then
      return value
    end
    local size = #value
    local before, after = pads[dim][1], pads[dim][2]
    local result = {}
    for position = 1, before + size + after do
      result[position] = build(value[wrap_index(position, size, before)],
        dim + 1)
    end
    return result
  end
  return build(values, 1)
end

test("pad wrap repeats each axis periodically", function()
  local vector = numlua.array({ 10, 20, 30 })
  for _, widths in ipairs({
    { 0, 0 }, { 1, 1 }, { 2, 2 }, { 5, 4 }, { 7, 1 },
  }) do
    local expected = wrap_expected(vector:totable(),
      { { widths[1], widths[2] } })
    local padded = numlua.pad(vector, { widths[1], widths[2] }, "wrap")
    assert_tensor(padded, expected)
    assert_shape(padded, { widths[1] + 3 + widths[2] })
  end

  local matrix = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  })
  local cases = {
    { 1, { { 1, 1 }, { 1, 1 } } },
    { { 2, 1 }, { { 2, 1 }, { 2, 1 } } },
    { { { 1, 2 }, { 2, 1 } }, { { 1, 2 }, { 2, 1 } } },
  }
  for _, case in ipairs(cases) do
    local padded = numlua.pad(matrix, case[1], "wrap")
    assert_tensor(padded, wrap_expected(matrix:totable(), case[2]))
    assert_shape(padded, { 2 + case[2][1][1] + case[2][1][2],
      3 + case[2][2][1] + case[2][2][2] })
  end

  local cube = numlua.arange(24):reshape(2, 3, 4)
  local pads = { { 1, 0 }, { 0, 2 }, { 2, 1 } }
  local padded_cube = numlua.pad(cube, pads, "wrap")
  assert_tensor(padded_cube, wrap_expected(cube:totable(), pads))
  assert_shape(padded_cube, { 3, 5, 7 })
end)

test("pad wrap works on non-contiguous input and preserves dtype", function()
  local base = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = "int64" })
  local view = numlua.transpose(base)
  assert_false(view:is_contiguous())
  local padded = numlua.pad(view, { { 1, 2 }, { 0, 1 } }, "wrap")
  assert_dtype(padded, "int64")
  assert_shape(padded, { 6, 3 })
  assert_tensor(padded, wrap_expected(view:totable(),
    { { 1, 2 }, { 0, 1 } }))

  local mask = numlua.array({ true, false, true })
  local wrapped_mask = numlua.pad(mask, 1, "wrap")
  assert_dtype(wrapped_mask, "bool")
  assert_tensor(wrapped_mask, { true, true, false, true, true })

  local half = numlua.array({ 1.5, -2.25, 0 }):to("float16")
  local wrapped_half = numlua.pad(half, { 1, 2 }, "wrap")
  assert_dtype(wrapped_half, "float16")
  assert_tensor(wrapped_half, { 0, 1.5, -2.25, 0, 1.5, -2.25 }, 0)

  -- The result owns independent storage.
  padded:set(1, 1, 99)
  assert_eq(view:get(1, 1), 1)
end)

test("pad wrap allows zero-width pads on empty axes but rejects extension",
    function()
  local empty = numlua.zeros(0, 3)
  local unchanged = numlua.pad(empty, { { 0, 0 }, { 0, 0 } }, "wrap")
  assert_shape(unchanged, { 0, 3 })
  assert_eq(unchanged:numel(), 0)

  assert_error(function() numlua.pad(empty, { 1, 0 }, "wrap") end,
    "empty axis")
  assert_error(function()
    numlua.pad(empty, { { 0, 1 }, { 0, 0 } }, "wrap")
  end, "empty axis")
  -- A non-empty axis may still be padded when another axis is empty and has
  -- zero padding.
  local columns = numlua.pad(empty, { { 0, 0 }, { 1, 1 } }, "wrap")
  assert_shape(columns, { 0, 5 })
end)

test("pad constant values accept every NumPy form", function()
  local vector = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.pad(vector, { 1, 0 }, "constant", 9), { 9, 1, 2, 3 })
  assert_tensor(numlua.pad(vector, { 1, 0 }, "constant", { 9 }),
    { 9, 1, 2, 3 })
  assert_tensor(numlua.pad(vector, { 1, 2 }, "constant", { 9, 8 }),
    { 9, 1, 2, 3, 8, 8 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int32" })
  assert_tensor(numlua.pad(matrix, { { 1, 0 }, { 0, 1 } }, "constant",
    { { 10, 11 }, { 20, 21 } }), {
    { 10, 10, 10 },
    { 1, 2, 21 },
    { 3, 4, 21 },
  })
  assert_dtype(numlua.pad(matrix, 1, "constant", 5), "int32")
  assert_tensor(numlua.pad(matrix, 1, "constant", { { 5 }, { 6 } }), {
    { 5, 5, 5, 5 },
    { 6, 1, 2, 6 },
    { 6, 3, 4, 6 },
    { 5, 5, 5, 5 },
  })

  -- Zero fill stays the default, including its dtype.
  local default = numlua.pad(matrix, 1)
  assert_dtype(default, "int32")
  assert_tensor(default, { { 0, 0, 0, 0 }, { 0, 1, 2, 0 }, { 0, 3, 4, 0 },
    { 0, 0, 0, 0 } })

  -- Per-axis values work for 3-D input and non-contiguous views.
  local cube = numlua.arange(8):reshape(2, 2, 2)
  local view = numlua.transpose(cube, { 3, 1, 2 })
  local padded = numlua.pad(view, { { 1, 0 }, { 0, 1 }, { 1, 1 } },
    "constant", { { 100, 101 }, { 102, 103 }, { 104, 105 } })
  assert_shape(padded, { 3, 3, 4 })
  assert_eq(padded:get(1, 1, 1), 100)
  assert_eq(padded:get(2, 1, 1), 104)
  assert_eq(padded:get(2, 1, 2), view:get(1, 1, 1))
  assert_eq(padded:get(3, 1, 4), 105)
  assert_eq(padded:get(2, 3, 1), 103)
  assert_eq(padded:get(2, 2, 4), 105)

  padded:set(1, 1, 1, -1)
  assert_eq(view:get(1, 1, 1), 0)
end)

test("pad validates modes, widths, and constant values", function()
  local vector = numlua.array({ 1, 2, 3 })
  local matrix = numlua.zeros(2, 2)

  assert_error(function() numlua.pad(vector, 1, "wrapx") end,
    "invalid mode")
  assert_error(function() numlua.pad(vector, -1) end,
    "non-negative integers")
  assert_error(function() numlua.pad(vector, 1.5) end,
    "non-negative integers")
  assert_error(function() numlua.pad(vector, {}) end,
    "non-empty table")
  assert_error(function() numlua.pad(vector, { { 1, 2 }, { 3, 4 } }) end,
    "one pair per dimension")
  assert_error(function() numlua.pad(3, 1) end, "expects a tensor")

  assert_error(function()
    numlua.pad(vector, 1, "constant", true)
  end, "constant value must")
  assert_error(function()
    numlua.pad(vector, 1, "constant", {})
  end, "constant value must")
  assert_error(function()
    numlua.pad(vector, 1, "constant", { 1, 2, 3 })
  end, "constant value must")
  assert_error(function()
    numlua.pad(vector, 1, "constant", { 1, true })
  end, "constant values must be numeric")
  assert_error(function()
    numlua.pad(matrix, 1, "constant", { { 1, 2 } })
  end, "one entry per dimension")
  assert_error(function()
    numlua.pad(matrix, 1, "constant", { { 1, true }, { 2, 3 } })
  end, "constant values must be numeric")
end)
