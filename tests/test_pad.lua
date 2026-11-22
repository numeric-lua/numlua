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

-- Flatten a nested Lua array into C-order storage plus its shape, and rebuild
-- it afterwards.  The reference pad below walks this form directly, so it does
-- not reuse any of the implementation's index or reduction helpers.
local function flatten_nested(values)
  local flat, shape = {}, {}
  local function walk(value, depth)
    if type(value) ~= "table" then
      flat[#flat + 1] = value
      return
    end
    shape[depth] = #value
    for index = 1, #value do walk(value[index], depth + 1) end
  end
  walk(values, 1)
  return flat, shape
end

local function rebuild_nested(flat, shape)
  local position = 0
  local function build(depth)
    if depth > #shape then
      position = position + 1
      return flat[position]
    end
    local branch = {}
    for index = 1, shape[depth] do branch[index] = build(depth + 1) end
    return branch
  end
  return build(1)
end

local function flat_index(shape, coords)
  local index = 1
  for dim = 1, #shape do
    index = (index - 1) * shape[dim] + coords[dim]
  end
  return index
end

local function coords_from_flat(shape, index)
  local coords = {}
  local remainder = index - 1
  for dim = #shape, 1, -1 do
    coords[dim] = remainder % shape[dim] + 1
    remainder = math.floor(remainder / shape[dim])
  end
  return coords
end

local function statistic(values, mode)
  if mode == "maximum" or mode == "minimum" then
    local result = values[1]
    for index = 2, #values do
      if mode == "maximum" and values[index] > result then
        result = values[index]
      elseif mode == "minimum" and values[index] < result then
        result = values[index]
      end
    end
    return result
  end
  if mode == "mean" then
    local total = 0
    for index = 1, #values do total = total + values[index] end
    return total / #values
  end
  -- median: sort a copy, then average the middle pair when the count is even.
  local sorted = {}
  for index = 1, #values do sorted[index] = values[index] end
  table.sort(sorted)
  local middle = math.floor((#sorted + 1) / 2)
  if #sorted % 2 == 1 then
    return sorted[middle]
  end
  return (sorted[middle] + sorted[middle + 1]) / 2
end

-- Reference implementation for the linear_ramp and statistic pad modes.
-- `option` is a per-axis table of {before, after} values: end values for
-- linear_ramp, stat_length pairs for the statistic modes.  Integer rounding
-- is intentionally left out; callers use float inputs for the walker and
-- separate literal expectations for integer dtype behavior.
local function reference_pad(values, pads, mode, option)
  local flat, shape = flatten_nested(values)
  for dim = 1, #pads do
    local before, after = pads[dim][1], pads[dim][2]
    if before + after > 0 then
      local old_shape, old_flat, size = shape, flat, shape[dim]
      local new_shape = {}
      for index = 1, #old_shape do
        new_shape[index] = old_shape[index] +
          (index == dim and before + after or 0)
      end
      local lengths = option and option[dim] or nil
      local left_length = lengths and math.min(size, lengths[1] or size) or size
      local right_length = lengths and math.min(size, lengths[2] or size) or size

      local function statistic_for(other_coords, side)
        local count = side == "left" and left_length or right_length
        local start = side == "left" and 1 or size - count + 1
        local sample = {}
        for offset = 0, count - 1 do
          local coords = {}
          for index = 1, #old_shape do coords[index] = other_coords[index] end
          coords[dim] = start + offset
          sample[#sample + 1] = old_flat[flat_index(old_shape, coords)]
        end
        return statistic(sample, mode)
      end

      local total = 1
      for index = 1, #new_shape do total = total * new_shape[index] end
      local new_flat = {}
      for position = 1, total do
        local coords = coords_from_flat(new_shape, position)
        local axis_position = coords[dim]
        local other = {}
        for index = 1, #old_shape do other[index] = coords[index] end
        if axis_position >= before + 1 and
            axis_position <= before + size then
          local source = {}
          for index = 1, #old_shape do source[index] = coords[index] end
          source[dim] = axis_position - before
          new_flat[position] = old_flat[flat_index(old_shape, source)]
        elseif mode == "linear_ramp" then
          local side, offset, width, edge_position
          if axis_position <= before then
            side, offset, width, edge_position = "left", axis_position, before, 1
          else
            side = "right"
            offset = axis_position - before - size
            width = after
            edge_position = size
          end
          local edge_coords = {}
          for index = 1, #old_shape do edge_coords[index] = coords[index] end
          edge_coords[dim] = edge_position
          local edge = old_flat[flat_index(old_shape, edge_coords)]
          local end_value = option[dim][side == "left" and 1 or 2]
          local weight = side == "left" and (offset - 1) / width or
            (width - offset) / width
          new_flat[position] = end_value + (edge - end_value) * weight
        else
          local side
          if axis_position <= before then
            side = "left"
          else
            side = "right"
          end
          new_flat[position] = statistic_for(other, side)
        end
      end
      flat, shape = new_flat, new_shape
    end
  end
  return rebuild_nested(flat, shape)
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

test("pad linear_ramp interpolates from per-axis end values", function()
  local vector = numlua.array({ 1, 2, 3, 4, 5 })
  -- NumPy's np.pad example: end_values=(5, -4) on a 1..5 vector.
  assert_tensor(numlua.pad(vector, { 2, 3 }, "linear_ramp", { 5, -4 }),
    { 5, 3, 1, 2, 3, 4, 5, 2, -1, -4 })

  for _, ends in ipairs({ { 0, 0 }, { 5, -4 }, { -2.5, 7.25 } }) do
    assert_tensor(numlua.pad(vector, { 2, 3 }, "linear_ramp", ends),
      reference_pad(vector:totable(), { { 2, 3 } }, "linear_ramp",
        { ends }))
  end

  -- The default end value is zero, matching end_values=0.
  assert_tensor(numlua.pad(vector, { 2, 3 }, "linear_ramp"),
    { 0, 0.5, 1, 2, 3, 4, 5, 10 / 3, 5 / 3, 0 })

  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local pads = { { 1, 2 }, { 2, 1 } }
  local ends = { { 10, 20 }, { 30, 40 } }
  assert_tensor(numlua.pad(matrix, pads, "linear_ramp", ends),
    reference_pad(matrix:totable(), pads, "linear_ramp", ends))

  local cube = numlua.arange(24):reshape(2, 3, 4):to("float32")
  local cube_pads = { { 2, 1 }, { 0, 3 }, { 1, 1 } }
  local cube_ends = { { -5, 15 }, { 0, 0 }, { 2, -2 } }
  assert_tensor(numlua.pad(cube, cube_pads, "linear_ramp", cube_ends),
    reference_pad(cube:totable(), cube_pads, "linear_ramp", cube_ends))

  -- Widths larger than the axis simply place the end value further out.
  local wide_pads = { { 7, 5 } }
  local wide_ends = { { 9, -9 } }
  assert_tensor(numlua.pad(vector, wide_pads, "linear_ramp", wide_ends),
    reference_pad(vector:totable(), wide_pads, "linear_ramp", wide_ends))

  -- Integer ramps cast toward zero after the end values are rounded.
  local ints = numlua.array({ 1, 2, 3 }, { dtype = "int32" })
  assert_tensor(numlua.pad(ints, { 2, 2 }, "linear_ramp", { 4, -4 }),
    { 4, 2, 1, 2, 3, 0, -4 })
  assert_dtype(numlua.pad(ints, { 2, 2 }, "linear_ramp", { 4, -4 }), "int32")

  local base = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = "float32" })
  local view = numlua.transpose(base)
  local padded = numlua.pad(view, { { 1, 2 }, { 0, 1 } }, "linear_ramp",
    { { 4, 5 }, { 6, 7 } })
  assert_dtype(padded, "float32")
  assert_tensor(padded, reference_pad(view:totable(), { { 1, 2 }, { 0, 1 } },
    "linear_ramp", { { 4, 5 }, { 6, 7 } }))
  padded:set(1, 1, 99)
  assert_eq(view:get(1, 1), 1)
end)

test("pad statistic modes fill each side from the axis", function()
  local vector = numlua.array({ 1, 2, 3, 4 })
  local modes = { "maximum", "minimum", "mean", "median" }
  for _, mode in ipairs(modes) do
    for _, widths in ipairs({ { 1, 1 }, { 2, 3 }, { 6, 5 } }) do
      local pads = { widths }
      assert_tensor(numlua.pad(vector, widths, mode),
        reference_pad(vector:totable(), pads, mode))
    end
    for _, lengths in ipairs({ 1, { 2, 1 }, { 99, 99 } }) do
      local option
      if type(lengths) == "number" then
        option = { { lengths, lengths } }
      else
        option = { lengths }
      end
      assert_tensor(numlua.pad(vector, { 3, 2 }, mode, lengths),
        reference_pad(vector:totable(), { { 3, 2 } }, mode, option))
    end
  end

  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local pads = { { 1, 2 }, { 2, 1 } }
  for _, mode in ipairs(modes) do
    assert_tensor(numlua.pad(matrix, pads, mode),
      reference_pad(matrix:totable(), pads, mode))
    local lengths = { { 1, 2 }, { 2, 1 } }
    assert_tensor(numlua.pad(matrix, pads, mode, lengths),
      reference_pad(matrix:totable(), pads, mode, lengths))
  end

  local cube = numlua.arange(24):reshape(2, 3, 4):to("float32")
  local cube_pads = { { 2, 1 }, { 1, 2 }, { 0, 2 } }
  for _, mode in ipairs(modes) do
    assert_tensor(numlua.pad(cube, cube_pads, mode),
      reference_pad(cube:totable(), cube_pads, mode))
  end

  -- Bool reductions go through a supported kernel and cast back.
  local mask = numlua.array({ true, false, true })
  assert_tensor(numlua.pad(mask, 1, "maximum"),
    { true, true, false, true, true })
  assert_tensor(numlua.pad(mask, 1, "minimum"),
    { false, true, false, true, false })
  assert_tensor(numlua.pad(mask, 1, "mean"),
    { true, true, false, true, true })
  assert_tensor(numlua.pad(mask, 1, "median"),
    { true, true, false, true, true })
  assert_dtype(numlua.pad(mask, 1, "mean"), "bool")

  -- Integer statistics use NumPy's round-half-to-even cast.
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int32" })
  local int_pads = { { 1, 1 }, { 1, 1 } }
  local rounded = {
    { 2, 2, 3, 2 },
    { 2, 1, 2, 2 },
    { 4, 3, 4, 4 },
    { 2, 2, 3, 2 },
  }
  for _, mode in ipairs({ "mean", "median" }) do
    local padded = numlua.pad(ints, int_pads, mode)
    assert_tensor(padded, rounded)
    assert_dtype(padded, "int32")
  end

  local base = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = "int64" })
  local view = numlua.transpose(base)
  local padded = numlua.pad(view, { { 1, 1 }, { 1, 1 } }, "maximum")
  assert_dtype(padded, "int64")
  assert_tensor(padded, reference_pad(view:totable(),
    { { 1, 1 }, { 1, 1 } }, "maximum"))
  padded:set(1, 1, -1)
  assert_eq(view:get(1, 1), 1)
end)

test("pad statistic corners include earlier-axis padding", function()
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  local pads = { { 1, 1 }, { 1, 1 } }
  assert_tensor(numlua.pad(matrix, pads, "maximum"), {
    { 4, 3, 4, 4 },
    { 2, 1, 2, 2 },
    { 4, 3, 4, 4 },
    { 4, 3, 4, 4 },
  })
  assert_tensor(numlua.pad(matrix, pads, "minimum"), {
    { 1, 1, 2, 1 },
    { 1, 1, 2, 1 },
    { 3, 3, 4, 3 },
    { 1, 1, 2, 1 },
  })
  assert_tensor(numlua.pad(matrix, pads, "mean"), {
    { 2.5, 2, 3, 2.5 },
    { 1.5, 1, 2, 1.5 },
    { 3.5, 3, 4, 3.5 },
    { 2.5, 2, 3, 2.5 },
  })
  assert_tensor(numlua.pad(matrix, pads, "linear_ramp",
    { { 10, 20 }, { 30, 40 } }), {
    { 30, 10, 10, 40 },
    { 30, 1, 2, 40 },
    { 30, 3, 4, 40 },
    { 30, 20, 20, 40 },
  })
end)

test("pad new modes validate end values and stat lengths", function()
  local vector = numlua.array({ 1, 2, 3 })
  local matrix = numlua.zeros(2, 2)

  assert_error(function() numlua.pad(vector, 1, "edge", 5) end,
    "does not accept a value")
  assert_error(function() numlua.pad(vector, 1, "wrap", { 1, 2 }) end,
    "does not accept a value")

  assert_error(function()
    numlua.pad(vector, 1, "linear_ramp", true)
  end, "end value must")
  assert_error(function()
    numlua.pad(vector, 1, "linear_ramp", {})
  end, "end value must")
  assert_error(function()
    numlua.pad(vector, 1, "linear_ramp", { 1, 2, 3 })
  end, "end value must")
  assert_error(function()
    numlua.pad(vector, 1, "linear_ramp", { 1, true })
  end, "end values must be numeric")
  assert_error(function()
    numlua.pad(matrix, 1, "linear_ramp", { { 1, 2 } })
  end, "one entry per dimension")

  for _, mode in ipairs({ "maximum", "minimum", "mean", "median" }) do
    assert_error(function() numlua.pad(vector, 1, mode, 0) end,
      "stat lengths must be positive integers")
    assert_error(function() numlua.pad(vector, 1, mode, 1.5) end,
      "stat lengths must be positive integers")
    assert_error(function() numlua.pad(vector, 1, mode, { 2, -1 }) end,
      "stat lengths must be positive integers")
    assert_error(function() numlua.pad(vector, 1, mode, { 1, 2, 3 }) end,
      "stat length must")
    assert_error(function() numlua.pad(matrix, 1, mode, { { 1, 2 } }) end,
      "one entry per dimension")
  end
end)
