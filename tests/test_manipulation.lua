local numlua = require("numlua")

test("broadcast_to and broadcast_arrays share a shape", function()
  local row = numlua.array({ { 1, 2, 3 } })
  local wide = numlua.broadcast_to(row, { 2, 3 })
  assert_shape(wide, { 2, 3 })
  assert_tensor(wide, { { 1, 2, 3 }, { 1, 2, 3 } })

  local column = numlua.array({ { 10 }, { 20 } })
  local a, b = numlua.broadcast_arrays(row, column)
  assert_shape(a, { 2, 3 })
  assert_shape(b, { 2, 3 })
  assert_error(function()
    numlua.broadcast_arrays(numlua.zeros(2), numlua.zeros(3))
  end, "incompatible")
end)

test("broadcast handles 0-D, zero-size, and high-rank shapes", function()
  -- A 0-D scalar expands to any shape, including an empty one.
  local scalar = numlua.array(7)
  local wide = numlua.broadcast_to(scalar, { 2, 3 })
  assert_shape(wide, { 2, 3 })
  assert_tensor(wide, { { 7, 7, 7 }, { 7, 7, 7 } })
  local empty_scalar = numlua.broadcast_to(scalar, { 0, 2 })
  assert_shape(empty_scalar, { 0, 2 })

  -- Size-1 axes grow, and a zero-size axis wins over size 1 (NumPy rules).
  local empty = numlua.zeros(0, 1)
  local single = numlua.zeros(1, 4)
  local a, b = numlua.broadcast_arrays(empty, single)
  assert_shape(a, { 0, 4 })
  assert_shape(b, { 0, 4 })
  assert_eq(a:numel(), 0)

  -- Right alignment across ranks: {2,1,4,1} against {3,1,5}.
  local high, low = numlua.broadcast_arrays(
    numlua.zeros(2, 1, 4, 1), numlua.zeros(3, 1, 5))
  assert_shape(high, { 2, 3, 4, 5 })
  assert_shape(low, { 2, 3, 4, 5 })

  -- A single input broadcasts to its own shape.
  local only = numlua.broadcast_arrays(numlua.zeros(2, 3))
  assert_shape(only, { 2, 3 })
end)

test("broadcast rejects incompatible and malformed shapes", function()
  assert_error(function()
    numlua.broadcast_arrays(numlua.zeros(0), numlua.zeros(2))
  end, "incompatible")
  assert_error(function() numlua.broadcast_arrays() end,
    "at least one tensor")
  assert_error(function()
    numlua.broadcast_arrays(numlua.zeros(2), 3)
  end, "expects a tensor")
  assert_error(function() numlua.broadcast_to(3, { 2 }) end,
    "expects a tensor")
  assert_error(function() numlua.broadcast_to(numlua.zeros(2), {}) end,
    "non-empty shape")
  assert_error(function() numlua.broadcast_to(numlua.zeros(2), { -1 }) end,
    "non-negative")
  assert_error(function() numlua.broadcast_to(numlua.zeros(2), 2) end,
    "shape table")
  assert_error(function()
    numlua.broadcast_to(numlua.zeros(2, 3), { 3, 2 })
  end)
  assert_error(function()
    numlua.broadcast_to(numlua.zeros(2, 3), { 3 })
  end)
end)

test("tile repeats whole blocks", function()
  local vector = numlua.array({ 1, 2 })
  assert_tensor(numlua.tile(vector, 3), { 1, 2, 1, 2, 1, 2 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.tile(matrix, { 2, 1 }), {
    { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 },
  })
end)

test("repeat repeats elements along an axis", function()
  local vector = numlua.array({ 1, 2 })
  assert_tensor(numlua.repeat_(vector, 2), { 1, 1, 2, 2 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.repeat_(matrix, 2, 1), {
    { 1, 2 }, { 1, 2 }, { 3, 4 }, { 3, 4 },
  })
  assert_tensor(numlua.repeat_(matrix, { 1, 2 }, 2), {
    { 1, 2, 2 }, { 3, 4, 4 },
  })
end)

test("flip and roll preserve shapes", function()
  local t = numlua.array({ 1, 2, 3, 4 })
  assert_tensor(numlua.flip(t, 1), { 4, 3, 2, 1 })
  assert_tensor(numlua.roll(t, 1, 1), { 4, 1, 2, 3 })
  assert_tensor(numlua.roll(t, -1), { 2, 3, 4, 1 })

  local m = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.flip(m, { 1, 2 }), { { 4, 3 }, { 2, 1 } })
end)

test("pad adds constant borders", function()
  local vector = numlua.array({ 1, 2 })
  assert_tensor(numlua.pad(vector, { 1, 2 }, "constant", 9),
    { 9, 1, 2, 9, 9 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  local padded = numlua.pad(matrix, { { 1, 0 }, { 0, 1 } }, "constant", 0)
  assert_shape(padded, { 3, 3 })
  assert_near(padded:sum():item(), 10, 1e-6)
end)

test("diff, cumsum, and cumprod follow axis semantics", function()
  local vector = numlua.array({ 1, 4, 9, 16 })
  assert_tensor(numlua.diff(vector), { 3, 5, 7 })
  assert_tensor(numlua.diff(vector, 2), { 2, 2 })
  assert_tensor(numlua.cumsum(vector), { 1, 5, 14, 30 })
  assert_tensor(numlua.cumprod(vector), { 1, 4, 36, 576 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.cumsum(matrix, 1), { { 1, 2 }, { 4, 6 } })
  assert_tensor(numlua.diff(matrix, 1, 2), { { 1 }, { 1 } })
end)

test("argsort and take use 1-based indices", function()
  local t = numlua.array({ 30, 10, 20 })
  local order = numlua.argsort(t)
  assert_tensor(order, { 2, 3, 1 })
  assert_tensor(numlua.take(t, order), { 10, 20, 30 })

  local matrix = numlua.array({ { 3, 1 }, { 2, 4 } })
  local indices = numlua.argsort(matrix, 2)
  assert_tensor(indices, { { 2, 1 }, { 1, 2 } })
  assert_error(function() numlua.take(t, { 1, 2 }) end, "indices must")
end)

test("repeat with a table and no axis flattens", function()
  local t = numlua.array({{1, 2}, {3, 4}})
  local r = numlua.repeat_(t, { 1, 2, 1, 2 })
  assert_shape(r, { 6 })
  assert_tensor(r, { 1, 2, 2, 3, 4, 4 })
end)

test("compress selects elements with a boolean mask", function()
  local t = numlua.array({ 10, 20, 30, 40 })
  local mask = numlua.array({ true, false, true, false }, { dtype = "bool" })
  local selected = numlua.compress(mask, t)
  assert_tensor(selected, { 10, 30 })
  assert_error(function() numlua.compress(mask, t, 1) end, "axis")
end)

test("take selects flat and axis-aligned values", function()
  local t = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  assert_tensor(numlua.take(t, numlua.array({ 1, 3, 5 }, { dtype = "int64" })), { 10, 30, 50 })
  local row = numlua.take(t, numlua.array({ 3, 1 }, { dtype = "int64" }), 2)
  assert_shape(row, { 2, 2 })
  assert_tensor(row, { { 30, 10 }, { 60, 40 } })
end)

test("take handles 0-D, empty, and zero-size-axis shapes", function()
  local scalar = numlua.array(7)
  local one = numlua.take(scalar, numlua.array({ 1 }, { dtype = "int64" }))
  assert_shape(one, { 1 })
  assert_tensor(one, { 7 })
  assert_eq(numlua.take(scalar,
    numlua.array({}, { dtype = "int64" })):numel(), 0)

  local empty = numlua.zeros(0)
  assert_eq(numlua.take(empty,
    numlua.array({}, { dtype = "int64" })):numel(), 0)
  assert_error(function()
    numlua.take(empty, numlua.array({ 1 }, { dtype = "int64" }))
  end, "empty tensor")

  local t = numlua.arange(24):reshape(2, 3, 4)
  local flat = numlua.take(t, numlua.array({ 4, 1, 4 }, { dtype = "int64" }))
  assert_shape(flat, { 3 })
  assert_tensor(flat, { 3, 0, 3 })
  local middle = numlua.take(t,
    numlua.array({ 2, 1 }, { dtype = "int32" }), 2)
  assert_shape(middle, { 2, 2, 4 })
  local negated = numlua.take(t,
    numlua.array({ 2, 1 }, { dtype = "int32" }), -2)
  assert_shape(negated, { 2, 2, 4 })
  assert_true(negated:equal(middle))

  local zero = numlua.zeros(0, 3)
  assert_shape(numlua.take(zero,
    numlua.array({ 1, 3 }, { dtype = "int64" }), 2), { 0, 2 })
  assert_shape(numlua.take(zero,
    numlua.array({}, { dtype = "int64" }), 1), { 0, 3 })
  assert_error(function()
    numlua.take(zero, numlua.array({ 1 }, { dtype = "int64" }), 1)
  end, "empty axis")

  -- A size-1 selection axis is a normal shape, repeated indices are allowed,
  -- and a non-contiguous input is flattened in C order before selection.
  assert_shape(numlua.take(numlua.zeros(2, 1, 3),
    numlua.array({ 1, 1 }, { dtype = "int64" }), 2), { 2, 2, 3 })
  assert_tensor(numlua.take(numlua.array({ 10, 20 }),
    numlua.array({ 2, 2, 1 }, { dtype = "int64" })), { 20, 20, 10 })
  local transposed = numlua.transpose(numlua.array({
    { 10, 20, 30 },
    { 40, 50, 60 },
  }))
  assert_tensor(numlua.take(transposed,
    numlua.array({ 2, 4 }, { dtype = "int64" })), { 40, 50 })
end)

test("take validates index tensors and axes", function()
  local t = numlua.array({ 10, 20, 30 })
  assert_error(function() numlua.take(t, numlua.array({ 1.5 })) end,
    "integer Tensor")
  assert_error(function() numlua.take(t, numlua.array({ true })) end,
    "integer Tensor")
  assert_error(function()
    numlua.take(t, numlua.array({ 1 }, { dtype = "int64" }), 0)
  end, "non-zero integer")
  assert_error(function()
    numlua.take(t, numlua.array({ 1 }, { dtype = "int64" }), 2)
  end, "out of range")
  assert_error(function()
    numlua.take(t, numlua.array({ 4 }, { dtype = "int64" }))
  end, "out of range")

  -- Every integer dtype is accepted and treated as 1-based.
  for _, dtype in ipairs({ "int64", "int32", "int16", "int8", "uint8" }) do
    assert_tensor(numlua.take(t,
      numlua.array({ 2 }, { dtype = dtype })), { 20 })
  end
end)

test("take accepts scalar and N-D index tensors", function()
  local t = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })

  -- A 0-D index behaves like NumPy's scalar index: the selected axis (or the
  -- flattened input) disappears from the result shape.
  local flat_scalar = numlua.take(t, numlua.array(2, { dtype = "int64" }))
  assert_eq(flat_scalar:dim(), 0)
  assert_eq(flat_scalar:item(), 20)
  local axis_scalar = numlua.take(t, numlua.array(1, { dtype = "int32" }), 2)
  assert_shape(axis_scalar, { 2 })
  assert_tensor(axis_scalar, { 10, 40 })

  -- An N-D index tensor contributes its own shape in place of the axis.
  local picked = numlua.take(t,
    numlua.array({ { 3, 1 }, { 1, 3 } }, { dtype = "int64" }), 2)
  assert_shape(picked, { 2, 2, 2 })
  assert_tensor(picked, {
    { { 30, 10 }, { 10, 30 } },
    { { 60, 40 }, { 40, 60 } },
  })
  local flat_picked = numlua.take(t,
    numlua.array({ { 5, 1 }, { 6, 2 } }, { dtype = "uint8" }))
  assert_shape(flat_picked, { 2, 2 })
  assert_tensor(flat_picked, { { 50, 10 }, { 60, 20 } })

  -- Empty N-D indices keep their leading dimensions.
  local empty_indices = numlua.zeros(0, 2):to("int64")
  assert_shape(numlua.take(t, empty_indices), { 0, 2 })
  assert_shape(numlua.take(t, empty_indices, 2), { 2, 0, 2 })

  -- 0-D inputs still flatten correctly with scalar and N-D indices.
  local scalar = numlua.array(7)
  assert_eq(numlua.take(scalar,
    numlua.array(1, { dtype = "int64" })):dim(), 0)
  assert_shape(numlua.take(scalar,
    numlua.array({ { 1, 1 } }, { dtype = "int64" })), { 1, 2 })
end)

test("take reports 1-based index range errors explicitly", function()
  local t = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  local function indices(...)
    return numlua.array(...)
  end

  assert_error(function()
    numlua.take(t, indices(0, { dtype = "int64" }))
  end, "index 0 is out of range for the flattened input with size 6")
  assert_error(function()
    numlua.take(t, indices(7, { dtype = "int64" }))
  end, "index 7 is out of range for the flattened input with size 6")
  assert_error(function()
    numlua.take(t, indices(-1, { dtype = "int64" }))
  end, "index -1 is out of range for the flattened input with size 6")
  assert_error(function()
    numlua.take(t, indices({ { 1, 4 } }, { dtype = "int32" }), 2)
  end, "index 4 is out of range for axis 2 with size 3")
  assert_true(tostring(select(2, pcall(numlua.take, t,
    indices(0, { dtype = "int64" })))):find("indices are 1-based", 1, true)
    ~= nil)
end)

test("take_along_axis selects one value per position along an axis", function()
  local t = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
  local order = numlua.argsort(t, 2)
  assert_tensor(order, { { 1, 3, 2 }, { 2, 3, 1 } })
  local sorted = numlua.take_along_axis(t, order, 2)
  assert_tensor(sorted, { { 10, 20, 30 }, { 40, 50, 60 } })
  assert_dtype(sorted, "float32")

  -- argmax(..., keepdims=true) produces the gather shape used by NumPy's
  -- "maximum values" recipe.
  local maxima = numlua.take_along_axis(t, numlua.argmax(t, 2, true), 2)
  assert_shape(maxima, { 2, 1 })
  assert_tensor(maxima, { { 30 }, { 60 } })

  -- Negative axes count from the end, and the axis length comes from indices.
  assert_tensor(numlua.take_along_axis(t,
    numlua.array({ { 3, 1 }, { 2, 1 } }, { dtype = "int32" }), -1),
    { { 20, 10 }, { 40, 60 } })
  local wide = numlua.take_along_axis(t,
    numlua.array({ { 1, 2, 3, 1, 2 }, { 3, 2, 1, 3, 2 } },
      { dtype = "int64" }), 2)
  assert_shape(wide, { 2, 5 })
  assert_tensor(wide, { { 10, 30, 20, 10, 30 }, { 50, 40, 60, 50, 40 } })
  assert_tensor(numlua.take_along_axis(t,
    numlua.array({ { 2 }, { 3 } }, { dtype = "int64" }), 2),
    { { 30 }, { 50 } })

  -- Non-axis size-1 dimensions broadcast in either operand.
  local row = numlua.array({ { 10, 20, 30 } })
  assert_tensor(numlua.take_along_axis(row,
    numlua.array({ { 3, 1 }, { 2, 1 } }, { dtype = "int64" }), 2),
    { { 30, 10 }, { 20, 10 } })
  local column = numlua.array({ { 10, 20 }, { 30, 40 }, { 50, 60 } })
  assert_shape(numlua.take_along_axis(column,
    numlua.array({ { 1, 2 } }, { dtype = "int64" }), 1), { 1, 2 })
  assert_tensor(numlua.take_along_axis(column,
    numlua.array({ { 1, 2 }, { 3, 1 }, { 2, 3 } }, { dtype = "int64" }), 1),
    { { 10, 40 }, { 50, 20 }, { 30, 60 } })

  -- axis=nil flattens in C order, as in np.take_along_axis.
  assert_tensor(numlua.take_along_axis(t,
    numlua.array({ 6, 1, 4 }, { dtype = "int64" })), { 50, 10, 60 })
  assert_tensor(numlua.take_along_axis(numlua.array(7),
    numlua.array({ 1, 1 }, { dtype = "int64" })), { 7, 7 })

  -- 3-D gather, every integer index dtype, empty selections, and a
  -- non-contiguous input.
  local cube = numlua.arange(24):reshape(2, 3, 4)
  local indices = numlua.array({ { { 1, 2 }, { 1, 2 }, { 1, 2 } },
    { { 3, 4 }, { 3, 4 }, { 3, 4 } } }, { dtype = "int64" })
  local picked = numlua.take_along_axis(cube, indices, 3)
  assert_shape(picked, { 2, 3, 2 })
  assert_tensor(picked, { { { 0, 1 }, { 4, 5 }, { 8, 9 } },
    { { 14, 15 }, { 18, 19 }, { 22, 23 } } })
  for _, dtype in ipairs({ "int64", "int32", "int16", "int8", "uint8" }) do
    assert_tensor(numlua.take_along_axis(t,
      numlua.array({ { 1 } }, { dtype = dtype }), 2), { { 10 }, { 60 } })
  end
  local empty = numlua.take_along_axis(t,
    numlua.zeros(2, 0, { dtype = "int64" }), 2)
  assert_shape(empty, { 2, 0 })
  assert_eq(empty:numel(), 0)
  assert_shape(numlua.take_along_axis(numlua.zeros(2, 0),
    numlua.zeros(2, 0, { dtype = "int64" }), 2), { 2, 0 })
  local transposed = numlua.transpose(t)
  assert_tensor(numlua.take_along_axis(transposed,
    numlua.array({ { 1, 2 }, { 1, 2 }, { 1, 2 } }, { dtype = "int64" }), 2),
    { { 10, 60 }, { 30, 40 }, { 20, 50 } })

  -- The result owns independent storage.
  local result = numlua.take_along_axis(t,
    numlua.array({ { 1, 3 } }, { dtype = "int64" }), 2)
  result:set(1, 1, 999)
  assert_eq(t:get(1, 1), 10)
end)

test("take_along_axis validates rank, axes, shapes, and indices", function()
  local t = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
  local one = numlua.array({ { 1 } }, { dtype = "int64" })
  assert_error(function() numlua.take_along_axis({}, one, 2) end,
    "expects a tensor")
  assert_error(function() numlua.take_along_axis(t, { 1 }) end,
    "integer Tensor")
  assert_error(function()
    numlua.take_along_axis(t, numlua.array({ { 1.5 } }))
  end, "integer Tensor")
  assert_error(function()
    numlua.take_along_axis(t, numlua.array({ { true } }))
  end, "integer Tensor")
  assert_error(function() numlua.take_along_axis(t, one, 0) end,
    "non-zero integer")
  assert_error(function() numlua.take_along_axis(t, one, 3) end,
    "out of range")
  assert_error(function()
    numlua.take_along_axis(t,
      numlua.array({ 1 }, { dtype = "int64" }), 2)
  end, "same rank")
  assert_error(function()
    numlua.take_along_axis(t,
      numlua.array({ { 1 }, { 1 }, { 1 } }, { dtype = "int64" }), 2)
  end, "broadcast")
  assert_error(function()
    numlua.take_along_axis(t,
      numlua.array({ { 4 } }, { dtype = "int64" }), 2)
  end, "out of range")
  assert_error(function()
    numlua.take_along_axis(t,
      numlua.array({ { 0 } }, { dtype = "int64" }), 2)
  end, "out of range")
  assert_error(function()
    numlua.take_along_axis(t,
      numlua.array({ { -1 } }, { dtype = "int64" }), 2)
  end, "out of range")
  assert_error(function() numlua.take_along_axis(t, one) end, "must be 1-D")
  assert_error(function()
    numlua.take_along_axis(t,
      numlua.array({ { 1 }, { 1 } }, { dtype = "int64" }), nil)
  end, "must be 1-D")
  assert_error(function()
    numlua.take_along_axis(numlua.zeros(2, 0),
      numlua.array({ { 1 }, { 1 } }, { dtype = "int64" }), 2)
  end, "out of range")
end)

test("flatnonzero returns 1-based flat indices", function()
  local t = numlua.array({ 0, 5, 0, 7 })
  assert_tensor(numlua.flatnonzero(t), { 2, 4 })
end)

test("put writes flat positions with cycling values", function()
  local t = numlua.arange(6)
  local returned = numlua.put(t, { 1, 3, 5 }, { 10, 30 })
  assert_true(returned == t)
  assert_tensor(t, { 10, 1, 30, 3, 10, 5 })

  -- Duplicate indices apply in order, so the last write wins.
  numlua.put(t, { 2, 2, 2 }, { 7, 8, 9 })
  assert_eq(t:get(2), 9)

  -- Tensor indices and values behave like flat Lua tables.
  local u = numlua.zeros(4)
  numlua.put(u, numlua.array({ 2, 4 }, { dtype = "int32" }),
    numlua.array({ 5, 6 }))
  assert_tensor(u, { 0, 5, 0, 6 })

  -- Every integer dtype accepted by take is accepted here too.
  for _, dtype in ipairs({ "int64", "int32", "int16", "int8", "uint8" }) do
    local v = numlua.zeros(2)
    numlua.put(v, numlua.array({ 1 }, { dtype = dtype }), 3)
    assert_tensor(v, { 3, 0 })
  end
end)

test("put modes raise, wrap, and clip out-of-range indices", function()
  local t = numlua.arange(5)
  assert_error(function() numlua.put(t, { 6 }, { 9 }) end, "out of range")
  assert_error(function() numlua.put(t, { 0 }, { 9 }) end, "out of range")
  assert_error(function() numlua.put(t, { -1 }, { 9 }) end, "out of range")

  -- 1-based wrap: 6 -> 1, 0 -> 5, -1 -> 4.
  numlua.put(t, { 6, 0, -1 }, { 60, 0, -1 }, "wrap")
  assert_eq(t:get(1), 60)
  assert_eq(t:get(5), 0)
  assert_eq(t:get(4), -1)

  local clipped = numlua.arange(5)
  numlua.put(clipped, { 0, 6, -3 }, { 10, 20, 30 }, "clip")
  assert_eq(clipped:get(1), 30) -- -3 clips to 1, and the last write wins
  assert_eq(clipped:get(5), 20)

  assert_error(function() numlua.put(t, { 1 }, { 9 }, "reflect") end,
    "mode must")
end)

test("put handles 0-D, non-contiguous, bool, and integer targets", function()
  local scalar = numlua.array(1.5)
  numlua.put(scalar, 1, 2.5)
  assert_eq(scalar:item(), 2.5)

  local base = numlua.arange(6):reshape(2, 3)
  local view = numlua.transpose(base)
  assert_false(view:is_contiguous())
  numlua.put(view, { 2, 5 }, { 20, 50 })
  assert_tensor(view, { { 0, 20 }, { 1, 4 }, { 50, 5 } })
  assert_tensor(base, { { 0, 1, 50 }, { 20, 4, 5 } })

  local mask = numlua.array({ false, false, false })
  numlua.put(mask, { 1, 2 }, { 0, 2 })
  assert_tensor(mask, { false, true, false })
  numlua.put(mask, { 3 }, 1)
  assert_tensor(mask, { false, true, true })

  -- Integer targets truncate fractional values toward zero.
  local ints = numlua.array({ 1, 2 }, { dtype = "int64" })
  numlua.put(ints, { 1, 2 }, { -2.9, 2.9 })
  assert_tensor(ints, { -2, 2 })

  -- A bool target truth-tests integer values from an integer tensor.
  local flags = numlua.array({ 0, 1 }, { dtype = "int64" })
  numlua.put(mask, { 1, 3 }, flags)
  assert_tensor(mask, { false, true, true })
end)

test("put validates indices, values, modes, and empty inputs", function()
  local t = numlua.zeros(3)
  assert_error(function() numlua.put(t, { 1.5 }, { 1 }) end, "integers")
  assert_error(function() numlua.put(t, numlua.array({ 1.5 }), { 1 }) end,
    "integer Tensor")
  assert_error(function() numlua.put(t, numlua.array({ true }), { 1 }) end,
    "integer Tensor")
  assert_error(function() numlua.put(t, "1", { 1 }) end, "integer Tensor")
  assert_error(function() numlua.put(t, { 1 }, "x") end, "values must be")
  assert_error(function() numlua.put(t, { 1 }, {}) end, "must not be empty")
  assert_error(function() numlua.put(numlua.zeros(0), { 1 }, { 1 }) end,
    "empty tensor")
  assert_error(function() numlua.put(t, { 1 }, { 1 }, "bad") end, "mode must")

  -- Empty selections are a no-op, including empty index tensors.
  assert_true(numlua.put(t, {}, {}) == t)
  assert_true(numlua.put(t, numlua.array({}, { dtype = "int64" }), {}) == t)
  assert_tensor(t, { 0, 0, 0 })

  -- Empty targets still reject non-empty indices in every mode.
  assert_error(function()
    numlua.put(numlua.zeros(0), { 1 }, { 1 }, "wrap")
  end, "empty tensor")
end)

test("put flattens 3-D inputs and nested values in C order", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  numlua.put(cube, { 2, 24 }, { -1, -24 })
  assert_eq(cube:get(1, 1, 2), -1)
  assert_eq(cube:get(2, 3, 4), -24)

  -- Nested value tables are flattened in C order, like np.put.
  numlua.put(cube, { 3, 4 }, { { 30 }, { 40 } })
  assert_eq(cube:get(1, 1, 3), 30)
  assert_eq(cube:get(1, 1, 4), 40)

  -- Values longer than the index list are ignored, matching np.put.
  numlua.put(cube, { 1 }, { 100, 200, 300 })
  assert_eq(cube:get(1, 1, 1), 100)
end)

test("put wrap and clip accept extreme indices", function()
  local t = numlua.zeros(3)
  numlua.put(t, { -1000000, 1000000 }, { 1, 2 }, "wrap")
  assert_tensor(t, { 2, 1, 0 })
  numlua.put(t, { -100, 100 }, { 10, 30 }, "clip")
  assert_tensor(t, { 10, 1, 30 })
end)

test("put accepts 0-D and empty N-D index tensors", function()
  local t = numlua.arange(6)
  numlua.put(t, numlua.array(2, { dtype = "int64" }), 20)
  assert_eq(t:get(2), 20)

  numlua.put(t, numlua.array({ { 4, 6 } }, { dtype = "int32" }),
    { 40, 60 })
  assert_eq(t:get(4), 40)
  assert_eq(t:get(6), 60)

  -- Empty index tensors of any rank are a no-op, with empty values.
  local empty_indices = numlua.zeros(0, 2):to("int64")
  assert_true(numlua.put(t, empty_indices, {}) == t)
  assert_true(numlua.put(t, empty_indices, numlua.zeros(0)) == t)
  assert_tensor(t, { 0, 20, 2, 40, 4, 60 })
end)

test("put rejects non-finite indices in every mode", function()
  local t = numlua.arange(6)
  for _, mode in ipairs({ "raise", "wrap", "clip" }) do
    assert_error(function() numlua.put(t, { math.huge }, 9, mode) end,
      "finite integers")
    assert_error(function() numlua.put(t, { -math.huge }, 9, mode) end,
      "finite integers")
    assert_error(function() numlua.put(t, { 0 / 0 }, 9, mode) end,
      "finite integers")
    assert_error(function() numlua.put(t, math.huge, 9, mode) end,
      "finite integers")
    assert_error(function() numlua.put(t, -math.huge, 9, mode) end,
      "finite integers")
  end
  -- Rejected indices must not partially mutate the tensor.
  assert_tensor(t, { 0, 1, 2, 3, 4, 5 })
end)

