--
-- NumPy rotation helpers: rot90, fliplr, flipud.
--

local numlua = require("numlua")

test("rot90 rotates a matrix in quarter turns", function()
  local m = numlua.array({
    { 1, 2 },
    { 3, 4 },
  })

  assert_tensor(numlua.rot90(m), { { 2, 4 }, { 1, 3 } })
  assert_tensor(numlua.rot90(m, 2), { { 4, 3 }, { 2, 1 } })
  assert_tensor(numlua.rot90(m, 3), { { 3, 1 }, { 4, 2 } })
  assert_tensor(numlua.rot90(m, 4), { { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.rot90(m, 0), { { 1, 2 }, { 3, 4 } })

  -- Negative quarter turns rotate clockwise and are reduced modulo four.
  assert_tensor(numlua.rot90(m, -1), { { 3, 1 }, { 4, 2 } })
  assert_tensor(numlua.rot90(m, -2), { { 4, 3 }, { 2, 1 } })
  assert_tensor(numlua.rot90(m, -3), { { 2, 4 }, { 1, 3 } })
  assert_tensor(numlua.rot90(m, 5), { { 2, 4 }, { 1, 3 } })

  -- Rectangular input changes shape on odd turns.
  local rectangular = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  })
  local odd = numlua.rot90(rectangular)
  assert_shape(odd, { 3, 2 })
  assert_tensor(odd, {
    { 3, 6 },
    { 2, 5 },
    { 1, 4 },
  })
  assert_shape(numlua.rot90(rectangular, 2), { 2, 3 })

  -- Inputs below two dimensions are rejected, including k = 0.
  assert_error(function() numlua.rot90(numlua.array({ 1, 2 })) end,
    "at least a 2-D tensor")
  assert_error(function() numlua.rot90(numlua.array(1), 0) end,
    "at least a 2-D tensor")
end)

test("rot90 accepts explicit axes and preserves dtypes", function()
  local cube = numlua.arange(1, 9):reshape(2, 2, 2)
  assert_tensor(numlua.rot90(cube, 1, { 1, 3 }):reshape(-1),
    { 2, 6, 4, 8, 1, 5, 3, 7 })
  assert_tensor(numlua.rot90(cube, 2, { 1, 3 }):reshape(-1),
    { 6, 5, 8, 7, 2, 1, 4, 3 })
  assert_tensor(numlua.rot90(cube, 3, { 1, 3 }):reshape(-1),
    { 5, 1, 7, 3, 6, 2, 8, 4 })

  -- Reversing the axes rotates the other way.
  local m = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.rot90(m, 1, { 2, 1 }), { { 3, 1 }, { 4, 2 } })
  assert_tensor(numlua.rot90(m, 1, { -1, -2 }), { { 3, 1 }, { 4, 2 } })

  local bools = numlua.array({
    { true, false },
    { false, true },
  })
  local rotated = numlua.rot90(bools)
  assert_dtype(rotated, "bool")
  assert_tensor(rotated, {
    { false, true },
    { true, false },
  })

  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_dtype(numlua.rot90(ints), "int64")
end)

test("rot90 allocates a result except for k = 0", function()
  local m = numlua.array({ { 1, 2 }, { 3, 4 } })
  local rotated = numlua.rot90(m)
  rotated:set(1, 1, 99)
  assert_eq(m:get(1, 2), 2)

  -- k = 0 is the input itself, matching NumPy's view for a no-op rotation.
  local same = numlua.rot90(m, 0)
  same:set(2, 2, 77)
  assert_eq(m:get(2, 2), 77)
end)

test("fliplr and flipud reverse one axis", function()
  local m = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  })
  assert_tensor(numlua.fliplr(m), {
    { 3, 2, 1 },
    { 6, 5, 4 },
  })
  assert_tensor(numlua.flipud(m), {
    { 4, 5, 6 },
    { 1, 2, 3 },
  })

  local vector = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.flipud(vector), { 3, 2, 1 })
  assert_error(function() numlua.fliplr(vector) end,
    "at least a 2-D tensor")
  assert_error(function() numlua.flipud(numlua.array(1)) end,
    "at least a 1-D tensor")

  -- Both helpers materialize independent results, like flip().
  local flipped = numlua.fliplr(m)
  flipped:set(1, 1, 42)
  assert_eq(m:get(1, 3), 3)
  local column = numlua.flipud(m)
  column:set(1, 1, 24)
  assert_eq(m:get(2, 1), 4)
end)

test("rotation helpers validate k, axes, and offsets", function()
  local m = numlua.zeros(2, 3)
  for _, invalid in ipairs({ 1.5, math.huge, -math.huge, 0 / 0, "1" }) do
    assert_error(function() numlua.rot90(m, invalid) end,
      "k must be a finite integer")
  end
  assert_error(function() numlua.rot90(m, 1, 1) end,
    "axes must be a table of two axes")
  assert_error(function() numlua.rot90(m, 1, { 1 }) end,
    "axes must be a table of two axes")
  assert_error(function() numlua.rot90(m, 1, { 1, 2, 3 }) end,
    "axes must be a table of two axes")
  assert_error(function() numlua.rot90(m, 1, { 1, 1 }) end,
    "axes must be different")
  assert_error(function() numlua.rot90(m, 1, { 0, 1 }) end,
    "axis must be a non-zero integer")
  assert_error(function() numlua.rot90(m, 1, { 1, 3 }) end,
    "out of range")
  assert_error(function() numlua.fliplr(numlua.zeros(3)) end,
    "expects at least a 2-D tensor")
end)
