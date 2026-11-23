--
-- NumPy diagonal extraction and trace: diagonal, trace.
--

local numlua = require("numlua")

test("diagonal extracts offsets from matrices", function()
  local m = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  })
  assert_tensor(numlua.diagonal(m), { 1, 5, 9 })
  assert_tensor(numlua.diagonal(m, 0), { 1, 5, 9 })
  assert_tensor(numlua.diagonal(m, 1), { 2, 6 })
  assert_tensor(numlua.diagonal(m, 2), { 3 })
  assert_tensor(numlua.diagonal(m, -1), { 4, 8 })
  assert_tensor(numlua.diagonal(m, -2), { 7 })

  local empty = numlua.diagonal(m, 4)
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)
  assert_shape(numlua.diagonal(m, -4), { 0 })

  local rectangular = numlua.array({
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
  })
  assert_tensor(numlua.diagonal(rectangular), { 1, 6 })
  assert_tensor(numlua.diagonal(rectangular, 1), { 2, 7 })
  assert_tensor(numlua.diagonal(rectangular, -1), { 5 })
  assert_tensor(numlua.diagonal(rectangular, 3), { 4 })

  -- Non-empty numeric diagonals are writable views, like NumPy's (read-only)
  -- diagonal view.
  local diagonal = numlua.diagonal(m)
  diagonal:set(1, 99)
  assert_eq(m:get(1, 1), 99)

  -- Bool diagonals are materialized through uint8.
  local bools = numlua.array({ { true, false }, { false, true } })
  local bool_diagonal = numlua.diagonal(bools)
  bool_diagonal:set(1, false)
  assert_true(bools:get(1, 1))
end)

test("diagonal accepts explicit axis pairs and N-D input", function()
  local cube = numlua.arange(2 * 3 * 4):reshape(2, 3, 4)
  local sample = numlua.diagonal(cube, 0, 1, 3)
  assert_shape(sample, { 3, 2 })
  assert_tensor(sample, {
    { 0, 13 },
    { 4, 17 },
    { 8, 21 },
  })

  local offset = numlua.diagonal(cube, 1, 1, 3)
  assert_shape(offset, { 3, 2 })
  assert_tensor(offset, {
    { 1, 14 },
    { 5, 18 },
    { 9, 22 },
  })

  local empty_batch = numlua.diagonal(numlua.zeros(2, 0, 3), 0, 1, 3)
  assert_shape(empty_batch, { 0, 2 })
  assert_eq(empty_batch:numel(), 0)

  local bools = numlua.array({
    { true, false, true },
    { false, true, false },
  })
  local diagonal = numlua.diagonal(bools, 0, 1, 2)
  assert_dtype(diagonal, "bool")
  assert_tensor(diagonal, { true, true })
  assert_tensor(numlua.diagonal(bools, 1), { false, false })
end)

test("trace sums the selected diagonal into a 0-D tensor", function()
  local m = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  })
  local total = numlua.trace(m)
  assert_shape(total, {})
  assert_dtype(total, "float32")
  assert_eq(total:item(), 15)
  assert_eq(numlua.trace(m, 1):item(), 8)
  assert_eq(numlua.trace(m, -1):item(), 12)
  assert_eq(numlua.trace(m, 3):item(), 0)

  local cube = numlua.arange(2 * 3 * 4):reshape(2, 3, 4)
  local sample = numlua.trace(cube, 0, 1, 3)
  assert_shape(sample, { 3 })
  assert_tensor(sample, { 13, 21, 29 })

  local bools = numlua.array({
    { true, false },
    { false, true },
  })
  local bool_total = numlua.trace(bools)
  assert_shape(bool_total, {})
  assert_dtype(bool_total, "int64")
  assert_eq(bool_total:item(), 2)

  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_eq(numlua.trace(ints):item(), 5)
end)

test("diagonal and trace reject invalid ranks, axes, and offsets", function()
  local vector = numlua.array({ 1, 2, 3 })
  assert_error(function() numlua.diagonal(vector) end,
    "at least a 2-D tensor")
  assert_error(function() numlua.trace(vector) end, "at least a 2-D tensor")
  assert_error(function() numlua.diagonal(nil) end, "expects a tensor")
  assert_error(function() numlua.trace(nil) end, "expects a tensor")

  local m = numlua.zeros(2, 3)
  assert_error(function() numlua.diagonal(m, 0, 0, 1) end,
    "axis must be a non-zero integer")
  assert_error(function() numlua.diagonal(m, 0, 1, 3) end, "out of range")
  assert_error(function() numlua.diagonal(m, 0, 1, 1) end,
    "axes must be different")
  assert_error(function() numlua.trace(m, 0, 2, 2) end,
    "axes must be different")
  for _, invalid in ipairs({ 1.5, math.huge, -math.huge, 0 / 0 }) do
    assert_error(function() numlua.diagonal(m, invalid) end,
      "offset must be a finite integer")
    assert_error(function() numlua.trace(m, invalid) end,
      "offset must be a finite integer")
  end
end)
