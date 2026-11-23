--
-- NumPy split family: split, array_split, hsplit, vsplit, dsplit.
--

local numlua = require("numlua")

test("split divides an axis into equal sections", function()
  local vector = numlua.array({ 1, 2, 3, 4, 5, 6 })
  local parts = numlua.split(vector, 3)
  assert_eq(#parts, 3)
  assert_tensor(parts[1], { 1, 2 })
  assert_tensor(parts[2], { 3, 4 })
  assert_tensor(parts[3], { 5, 6 })
  for i = 1, #parts do assert_shape(parts[i], { 2 }) end

  local matrix = numlua.array({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 } })
  local columns = numlua.split(matrix, 2, 2)
  assert_eq(#columns, 2)
  assert_tensor(columns[1], { { 1, 2 }, { 5, 6 } })
  assert_tensor(columns[2], { { 3, 4 }, { 7, 8 } })

  -- The default axis is the first one, matching np.split.
  local rows = numlua.split(matrix, 1)
  assert_eq(#rows, 1)
  assert_true(rows[1]:equal(matrix))

  -- Negative axes are normalized like every other facade axis.
  assert_true(numlua.split(matrix, 2, -1)[2]:equal(columns[2]))
end)

test("array_split distributes the remainder to leading chunks", function()
  local vector = numlua.array({ 1, 2, 3, 4, 5, 6, 7 })
  local parts = numlua.array_split(vector, 3)
  assert_eq(#parts, 3)
  assert_tensor(parts[1], { 1, 2, 3 })
  assert_tensor(parts[2], { 4, 5 })
  assert_tensor(parts[3], { 6, 7 })

  local exact = numlua.array_split(vector, 7)
  assert_eq(#exact, 7)
  assert_tensor(exact[7], { 7 })
end)

test("split tables use 1-based boundaries", function()
  local vector = numlua.array({ 10, 20, 30, 40, 50 })
  local parts = numlua.split(vector, { 2, 4 })
  assert_eq(#parts, 3)
  assert_tensor(parts[1], { 10 })
  assert_tensor(parts[2], { 20, 30 })
  assert_tensor(parts[3], { 40, 50 })

  -- An empty boundary table returns the whole input as one chunk.
  local whole = numlua.split(vector, {})
  assert_eq(#whole, 1)
  assert_true(whole[1]:equal(vector))

  -- Negative boundaries count from the end; -1 splits before the last item.
  local tail = numlua.split(vector, { -1 })
  assert_eq(#tail, 2)
  assert_tensor(tail[1], { 10, 20, 30, 40 })
  assert_tensor(tail[2], { 50 })

  -- Out-of-range boundaries are clamped, and duplicate/reversed boundaries
  -- produce empty chunks instead of errors, following NumPy.
  local clamped = numlua.split(vector, { 0, 99 })
  assert_eq(#clamped, 3)
  assert_eq(clamped[1]:numel(), 0)
  assert_tensor(clamped[2], { 10, 20, 30, 40, 50 })
  assert_eq(clamped[3]:numel(), 0)

  local reversed = numlua.split(vector, { 4, 2 })
  assert_eq(#reversed, 3)
  assert_tensor(reversed[1], { 10, 20, 30 })
  assert_eq(reversed[2]:numel(), 0)
  assert_tensor(reversed[3], { 20, 30, 40, 50 })
end)

test("split handles empty axes and empty chunks", function()
  local empty = numlua.zeros(2, 0)
  local parts = numlua.split(empty, 2, 2)
  assert_eq(#parts, 2)
  assert_shape(parts[1], { 2, 0 })
  assert_shape(parts[2], { 2, 0 })

  local zero = numlua.zeros(0)
  local zero_parts = numlua.array_split(zero, 3)
  assert_eq(#zero_parts, 3)
  for i = 1, 3 do assert_shape(zero_parts[i], { 0 }) end
end)

test("hsplit, vsplit, and dsplit use NumPy's axes", function()
  local vector = numlua.array({ 1, 2, 3, 4 })
  local halves = numlua.hsplit(vector, 2)
  assert_eq(#halves, 2)
  assert_tensor(halves[1], { 1, 2 })
  assert_tensor(halves[2], { 3, 4 })

  local matrix = numlua.array({
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
  })
  local columns = numlua.hsplit(matrix, 2)
  assert_tensor(columns[1], { { 1, 2 }, { 5, 6 } })
  assert_tensor(columns[2], { { 3, 4 }, { 7, 8 } })

  local rows = numlua.vsplit(matrix, 2)
  assert_tensor(rows[1], { { 1, 2, 3, 4 } })
  assert_tensor(rows[2], { { 5, 6, 7, 8 } })

  local cube = numlua.arange(24):reshape(2, 3, 4)
  local planes = numlua.dsplit(cube, 2)
  assert_eq(#planes, 2)
  assert_shape(planes[1], { 2, 3, 2 })
  assert_shape(planes[2], { 2, 3, 2 })
  assert_tensor(planes[1]:select(1, 1), { { 0, 1 }, { 4, 5 }, { 8, 9 } })
  assert_tensor(planes[2]:select(1, 2), { { 14, 15 }, { 18, 19 }, { 22, 23 } })
end)

test("split family validates sections, boundaries, axes, and ranks", function()
  local vector = numlua.array({ 1, 2, 3, 4 })
  assert_error(function() numlua.split(vector, 0) end,
    "positive integer")
  assert_error(function() numlua.split(vector, 1.5) end,
    "positive integer")
  assert_error(function() numlua.split(vector, 3) end,
    "cannot be split into 3 equal parts")
  assert_error(function() numlua.split(vector, { 1.5 }) end,
    "finite integers")
  assert_error(function() numlua.split(vector, { math.huge }) end,
    "finite integers")
  assert_error(function() numlua.split(vector, "2") end,
    "integer section count or a table")
  assert_error(function() numlua.split(vector, 2, 0) end,
    "axis must be a non-zero integer")
  assert_error(function() numlua.split(vector, 2, 2) end,
    "out of range")
  assert_error(function() numlua.split({ 1, 2 }, 1) end,
    "expects a tensor")
  assert_error(function() numlua.vsplit(vector, 2) end,
    "at least a 2-D tensor")
  assert_error(function() numlua.dsplit(numlua.zeros(2, 2), 2) end,
    "at least a 3-D tensor")
end)
