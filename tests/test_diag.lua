--
-- NumPy diagonal and triangular helpers: diag, diagflat, tril, triu.
--

local numlua = require("numlua")

test("diag constructs a matrix from a vector", function()
  local v = numlua.array({ 1, 2, 3 })
  local d = numlua.diag(v)
  assert_dtype(d, "float32")
  assert_shape(d, { 3, 3 })
  assert_tensor(d, {
    { 1, 0, 0 },
    { 0, 2, 0 },
    { 0, 0, 3 },
  })

  assert_tensor(numlua.diag(v, 1), {
    { 0, 1, 0, 0 },
    { 0, 0, 2, 0 },
    { 0, 0, 0, 3 },
    { 0, 0, 0, 0 },
  })
  assert_tensor(numlua.diag(v, -2), {
    { 1, 0, 0, 0, 0 },
    { 0, 2, 0, 0, 0 },
    { 0, 0, 3, 0, 0 },
    { 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0 },
  })

  -- Integer and bool dtypes are preserved.
  local ints = numlua.diag(numlua.array({ 4, 5 }, { dtype = "int64" }))
  assert_dtype(ints, "int64")
  assert_tensor(ints, { { 4, 0 }, { 0, 5 } })

  local bools = numlua.diag(numlua.array({ true, false, true }))
  assert_dtype(bools, "bool")
  assert_tensor(bools, {
    { true, false, false },
    { false, false, false },
    { false, false, true },
  })
  assert_tensor(numlua.diag(numlua.array({ true, false }), -1), {
    { false, false, false },
    { true, false, false },
    { false, false, false },
  })
end)

test("diag extracts diagonals from a matrix", function()
  local m = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  })
  assert_tensor(numlua.diag(m), { 1, 5, 9 })
  assert_tensor(numlua.diag(m, 1), { 2, 6 })
  assert_tensor(numlua.diag(m, 2), { 3 })
  assert_tensor(numlua.diag(m, -1), { 4, 8 })
  assert_tensor(numlua.diag(m, -2), { 7 })

  -- Offsets beyond either dimension produce an empty 1-D result.
  local empty = numlua.diag(m, 4)
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)
  assert_shape(numlua.diag(m, -4), { 0 })

  -- Extraction is not limited to square matrices.
  local rectangular = numlua.array({
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
  })
  assert_tensor(numlua.diag(rectangular), { 1, 6 })
  assert_tensor(numlua.diag(rectangular, 1), { 2, 7 })
  assert_tensor(numlua.diag(rectangular, -1), { 5 })

  local bools = numlua.array({
    { true, false },
    { false, true },
  })
  assert_tensor(numlua.diag(bools), { true, true })
  assert_tensor(numlua.diag(bools, 1), { false })
end)

test("diag and diagflat handle empty, scalar, and flattened inputs", function()
  local empty = numlua.diag(numlua.zeros(0))
  assert_shape(empty, { 0, 0 })
  assert_eq(empty:numel(), 0)

  local bool_empty = numlua.diag(numlua.array({}, { dtype = "bool" }))
  assert_dtype(bool_empty, "bool")
  assert_shape(bool_empty, { 0, 0 })
  assert_eq(bool_empty:numel(), 0)

  local empty_offset = numlua.diag(numlua.zeros(0), 2)
  assert_shape(empty_offset, { 2, 2 })
  assert_eq(empty_offset:numel(), 4)

  local scalar = numlua.diagflat(numlua.array(7))
  assert_shape(scalar, { 1, 1 })
  assert_eq(scalar:get(1, 1), 7)

  local flat = numlua.diagflat(numlua.array({ { 1, 2 }, { 3, 4 } }))
  assert_shape(flat, { 4, 4 })
  assert_tensor(numlua.diag(flat), { 1, 2, 3, 4 })

  assert_tensor(numlua.diagflat(numlua.array({ 1, 2 }), 2), {
    { 0, 0, 1, 0 },
    { 0, 0, 0, 2 },
    { 0, 0, 0, 0 },
    { 0, 0, 0, 0 },
  })
end)

test("tril and triu zero the opposite triangle", function()
  local m = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  })
  assert_tensor(numlua.tril(m), {
    { 1, 0, 0 },
    { 4, 5, 0 },
    { 7, 8, 9 },
  })
  assert_tensor(numlua.tril(m, 1), {
    { 1, 2, 0 },
    { 4, 5, 6 },
    { 7, 8, 9 },
  })
  assert_tensor(numlua.tril(m, -1), {
    { 0, 0, 0 },
    { 4, 0, 0 },
    { 7, 8, 0 },
  })
  assert_tensor(numlua.triu(m), {
    { 1, 2, 3 },
    { 0, 5, 6 },
    { 0, 0, 9 },
  })
  assert_tensor(numlua.triu(m, 1), {
    { 0, 2, 3 },
    { 0, 0, 6 },
    { 0, 0, 0 },
  })
  assert_tensor(numlua.triu(m, -1), {
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 0, 8, 9 },
  })

  -- Rectangular matrices use the last two dimensions as rows and columns.
  local rectangular = numlua.array({
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
  })
  assert_tensor(numlua.tril(rectangular), {
    { 1, 0, 0, 0 },
    { 5, 6, 0, 0 },
  })
  assert_tensor(numlua.triu(rectangular, 2), {
    { 0, 0, 3, 4 },
    { 0, 0, 0, 8 },
  })
end)

test("tril and triu preserve dtype and handle N-D and bool inputs", function()
  local ints = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" })
  assert_dtype(numlua.tril(ints), "int64")
  assert_tensor(numlua.tril(ints), { { 1, 0 }, { 3, 4 } })

  local cube = numlua.arange(24):reshape(2, 3, 4)
  local lower = numlua.tril(cube)
  assert_shape(lower, { 2, 3, 4 })
  assert_tensor(lower:select(1, 1), {
    { 0, 0, 0, 0 },
    { 4, 5, 0, 0 },
    { 8, 9, 10, 0 },
  })
  assert_tensor(lower:select(1, 2), {
    { 12, 0, 0, 0 },
    { 16, 17, 0, 0 },
    { 20, 21, 22, 0 },
  })

  local bools = numlua.array({
    { true, true },
    { true, false },
  })
  assert_dtype(numlua.triu(bools), "bool")
  assert_tensor(numlua.triu(bools), {
    { true, true },
    { false, false },
  })
  assert_tensor(numlua.tril(bools), {
    { true, false },
    { true, false },
  })

  local empty = numlua.tril(numlua.zeros(2, 0))
  assert_shape(empty, { 2, 0 })
  assert_eq(empty:numel(), 0)
end)

test("diag helpers validate rank and offsets", function()
  assert_error(function() numlua.diag(numlua.zeros(2, 2, 2)) end,
    "expects a 1-D or 2-D tensor")
  assert_error(function() numlua.diag(numlua.zeros(2), 1.5) end,
    "offset must be a finite integer")
  assert_error(function() numlua.diag(numlua.zeros(2), math.huge) end,
    "offset must be a finite integer")
  assert_error(function() numlua.diagflat({ 1, 2 }) end,
    "expects a tensor")
  assert_error(function() numlua.tril(numlua.zeros(3)) end,
    "at least a 2-D tensor")
  assert_error(function() numlua.triu(numlua.zeros(2, 2), 0.5) end,
    "offset must be a finite integer")
  assert_error(function() numlua.tril({ { 1 } }) end,
    "expects a tensor")
end)

test("diag and triangular results own independent storage", function()
  local vector = numlua.array({ 1, 2 })
  local matrix = numlua.diag(vector)
  matrix:fill_(9)
  assert_tensor(vector, { 1, 2 })

  local input = numlua.array({ { 1, 2 }, { 3, 4 } })
  local lower = numlua.tril(input)
  lower:fill_(0)
  assert_tensor(input, { { 1, 2 }, { 3, 4 } })
end)
