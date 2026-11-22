local numlua = require("numlua")

test("swapaxes and moveaxis are zero-copy axis permutations", function()
  local cube = numlua.reshape(numlua.arange(24), 2, 3, 4)

  local swapped = numlua.swapaxes(cube, 1, 3)
  assert_shape(swapped, { 4, 3, 2 })
  assert_eq(swapped:get(1, 1, 1), cube:get(1, 1, 1))
  assert_eq(swapped:get(1, 1, 2), cube:get(2, 1, 1))
  assert_shape(numlua.swapaxes(cube, -1, -3), { 4, 3, 2 })
  swapped:set(1, 1, 1, 99)
  assert_eq(cube:get(1, 1, 1), 99)

  local moved = numlua.moveaxis(cube, 1, 3)
  assert_shape(moved, { 3, 4, 2 })
  assert_eq(moved:get(1, 1, 1), cube:get(1, 1, 1))
  assert_eq(moved:get(1, 1, 2), cube:get(2, 1, 1))
  moved:set(1, 1, 1, 77)
  assert_eq(cube:get(1, 1, 1), 77)

  assert_shape(numlua.moveaxis(cube, -1, 1), { 4, 2, 3 })
  assert_shape(numlua.moveaxis(cube, { 1, 2 }, { 3, 1 }), { 3, 4, 2 })
end)

test("atleast_* add singleton dimensions as views", function()
  local scalar = numlua.array(5)
  assert_shape(numlua.atleast_1d(scalar), { 1 })
  assert_shape(numlua.atleast_2d(scalar), { 1, 1 })
  assert_shape(numlua.atleast_3d(scalar), { 1, 1, 1 })

  local vector = numlua.array({ 1, 2, 3 })
  assert_shape(numlua.atleast_1d(vector), { 3 })
  assert_shape(numlua.atleast_2d(vector), { 1, 3 })
  assert_shape(numlua.atleast_3d(vector), { 1, 3, 1 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_shape(numlua.atleast_3d(matrix), { 2, 2, 1 })

  local first, second = numlua.atleast_1d(scalar, vector)
  assert_shape(first, { 1 })
  assert_shape(second, { 3 })

  local promoted = numlua.atleast_2d(vector)
  promoted:set(1, 1, 8)
  assert_eq(vector:get(1), 8)

  assert_shape(numlua.atleast_2d(numlua.zeros(0)), { 1, 0 })
  assert_shape(numlua.atleast_3d(numlua.zeros(0, 3)), { 0, 3, 1 })
end)

test("resize zero-fills, truncates, and allocates independent storage", function()
  local vector = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.resize(vector, { 2, 3 }), {
    { 1, 2, 3 },
    { 0, 0, 0 },
  })
  assert_tensor(numlua.resize(vector, { 5 }), { 1, 2, 3, 0, 0 })
  assert_tensor(numlua.resize(vector, { 2 }), { 1, 2 })
  assert_shape(numlua.resize(vector, { 0, 4 }), { 0, 4 })
  assert_eq(numlua.resize(vector, { 0, 4 }):numel(), 0)

  assert_tensor(numlua.resize(numlua.zeros(0), { 3 }), { 0, 0, 0 })
  local transposed = numlua.transpose(numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }))
  assert_tensor(numlua.resize(transposed, { 4 }), { 1, 4, 2, 5 })
  assert_dtype(numlua.resize(numlua.array({ 1, 2 }, { dtype = "int64" }),
    { 3 }), "int64")

  local copy = numlua.resize(vector, { 3 })
  copy:set(1, 99)
  assert_eq(vector:get(1), 1)
end)

test("pad supports edge, reflect, and symmetric modes", function()
  local vector = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.pad(vector, 2, "edge"), { 1, 1, 1, 2, 3, 3, 3 })
  assert_tensor(numlua.pad(vector, 2, "reflect"), { 3, 2, 1, 2, 3, 2, 1 })
  assert_tensor(numlua.pad(vector, { 2, 3 }, "symmetric"),
    { 2, 1, 1, 2, 3, 3, 2, 1 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.pad(matrix, 1, "edge"), {
    { 1, 1, 2, 2 },
    { 1, 1, 2, 2 },
    { 3, 3, 4, 4 },
    { 3, 3, 4, 4 },
  })
  assert_tensor(numlua.pad(matrix, { 1, 1 }, "reflect"), {
    { 4, 3, 4, 3 },
    { 2, 1, 2, 1 },
    { 4, 3, 4, 3 },
    { 2, 1, 2, 1 },
  })
  assert_tensor(numlua.pad(matrix, { { 1, 0 }, { 0, 1 } }, "symmetric"), {
    { 1, 2, 2 },
    { 1, 2, 2 },
    { 3, 4, 4 },
  })
  assert_tensor(numlua.pad(numlua.array({ 7 }), 3, "reflect"),
    { 7, 7, 7, 7, 7, 7, 7 })

  local constant = numlua.pad(vector, { 0, 0 }, "constant")
  assert_tensor(constant, { 1, 2, 3 })
  constant:set(1, 9)
  assert_eq(vector:get(1), 1)
end)

test("pad supports wrap and per-axis constant values", function()
  local vector = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.pad(vector, 2, "wrap"), { 2, 3, 1, 2, 3, 1, 2 })
  assert_tensor(numlua.pad(vector, { 2, 3 }, "wrap"),
    { 2, 3, 1, 2, 3, 1, 2, 3 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.pad(matrix, 1, "wrap"), {
    { 4, 3, 4, 3 },
    { 2, 1, 2, 1 },
    { 4, 3, 4, 3 },
    { 2, 1, 2, 1 },
  })

  -- A numeric pair fills before/after on every axis; a table of per-axis
  -- entries can give each axis its own pair.
  assert_tensor(numlua.pad(vector, { 1, 2 }, "constant", { 10, 20 }),
    { 10, 1, 2, 3, 20, 20 })
  assert_tensor(numlua.pad(matrix, { { 1, 0 }, { 0, 1 } }, "constant",
    { { 10, 11 }, { 20, 21 } }), {
    { 10, 10, 10 },
    { 1, 2, 21 },
    { 3, 4, 21 },
  })
  assert_tensor(numlua.pad(matrix, 1, "constant", { 7 }),
    { { 7, 7, 7, 7 }, { 7, 1, 2, 7 }, { 7, 3, 4, 7 },
      { 7, 7, 7, 7 } })
end)

test("pad handles zero-size axes and rejects invalid modes", function()
  local empty = numlua.zeros(0, 3)
  local padded = numlua.pad(empty, { 1, 1 }, "constant", 5)
  assert_shape(padded, { 2, 5 })
  assert_near(padded:sum():item(), 50, 1e-6)

  assert_shape(numlua.pad(empty, { { 0, 0 }, { 1, 1 } }, "edge"), { 0, 5 })
  assert_error(function() numlua.pad(empty, { 0, 1 }, "edge") end,
    "empty axis")
  assert_error(function() numlua.pad(empty, { 0, 1 }, "reflect") end,
    "empty axis")
  assert_error(function() numlua.pad(empty, { 0, 1 }, "symmetric") end,
    "empty axis")
  assert_error(function()
    numlua.pad(numlua.array({ 1, 2 }), 1, "mirror")
  end, "invalid mode")
  assert_error(function() numlua.pad(empty, { 0, 1 }, "wrap") end,
    "empty axis")
end)

test("layout extensions reject invalid axes, shapes, and modes", function()
  assert_error(function()
    numlua.swapaxes(numlua.zeros(2, 3), 0, 1)
  end, "axis must")
  assert_error(function()
    numlua.swapaxes(numlua.zeros(2, 3), 1, 4)
  end, "out of range")
  assert_error(function()
    numlua.moveaxis(numlua.zeros(2, 3), { 1, 1 }, { 1, 2 })
  end, "unique")
  assert_error(function()
    numlua.moveaxis(numlua.zeros(2, 3), { 1, 2 }, { 1 })
  end, "same number")
  assert_error(function()
    numlua.moveaxis(numlua.zeros(2, 3), 3, 1)
  end, "out of range")
  assert_error(function() numlua.resize(numlua.zeros(2)) end, "shape")
  assert_error(function() numlua.resize(numlua.zeros(2), { -1 }) end,
    "non-negative")
end)
