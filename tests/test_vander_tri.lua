--
-- Vandermonde, triangular-ones, and identity constructors.
--

local numlua = require("numlua")

test("vander builds NumPy-shaped Vandermonde matrices", function()
  local values = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.vander(values),
    { { 1, 1, 1 }, { 4, 2, 1 }, { 9, 3, 1 } })
  assert_tensor(numlua.vander(values, 4),
    { { 1, 1, 1, 1 }, { 8, 4, 2, 1 }, { 27, 9, 3, 1 } })
  assert_shape(numlua.vander(values, 0), { 3, 0 })
  assert_shape(numlua.vander(values, 1), { 3, 1 })
  assert_dtype(numlua.vander(values, 4), "float32")
end)

test("vander supports increasing order and table input", function()
  assert_tensor(numlua.vander({ 1, 2 }, 2, true), { { 1, 1 }, { 1, 2 } })
  assert_tensor(numlua.vander(numlua.array({ 2, 3 }, { dtype = "float64" }),
    2, true), { { 1, 2 }, { 1, 3 } })
  assert_dtype(numlua.vander(numlua.array({ 2, 3 }, { dtype = "float64" })),
    "float64")
  assert_shape(numlua.vander(numlua.zeros(0), 2), { 0, 2 })
end)

test("tri builds lower-triangular ones", function()
  assert_tensor(numlua.tri(3),
    { { 1, 0, 0 }, { 1, 1, 0 }, { 1, 1, 1 } })
  assert_shape(numlua.tri(2, 4), { 2, 4 })
  assert_tensor(numlua.tri(2, 4), { { 1, 0, 0, 0 }, { 1, 1, 0, 0 } })
  assert_tensor(numlua.tri(2, 4, 1),
    { { 1, 1, 0, 0 }, { 1, 1, 1, 0 } })
  assert_tensor(numlua.tri(3, 3, -1),
    { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 } })
  assert_eq(numlua.tri(0, 3):numel(), 0)
  assert_eq(numlua.tri(3, 0):numel(), 0)
end)

test("tri accepts dtype and option-table positions", function()
  local ints = numlua.tri(2, { dtype = "int64" })
  assert_dtype(ints, "int64")
  assert_tensor(ints, { { 1, 0 }, { 1, 1 } })
  assert_dtype(numlua.tri(2, 3, { dtype = "int32" }), "int32")
  assert_dtype(numlua.tri(2, 3, 1, { dtype = "float64" }), "float64")
  assert_dtype(numlua.tri(2, 3, { dtype = "float64" }), "float64")
end)

test("identity is a square eye", function()
  assert_tensor(numlua.identity(2), { { 1, 0 }, { 0, 1 } })
  assert_dtype(numlua.identity(2), "float32")
  assert_dtype(numlua.identity(2, { dtype = "int64" }), "int64")
  assert_shape(numlua.identity(0), { 0, 0 })
end)

test("vander, tri, and identity validate arguments", function()
  assert_error(function() numlua.vander(1) end, "1-D tensor or table")
  assert_error(function() numlua.vander({ { 1, 2 } }) end, "1-D")
  assert_error(function() numlua.vander({ 1, 2 }, -1) end, "N")
  assert_error(function() numlua.vander({ 1, 2 }, 2.5) end, "N")
  assert_error(function() numlua.vander({ 1, 2 }, 2, 1) end, "increasing")

  assert_error(function() numlua.tri(-1) end, "N")
  assert_error(function() numlua.tri(2, -1) end, "M")
  assert_error(function() numlua.tri(2, 3, 0.5) end, "k")
  assert_error(function() numlua.tri(2, { nope = true }) end,
    "unknown option")
  assert_error(function() numlua.tri(2, 3, false) end, "k")
  assert_error(function() numlua.tri(2, { dtype = "int64" }, 1) end,
    "options must be the last")
  assert_error(function() numlua.tri(2, 3, { dtype = "int64" },
    { dtype = "int64" }) end, "options must be the last")

  assert_error(function() numlua.identity(-1) end, "non-negative")
  assert_error(function() numlua.identity(1.5) end, "non-negative")
end)
