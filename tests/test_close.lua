--
-- NumPy closeness/equality predicates: isclose, allclose, array_equal,
-- array_equiv.
--

local numlua = require("numlua")

test("isclose uses NumPy's relative and absolute tolerances", function()
  local nan = 0 / 0
  local actual = numlua.array({ 1, 2, 3, math.huge, -math.huge, nan })
  local expected = numlua.array({
    1 + 1e-6,
    2 + 1e-4,
    3.1,
    math.huge,
    -math.huge,
    nan,
  })
  local close = numlua.isclose(actual, expected)
  assert_dtype(close, "bool")
  assert_tensor(close, { true, false, false, true, true, false })

  -- equal_nan opts NaN in, while NaN against a number remains unequal.
  assert_tensor(numlua.isclose(actual, expected, nil, nil, true),
    { true, false, false, true, true, true })

  -- atol dominates near zero; rtol dominates for large magnitudes.
  assert_eq(numlua.isclose(0, 1e-9):item(), true)
  assert_eq(numlua.isclose(0, 1e-7):item(), false)
  assert_eq(numlua.isclose(1e6, 1e6 + 0.5):item(), true)
  assert_eq(numlua.isclose(1e6, 1e6 + 20):item(), false)
  assert_eq(numlua.isclose(1, 1.000001, 0, 1e-6):item(), true)
  assert_eq(numlua.isclose(1, 1.1, 0, 0):item(), false)
end)

test("isclose treats infinities exactly and never mixes them with finite values", function()
  local inf = math.huge
  assert_eq(numlua.isclose(inf, inf):item(), true)
  assert_eq(numlua.isclose(-inf, -inf):item(), true)
  assert_eq(numlua.isclose(inf, -inf):item(), false)
  assert_eq(numlua.isclose(1, inf):item(), false)
  assert_eq(numlua.isclose(inf, 1):item(), false)
  assert_eq(numlua.isclose(-inf, 0):item(), false)
  -- An infinite tolerance must not make a finite/infinite pair close.
  assert_eq(numlua.isclose(1, inf, 1, 1):item(), false)
end)

test("isclose broadcasts and accepts scalars and mixed dtypes", function()
  local column = numlua.array({ { 1 }, { 2 }, { 3 } })
  local row = numlua.array({ { 1, 2, 3 } })
  local close = numlua.isclose(column, row)
  assert_shape(close, { 3, 3 })
  assert_tensor(close, {
    { true, false, false },
    { false, true, false },
    { false, false, true },
  })

  assert_eq(numlua.isclose(2, 2):item(), true)
  assert_tensor(numlua.isclose(numlua.array({ 1, 2 }), 2),
    { false, true })
  assert_tensor(numlua.isclose(
    numlua.array({ 1, 2 }, { dtype = "int64" }), 1.5), { false, false })
  assert_tensor(numlua.isclose(
    numlua.array({ true, false }), numlua.array({ 1, 0 })),
    { true, true })
end)

test("allclose returns a Lua boolean", function()
  assert_eq(numlua.allclose(numlua.array({ 1, 2 }),
    numlua.array({ 1, 2 + 1e-9 })), true)
  assert_eq(numlua.allclose(numlua.array({ 1, 2 }),
    numlua.array({ 1, 3 })), false)
  assert_eq(numlua.allclose(numlua.array({ math.huge }),
    numlua.array({ math.huge })), true)
  assert_eq(numlua.allclose(numlua.array({ 1 }),
    numlua.array({ math.huge })), false)

  local nan = 0 / 0
  assert_eq(numlua.allclose(numlua.array({ nan }),
    numlua.array({ nan })), false)
  assert_eq(numlua.allclose(numlua.array({ nan }),
    numlua.array({ nan }), nil, nil, true), true)
  assert_eq(numlua.allclose(numlua.array({ { 1 }, { 2 } }),
    numlua.array({ { 1, 1 } })), false)
  assert_eq(numlua.allclose(2, numlua.array({ 2, 2 })), true)
end)

test("array_equal requires matching shapes and exact values", function()
  assert_eq(numlua.array_equal(numlua.array({ 1, 2 }),
    numlua.array({ 1, 2 })), true)
  assert_eq(numlua.array_equal(numlua.array({ 1, 2 }),
    numlua.array({ 1, 3 })), false)
  assert_eq(numlua.array_equal(numlua.array({ 1, 2 }),
    numlua.array({ { 1, 2 } })), false)
  assert_eq(numlua.array_equal(numlua.array({ 1, 2 }),
    numlua.array({ 1, 2, 3 })), false)
  assert_eq(numlua.array_equal(numlua.array(1), numlua.array({ 1 })), false)
  assert_eq(numlua.array_equal(numlua.array({ 1 }, { dtype = "int64" }),
    numlua.array({ 1 })), true)

  local nan = 0 / 0
  assert_eq(numlua.array_equal(numlua.array({ nan }),
    numlua.array({ nan })), false)
  assert_eq(numlua.array_equal(numlua.array({ nan }),
    numlua.array({ nan }), true), true)
  assert_eq(numlua.array_equal(
    numlua.array({ nan, 1 }), numlua.array({ nan, 1 }), true), true)
  assert_eq(numlua.array_equal(
    numlua.array({ nan, 1 }), numlua.array({ nan, 2 }), true), false)
end)

test("array_equiv allows broadcasting and rejects incompatible shapes", function()
  assert_eq(numlua.array_equiv(1, numlua.array({ 1, 1, 1 })), true)
  assert_eq(numlua.array_equiv(numlua.array({ 1, 2 }),
    numlua.array({ { 1, 2 }, { 1, 2 } })), true)
  assert_eq(numlua.array_equiv(numlua.array({ 1, 2 }),
    numlua.array({ { 1 }, { 2 } })), false)
  assert_eq(numlua.array_equiv(numlua.array({ 1, 2 }),
    numlua.array({ 1, 2, 3 })), false)
  assert_eq(numlua.array_equiv(numlua.array({ true, false }),
    numlua.array({ 1, 0 })), true)

  local nan = 0 / 0
  assert_eq(numlua.array_equiv(numlua.array({ nan }),
    numlua.array({ nan })), false)
end)

test("closeness helpers validate options and operands", function()
  local a = numlua.array({ 1 })
  local b = numlua.array({ 1 })
  assert_error(function() numlua.isclose(a, b, -1) end,
    "rtol must be a finite non-negative number")
  assert_error(function() numlua.isclose(a, b, 0 / 0) end,
    "rtol must be a finite non-negative number")
  assert_error(function() numlua.isclose(a, b, math.huge) end,
    "rtol must be a finite non-negative number")
  assert_error(function() numlua.isclose(a, b, nil, -1) end,
    "atol must be a finite non-negative number")
  assert_error(function() numlua.isclose(a, b, nil, nil, 1) end,
    "equal_nan must be a boolean or nil")
  assert_error(function() numlua.allclose(a, b, "1") end,
    "rtol must be a finite non-negative number")
  assert_error(function() numlua.isclose(a, {}) end,
    "expects tensors or numbers")
  assert_error(function() numlua.array_equal(a, {}) end,
    "expects tensors or numbers")
  assert_error(function() numlua.array_equal(a, b, 1) end,
    "equal_nan must be a boolean or nil")
  assert_error(function() numlua.array_equiv({}, b) end,
    "expects tensors or numbers")
end)
