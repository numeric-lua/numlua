--
-- Focused parity checks for numlua.select (np.select).
--
-- Conditions are tested in order, so these tests pin the first-true-wins
-- rule, default broadcasting, scalar handling, and dtype behavior instead of
-- only re-checking a single literal output.
--
local numlua = require("numlua")

test("select uses the first true condition and fills with the default",
    function()
  local values = numlua.array({ -2, -1, 0, 1, 2 })
  local result = numlua.select({
    values:lt(0),
    values:gt(0),
  }, { -1, 1 }, 0)
  assert_tensor(result, { -1, -1, 0, 1, 1 })
  assert_dtype(result, "float32")

  -- Both conditions are true at positions 1 and 2; the first must win.
  local first = numlua.array({ true, true, false, false })
  local second = numlua.array({ true, false, true, false })
  assert_tensor(numlua.select({ first, second }, { 10, 20 }, 30),
    { 10, 10, 20, 30 })
end)

test("select broadcasts conditions, choices, and the default", function()
  local rows = numlua.array({ { true }, { false } })
  local columns = numlua.array({ { false, true, false } })
  local result = numlua.select({
    rows,
    columns,
  }, {
    numlua.array({ { 1, 2, 3 } }),
    numlua.array({ { 10 }, { 20 } }),
  }, -1)
  assert_shape(result, { 2, 3 })
  assert_tensor(result, {
    { 1, 2, 3 },
    { -1, 20, -1 },
  })

  -- Numeric conditions are truth-tested, and a false scalar condition leaves
  -- a broadcast tensor default untouched.
  assert_eq(numlua.select({ 2 }, { 5 }):item(), 5)
  assert_eq(numlua.select({ 0 }, { 5 }, 9):item(), 9)
  assert_tensor(numlua.select({ false }, { 1 },
    numlua.array({ 7, 8, 9 })), { 7, 8, 9 })
end)

test("select preserves tensor dtype and materializes scalar branches",
    function()
  local condition = numlua.array({ true, false })
  local integers = numlua.array({ 5, 6 }, { dtype = "int64" })

  local typed_default = numlua.select({ condition }, { integers },
    numlua.array(0, { dtype = "int64" }))
  assert_dtype(typed_default, "int64")
  assert_tensor(typed_default, { 5, 0 })

  -- Like where(), an omitted scalar default is a 0-D float32 tensor.
  local scalar_default = numlua.select({ condition }, { integers })
  assert_dtype(scalar_default, "float32")
  assert_tensor(scalar_default, { 5, 0 })

  assert_dtype(numlua.select({ condition }, { numlua.array({ true, false }) },
    false), "bool")
  assert_tensor(numlua.select({ condition }, { numlua.array({ true, false }) },
    false), { true, false })
end)

test("select accepts non-contiguous conditions and choices", function()
  local source = numlua.arange(6):reshape(2, 3)
  local view = numlua.transpose(source)
  assert_false(view:is_contiguous())
  local result = numlua.select({ view:gt(3) }, { view },
    numlua.zeros(3, 2))
  assert_shape(result, { 3, 2 })
  assert_tensor(result, {
    { 0, 0 },
    { 0, 4 },
    { 0, 5 },
  })
end)

test("select validates lists, operands, defaults, and broadcasting",
    function()
  local condition = numlua.array({ true, false })
  assert_error(function() numlua.select(nil, { 1 }) end,
    "non-empty table of conditions")
  assert_error(function() numlua.select({}, { 1 }) end,
    "non-empty table of conditions")
  assert_error(function() numlua.select(condition, { 1 }) end,
    "non-empty table of conditions")
  assert_error(function() numlua.select({ true }, {}) end,
    "non-empty table of choices")
  assert_error(function() numlua.select({ true }, { 1, 2 }) end,
    "same length")
  assert_error(function() numlua.select({ {} }, { 1 }) end,
    "condition 1 must be a tensor, number, or boolean")
  assert_error(function() numlua.select({ true }, { {} }) end,
    "choice 1 must be a tensor, number, or boolean")
  assert_error(function() numlua.select({ true }, { 1 }, {}) end,
    "default must be a tensor, number, or boolean")
  assert_error(function()
    numlua.select({ condition }, { numlua.array({ 1, 2, 3 }) })
  end, "do not broadcast")
end)

test("select handles empty tensors and 0-D branches", function()
  local empty_condition = numlua.array({}, { dtype = "bool" })
  local empty_choice = numlua.array({}, { dtype = "int64" })
  local empty = numlua.select({ empty_condition }, { empty_choice }, 7)
  assert_shape(empty, { 0 })
  assert_eq(empty:numel(), 0)

  -- A 0-D true condition selects a whole 1-D choice, and a scalar default
  -- broadcasts against a tensor default choice.
  assert_tensor(numlua.select({ numlua.array(true) },
    { numlua.array({ 1, 2, 3 }) }, 9), { 1, 2, 3 })
  assert_tensor(numlua.select({ false }, { numlua.array(5) },
    numlua.array({ 7, 8 })), { 7, 8 })

  -- A size-1 choice still broadcasts against an empty condition, but a
  -- non-broadcastable choice is rejected.
  assert_shape(numlua.select({ empty_condition },
    { numlua.array({ 1 }) }, 0), { 0 })
  assert_error(function()
    numlua.select({ empty_condition }, { numlua.array({ 1, 2 }) })
  end, "do not broadcast")
end)

test("select truth-tests numeric conditions and keeps integer dtypes",
    function()
  local condition = numlua.array({ 0, 1, -2, 0 }, { dtype = "int32" })
  local choices = numlua.array({ 10, 20, 30, 40 }, { dtype = "int64" })
  local default = numlua.array(-1, { dtype = "int64" })
  local result = numlua.select({ condition }, { choices }, default)
  assert_dtype(result, "int64")
  assert_tensor(result, { -1, 20, 30, -1 })

  -- Float NaN is truthy because it is non-zero.
  local nan = 0 / 0
  assert_tensor(numlua.select({ numlua.array({ nan, 0 }) },
    { numlua.array({ 1, 2 }, { dtype = "int64" }) }, 0), { 1, 0 })
end)

test("select broadcasts non-contiguous defaults and 0-D choices", function()
  local view = numlua.transpose(numlua.arange(6):reshape(2, 3))
  assert_false(view:is_contiguous())
  local defaulted = numlua.select({ view:gt(100) }, { numlua.array(1) }, view)
  assert_shape(defaulted, { 3, 2 })
  assert_true(defaulted:equal(view))

  local result = numlua.select({
    numlua.array({ { true }, { false } }),
  }, {
    numlua.array({ { 5, 6, 7 } }),
  }, numlua.array({ 9, 8, 7 }))
  assert_shape(result, { 2, 3 })
  assert_tensor(result, {
    { 5, 6, 7 },
    { 9, 8, 7 },
  })
end)
