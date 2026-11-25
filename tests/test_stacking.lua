--
-- NumPy stacking and append helpers.
--

local numlua = require("numlua")

test("vstack and row_stack promote and stack along rows", function()
  local vectors = numlua.array({ { 1, 2 }, { 3, 4 } })
  local stacked = numlua.vstack({ numlua.array({ 1, 2 }),
    numlua.array({ 3, 4 }) })
  assert_tensor(stacked, { { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.row_stack({ numlua.array({ 1, 2 }),
    numlua.array({ 3, 4 }) }), stacked:totable())
  assert_tensor(numlua.vstack({ vectors }),
    { { 1, 2 }, { 3, 4 } })
  assert_shape(numlua.vstack({ numlua.array(1), numlua.array(2) }),
    { 2, 1 })
end)

test("hstack concatenates vectors and columns", function()
  assert_tensor(numlua.hstack({ numlua.array({ 1, 2 }),
    numlua.array({ 3, 4 }) }), { 1, 2, 3, 4 })
  assert_tensor(numlua.hstack({ numlua.array({ { 1 }, { 2 } }),
    numlua.array({ { 3 }, { 4 } }) }),
    { { 1, 3 }, { 2, 4 } })
  assert_error(function()
    numlua.hstack({ numlua.array({ 1, 2 }), numlua.array({ { 3, 4 } }) })
  end, "mixed")
end)

test("dstack and column_stack follow NumPy rank promotion", function()
  local stacked = numlua.dstack({ numlua.array({ 1, 2 }),
    numlua.array({ 3, 4 }) })
  assert_shape(stacked, { 1, 2, 2 })
  assert_tensor(stacked, { { { 1, 3 }, { 2, 4 } } })

  local columns = numlua.column_stack({ numlua.array({ 1, 2, 3 }),
    numlua.array({ 4, 5, 6 }) })
  assert_shape(columns, { 3, 2 })
  assert_tensor(columns, { { 1, 4 }, { 2, 5 }, { 3, 6 } })
  assert_tensor(numlua.column_stack({ numlua.array(1), numlua.array(2) }),
    { { 1, 2 } })
  assert_error(function() numlua.column_stack({ numlua.zeros(1, 1, 1) }) end,
    "rank at most 2")
end)

test("append flattens or concatenates along an axis", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.append(values, { 5, 6 }), { 1, 2, 3, 4, 5, 6 })
  assert_tensor(numlua.append(values, numlua.array({ { 5, 6 } }), 1),
    { { 1, 2 }, { 3, 4 }, { 5, 6 } })
  assert_tensor(numlua.append(values, numlua.array({ { 5 }, { 6 } }), 2),
    { { 1, 2, 5 }, { 3, 4, 6 } })
  local ints = numlua.array({ 1, 2 }, { dtype = "int64" })
  assert_dtype(numlua.append(ints, { 3, 4 }), "int64")
end)

test("stacking helpers validate arguments", function()
  assert_error(function() numlua.vstack({}) end, "non-empty table")
  assert_error(function() numlua.vstack({{ 1, 2 }}) end, "not a tensor")
  assert_error(function() numlua.append({ 1, 2 }, { 3 }) end,
    "expects a tensor")
  assert_error(function() numlua.append(numlua.zeros(2), { 3 }, 0) end,
    "1-based")
  assert_error(function() numlua.append(numlua.zeros(2), true) end,
    "values")
end)
