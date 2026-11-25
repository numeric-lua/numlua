--
-- Second-order gradient edge differences.
--

local numlua = require("numlua")

test("gradient edge_order=2 uses three-point edges", function()
  local values = numlua.array({ 0, 1, 4, 9 }, { dtype = "float64" })
  local result = numlua.gradient(values, nil, nil, { edge_order = 2 })
  assert_near(result:totable(), { 0, 2, 4, 6 }, 1e-10)

  local nonuniform = numlua.gradient(
    numlua.array({ 1, 3, 7, 13 }, { dtype = "float64" }),
    numlua.array({ 0, 1, 3, 6 }), nil, { edge_order = 2 })
  assert_near(nonuniform:totable(), { 2, 2, 2, 2 }, 1e-10)
end)

test("gradient edge_order=2 works on 2-D axes", function()
  local values = numlua.array({
    { 0, 1, 4 },
    { 0, 2, 8 },
    { 0, 3, 12 },
  }, { dtype = "float64" })
  local columns = numlua.gradient(values, nil, 1, { edge_order = 2 })
  assert_near(columns:totable(), {
    { 0, 1, 4 }, { 0, 1, 4 }, { 0, 1, 4 },
  }, 1e-10)
  local rows = numlua.gradient(values, nil, 2, { edge_order = 2 })
  assert_near(rows:totable(), {
    { 0, 2, 4 }, { 0, 4, 8 }, { 0, 6, 12 },
  }, 1e-10)
  local all = numlua.gradient(values, nil, nil, { edge_order = 2 })
  assert_eq(#all, 2)
end)

test("gradient edge_order validation and legacy default", function()
  local values = numlua.array({ 0, 1, 4, 9 }, { dtype = "float64" })
  local first_order = numlua.gradient(values)
  assert_near(first_order:totable(), { 1, 2, 4, 5 }, 1e-10)

  assert_error(function()
    numlua.gradient(values, nil, nil, { edge_order = 3 })
  end, "edge_order")
  assert_error(function()
    numlua.gradient(values, nil, nil, { nope = 1 })
  end, "unknown option")
  assert_error(function() numlua.gradient(values, nil, nil, 2) end,
    "options")
  assert_error(function()
    numlua.gradient(numlua.array({ 0, 1 }), nil, nil, { edge_order = 2 })
  end, "at least 3")
end)
