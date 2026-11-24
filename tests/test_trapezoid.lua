--
-- np.trapezoid/np.trapz-shaped numerical integration.
--

local numlua = require("numlua")

test("trapezoid integrates 1-D values with uniform spacing", function()
  local values = numlua.array({ 1, 2, 3 })
  local result = numlua.trapezoid(values)
  assert_eq(result:dim(), 0)
  assert_near(result:item(), 4, 0)
  assert_near(numlua.trapezoid(values, nil, 2):item(), 8, 0)
  assert_near(numlua.trapz(values):item(), 4, 0)
  assert_near(numlua.trapz(values, nil, 2):item(), 8, 0)
end)

test("trapezoid accepts coordinate arrays", function()
  assert_near(numlua.trapezoid(
    numlua.array({ 1, 2, 3 }), numlua.array({ 0, 1, 3 })):item(),
    6.5, 0)
  assert_near(numlua.trapezoid(
    numlua.array({ 1, 2, 3 }), { 0, 1, 3 }):item(), 6.5, 0)
end)

test("trapezoid reduces along the requested axis", function()
  local values = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(numlua.trapezoid(values), { 4, 10 })
  assert_tensor(numlua.trapezoid(values, nil, 1, 2), { 4, 10 })
  assert_tensor(numlua.trapezoid(values, nil, 2, 2), { 8, 20 })
  assert_tensor(numlua.trapezoid(values, nil, 1, 1),
    { 2.5, 3.5, 4.5 })
  assert_tensor(numlua.trapezoid(values, nil, 1, -1), { 4, 10 })
end)

test("trapezoid broadcasts a 1-D coordinate vector and accepts full coordinates", function()
  local values = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local x = numlua.array({ 0, 1, 3 })
  assert_tensor(numlua.trapezoid(values, x), { 6.5, 15.5 })
  local full = numlua.array({ { 0, 1, 3 }, { 0, 2, 5 } })
  assert_tensor(numlua.trapezoid(values, full), { 6.5, 25.5 })
end)

test("trapezoid handles empty and single-point intervals", function()
  assert_eq(numlua.trapezoid(numlua.zeros(0)):item(), 0)
  assert_eq(numlua.trapezoid(numlua.array({ 5 })):item(), 0)
  assert_tensor(numlua.trapezoid(numlua.zeros(2, 0), nil, 1, 2),
    { 0, 0 })
  assert_tensor(numlua.trapezoid(numlua.zeros(2, 1), nil, 1, 2),
    { 0, 0 })
end)

test("trapezoid preserves floating dtypes and promotes integer input", function()
  local floats = numlua.trapezoid(
    numlua.array({ 1, 2, 3 }, { dtype = "float64" }))
  assert_dtype(floats, "float64")
  assert_near(floats:item(), 4, 0)
  local ints = numlua.trapezoid(
    numlua.array({ 1, 2, 3 }, { dtype = "int64" }))
  assert_dtype(ints, "float32")
  assert_near(ints:item(), 4, 0)
end)

test("trapezoid validates axes, spacing, and coordinates", function()
  assert_error(function() numlua.trapezoid(numlua.array(1)) end,
    "at least a 1-D")
  assert_error(function() numlua.trapezoid({ 1, 2, 3 }) end,
    "expects a tensor")
  assert_error(function() numlua.trapezoid(numlua.zeros(2), nil, 1, 0) end,
    "1-based")
  assert_error(function() numlua.trapezoid(numlua.zeros(2), nil, 1, 2) end,
    "out of range")
  assert_error(function() numlua.trapezoid(numlua.zeros(2), nil, "x") end,
    "dx")
  assert_error(function() numlua.trapezoid(numlua.zeros(3),
    numlua.zeros(2)) end, "x has 2 elements")
  assert_error(function() numlua.trapezoid(numlua.zeros(2, 3),
    numlua.zeros(2, 2)) end, "x shape does not match")
  assert_error(function() numlua.trapezoid(numlua.zeros(2), 1) end,
    "x must be")
end)
