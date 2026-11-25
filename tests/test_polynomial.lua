--
-- Polynomial evaluation and least-squares fitting.
--

local numlua = require("numlua")

test("polyval evaluates descending-degree coefficients", function()
  local p = numlua.array({ 1, -2, 1 })
  assert_near(numlua.polyval(p, 3):item(), 4, 1e-6)
  assert_near(numlua.polyval(p, numlua.array({ 0, 1, 2, 3 })):totable(),
    { 1, 0, 1, 4 }, 1e-6)
  assert_shape(numlua.polyval(p,
    numlua.array({ { 0, 1 }, { 2, 3 } })), { 2, 2 })
  assert_near(numlua.polyval(p,
    numlua.array({ { 0, 1 }, { 2, 3 } })):totable(),
    { { 1, 0 }, { 1, 4 } }, 1e-6)
end)

test("polyval handles degrees, tables, and dtypes", function()
  assert_near(numlua.polyval({ 5 }, numlua.array({ 1, 2, 3 })):totable(),
    { 5, 5, 5 }, 0)
  assert_near(numlua.polyval({ 2, 3 }, { 0, 1, 2 }):totable(),
    { 3, 5, 7 }, 0)
  assert_dtype(numlua.polyval(
    numlua.array({ 1, 1 }, { dtype = "float32" }),
    numlua.array({ 1, 2 }, { dtype = "float32" })), "float32")
  assert_dtype(numlua.polyval(
    numlua.array({ 1, 1 }, { dtype = "int64" }),
    numlua.array({ 1, 2 }, { dtype = "int64" })), "float64")
end)

test("polyfit recovers exact low-degree polynomials", function()
  local x = numlua.array({ 0, 1, 2, 3 })
  local line = numlua.polyfit(x, numlua.array({ 3, 5, 7, 9 }), 1)
  assert_dtype(line, "float64")
  assert_near(line:totable(), { 2, 3 }, 1e-8)
  assert_tensor(numlua.polyval(line, x), { 3, 5, 7, 9 }, 1e-8)

  local quadratic = numlua.polyfit(x, numlua.array({ 1, 0, 1, 4 }), 2)
  assert_near(quadratic:totable(), { 1, -2, 1 }, 1e-8)
  assert_near(numlua.polyval(quadratic, 2):item(), 1, 1e-8)
end)

test("polyfit fits noisy data by least squares", function()
  local coefficients = numlua.polyfit(
    numlua.array({ 0, 1, 2, 3 }), numlua.array({ 0, 2, 4, 7 }), 1)
  assert_near(coefficients:get(1), 2.3, 1e-8)
  assert_near(coefficients:get(2), -0.2, 1e-8)
end)

test("polynomial helpers validate their arguments", function()
  assert_error(function() numlua.polyval({}, 1) end, "at least one")
  assert_error(function() numlua.polyval({ { 1, 2 } }, 1) end, "1-D")
  assert_error(function() numlua.polyval({ 1 }, true) end, "x to be")
  assert_error(function() numlua.polyval({ "x" }, 1) end, "not numeric")
  assert_error(function() numlua.polyfit({ 1, 2 }, { 1 }, 1) end,
    "same length")
  assert_error(function() numlua.polyfit({ 1, 2 }, { 1, 2 }, -1) end,
    "degree")
  assert_error(function() numlua.polyfit({ 1, 2 }, { 1, 2 }, 2) end,
    "smaller than")
  assert_error(function() numlua.polyfit({ { 1, 2 } }, { 1, 2 }, 1) end,
    "1-D")
  assert_error(function() numlua.polyfit({ 1, 2 }, { 1, 2 }, 1.5) end,
    "degree")
end)
