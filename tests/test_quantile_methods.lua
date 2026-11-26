--
-- quantile interpolation methods.
--

local numlua = require("numlua")

test("quantile supports linear/lower/higher/midpoint/nearest", function()
  local values = numlua.array({ 1, 2, 3, 4 })
  assert_near(numlua.quantile(values, 0.5), 2.5, 1e-6)
  assert_eq(numlua.quantile(values, 0.5, nil, false, "lower"), 2)
  assert_eq(numlua.quantile(values, 0.5, nil, false, "higher"), 3)
  assert_eq(numlua.quantile(values, 0.5, nil, false, "midpoint"), 2.5)
  assert_eq(numlua.quantile(values, 0.4, nil, false, "nearest"), 2)
  assert_eq(numlua.quantile(values, 0.6, nil, false, "nearest"), 3)
end)

test("quantile methods work along axes and with q tables", function()
  local matrix = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_tensor(numlua.quantile(matrix, 0.5, 2, false, "lower"),
    { 2, 5 })
  assert_tensor(numlua.quantile(matrix, 0.5, 2, false, "higher"),
    { 2, 5 })
  assert_tensor(numlua.quantile(matrix, { 0.25, 0.75 }, 2, false,
    "midpoint"), { { 1.5, 4.5 }, { 2.5, 5.5 } })
  assert_tensor(numlua.percentile(matrix, 50, 2, false, "lower"),
    { 2, 5 })
end)

test("quantile rejects unknown methods", function()
  assert_error(function()
    numlua.quantile(numlua.array({ 1, 2 }), 0.5, nil, false, "cubic")
  end, "method")
  assert_error(function()
    numlua.percentile(numlua.array({ 1, 2 }), 50, nil, false, "cubic")
  end, "method")
end)
