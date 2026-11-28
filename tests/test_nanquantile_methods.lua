--
-- NaN-aware quantile interpolation methods.
--

local numlua = require("numlua")

test("nanquantile supports interpolation methods", function()
  local values = numlua.array({ 1, 0 / 0, 3, 4 })
  assert_eq(numlua.nanquantile(values, 0.25, nil, false, "lower"):item(), 1)
  assert_eq(numlua.nanquantile(values, 0.25, nil, false, "higher"):item(), 3)
  assert_eq(numlua.nanquantile(values, 0.25, nil, false, "midpoint"):item(), 2)
  assert_eq(numlua.nanquantile(values, 0.25, nil, false, "nearest"):item(), 3)
  assert_eq(numlua.nanquantile(values, 0.75, nil, false, "midpoint"):item(), 3.5)
  assert_eq(numlua.nanpercentile(values, 25, nil, false, "higher"):item(), 3)
end)

test("nanquantile methods work along axes and q tables", function()
  local matrix = numlua.array({ { 1, 0 / 0, 3 }, { 4, 5, 6 } })
  assert_tensor(numlua.nanquantile(matrix, 0.5, 2, false, "lower"),
    { 1, 5 })
  assert_tensor(numlua.nanquantile(matrix, { 0.25, 0.75 }, 2, false,
    "midpoint"), { { 2, 4.5 }, { 2, 5.5 } })
  assert_error(function()
    numlua.nanquantile(matrix, 0.5, 2, false, "cubic")
  end, "method")
end)
