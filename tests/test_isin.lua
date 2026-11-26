--
-- isin/in1d boolean membership.
--

local numlua = require("numlua")

test("isin preserves input shape and supports inversion", function()
  local element = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.isin(element, { 2, 4 }),
    { { false, true }, { false, true } })
  assert_tensor(numlua.isin(element, { 2, 4 }, true),
    { { true, false }, { true, false } })
  local scalar = numlua.isin(numlua.array(2), { 2, 3 })
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), true)
  assert_tensor(numlua.isin(numlua.zeros(2), {}),
    { false, false })
end)

test("in1d flattens and accepts tables", function()
  assert_tensor(numlua.in1d({ { 1, 2 }, { 3, 4 } }, { 2, 4 }),
    { false, true, false, true })
  assert_tensor(numlua.in1d({ 1, 2, 3 }, { 1, 3 }, true),
    { false, true, false })
end)

test("isin validates inputs", function()
  assert_error(function() numlua.isin({ 1, 2 }, { 1 }) end,
    "expects a tensor")
  assert_error(function() numlua.isin(numlua.array({ 1 }), { 1 }, 1)
    end, "invert")
end)
