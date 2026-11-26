--
-- Polynomial arithmetic.
--

local numlua = require("numlua")

test("polyadd and polysub align descending coefficients", function()
  assert_tensor(numlua.polyadd({ 1, 2 }, { 1 }), { 1, 3 })
  assert_tensor(numlua.polysub({ 1, 2 }, { 1 }), { 1, 1 })
  assert_tensor(numlua.polyadd({ 1, 1 }, { -1, -1 }), { 0 })
  assert_tensor(numlua.polyadd(numlua.array({ 2, 3 }), { 4 }),
    { 2, 7 })
  assert_dtype(numlua.polyadd({ 1 }, { 2 }), "float64")
end)

test("polymul convolves coefficients", function()
  assert_tensor(numlua.polymul({ 1, 1 }, { 1, -1 }), { 1, 0, -1 })
  assert_tensor(numlua.polymul({ 2, 3 }, { 4 }), { 8, 12 })
  assert_tensor(numlua.polymul({ 0 }, { 1, 2 }), { 0 })
end)

test("polydiv returns quotient and remainder", function()
  local quotient, remainder = numlua.polydiv({ 1, 0, -1 }, { 1, -1 })
  assert_tensor(quotient, { 1, 1 })
  assert_tensor(remainder, { 0 })
  quotient, remainder = numlua.polydiv({ 1, 0, 0 }, { 1, 1 })
  assert_tensor(quotient, { 1, -1 })
  assert_tensor(remainder, { 1 })
  quotient, remainder = numlua.polydiv({ 1, 2 }, { 1, 0, 1 })
  assert_tensor(quotient, { 0 })
  assert_tensor(remainder, { 1, 2 })
  quotient, remainder = numlua.polydiv({ 1, 2 }, { 0, 1 })
  assert_tensor(quotient, { 1, 2 })
  assert_tensor(remainder, { 0 })
end)

test("polynomial arithmetic validates inputs", function()
  assert_error(function() numlua.polyadd({}, { 1 }) end, "coefficient")
  assert_error(function() numlua.polymul(numlua.zeros(2, 2), { 1 }) end,
    "1-D")
  assert_error(function() numlua.polydiv({ 1 }, { 0 }) end, "zero")
  assert_error(function() numlua.polysub({ 1 }, 2) end, "tensor or table")
end)
