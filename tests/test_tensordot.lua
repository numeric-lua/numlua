--
-- tensordot contractions.
--

local numlua = require("numlua")

test("tensordot contracts trailing/leading axes", function()
  local a = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local b = numlua.array({ { 1, 2 }, { 3, 4 }, { 5, 6 } })
  assert_tensor(numlua.tensordot(a, b, 1), a:matmul(b):totable())
  assert_tensor(numlua.tensordot(
    numlua.array({ 1, 2 }), numlua.array({ 3, 4 }), 0),
    { { 3, 4 }, { 6, 8 } })
end)

test("tensordot supports N-D and explicit axis pairs", function()
  local a = numlua.array({ { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } } })
  local b = numlua.array({ { 1, 0 }, { 0, 1 } })
  local result = numlua.tensordot(a, b, 1)
  assert_shape(result, { 2, 2, 2 })
  assert_tensor(result, { { { 1, 2 }, { 3, 4 } },
    { { 5, 6 }, { 7, 8 } } })

  local c = numlua.array({ { { 1, 2 }, { 3, 4 }, { 5, 6 } },
    { { 7, 8 }, { 9, 10 }, { 11, 12 } } })
  local d = numlua.array({ { 1, 2, 3, 4 }, { 5, 6, 7, 8 },
    { 9, 10, 11, 12 } })
  local paired = numlua.tensordot(c, d, { { 2 }, { 1 } })
  assert_shape(paired, { 2, 2, 4 })
end)

test("tensordot validates axes and dimensions", function()
  local a = numlua.array({ { 1, 2 } })
  local b = numlua.array({ { 1, 2 } })
  assert_error(function() numlua.tensordot(a, b, 3) end, "too many")
  assert_error(function() numlua.tensordot(a, b, 1) end, "differ")
  assert_error(function() numlua.tensordot(a, b, { { 1, 1 }, { 1 } }) end,
    "duplicate")
  assert_error(function() numlua.tensordot(a, b, { { 1 }, { 1, 2 } }) end,
    "same length")
  assert_error(function() numlua.tensordot(a, b, -1) end, "non-negative")
end)
