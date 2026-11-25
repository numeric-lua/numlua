--
-- piecewise conditional evaluation.
--

local numlua = require("numlua")

test("piecewise evaluates callbacks per condition", function()
  local x = numlua.array({ -2, -1, 0, 1, 2 })
  local result = numlua.piecewise(x, { x:lt(0), x:gt(0) }, {
    function(value) return numlua.negative(value) end,
    function(value) return value:mul(2) end,
  }, 0)
  assert_tensor(result, { 2, 1, 0, 2, 4 })
end)

test("piecewise supports constants, defaults, and extra arguments", function()
  local x = numlua.array({ -2, -1, 0, 1, 2 })
  assert_tensor(numlua.piecewise(x, { x:lt(0), x:gt(0) },
    { -1, 1 }), { -1, -1, 0, 1, 1 })
  assert_tensor(numlua.piecewise(x, { x:lt(0) },
    { function(value, scale) return value:mul(scale) end, 9 }, 3),
    { -6, -3, 9, 9, 9 })
  assert_tensor(numlua.piecewise(x, { x:ge(0) },
    { function(value) return value end }), { 0, 0, 0, 1, 2 })
end)

test("piecewise broadcasts scalar conditions and validates lists", function()
  local x = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.piecewise(x, { true }, { 5 }), { 5, 5, 5 })
  assert_error(function() numlua.piecewise({ 1, 2 }, { true }, { 1 }) end,
    "expects a tensor")
  assert_error(function() numlua.piecewise(x, {}, { 1 }) end,
    "non-empty condition")
  assert_error(function() numlua.piecewise(x, { true }, {}) end,
    "non-empty function")
  assert_error(function() numlua.piecewise(x, { true, false }, { 1 })
    end, "funclist")
  assert_error(function() numlua.piecewise(x, { true },
    { "bad" }) end, "choice")
end)
