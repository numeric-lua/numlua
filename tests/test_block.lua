--
-- 2-D block assembly.
--

local numlua = require("numlua")

test("block assembles nested 2-D tiles", function()
  local a = numlua.array({ { 1, 2 }, { 3, 4 } })
  local b = numlua.array({ { 5 }, { 6 } })
  local c = numlua.array({ { 7, 8 } })
  local d = numlua.array({ { 9 } })
  assert_tensor(numlua.block({ { a, b }, { c, d } }), {
    { 1, 2, 5 },
    { 3, 4, 6 },
    { 7, 8, 9 },
  })
end)

test("block accepts a single row and varies row heights", function()
  local a = numlua.array({ { 1 } })
  local b = numlua.array({ { 2 } })
  assert_tensor(numlua.block({ a, b }), { { 1, 2 } })
  assert_tensor(numlua.block({ { a }, { a }, { b } }),
    { { 1 }, { 1 }, { 2 } })
end)

test("block validates nesting and widths", function()
  local a = numlua.array({ { 1, 2 } })
  assert_error(function() numlua.block({}) end, "non-empty")
  assert_error(function() numlua.block({ {} }) end, "block rows")
  assert_error(function() numlua.block({ { a, a }, { a } }) end, "total width")
  assert_error(function() numlua.block({ { numlua.zeros(2) } }) end,
    "2-D")
  assert_error(function() numlua.block({ { { 1 } } }) end, "2-D")
end)
