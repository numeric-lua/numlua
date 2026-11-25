--
-- Public broadcast_shapes() shape arithmetic.
--

local numlua = require("numlua")

test("broadcast_shapes follows right-aligned broadcasting", function()
  assert_eq(numlua.broadcast_shapes({ 4, 1, 3 }, { 1, 5, 1 }, { 4, 5, 1 }),
    { 4, 5, 3 })
  assert_eq(numlua.broadcast_shapes({ 4, 5, 1 }, { 1, 5, 1 }, { 4, 1, 3 }),
    { 4, 5, 3 })
  assert_eq(numlua.broadcast_shapes({ 5 }, { 2, 3, 1 }), { 2, 3, 5 })
  assert_eq(numlua.broadcast_shapes({}, { 2, 3 }), { 2, 3 })
  assert_eq(numlua.broadcast_shapes({}), {})
  assert_eq(numlua.broadcast_shapes(), {})
end)

test("broadcast_shapes handles zero-size axes", function()
  assert_eq(numlua.broadcast_shapes({ 0, 1 }, { 1, 3 }), { 0, 3 })
  assert_eq(numlua.broadcast_shapes({ 0, 4 }, { 1, 4 }), { 0, 4 })
  assert_error(function()
    numlua.broadcast_shapes({ 0, 2 }, { 0, 3 })
  end, "incompatible shapes")
end)

test("broadcast_shapes rejects malformed shapes", function()
  assert_error(function() numlua.broadcast_shapes(2, { 3 }) end,
    "must be a shape table")
  assert_error(function() numlua.broadcast_shapes({ 2 }, { -1 }) end,
    "non-negative integers")
  assert_error(function() numlua.broadcast_shapes({ 2.5 }) end,
    "non-negative integers")
  assert_error(function() numlua.broadcast_shapes({ "2" }) end,
    "non-negative integers")
  assert_error(function() numlua.broadcast_shapes({ 2 }, { 3 }) end,
    "incompatible shapes")
end)
