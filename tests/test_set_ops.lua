--
-- Set operations.
--

local numlua = require("numlua")

test("intersect1d/union1d/setdiff1d/setxor1d return sorted results", function()
  assert_tensor(numlua.intersect1d({ 3, 1, 2, 2 }, { 2, 3, 4 }),
    { 2, 3 })
  assert_tensor(numlua.union1d({ 3, 1, 2 }, { 2, 4 }), { 1, 2, 3, 4 })
  assert_tensor(numlua.setdiff1d({ 1, 2, 3, 4 }, { 2, 4 }), { 1, 3 })
  assert_tensor(numlua.setxor1d({ 1, 2, 3 }, { 2, 3, 4 }), { 1, 4 })
  assert_tensor(numlua.intersect1d(numlua.array({ 1.5, 2.5 }),
    numlua.array({ 2.5, 3.5 })), { 2.5 })
  assert_dtype(numlua.union1d({ 1 }, { 2 }), "float32")
end)

test("set operations handle empty and disjoint inputs", function()
  assert_shape(numlua.intersect1d({ 1, 2 }, { 3, 4 }), { 0 })
  assert_tensor(numlua.union1d({}, { 1, 2 }), { 1, 2 })
  assert_tensor(numlua.setdiff1d({ 1, 2 }, { 1, 2 }), {})
  assert_tensor(numlua.setxor1d({ 1, 2 }, { 1, 2 }), {})
end)

test("set operations validate inputs and flags", function()
  assert_error(function() numlua.intersect1d({ 1 }, 2) end, "tensor or table")
  assert_error(function() numlua.intersect1d({ 1 }, { 1 }, 1) end,
    "assume_unique")
  assert_error(function() numlua.setdiff1d({ 1 }, { 1 }, "yes") end,
    "assume_unique")
  assert_error(function() numlua.setxor1d({ 1 }, { 1 }, 1) end,
    "assume_unique")
end)
