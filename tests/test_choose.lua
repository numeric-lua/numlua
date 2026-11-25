--
-- choose selection by integer index tensor.
--

local numlua = require("numlua")

test("choose selects from a list of choices", function()
  local a = numlua.array({ 0, 1, 0 }, { dtype = "int64" })
  assert_tensor(numlua.choose(a, { 10, 20 }), { 10, 20, 10 })

  local matrix = numlua.array({ { 0, 1 }, { 1, 0 } }, { dtype = "int64" })
  assert_tensor(numlua.choose(matrix, {
    numlua.array({ { 1, 2 }, { 3, 4 } }),
    numlua.array({ { 10, 20 }, { 30, 40 } }),
  }), { { 1, 20 }, { 30, 4 } })
  assert_tensor(numlua.choose(matrix, { 7, 9 }), { { 7, 9 }, { 9, 7 } })
end)

test("choose handles scalar indices and validates arguments", function()
  assert_eq(numlua.choose(
    numlua.array(1, { dtype = "int64" }), { 5, 6 }):item(), 6)
  assert_error(function() numlua.choose(numlua.zeros(2), { 1, 2 }) end,
    "integer")
  assert_error(function() numlua.choose(
    numlua.array({ 0, 2 }, { dtype = "int64" }), { 1, 2 }) end,
    "out of range")
  assert_error(function() numlua.choose(
    numlua.array({ -1 }, { dtype = "int64" }), { 1 }) end, "out of range")
  assert_error(function() numlua.choose(
    numlua.array({ 0 }, { dtype = "int64" }), {}) end, "non-empty")
  assert_error(function() numlua.choose({ 0 }, { 1 }) end, "expects a tensor")
end)
