--
-- copyto and putmask mutation helpers.
--

local numlua = require("numlua")

test("copyto broadcasts and casts into the destination", function()
  local destination = numlua.zeros(2, 2)
  local source = numlua.array({ { 1, 2 } })
  local result = numlua.copyto(destination, source)
  assert_true(result == destination)
  assert_tensor(destination, { { 1, 2 }, { 1, 2 } })

  local ints = numlua.zeros(3, { dtype = "int64" })
  numlua.copyto(ints, numlua.array({ 1.9, 2.1, 3.5 }))
  assert_dtype(ints, "int64")
  assert_tensor(ints, { 1, 2, 3 })
  assert_error(function() numlua.copyto({ 1 }, numlua.zeros(1)) end,
    "expects a tensor")
end)

test("putmask writes selected elements", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  local mask = values:gt(2):logical_or(values:eq(1))
  local result = numlua.putmask(values, mask, 9)
  assert_true(result == values)
  assert_tensor(values, { { 9, 2 }, { 9, 9 } })
  numlua.putmask(values, numlua.array({ { false, true }, { false, false } }),
    numlua.array({ { 0, 7 }, { 0, 0 } }))
  assert_tensor(values, { { 9, 7 }, { 9, 9 } })
  assert_error(function() numlua.putmask({ 1 }, mask, 0) end,
    "expects a tensor")
end)
