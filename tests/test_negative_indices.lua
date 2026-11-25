--
-- Negative indices count from the end in take/take_along_axis.
--

local numlua = require("numlua")

test("take resolves negative flat and axis indices", function()
  local values = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  local function ints(...)
    return numlua.array(..., { dtype = "int64" })
  end
  assert_eq(numlua.take(values, ints(-1)):item(), 60)
  assert_tensor(numlua.take(values, ints({ -6, -1 })), { 10, 60 })
  assert_tensor(numlua.take(values, ints({ -3, -1 }), 2),
    { { 10, 30 }, { 40, 60 } })
  assert_tensor(numlua.take(values, ints({ -2, -1 }), 1),
    { { 10, 20, 30 }, { 40, 50, 60 } })
end)

test("take_along_axis resolves negative indices", function()
  local values = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  local indices = numlua.array({ { -1, -2 }, { -3, -1 } },
    { dtype = "int64" })
  assert_tensor(numlua.take_along_axis(values, indices, 2),
    { { 30, 20 }, { 40, 60 } })
  assert_tensor(numlua.take_along_axis(
    numlua.array({ 10, 20, 30 }), numlua.array({ -1, -2 },
      { dtype = "int64" })), { 30, 20 })
end)

test("negative index bounds are validated", function()
  local values = numlua.array({ 10, 20, 30 })
  local function ints(...)
    return numlua.array(..., { dtype = "int64" })
  end
  assert_eq(numlua.take(values, ints(-3)):item(), 10)
  assert_error(function() numlua.take(values, ints(-4)) end,
    "index -4 is out of range")
  assert_error(function() numlua.take(values, ints(0)) end,
    "index 0 is out of range")
  assert_error(function()
    numlua.take_along_axis(values, ints({ -4 }), 1)
  end, "index -4 is out of range")
  assert_error(function()
    numlua.take_along_axis(values, ints({ 0 }), 1)
  end, "index 0 is out of range")
end)
