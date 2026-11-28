--
-- vectorize callback wrappers.
--

local numlua = require("numlua")

test("vectorize applies a scalar callback with broadcasting", function()
  local add = numlua.vectorize(function(left, right) return left + right end,
    "float64")
  assert_tensor(add(numlua.array({ 1, 2 }), 10), { 11, 12 })
  assert_tensor(add(numlua.array({ { 1 }, { 2 } }),
    numlua.array({ { 10, 20, 30 } })), {
    { 11, 21, 31 },
    { 12, 22, 32 },
  })
  assert_eq(add(2, 3):item(), 5)
end)

test("vectorize infers dtypes and accepts scalar tensors", function()
  local positive = numlua.vectorize(function(value) return value > 0 end)
  assert_tensor(positive(numlua.array({ -1, 0, 1 })), { false, false, true })
  local as_int = numlua.vectorize(function(value) return value * 2 end,
    "int64")
  assert_tensor(as_int(numlua.array({ 1, 2 })), { 2, 4 })
  assert_dtype(as_int(numlua.array({ 1 })), "int64")
end)

test("vectorize validates functions and callbacks", function()
  assert_error(function() numlua.vectorize(1) end, "function")
  local wrapper = numlua.vectorize(function() return { 1 } end)
  assert_error(function() wrapper(1) end, "number, boolean")
  local noargs = numlua.vectorize(function(value) return value end)
  assert_error(function() noargs() end, "at least one")
  assert_error(function() noargs("x") end, "must be")
end)
