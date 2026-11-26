--
-- fromiter constructor.
--

local numlua = require("numlua")

local function iterator(values)
  local index = 0
  return function()
    index = index + 1
    return values[index]
  end
end

test("fromiter consumes an iterator and optional count", function()
  local values = numlua.fromiter(iterator({ 1, 2, 3 }), "int64")
  assert_tensor(values, { 1, 2, 3 })
  assert_dtype(values, "int64")
  local limited = numlua.fromiter(iterator({ 1, 2, 3 }), "float64", 2)
  assert_tensor(limited, { 1, 2 })
  assert_dtype(limited, "float64")
  local empty = numlua.fromiter(iterator({}), "float32")
  assert_shape(empty, { 0 })
  assert_dtype(empty, "float32")
  assert_tensor(numlua.fromiter(iterator({ true, false }), "bool"),
    { true, false })
end)

test("fromiter validates iterator, dtype, count, and values", function()
  assert_error(function() numlua.fromiter({}, "float32") end, "iterator")
  assert_error(function() numlua.fromiter(iterator({ 1 }), 1) end, "dtype")
  assert_error(function() numlua.fromiter(iterator({ 1 }), "float32", -1)
    end, "count")
  assert_error(function() numlua.fromiter(iterator({ "x" }), "float32")
    end, "numbers")
end)
