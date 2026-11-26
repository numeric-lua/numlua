--
-- fromstring text parsing.
--

local numlua = require("numlua")

test("fromstring parses separated and multiline text", function()
  assert_tensor(numlua.fromstring("1 2 3", "int64", nil, " "),
    { 1, 2, 3 })
  assert_tensor(numlua.fromstring("1, 2,3", "float64", nil, ","),
    { 1, 2, 3 })
  assert_tensor(numlua.fromstring("1\n2\n3", "float32", nil, " "),
    { 1, 2, 3 })
  assert_tensor(numlua.fromstring("1 2 3 4", "int64", 2, " "),
    { 1, 2 })
  assert_tensor(numlua.fromstring("true false 1 0", "bool", nil, " "),
    { true, false, true, false })
  assert_dtype(numlua.fromstring("1 2", nil, nil, " "), "float64")
end)

test("fromstring parses special numeric tokens", function()
  local values = numlua.fromstring("nan inf -inf", "float64", nil, " ")
  assert_true(values:get(1) ~= values:get(1))
  assert_eq(values:get(2), math.huge)
  assert_eq(values:get(3), -math.huge)
end)

test("fromstring validates separator, dtype, count, and fields", function()
  assert_error(function() numlua.fromstring("1 2", "float64", nil, "")
    end, "separator")
  assert_error(function() numlua.fromstring("1 2", "nope", nil, " ")
    end, "dtype")
  assert_error(function() numlua.fromstring("1 2", "float64", -2, " ")
    end, "count")
  assert_error(function() numlua.fromstring("1 x", "float64", nil, " ")
    end, "cannot parse")
  assert_error(function() numlua.fromstring(1, "float64", nil, " ")
    end, "string")
end)
