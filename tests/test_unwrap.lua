--
-- Phase unwrapping along an axis.
--

local numlua = require("numlua")

test("unwrap removes phase jumps", function()
  local values = numlua.array({ 0, 3.5, 7, 10.5 })
  local result = numlua.unwrap(values)
  assert_dtype(result, "float32")
  assert_near(result:totable(),
    { 0, 3.5 - 2 * math.pi, 7 - 4 * math.pi, 10.5 - 6 * math.pi }, 1e-5)

  local smooth = numlua.array({ 0, 0.5, 1, 1.5 })
  assert_near(numlua.unwrap(smooth):totable(), smooth:totable(), 0)
end)

test("unwrap accepts a custom period and discont", function()
  assert_near(numlua.unwrap(numlua.array({ 0, 200, 400 }), nil, nil, 360)
    :totable(), { 0, -160, -320 }, 1e-5)
  assert_near(numlua.unwrap(numlua.array({ 0, 3, 6 }), 5):totable(),
    { 0, 3, 6 }, 1e-5)
  local values = numlua.array({ 0, 3.5, 7 })
  assert_near(numlua.unwrap(values, 5):totable(), values:totable(), 1e-6)
end)

test("unwrap works along an axis and promotes integer input", function()
  local matrix = numlua.array({
    { 0, 3.5, 7 },
    { 0, 1, 2 },
  })
  local rows = numlua.unwrap(matrix, nil, 2)
  assert_shape(rows, { 2, 3 })
  assert_near(rows:get(1, 2), 3.5 - 2 * math.pi, 1e-5)
  assert_near(rows:get(2, 3), 2, 1e-6)
  assert_near(numlua.unwrap(matrix, nil, -1):totable(), rows:totable(), 1e-6)

  local columns = numlua.unwrap(matrix, nil, 1)
  assert_shape(columns, { 2, 3 })
  assert_near(columns:get(1, 2), 3.5, 1e-6)

  local ints = numlua.unwrap(numlua.array({ 0, 1, 2 }, { dtype = "int64" }))
  assert_dtype(ints, "float64")
  assert_near(ints:totable(), { 0, 1, 2 }, 0)
end)

test("unwrap returns independent storage and validates arguments", function()
  local values = numlua.array({ 0, 3.5, 7 })
  local result = numlua.unwrap(values)
  result:set(1, 99)
  assert_eq(values:get(1), 0)

  assert_error(function() numlua.unwrap(numlua.array(1)) end,
    "at least a 1-D")
  assert_error(function() numlua.unwrap({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.unwrap(numlua.zeros(2), nil, 0) end,
    "1-based")
  assert_error(function() numlua.unwrap(numlua.zeros(2), nil, 2) end,
    "out of range")
  assert_error(function() numlua.unwrap(numlua.zeros(2), -1) end,
    "discont")
  assert_error(function() numlua.unwrap(numlua.zeros(2), nil, nil, 0) end,
    "period")
  assert_error(function() numlua.unwrap(numlua.zeros(2), nil, nil,
    math.huge) end, "period")
end)
