--
-- trim_zeros and extract conveniences.
--

local numlua = require("numlua")

test("trim_zeros removes selected edges", function()
  local values = numlua.array({ 0, 0, 1, 2, 0, 3, 0 })
  assert_tensor(numlua.trim_zeros(values), { 1, 2, 0, 3 })
  assert_tensor(numlua.trim_zeros(values, "f"), { 1, 2, 0, 3, 0 })
  assert_tensor(numlua.trim_zeros(values, "b"), { 0, 0, 1, 2, 0, 3 })
  assert_tensor(numlua.trim_zeros(values, "bf"), { 1, 2, 0, 3 })
  assert_shape(numlua.trim_zeros(numlua.zeros(3)), { 0 })
  assert_tensor(numlua.trim_zeros(numlua.array({ 1, 2, 3 })), { 1, 2, 3 })
  assert_tensor(numlua.trim_zeros(numlua.array({ false, true, false })),
    { true })
end)

test("trim_zeros validates shape and trim mode", function()
  assert_error(function() numlua.trim_zeros({ 0, 1 }) end, "expects a tensor")
  assert_error(function() numlua.trim_zeros(numlua.zeros(2, 2)) end, "1-D")
  assert_error(function() numlua.trim_zeros(numlua.zeros(2), "x") end,
    "trim")
end)

test("extract compresses the flattened input", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.extract(
    numlua.array({ true, false, true }, { dtype = "bool" }), values),
    { 1, 3 })
  assert_tensor(numlua.extract({ 1, 0, 0, 1 }, values), { 1, 4 })
  assert_error(function()
    numlua.extract(numlua.array({ true, false, true, false, true },
      { dtype = "bool" }), values)
  end, "condition has 5")
  assert_error(function() numlua.extract({ 1 }, { 1, 2 }) end,
    "expects a tensor")
end)
