--
-- rollaxis and msort compatibility helpers.
--

local numlua = require("numlua")

test("rollaxis moves one axis to a 1-based start", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  assert_shape(numlua.rollaxis(cube, 2), { 3, 2, 4 })
  assert_shape(numlua.rollaxis(cube, 1, 3), { 3, 4, 2 })
  assert_shape(numlua.rollaxis(cube, 3, 1), { 4, 2, 3 })
  assert_shape(numlua.rollaxis(cube, -1, 2), { 2, 4, 3 })
  assert_shape(numlua.rollaxis(cube, 2, 4), { 2, 4, 3 })
  assert_eq(numlua.rollaxis(cube, 2, 2):get(2, 1, 3), cube:get(2, 1, 3))
end)

test("rollaxis returns a view", function()
  local cube = numlua.arange(8):reshape(2, 2, 2)
  local rolled = numlua.rollaxis(cube, 2, 1)
  rolled:set(1, 1, 1, 99)
  assert_eq(cube:get(1, 1, 1), 99)
end)

test("msort sorts along the first axis", function()
  local matrix = numlua.array({ { 3, 1 }, { 2, 4 } })
  assert_tensor(numlua.msort(matrix), { { 2, 1 }, { 3, 4 } })
  assert_error(function() numlua.msort(numlua.array(1)) end,
    "at least a 1-D")
end)

test("rollaxis validates axes and start", function()
  local cube = numlua.arange(24):reshape(2, 3, 4)
  assert_error(function() numlua.rollaxis(cube, 0) end, "axis")
  assert_error(function() numlua.rollaxis(cube, 4) end, "out of range")
  assert_error(function() numlua.rollaxis(cube, 1, 0) end, "start")
  assert_error(function() numlua.rollaxis(cube, 1, 5) end, "start")
  assert_error(function() numlua.rollaxis({ 1, 2 }, 1) end, "expects a tensor")
  assert_error(function() numlua.rollaxis(numlua.array(1), 1) end,
    "out of range")
end)
