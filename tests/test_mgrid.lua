--
-- mgrid/ogrid coordinate grids.
--

local numlua = require("numlua")

test("mgrid returns dense coordinate grids", function()
  local x, y = numlua.mgrid({ 0, 3 }, { 0, 2 })
  assert_shape(x, { 3, 2 })
  assert_shape(y, { 3, 2 })
  assert_tensor(x, { { 0, 0 }, { 1, 1 }, { 2, 2 } })
  assert_tensor(y, { { 0, 1 }, { 0, 1 }, { 0, 1 } })

  local a, b = numlua.mgrid({ { 0, 2 }, { 0, 3 } })
  assert_shape(a, { 2, 3 })
  assert_shape(b, { 2, 3 })
end)

test("ogrid returns broadcastable coordinate grids", function()
  local x, y = numlua.ogrid({ 0, 3 }, { 0, 2 })
  assert_shape(x, { 3, 1 })
  assert_shape(y, { 1, 2 })
  assert_tensor(x, { { 0 }, { 1 }, { 2 } })
  assert_tensor(y, { { 0, 1 } })
  local z = numlua.ogrid({ 0, 1, 0.5 })
  assert_shape(z, { 2 })
  assert_tensor(z, { 0, 0.5 })
end)

test("mgrid/ogrid validate slices", function()
  assert_error(function() numlua.mgrid() end, "at least one")
  assert_error(function() numlua.mgrid({ 0 }) end, "slice")
  assert_error(function() numlua.mgrid({ 0, 1, 0 }) end, "non-zero")
  assert_error(function() numlua.ogrid({ "a", 1 }) end, "numbers")
end)
