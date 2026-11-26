--
-- Angle conversions and sinc.
--

local numlua = require("numlua")

test("deg2rad/rad2deg and aliases convert angles", function()
  assert_tensor(numlua.deg2rad(numlua.array({ 0, 180 })),
    { 0, math.pi }, 1e-6)
  assert_tensor(numlua.rad2deg(numlua.array({ 0, math.pi })),
    { 0, 180 }, 1e-6)
  assert_tensor(numlua.radians(numlua.array({ 90 })), { math.pi / 2 }, 1e-6)
  assert_tensor(numlua.degrees(numlua.array({ math.pi / 2 })), { 90 }, 1e-6)
  assert_dtype(numlua.deg2rad(numlua.array({ 180 }, { dtype = "int64" })),
    "float64")
end)

test("sinc handles zero and ordinary points", function()
  local values = numlua.sinc(numlua.array({ 0, 1, 2 }))
  assert_near(values:get(1), 1, 1e-6)
  assert_near(values:get(2), 0, 1e-6)
  assert_near(values:get(3), 0, 1e-6)
  assert_dtype(numlua.sinc(numlua.array({ 1 }, { dtype = "int64" })),
    "float64")
  assert_error(function() numlua.sinc({ 1 }) end, "expects a tensor")
  assert_error(function() numlua.deg2rad({ 1 }) end, "expects a tensor")
end)
