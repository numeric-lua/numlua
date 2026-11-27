--
-- ndenumerate iterator.
--

local numlua = require("numlua")

test("ndenumerate yields coordinates and values", function()
  local iterator = numlua.ndenumerate(numlua.array({ { 10, 20 },
    { 30, 40 } }))
  local values = {}
  while true do
    local coordinates, value = iterator()
    if coordinates == nil then break end
    values[#values + 1] = table.concat(coordinates, ",") .. "=" .. tostring(value)
  end
  assert_eq(table.concat(values, ";"), "1,1=10;1,2=20;2,1=30;2,2=40")
end)

test("ndenumerate handles 0-D and validates input", function()
  local iterator = numlua.ndenumerate(numlua.array(7))
  local coordinates, value = iterator()
  assert_eq(#coordinates, 0)
  assert_eq(value, 7)
  assert_eq(iterator(), nil)
  assert_error(function() numlua.ndenumerate({ 1, 2 }) end,
    "expects a tensor")
end)
