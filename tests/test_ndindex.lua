--
-- ndindex iterator.
--

local numlua = require("numlua")

local function collect(iterator)
  local values = {}
  while true do
    local coordinates = iterator()
    if coordinates == nil then break end
    values[#values + 1] = table.concat(coordinates, ",")
  end
  return table.concat(values, ";")
end

test("ndindex enumerates C-order coordinates", function()
  assert_eq(collect(numlua.ndindex({ 2, 3 })),
    "1,1;1,2;1,3;2,1;2,2;2,3")
  assert_eq(collect(numlua.ndindex(2, 2)), "1,1;1,2;2,1;2,2")
  assert_eq(collect(numlua.ndindex({ 2, 0 })), "")
end)

test("ndindex validates dimensions", function()
  assert_error(function() numlua.ndindex() end, "one or more")
  assert_error(function() numlua.ndindex({}) end, "one or more")
  assert_error(function() numlua.ndindex({ 2, -1 }) end, "non-negative")
  assert_error(function() numlua.ndindex({ 2, 1.5 }) end, "integers")
end)
