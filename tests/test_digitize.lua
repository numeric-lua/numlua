--
-- 1-based digitize bin positions.
--

local numlua = require("numlua")

test("digitize returns searchsorted positions", function()
  local bins = numlua.array({ 1, 3, 5 })
  local values = numlua.array({ 0, 1, 2, 3, 4, 5, 6 })
  assert_dtype(numlua.digitize(values, bins), "int64")
  assert_tensor(numlua.digitize(values, bins), { 1, 1, 2, 2, 3, 3, 4 })
  assert_tensor(numlua.digitize(values, bins, true),
    { 1, 2, 2, 3, 3, 4, 4 })
end)

test("digitize handles scalars, tables, and N-D shapes", function()
  local bins = numlua.array({ 1, 3, 5 })
  local scalar = numlua.digitize(2, bins)
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 2)

  local matrix = numlua.digitize(
    numlua.array({ { 0, 2 }, { 4, 6 } }), bins)
  assert_shape(matrix, { 2, 2 })
  assert_tensor(matrix, { { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.digitize({ 0, 2, 6 }, bins), { 1, 2, 4 })
end)

test("digitize validates bins, x, and right", function()
  local bins = numlua.array({ 1, 3, 5 })
  assert_error(function() numlua.digitize(1, numlua.array({ { 1, 3 } }))
    end, "1-D bins")
  assert_error(function() numlua.digitize(1, { 1, 3 }) end,
    "expects a tensor")
  assert_error(function() numlua.digitize(1, bins, 1) end, "right")
  assert_error(function() numlua.digitize(true, bins) end, "x to be")
end)
