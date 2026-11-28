--
-- copy/ascontiguousarray helpers.
--

local numlua = require("numlua")

test("copy returns an independent tensor", function()
  local source = numlua.array({ { 1, 2 }, { 3, 4 } })
  local result = numlua.copy(source)
  assert_tensor(result, source:totable())
  assert_true(result ~= source)
  result:set(1, 1, 99)
  assert_eq(source:get(1, 1), 1)
  assert_error(function() numlua.copy({ 1, 2 }) end, "expects a tensor")
end)

test("ascontiguousarray makes transposed views contiguous", function()
  local source = numlua.array({ { 1, 2 }, { 3, 4 } })
  local view = source:t()
  local contiguous = numlua.ascontiguousarray(view)
  assert_true(contiguous:is_contiguous())
  assert_tensor(contiguous, { { 1, 3 }, { 2, 4 } })
  assert_error(function() numlua.ascontiguousarray({ 1 }) end,
    "expects a tensor")
end)
