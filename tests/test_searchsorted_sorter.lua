--
-- searchsorted sorter support.
--

local numlua = require("numlua")

test("searchsorted uses a sorter permutation", function()
  local unsorted = numlua.array({ 3, 1, 2 })
  local sorter = numlua.argsort(unsorted)
  assert_tensor(sorter, { 2, 3, 1 })
  local values = numlua.array({ 0, 1, 2, 3, 4 })
  assert_tensor(numlua.searchsorted(unsorted, values, "left", sorter),
    { 1, 1, 2, 3, 4 })
  assert_tensor(numlua.searchsorted(unsorted, values, "right", sorter),
    { 1, 2, 3, 4, 4 })
  assert_tensor(numlua.searchsorted(unsorted, { 1, 2 }, "left", { 2, 3, 1 }),
    { 1, 2 })
  assert_eq(numlua.searchsorted(unsorted, 2, "left", sorter), 2)
end)

test("searchsorted validates sorter shape and range", function()
  local values = numlua.array({ 3, 1, 2 })
  assert_error(function()
    numlua.searchsorted(values, 1, "left", numlua.array({ 1.5, 2, 3 }))
  end, "integer")
  assert_error(function()
    numlua.searchsorted(values, 1, "left",
      numlua.array({ 1, 2 }, { dtype = "int64" }))
  end, "match")
  assert_error(function()
    numlua.searchsorted(values, 1, "left",
      numlua.array({ 1, 2, 4 }, { dtype = "int64" }))
  end, "out of range")
  assert_error(function()
    numlua.searchsorted(values, 1, "left", "bad")
  end, "integer tensor")
end)
