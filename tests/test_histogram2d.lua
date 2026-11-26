--
-- Two-dimensional histogram.
--

local numlua = require("numlua")

test("histogram2d counts points and returns edges", function()
  local x = numlua.array({ 0.5, 1.5, 2.5 })
  local y = numlua.array({ 0.5, 1.5, 2.5 })
  local counts, xedges, yedges = numlua.histogram2d(x, y, {
    bins = 2,
    range = { { 0, 2 }, { 0, 2 } },
  })
  assert_tensor(counts, { { 1, 0 }, { 0, 1 } })
  assert_tensor(xedges, { 0, 1, 2 })
  assert_tensor(yedges, { 0, 1, 2 })
end)

test("histogram2d supports explicit edges, distinct bins, and weights", function()
  local x = numlua.array({ 0.5, 1.5, 2.5 })
  local y = numlua.array({ 0.5, 1.5, 2.5 })
  local edges = numlua.array({ 0, 1, 2, 3 })
  local counts = numlua.histogram2d(x, y, { bins = edges })
  assert_tensor(counts, {
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
  })
  local weighted = numlua.histogram2d(x, y, {
    bins = 2,
    range = { { 0, 2 }, { 0, 2 } },
    weights = { 2, 3, 4 },
  })
  assert_tensor(weighted, { { 2, 0 }, { 0, 3 } })
end)

test("histogram2d density normalizes by area", function()
  local counts = numlua.histogram2d(
    numlua.array({ 0.5 }), numlua.array({ 0.5 }), {
      bins = 2,
      range = { { 0, 2 }, { 0, 2 } },
      density = true,
    })
  assert_tensor(counts, { { 1, 0 }, { 0, 0 } })
end)

test("histogram2d validates samples, bins, and ranges", function()
  local x = numlua.array({ 0.5, 1.5 })
  local y = numlua.array({ 0.5 })
  assert_error(function() numlua.histogram2d(x, y) end, "same length")
  assert_error(function() numlua.histogram2d(x, numlua.zeros(2, 2)) end,
    "1-D")
  assert_error(function() numlua.histogram2d(x, numlua.zeros(2), {
    bins = { 0, 1, 2 },
    range = { { 0, 1 }, { 0, 1 } },
  }) end, "range")
  assert_error(function() numlua.histogram2d(x, numlua.zeros(2), {
    bins = { 0, 1, 2 },
    weights = { 1 },
  }) end, "weights")
end)
