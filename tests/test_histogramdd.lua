--
-- N-dimensional histogram.
--

local numlua = require("numlua")

test("histogramdd counts 2-D samples", function()
  local samples = numlua.array({
    { 0.5, 0.5 },
    { 1.5, 1.5 },
    { 2.5, 2.5 },
  })
  local counts, edges = numlua.histogramdd(samples, {
    bins = 2,
    range = { { 0, 2 }, { 0, 2 } },
  })
  assert_tensor(counts, { { 1, 0 }, { 0, 1 } })
  assert_eq(#edges, 2)
  assert_tensor(edges[1], { 0, 1, 2 })
  assert_tensor(edges[2], { 0, 1, 2 })
end)

test("histogramdd accepts column tables, weights, and density", function()
  local counts = numlua.histogramdd({
    numlua.array({ 0.5, 1.5 }),
    numlua.array({ 0.5, 1.5 }),
  }, { bins = { 2, 2 }, range = { { 0, 2 }, { 0, 2 } } })
  assert_tensor(counts, { { 1, 0 }, { 0, 1 } })

  local weighted = numlua.histogramdd({
    { 0.5, 0.5 },
    { 1.5, 1.5 },
  }, { bins = 2, range = { { 0, 2 }, { 0, 2 } }, weights = { 2, 3 } })
  assert_tensor(weighted, { { 2, 0 }, { 0, 3 } })

  local dense = numlua.histogramdd(numlua.array({ { 0.5, 0.5 } }), {
    bins = 2,
    range = { { 0, 2 }, { 0, 2 } },
    density = true,
  })
  assert_tensor(dense, { { 1, 0 }, { 0, 0 } })
end)

test("histogramdd validates samples and options", function()
  assert_error(function() numlua.histogramdd(numlua.zeros(2)) end, "2-D")
  assert_error(function() numlua.histogramdd({}) end, "2-D")
  assert_error(function() numlua.histogramdd(
    numlua.array({ { 0.5, 0.5 } }), { bins = 2, range = { { 0, 1 } } })
    end, "range")
  assert_error(function() numlua.histogramdd(
    numlua.array({ { 0.5, 0.5 } }), { bins = 2, weights = { 1, 2 } })
    end, "weights")
  assert_error(function() numlua.histogramdd(
    numlua.array({ { 0.5, 0.5 } }), { bins = 0 }) end, "bins")
end)
