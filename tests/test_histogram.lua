--
-- Histogram counts and bin edges.
--

local numlua = require("numlua")

test("histogram_bin_edges builds integer bins over the input range", function()
  local edges = numlua.histogram_bin_edges(
    numlua.array({ 0, 1, 2, 3, 4 }))
  assert_shape(edges, { 11 })
  assert_dtype(edges, "float64")
  assert_near(edges:get(1), 0, 0)
  assert_near(edges:get(11), 4, 0)
  assert_near(edges:get(2), 0.4, 1e-12)

  local explicit = numlua.histogram_bin_edges(
    numlua.array({ 5 }), { bins = { 0, 1, 2, 3 } })
  assert_near(explicit:totable(), { 0, 1, 2, 3 }, 0)
  local range_edges = numlua.histogram_bin_edges(
    numlua.zeros(0), { bins = 2, range = { -1, 1 } })
  assert_near(range_edges:totable(), { -1, 0, 1 }, 0)
end)

test("histogram counts values with left-closed, right-closed-last bins", function()
  local counts, edges = numlua.histogram(numlua.array({ 0, 1, 2, 3, 4 }),
    { bins = 2 })
  assert_dtype(counts, "int64")
  assert_near(edges:totable(), { 0, 2, 4 }, 0)
  assert_tensor(counts, { 2, 3 })

  local explicit_counts = numlua.histogram(
    numlua.array({ 0, 1, 2, 3 }), { bins = { 0, 1, 2, 3 } })
  assert_tensor(explicit_counts, { 1, 1, 2 })

  local trimmed = numlua.histogram(
    numlua.array({ -1, 0, 1, 2, 3, 4 }), { bins = { 0, 1, 2, 3 } })
  assert_tensor(trimmed, { 1, 1, 2 })
end)

test("histogram supports weights and density", function()
  local counts, edges = numlua.histogram(
    numlua.array({ 0, 1, 2, 3 }),
    { bins = { 0, 2, 4 }, weights = { 1, 2, 3, 4 } })
  assert_near(edges:totable(), { 0, 2, 4 }, 0)
  assert_near(counts:totable(), { 3, 7 }, 0)
  assert_dtype(counts, "float64")

  local density = numlua.histogram(
    numlua.array({ 0, 1, 2, 3 }), { bins = { 0, 2, 4 }, density = true })
  assert_dtype(density, "float64")
  assert_near(density:totable(), { 0.25, 0.25 }, 1e-12)
  assert_near(density:sum():item() * 2, 1, 1e-12)
end)

test("histogram flattens N-D input and handles empty ranges", function()
  local counts = numlua.histogram(
    numlua.array({ { 0, 1 }, { 2, 3 } }), { bins = { 0, 2, 4 } })
  assert_tensor(counts, { 2, 2 })
  local empty = numlua.histogram(numlua.zeros(0),
    { bins = 2, range = { 0, 2 } })
  assert_tensor(empty, { 0, 0 })
end)

test("histogram helpers validate bins, ranges, and weights", function()
  local values = numlua.array({ 1, 2, 3 })
  assert_error(function() numlua.histogram_bin_edges(values, { bins = 0 }) end,
    "positive integer")
  assert_error(function() numlua.histogram_bin_edges(values, { bins = 2.5 }) end,
    "positive integer")
  assert_error(function() numlua.histogram_bin_edges(values,
    { bins = { 0, 0, 1 } }) end, "strictly increasing")
  assert_error(function() numlua.histogram_bin_edges(values,
    { bins = { 0 } }) end, "at least two edges")
  assert_error(function() numlua.histogram_bin_edges(values,
    { bins = { 0, 1 }, range = { 0, 1 } }) end, "cannot be combined")
  assert_error(function() numlua.histogram_bin_edges(values,
    { bins = 2, range = { 2, 1 } }) end, "minimum exceeds")
  assert_error(function() numlua.histogram_bin_edges(numlua.zeros(0)) end,
    "cannot infer")
  assert_error(function() numlua.histogram(values, { density = 1 }) end,
    "density")
  assert_error(function() numlua.histogram(values,
    { weights = { 1, 2 } }) end, "weights must match")
  assert_error(function() numlua.histogram(values, { unknown = true }) end,
    "unknown option")
  assert_error(function() numlua.histogram(values,
    { bins = { 0, 2 }, weights = { 1, -1, 0 }, density = true }) end,
    "non-zero total weight")
end)
