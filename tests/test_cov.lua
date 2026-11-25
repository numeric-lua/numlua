--
-- Covariance and correlation matrices.
--

local numlua = require("numlua")

test("cov computes sample covariance for 1-D and 2-D data", function()
  local scalar = numlua.cov(numlua.array({ 1, 2, 3 }))
  assert_eq(scalar:dim(), 0)
  assert_near(scalar:item(), 1, 1e-8)
  assert_dtype(scalar, "float32")

  local matrix = numlua.array({
    { 1, 2, 3 },
    { 2, 4, 6 },
  })
  local result = numlua.cov(matrix)
  assert_shape(result, { 2, 2 })
  assert_near(result:totable(), { { 1, 2 }, { 2, 4 } }, 1e-8)
end)

test("cov supports rowvar=false, y observations, and ddof", function()
  local observations = numlua.array({
    { 1, 2 },
    { 2, 4 },
    { 3, 6 },
  })
  assert_near(numlua.cov(observations, nil, false):totable(),
    { { 1, 2 }, { 2, 4 } }, 1e-8)
  assert_near(numlua.cov(numlua.array({ 1, 2, 3 }),
    numlua.array({ 2, 4, 6 })):totable(),
    { { 1, 2 }, { 2, 4 } }, 1e-8)
  assert_near(numlua.cov(numlua.array({ 1, 2, 3 }), nil, true, 0):item(),
    2 / 3, 1e-6)
end)

test("cov promotes integer input and validates shapes", function()
  local ints = numlua.array({ { 1, 2, 3 }, { 2, 4, 6 } }, { dtype = "int64" })
  assert_dtype(numlua.cov(ints), "float64")
  assert_near(numlua.cov(ints):totable(), { { 1, 2 }, { 2, 4 } }, 1e-8)

  assert_error(function() numlua.cov({ 1, 2, 3 }) end, "expects a tensor")
  assert_error(function() numlua.cov(numlua.zeros(2, 2, 2)) end, "1-D or 2-D")
  assert_error(function() numlua.cov(numlua.zeros(2), nil, "yes") end,
    "rowvar")
  assert_error(function() numlua.cov(numlua.zeros(2), nil, true, "x") end,
    "ddof")
  assert_error(function() numlua.cov(numlua.zeros(1, 3),
    numlua.zeros(1, 2)) end, "same number of observations")
  assert_error(function() numlua.cov(numlua.zeros(1, 0)) end,
    "at least one observation")
end)

test("corrcoef returns unit-diagonal correlation matrices", function()
  local perfect = numlua.corrcoef(numlua.array({
    { 1, 2, 3 },
    { 2, 4, 6 },
  }))
  assert_shape(perfect, { 2, 2 })
  assert_near(perfect:totable(), { { 1, 1 }, { 1, 1 } }, 1e-8)

  local negative = numlua.corrcoef(numlua.array({
    { 1, 2, 3 },
    { 3, 2, 1 },
  }))
  assert_near(negative:totable(), { { 1, -1 }, { -1, 1 } }, 1e-8)
  assert_near(numlua.corrcoef(numlua.array({ 1, 2, 3 })):item(), 1, 1e-8)
end)

test("corrcoef handles rowvar and validation", function()
  local observations = numlua.array({
    { 1, 2 },
    { 2, 4 },
    { 3, 6 },
  })
  assert_near(numlua.corrcoef(observations, nil, false):totable(),
    { { 1, 1 }, { 1, 1 } }, 1e-8)
  assert_error(function() numlua.corrcoef(numlua.zeros(2, 2, 2)) end,
    "1-D or 2-D")
end)
