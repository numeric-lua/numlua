--
-- NumPy-parity variance and standard deviation reductions.
--

local numlua = require("numlua")

local function square_root(value)
  return math.sqrt(value)
end

test("var and std default to ddof = 0", function()
  local t = numlua.array({ 1, 2, 3, 4 })
  local variance = numlua.var(t)
  assert_dtype(variance, "float32")
  assert_eq(variance:dim(), 0)
  assert_near(variance:item(), 1.25)

  local deviation = numlua.std(t)
  assert_dtype(deviation, "float32")
  assert_near(deviation:item(), square_root(1.25))

  assert_near(numlua.var(t, nil, false, 1):item(), 5 / 3)
  assert_near(numlua.std(t, nil, false, 1):item(), square_root(5 / 3))
end)

test("var and std reduce along 1-based axes and keep dimensions", function()
  local t = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = "float64" })

  assert_near(numlua.var(t, 1):totable(), { 2.25, 2.25, 2.25 })
  assert_near(numlua.var(t, 2):totable(), { 2 / 3, 2 / 3 })
  assert_near(numlua.var(t, -1):totable(), { 2 / 3, 2 / 3 })
  assert_near(numlua.std(t, 2):totable(),
    { square_root(2 / 3), square_root(2 / 3) })

  assert_shape(numlua.var(t, 1, true), { 1, 3 })
  assert_shape(numlua.var(t, 2, true), { 2, 1 })
  assert_shape(numlua.std(t, 2, true), { 2, 1 })
  assert_shape(numlua.var(t, nil, true), { 1, 1 })
end)

test("var and std promote integer and bool input to float64", function()
  local ints = numlua.array({ 1, 2, 3, 4 }, { dtype = "int64" })
  local int_variance = numlua.var(ints)
  assert_dtype(int_variance, "float64")
  assert_near(int_variance:item(), 1.25)

  local flags = numlua.array({ true, false, true, false })
  local bool_variance = numlua.var(flags)
  assert_dtype(bool_variance, "float64")
  assert_near(bool_variance:item(), 0.25)
  assert_near(numlua.std(flags):item(), 0.5)
end)

test("var and std support arbitrary finite ddof values", function()
  local t = numlua.array({ 1, 2, 3, 4 }, { dtype = "float64" })
  assert_near(numlua.var(t, nil, false, 2):item(), 2.5)
  assert_near(numlua.std(t, nil, false, 2):item(), square_root(2.5))
  assert_near(numlua.var(t, nil, false, 0.5):item(), 5 / 3.5)

  local matrix = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = "float64" })
  assert_near(numlua.var(matrix, 2, false, 2):totable(), { 2, 2 })
end)

test("var and std validate arguments", function()
  assert_error(function() numlua.var(1) end, "expects a tensor")
  assert_error(function() numlua.std(nil) end, "expects a tensor")
  assert_error(function() numlua.var(numlua.array({ 1, 2 }), 0) end,
    "axis 0 is invalid")
  assert_error(function()
    numlua.var(numlua.array({ 1, 2 }), 2, false, 2)
  end, "out of range")
  assert_error(function()
    numlua.var(numlua.array({ 1, 2 }), 1, 1)
  end, "keepdims must be a boolean")
  assert_error(function()
    numlua.std(numlua.array({ 1, 2 }), 1, false, "two")
  end, "ddof must be a finite number")
  assert_error(function()
    numlua.var(numlua.array({ 1, 2 }), 1, false, 0 / 0)
  end, "ddof must be a finite number")
  assert_error(function()
    numlua.std(numlua.array({ 1, 2 }), 1, false, math.huge)
  end, "ddof must be a finite number")
end)
