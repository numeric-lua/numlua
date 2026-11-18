local numlua = require("numlua")

test("where selects tensor branches with broadcasting", function()
  local condition = numlua.array({
    { true, false, true },
    { false, true, false },
  })
  local x = numlua.array({ { 1 }, { 2 } })
  local y = numlua.array({ { 10, 20, 30 } })

  local result = numlua.where(condition, x, y)
  assert_dtype(result, "float32")
  assert_tensor(result, {
    { 1, 20, 1 },
    { 10, 2, 30 },
  })

  -- A broadcast condition and a broadcast branch combine with a scalar branch.
  local column = numlua.array({ { true }, { false } })
  assert_tensor(numlua.where(column, numlua.array({ { 1, 2, 3 } }), -1), {
    { 1, 2, 3 },
    { -1, -1, -1 },
  })
end)

test("where supports scalar conditions and truthy numeric conditions", function()
  local condition = numlua.array({ true, false, true })
  assert_tensor(numlua.where(condition, 1, 0), { 1, 0, 1 })

  local scalar = numlua.where(false,
    numlua.array({ { 1, 2 } }), numlua.array({ { 3, 4 } }))
  assert_tensor(scalar, { { 3, 4 } })

  local zero_d = numlua.where(0, 5, 6)
  assert_eq(zero_d:dim(), 0)
  assert_eq(zero_d:item(), 6)

  assert_tensor(numlua.where(numlua.array({ 0, 1, 2 }), 5, -5),
    { -5, 5, 5 })
end)

test("clip uses scalar bounds and broadcasts tensor bounds", function()
  local t = numlua.array({ -2, -0.5, 0, 0.5, 2 })
  assert_tensor(numlua.clip(t, -1, 1), { -1, -0.5, 0, 0.5, 1 })
  assert_dtype(numlua.clip(t, -1, 1), "float32")

  local matrix = numlua.array({ { -5, 0, 10 }, { 5, 10, -5 } })
  local minimum = numlua.array({ { 0 }, { 2 } })
  local maximum = numlua.array({ { 1, 3, 5 } })
  assert_tensor(numlua.clip(matrix, minimum, maximum), {
    { 0, 0, 5 },
    { 1, 3, 2 },
  })

  -- NaN is preserved by clamp; clamp(min, max) with min > max yields max.
  local with_nan = numlua.clip(numlua.array({ 0 / 0, 3 }), 0, 1)
  assert_tensor(numlua.isnan(with_nan), { true, false })
  assert_eq(with_nan:get(2), 1)
  assert_tensor(numlua.clip(numlua.array({ -2, 0, 2 }), 1, -1),
    { -1, -1, -1 })
end)

test("isnan covers NaN, infinities, 0-D, integer, and bool inputs", function()
  local t = numlua.array({ 0 / 0, 1, math.huge, -math.huge })
  local mask = numlua.isnan(t)
  assert_dtype(mask, "bool")
  assert_tensor(mask, { true, false, false, false })

  assert_eq(numlua.isnan(numlua.array(0 / 0)):item(), true)
  assert_eq(numlua.isnan(numlua.array(math.huge)):item(), false)

  assert_tensor(numlua.isnan(numlua.array({ 1, 2 }, { dtype = "int64" })),
    { false, false })
  assert_tensor(numlua.isnan(numlua.array({ true, false })),
    { false, false })
end)

test("any and all reduce bool and numeric tensors", function()
  local t = numlua.array({
    { 0, 1, 0 },
    { 0, 0, 0 },
  })

  local any_all = numlua.any(t)
  assert_dtype(any_all, "bool")
  assert_shape(any_all, {})
  assert_eq(any_all:item(), true)
  assert_eq(numlua.all(t):item(), false)

  assert_tensor(numlua.any(t, 1), { false, true, false })
  assert_tensor(numlua.all(t, 1), { false, false, false })
  assert_tensor(numlua.any(t, 2), { true, false })
  assert_tensor(numlua.all(t, 2), { false, false })
  assert_tensor(numlua.any(t, -1), { true, false })

  assert_shape(numlua.any(t, 1, true), { 1, 3 })
  assert_shape(numlua.all(t, 2, true), { 2, 1 })
  assert_shape(numlua.any(t, nil, true), { 1, 1 })
  assert_shape(numlua.all(t, nil, true), { 1, 1 })

  assert_eq(numlua.any(numlua.array(false)):item(), false)
  assert_eq(numlua.all(numlua.array(true)):item(), true)
  assert_eq(numlua.all(numlua.array({ true, false })):item(), false)
end)

test("any and all treat NaN as truthy", function()
  local nan = 0 / 0
  assert_eq(numlua.any(numlua.array({ nan, 0 })):item(), true)
  assert_eq(numlua.all(numlua.array({ nan, 1 })):item(), true)
  assert_eq(numlua.all(numlua.array({ nan, 0 })):item(), false)
end)

test("count_nonzero counts all elements, axes, and NaN", function()
  local t = numlua.array({
    { 0, 1, 0 },
    { 2, 0, 3 },
  })

  local total = numlua.count_nonzero(t)
  assert_dtype(total, "int64")
  assert_shape(total, {})
  assert_eq(total:item(), 3)

  assert_tensor(numlua.count_nonzero(t, 1), { 1, 1, 1 })
  assert_tensor(numlua.count_nonzero(t, 2), { 1, 2 })
  assert_tensor(numlua.count_nonzero(t, -1), { 1, 2 })

  local nan = 0 / 0
  assert_eq(numlua.count_nonzero(numlua.array({ nan, 0, 2 })):item(), 2)
  assert_eq(numlua.count_nonzero(numlua.array(5)):item(), 1)
  assert_eq(numlua.count_nonzero(numlua.array(0)):item(), 0)
  assert_eq(numlua.count_nonzero(
    numlua.array({ true, false, true })):item(), 2)
end)

test("predicate and selection helpers validate their arguments", function()
  assert_error(function() numlua.where(nil, 1, 2) end,
    "numlua.where expects a tensor")
  assert_error(function() numlua.where(true, {}, 2) end,
    "numlua.where expects a tensor")
  assert_error(function() numlua.where(true, 1, {}) end,
    "numlua.where expects a tensor")

  assert_error(function() numlua.clip(1, 0, 1) end,
    "numlua.clip expects a tensor")
  assert_error(function() numlua.clip(numlua.array({ 1 })) end,
    "minimum and maximum")
  assert_error(function() numlua.clip(numlua.array({ 1 }), "low", 1) end,
    "minimum")
  assert_error(function() numlua.clip(numlua.array({ 1 }), 0, {}) end,
    "maximum")

  assert_error(function() numlua.isnan(1) end,
    "numlua.isnan expects a tensor")
  assert_error(function() numlua.any(1) end,
    "numlua.any expects a tensor")
  assert_error(function() numlua.all(1) end,
    "numlua.all expects a tensor")
  assert_error(function() numlua.count_nonzero(1) end,
    "numlua.count_nonzero expects a tensor")

  local t = numlua.array({ { 0, 1 }, { 2, 3 } })
  assert_error(function() numlua.any(t, 0) end, "axis 0 is invalid")
  assert_error(function() numlua.all(t, 1.5) end,
    "axis must be an integer")
  assert_error(function() numlua.count_nonzero(t, 0) end,
    "axis 0 is invalid")
  assert_error(function() numlua.any(t, 1, "yes") end,
    "keepdims must be a boolean")
  assert_error(function() numlua.all(t, 1, 1) end,
    "keepdims must be a boolean")
end)
