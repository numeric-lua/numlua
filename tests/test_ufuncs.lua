local numlua = require("numlua")

test("binary ufuncs operate elementwise and broadcast tensors", function()
  local a = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local row = numlua.array({ 10, 20, 30 })

  assert_tensor(numlua.add(a, row), {
    { 11, 22, 33 },
    { 14, 25, 36 },
  })
  assert_tensor(numlua.subtract(a, row), {
    { -9, -18, -27 },
    { -6, -15, -24 },
  })
  assert_tensor(numlua.multiply(a, row), {
    { 10, 40, 90 },
    { 40, 100, 180 },
  })
  assert_tensor(numlua.divide(a, row), {
    { 0.1, 0.1, 0.1 },
    { 0.4, 0.25, 0.2 },
  }, 1e-6)
  assert_tensor(numlua.power(a, numlua.array({ 2, 2, 2 })), {
    { 1, 4, 9 },
    { 16, 25, 36 },
  })
  assert_tensor(numlua.maximum(a, row), {
    { 10, 20, 30 },
    { 10, 20, 30 },
  })
  assert_tensor(numlua.minimum(a, row), {
    { 1, 2, 3 },
    { 4, 5, 6 },
  })

  -- A 2x1 column and a 1x3 row broadcast to a 2x3 result.
  local column = numlua.array({ { 1 }, { 2 } })
  local row2 = numlua.array({ { 10, 20, 30 } })
  assert_tensor(numlua.add(column, row2), {
    { 11, 21, 31 },
    { 12, 22, 32 },
  })

  local zero_d = numlua.array(5)
  assert_tensor(numlua.multiply(a, zero_d), {
    { 5, 10, 15 },
    { 20, 25, 30 },
  })
end)

test("binary ufuncs accept scalars on either side", function()
  local t = numlua.array({ 1, 2, 3 })

  assert_tensor(numlua.add(t, 2), { 3, 4, 5 })
  assert_tensor(numlua.add(2, t), { 3, 4, 5 })
  assert_tensor(numlua.subtract(t, 2), { -1, 0, 1 })
  assert_tensor(numlua.subtract(5, t), { 4, 3, 2 })
  assert_tensor(numlua.multiply(t, 3), { 3, 6, 9 })
  assert_tensor(numlua.multiply(3, t), { 3, 6, 9 })
  assert_tensor(numlua.divide(t, 2), { 0.5, 1, 1.5 })
  assert_tensor(numlua.divide(6, t), { 6, 3, 2 })
  assert_tensor(numlua.power(t, 2), { 1, 4, 9 })
  assert_tensor(numlua.power(2, t), { 2, 4, 8 })
  assert_tensor(numlua.maximum(t, 2), { 2, 2, 3 })
  assert_tensor(numlua.maximum(2, t), { 2, 2, 3 })
  assert_tensor(numlua.minimum(t, 2), { 1, 2, 2 })
  assert_tensor(numlua.minimum(2, t), { 1, 2, 2 })

  local sum = numlua.add(1, 2)
  assert_dtype(sum, "float32")
  assert_shape(sum, {})
  assert_eq(sum:item(), 3)

  local quotient = numlua.divide(1, 2)
  assert_dtype(quotient, "float32")
  assert_shape(quotient, {})
  assert_eq(quotient:item(), 0.5)

  local largest = numlua.maximum(1, 2)
  assert_dtype(largest, "float32")
  assert_shape(largest, {})
  assert_eq(largest:item(), 2)
end)

test("unary ufuncs cover arithmetic and transcendental functions", function()
  local values = numlua.array({ -2, -1, 0, 1, 2 })

  assert_tensor(numlua.negative(values), { 2, 1, -0, -1, -2 })
  assert_tensor(numlua.abs(values), { 2, 1, 0, 1, 2 })
  assert_tensor(numlua.square(values), { 4, 1, 0, 1, 4 })
  assert_tensor(numlua.sqrt(numlua.array({ 0, 1, 4, 9 })),
    { 0, 1, 2, 3 })
  assert_tensor(numlua.exp(numlua.array({ 0, 1 })),
    { 1, math.exp(1) }, 1e-6)
  assert_tensor(numlua.log(numlua.array({ 1, math.exp(1) })),
    { 0, 1 }, 1e-6)
  assert_tensor(numlua.sin(numlua.array({ 0, math.pi / 2 })),
    { 0, 1 }, 1e-6)
  assert_tensor(numlua.cos(numlua.array({ 0, math.pi })),
    { 1, -1 }, 1e-6)
  assert_tensor(numlua.tanh(numlua.array({ 0, 1 })),
    { 0, math.tanh(1) }, 1e-6)

  assert_dtype(numlua.sqrt(9), "float32")
  assert_shape(numlua.sqrt(9), {})
  assert_eq(numlua.sqrt(9):item(), 3)
  assert_eq(numlua.negative(2):item(), -2)
end)

test("ufunc dtypes follow luatorch/ATen promotion rules", function()
  local ints = numlua.array({ 1, 2, 3 }, { dtype = "int64" })
  local floats = numlua.array({ 1, 2, 3 })
  local doubles = numlua.array({ 1, 2, 3 }, { dtype = "float64" })

  assert_dtype(numlua.add(ints, ints), "int64")
  assert_dtype(numlua.add(ints, 2), "int64")
  assert_dtype(numlua.add(ints, 2.5), "float32")
  assert_dtype(numlua.add(ints, floats), "float32")
  assert_dtype(numlua.add(floats, doubles), "float64")
  assert_dtype(numlua.add(doubles, floats), "float64")

  -- This is ATen's result, not numlua.promote_types or NumPy promotion.
  assert_eq(numlua.promote_types("float32", "int64"), "float64")
  assert_dtype(numlua.add(floats, ints), "float32")

  -- ATen true-division promotes integer operands to float32, not float64.
  assert_dtype(numlua.divide(ints, ints), "float32")
  assert_dtype(numlua.divide(ints, 2), "float32")
  assert_dtype(numlua.divide(2, ints), "float32")
  assert_tensor(numlua.divide(ints, 2), { 0.5, 1, 1.5 })

  assert_dtype(numlua.power(ints, 2), "int64")
  assert_dtype(numlua.power(2, ints), "int64")
  assert_dtype(numlua.power(ints, 0.5), "float32")
  assert_dtype(numlua.power(2.5, ints), "float32")

  assert_dtype(numlua.maximum(ints, 2), "int64")
  assert_dtype(numlua.maximum(2, ints), "int64")
  assert_dtype(numlua.maximum(ints, 2.5), "float32")
  assert_dtype(numlua.minimum(doubles, 2), "float64")
  assert_dtype(numlua.minimum(2.5, doubles), "float64")

  -- Integral unary operations keep an integral dtype; transcendental ones use
  -- luatorch's float32 default rather than NumPy's float64.
  for _, name in ipairs({ "negative", "abs", "square" }) do
    assert_dtype(numlua[name](ints), "int64", name)
  end
  for _, name in ipairs({ "sqrt", "exp", "log", "sin", "cos", "tanh" }) do
    assert_dtype(numlua[name](ints), "float32", name)
    assert_dtype(numlua[name](doubles), "float64", name)
  end

  local bools = numlua.array({ true, false, true })
  assert_dtype(numlua.add(bools, ints), "int64")
end)

test("ufuncs return new tensors and do not mutate their inputs", function()
  local left = numlua.array({ 1, 2, 3 })
  local right = numlua.array({ 4, 5, 6 })
  local result = numlua.add(left, right)
  assert_tensor(left, { 1, 2, 3 })
  assert_tensor(right, { 4, 5, 6 })
  assert_tensor(result, { 5, 7, 9 })

  local unary_result = numlua.negative(left)
  assert_tensor(left, { 1, 2, 3 })
  assert_tensor(unary_result, { -1, -2, -3 })
end)

test("ufuncs reject invalid operands and incompatible shapes", function()
  local t = numlua.array({ 1, 2, 3 })
  local cases = {
    function() numlua.add(nil, t) end,
    function() numlua.add({}, t) end,
    function() numlua.add(t, "2") end,
    function() numlua.add(t, true) end,
    function() numlua.subtract(t, nil) end,
    function() numlua.multiply(t, {}) end,
    function() numlua.divide(t, false) end,
    function() numlua.power(t, "2") end,
    function() numlua.maximum(t, {}) end,
    function() numlua.minimum(t, nil) end,
  }
  for _, fn in ipairs(cases) do
    assert_error(fn, "tensor or number")
  end

  local unary_cases = {
    function() numlua.negative(nil) end,
    function() numlua.abs({}) end,
    function() numlua.sqrt("4") end,
    function() numlua.exp(true) end,
    function() numlua.log(nil) end,
    function() numlua.sin({}) end,
    function() numlua.cos("0") end,
    function() numlua.tanh(false) end,
    function() numlua.square(nil) end,
  }
  for _, fn in ipairs(unary_cases) do
    assert_error(fn, "tensor or number")
  end

  assert_error(function() numlua.add(t, numlua.ones(2)) end)
  assert_error(function()
    numlua.negative(numlua.array({ true, false }))
  end)
end)
