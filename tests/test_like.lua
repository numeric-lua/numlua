--
-- NumPy *_like constructors: zeros_like, ones_like, empty_like, full_like.
--

local numlua = require("numlua")

test("*_like constructors preserve shape and dtype", function()
  local prototype = numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }, { dtype = "int64" })

  local zeros = numlua.zeros_like(prototype)
  assert_shape(zeros, { 2, 3 })
  assert_dtype(zeros, "int64")
  assert_tensor(zeros, { { 0, 0, 0 }, { 0, 0, 0 } })

  local ones = numlua.ones_like(prototype)
  assert_shape(ones, { 2, 3 })
  assert_dtype(ones, "int64")
  assert_tensor(ones, { { 1, 1, 1 }, { 1, 1, 1 } })

  local empty = numlua.empty_like(prototype)
  assert_shape(empty, { 2, 3 })
  assert_dtype(empty, "int64")
  empty:fill_(9)
  assert_tensor(empty, { { 9, 9, 9 }, { 9, 9, 9 } })

  local full = numlua.full_like(prototype, 7)
  assert_shape(full, { 2, 3 })
  assert_dtype(full, "int64")
  assert_tensor(full, { { 7, 7, 7 }, { 7, 7, 7 } })

  -- Results own their storage.
  full:set(1, 1, 99)
  assert_eq(prototype:get(1, 1), 1)
  zeros:set(1, 1, 42)
  assert_eq(prototype:get(1, 1), 1)
end)

test("*_like constructors accept dtype and shape overrides", function()
  local prototype = numlua.array({ 1.5, 2.5 }, { dtype = "float32" })

  local converted = numlua.ones_like(prototype, { dtype = "float64" })
  assert_shape(converted, { 2 })
  assert_dtype(converted, "float64")
  assert_tensor(converted, { 1, 1 })

  local resized = numlua.zeros_like(prototype, { shape = { 2, 2 } })
  assert_shape(resized, { 2, 2 })
  assert_dtype(resized, "float32")
  assert_tensor(resized, { { 0, 0 }, { 0, 0 } })

  local vector = numlua.full_like(prototype, 3.5, { shape = 3 })
  assert_shape(vector, { 3 })
  assert_tensor(vector, { 3.5, 3.5, 3.5 })

  local scalar = numlua.ones_like(prototype, { shape = {} })
  assert_shape(scalar, {})
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 1)

  local scalar_dimension = numlua.zeros_like(prototype, { shape = 0 })
  assert_shape(scalar_dimension, { 0 })
  assert_eq(scalar_dimension:numel(), 0)

  local bools = numlua.full_like(prototype, 1, { dtype = "bool" })
  assert_dtype(bools, "bool")
  assert_tensor(bools, { true, true })
end)

test("*_like constructors handle 0-D and non-contiguous prototypes", function()
  local scalar = numlua.array(5, { dtype = "int64" })
  for _, name in ipairs({ "zeros_like", "ones_like", "empty_like" }) do
    local result = numlua[name](scalar)
    assert_shape(result, {})
    assert_eq(result:dim(), 0)
    assert_dtype(result, "int64")
  end
  local full = numlua.full_like(scalar, 9)
  assert_shape(full, {})
  assert_eq(full:item(), 9)

  local transposed = numlua.transpose(numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }))
  assert_false(transposed:is_contiguous())
  local zeros = numlua.zeros_like(transposed)
  assert_shape(zeros, { 3, 2 })
  assert_tensor(zeros, { { 0, 0 }, { 0, 0 }, { 0, 0 } })
  local empty = numlua.empty_like(transposed)
  assert_shape(empty, { 3, 2 })
  assert_dtype(empty, "float32")
end)

test("*_like constructors validate prototypes, options, and shapes", function()
  local prototype = numlua.zeros(2, 3)
  for _, name in ipairs({ "zeros_like", "ones_like", "empty_like" }) do
    assert_error(function() numlua[name](nil) end, "expects a tensor")
  end
  assert_error(function() numlua.full_like(nil, 1) end, "expects a tensor")
  assert_error(function() numlua.full_like(prototype, "x") end,
    "numeric fill value")
  assert_error(function() numlua.zeros_like(prototype, 1) end,
    "options must be a table or nil")
  assert_error(function() numlua.zeros_like(prototype, { nope = true }) end,
    "unknown option")
  assert_error(function()
    numlua.zeros_like(prototype, { shape = "2x3" })
  end, "shape must be a dimension or a table")
  assert_error(function()
    numlua.zeros_like(prototype, { shape = { 2, -1 } })
  end, "non-negative integers")
  assert_error(function()
    numlua.zeros_like(prototype, { shape = { 2, 3.5 } })
  end, "non-negative integers")
  assert_error(function()
    numlua.zeros_like(prototype, { shape = { 2, 0 / 0 } })
  end, "non-negative integers")
  assert_error(function()
    numlua.zeros_like(prototype, { dtype = {} })
  end, "dtype must be a name")
end)
