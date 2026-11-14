local torch = require("torch")
local numlua = require("numlua")

test("array builds a float32 tensor from nested tables", function()
  local t = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  assert_true(torch.is_tensor(t))
  assert_dtype(t, "float32")
  assert_shape(t, { 2, 3 })
  assert_tensor(t, { { 1, 2, 3 }, { 4, 5, 6 } })
end)

test("array infers bool and supports an explicit dtype", function()
  assert_dtype(numlua.array({ true, false }), "bool")
  assert_dtype(numlua.array({ 1, 2, 3 }, { dtype = "int64" }), "int64")
  assert_tensor(numlua.array({ { 1, 2 }, { 3, 4 } },
    { dtype = torch.int64 }), { { 1, 2 }, { 3, 4 } })
end)

test("array handles 3-D data and scalars", function()
  local t = numlua.array({ { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } })
  assert_shape(t, { 2, 2, 2 })
  assert_eq(t:get(2, 2, 1), 7)

  local scalar = numlua.array(3.5)
  assert_eq(scalar:dim(), 0)
  assert_eq(scalar:item(), 3.5)
end)

test("array clones an existing tensor", function()
  local source = numlua.array({ { 1, 2 }, { 3, 4 } })
  local copy = numlua.array(source)
  assert_false(copy == source)
  assert_tensor(copy, source:totable())
  copy:set(1, 1, 99)
  assert_eq(source:get(1, 1), 1)
end)

test("asarray preserves tensors and converts on request", function()
  local source = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_true(numlua.asarray(source) == source)

  local converted = numlua.asarray(source, { dtype = "float64" })
  assert_dtype(converted, "float64")
  assert_tensor(converted, source:totable())
  assert_dtype(numlua.asarray({ 1, 2 }), "float32")
end)

test("zeros and ones", function()
  local z = numlua.zeros(2, 3)
  assert_dtype(z, "float32")
  assert_shape(z, { 2, 3 })
  assert_tensor(z, { { 0, 0, 0 }, { 0, 0, 0 } })

  local o = numlua.ones({ 2, 2 }, { dtype = "int64" })
  assert_dtype(o, "int64")
  assert_tensor(o, { { 1, 1 }, { 1, 1 } })
end)

test("full", function()
  local t = numlua.full({ 2, 3 }, 7, { dtype = "int64" })
  assert_dtype(t, "int64")
  assert_shape(t, { 2, 3 })
  assert_tensor(t, { { 7, 7, 7 }, { 7, 7, 7 } })

  local vector = numlua.full(4, 1.5)
  assert_shape(vector, { 4 })
  assert_tensor(vector, { 1.5, 1.5, 1.5, 1.5 })
end)

test("eye", function()
  local t = numlua.eye(3)
  assert_dtype(t, "float32")
  assert_shape(t, { 3, 3 })
  assert_tensor(t, {
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
  })
end)

test("arange", function()
  local a = numlua.arange(5)
  assert_dtype(a, "int64")
  assert_tensor(a, { 0, 1, 2, 3, 4 })
  assert_tensor(numlua.arange(2, 10, 2), { 2, 4, 6, 8 })

  local f = numlua.arange(0, 1.5, 0.5)
  assert_dtype(f, "float32")
  assert_near(f:totable(), { 0, 0.5, 1.0 })
end)

test("linspace", function()
  local t = numlua.linspace(0, 1, 3)
  assert_dtype(t, "float32")
  assert_shape(t, { 3 })
  assert_near(t:totable(), { 0, 0.5, 1 }, 1e-7)

  assert_shape(numlua.linspace(0, 1), { 50 })
  assert_dtype(numlua.linspace(0, 1, { dtype = "float64" }), "float64")
end)
