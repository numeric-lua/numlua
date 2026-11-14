local numlua = require("numlua")

test("shape reports the tensor dimensions", function()
  assert_eq(numlua.shape(numlua.zeros(2, 3, 4)), { 2, 3, 4 })
  assert_eq(numlua.shape(numlua.array(1.25)), {})
end)

test("reshape accepts variadic dimensions, a table and -1", function()
  local flat = numlua.arange(6)
  assert_tensor(numlua.reshape(flat, 2, 3), {
    { 0, 1, 2 },
    { 3, 4, 5 },
  })
  assert_shape(numlua.reshape(flat, { 3, 2 }), { 3, 2 })
  assert_shape(numlua.reshape(flat, 2, -1), { 2, 3 })
end)

test("transpose reverses axes by default", function()
  local t = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  local transposed = numlua.transpose(t)
  assert_shape(transposed, { 3, 2 })
  assert_tensor(transposed, { { 1, 4 }, { 2, 5 }, { 3, 6 } })
  assert_shape(t, { 2, 3 })

  local cube = numlua.reshape(numlua.arange(24), 2, 3, 4)
  assert_shape(numlua.transpose(cube), { 4, 3, 2 })
  assert_shape(numlua.transpose(cube, { 2, 3, 1 }), { 3, 4, 2 })
end)

test("transpose is a no-op for zero- and one-dimensional tensors", function()
  assert_shape(numlua.transpose(numlua.array(1)), {})
  assert_shape(numlua.transpose(numlua.array({ 1, 2, 3 })), { 3 })
end)

test("ravel flattens in C order, including non-contiguous views", function()
  local transposed = numlua.transpose(numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }))
  local flat = numlua.ravel(transposed)
  assert_shape(flat, { 6 })
  assert_true(flat:is_contiguous())
  assert_tensor(flat, { 1, 4, 2, 5, 3, 6 })
  assert_tensor(numlua.ravel(numlua.array(2.5)), { 2.5 })
end)

test("squeeze and unsqueeze", function()
  local t = numlua.zeros(1, 3, 1)
  assert_shape(numlua.squeeze(t), { 3 })
  assert_shape(numlua.squeeze(t, 1), { 3, 1 })
  assert_shape(numlua.squeeze(t, 3), { 1, 3 })

  local vector = numlua.zeros(3)
  assert_shape(numlua.unsqueeze(vector, 1), { 1, 3 })
  assert_shape(numlua.unsqueeze(vector, 2), { 3, 1 })
  assert_shape(numlua.unsqueeze(vector, -1), { 3, 1 })
end)

test("concatenate joins along the requested axis", function()
  local a = numlua.array({ { 1, 2 }, { 3, 4 } })
  local b = numlua.array({ { 5, 6 }, { 7, 8 } })
  assert_shape(numlua.concatenate({ a, b }), { 4, 2 })
  assert_tensor(numlua.concatenate({ a, b }, 1), {
    { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 },
  })
  assert_tensor(numlua.concatenate({ a, b }, 2), {
    { 1, 2, 5, 6 },
    { 3, 4, 7, 8 },
  })
end)

test("stack inserts a new axis", function()
  local a = numlua.array({ 1, 2, 3 })
  local b = numlua.array({ 4, 5, 6 })
  assert_shape(numlua.stack({ a, b }), { 2, 3 })
  assert_tensor(numlua.stack({ a, b }, 1), {
    { 1, 2, 3 },
    { 4, 5, 6 },
  })
  assert_shape(numlua.stack({ a, b }, 2), { 3, 2 })
  assert_tensor(numlua.stack({ a, b }, 2), {
    { 1, 4 },
    { 2, 5 },
    { 3, 6 },
  })
end)

test("shape operations reject bad arguments", function()
  assert_error(function() numlua.shape(1) end,
    "numlua.shape expects a tensor")
  assert_error(function() numlua.reshape(numlua.zeros(2)) end,
    "numlua.reshape expects a shape")
  assert_error(function() numlua.transpose(numlua.zeros(2, 3), { 1, 1 }) end,
    "axes must be unique")
  assert_error(function() numlua.transpose(numlua.zeros(2, 3), { 1 }) end,
    "expects 2 axes")
  assert_error(function() numlua.unsqueeze(numlua.zeros(2)) end,
    "expects an axis")
  assert_error(function()
    numlua.concatenate({ numlua.zeros(2, 2), numlua.zeros(2, 3) }, 1)
  end)
  assert_error(function() numlua.concatenate({}, 1) end,
    "non-empty table of tensors")
  assert_error(function() numlua.stack({ numlua.zeros(2), 3 }, 1) end,
    "element 2 is not a tensor")
  assert_error(function() numlua.concatenate({ numlua.zeros(2) }, 0) end,
    "axis 0 is invalid")
end)
