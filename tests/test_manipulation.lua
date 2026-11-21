local numlua = require("numlua")

test("broadcast_to and broadcast_arrays share a shape", function()
  local row = numlua.array({ { 1, 2, 3 } })
  local wide = numlua.broadcast_to(row, { 2, 3 })
  assert_shape(wide, { 2, 3 })
  assert_tensor(wide, { { 1, 2, 3 }, { 1, 2, 3 } })

  local column = numlua.array({ { 10 }, { 20 } })
  local a, b = numlua.broadcast_arrays(row, column)
  assert_shape(a, { 2, 3 })
  assert_shape(b, { 2, 3 })
  assert_error(function()
    numlua.broadcast_arrays(numlua.zeros(2), numlua.zeros(3))
  end, "incompatible")
end)

test("tile repeats whole blocks", function()
  local vector = numlua.array({ 1, 2 })
  assert_tensor(numlua.tile(vector, 3), { 1, 2, 1, 2, 1, 2 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.tile(matrix, { 2, 1 }), {
    { 1, 2 }, { 3, 4 }, { 1, 2 }, { 3, 4 },
  })
end)

test("repeat repeats elements along an axis", function()
  local vector = numlua.array({ 1, 2 })
  assert_tensor(numlua.repeat_(vector, 2), { 1, 1, 2, 2 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.repeat_(matrix, 2, 1), {
    { 1, 2 }, { 1, 2 }, { 3, 4 }, { 3, 4 },
  })
  assert_tensor(numlua.repeat_(matrix, { 1, 2 }, 2), {
    { 1, 2, 2 }, { 3, 4, 4 },
  })
end)

test("flip and roll preserve shapes", function()
  local t = numlua.array({ 1, 2, 3, 4 })
  assert_tensor(numlua.flip(t, 1), { 4, 3, 2, 1 })
  assert_tensor(numlua.roll(t, 1, 1), { 4, 1, 2, 3 })
  assert_tensor(numlua.roll(t, -1), { 2, 3, 4, 1 })

  local m = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.flip(m, { 1, 2 }), { { 4, 3 }, { 2, 1 } })
end)

test("pad adds constant borders", function()
  local vector = numlua.array({ 1, 2 })
  assert_tensor(numlua.pad(vector, { 1, 2 }, "constant", 9),
    { 9, 1, 2, 9, 9 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  local padded = numlua.pad(matrix, { { 1, 0 }, { 0, 1 } }, "constant", 0)
  assert_shape(padded, { 3, 3 })
  assert_near(padded:sum():item(), 10, 1e-6)
end)

test("diff, cumsum, and cumprod follow axis semantics", function()
  local vector = numlua.array({ 1, 4, 9, 16 })
  assert_tensor(numlua.diff(vector), { 3, 5, 7 })
  assert_tensor(numlua.diff(vector, 2), { 2, 2 })
  assert_tensor(numlua.cumsum(vector), { 1, 5, 14, 30 })
  assert_tensor(numlua.cumprod(vector), { 1, 4, 36, 576 })

  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.cumsum(matrix, 1), { { 1, 2 }, { 4, 6 } })
  assert_tensor(numlua.diff(matrix, 1, 2), { { 1 }, { 1 } })
end)

test("argsort and take use 1-based indices", function()
  local t = numlua.array({ 30, 10, 20 })
  local order = numlua.argsort(t)
  assert_tensor(order, { 2, 3, 1 })
  assert_tensor(numlua.take(t, order), { 10, 20, 30 })

  local matrix = numlua.array({ { 3, 1 }, { 2, 4 } })
  local indices = numlua.argsort(matrix, 2)
  assert_tensor(indices, { { 2, 1 }, { 1, 2 } })
  assert_error(function() numlua.take(t, { 1, 2 }) end, "indices must")
end)

test("repeat with a table and no axis flattens", function()
  local t = numlua.array({{1, 2}, {3, 4}})
  local r = numlua.repeat_(t, { 1, 2, 1, 2 })
  assert_shape(r, { 6 })
  assert_tensor(r, { 1, 2, 2, 3, 4, 4 })
end)

test("compress selects elements with a boolean mask", function()
  local t = numlua.array({ 10, 20, 30, 40 })
  local mask = numlua.array({ true, false, true, false }, { dtype = "bool" })
  local selected = numlua.compress(mask, t)
  assert_tensor(selected, { 10, 30 })
  assert_error(function() numlua.compress(mask, t, 1) end, "axis")
end)
