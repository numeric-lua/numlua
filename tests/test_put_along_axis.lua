--
-- put_along_axis assignment.
--

local numlua = require("numlua")

test("put_along_axis writes per-slice values", function()
  local values = numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })
  local indices = numlua.array({ { 3, 1 }, { 2, 1 } }, { dtype = "int64" })
  local replacements = numlua.array({ { 99, 88 }, { 77, 66 } })
  local result = numlua.put_along_axis(values, indices, replacements, 2)
  assert_true(result == values)
  assert_tensor(values, { { 88, 20, 99 }, { 66, 77, 60 } })
end)

test("put_along_axis supports scalars, negative indices, and axis 1", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  numlua.put_along_axis(values,
    numlua.array({ { -1, -2 }, { -2, -1 } }, { dtype = "int64" }), 0, 2)
  assert_tensor(values, { { 0, 0 }, { 0, 0 } })
  numlua.put_along_axis(values,
    numlua.array({ { 1 }, { 2 } }, { dtype = "int64" }), 9, 2)
  assert_tensor(values, { { 9, 0 }, { 0, 9 } })

  local rows = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
  numlua.put_along_axis(rows,
    numlua.array({ { 2 }, { 1 } }, { dtype = "int64" }),
    numlua.array({ { 20 }, { 10 } }), 1)
  assert_tensor(rows, { { 10, 10, 10 }, { 20, 20, 20 } })
end)

test("put_along_axis broadcasts singleton input dimensions and casts values", function()
  local values = numlua.array({ { 1, 2, 3 } })
  numlua.put_along_axis(values,
    numlua.array({ { 3, 2, 1 }, { 1, 2, 3 } }, { dtype = "int64" }),
    numlua.array({ { 30, 20, 10 }, { 11, 12, 13 } }), 2)
  assert_tensor(values, { { 11, 12, 13 } })

  local ints = numlua.array({ { 1, 2 } }, { dtype = "int64" })
  numlua.put_along_axis(ints,
    numlua.array({ { 1 } }, { dtype = "int64" }), 3.9, 2)
  assert_dtype(ints, "int64")
  assert_tensor(ints, { { 3, 2 } })
end)

test("put_along_axis validates ranks, shapes, values, and indices", function()
  local values = numlua.array({ { 1, 2 }, { 3, 4 } })
  local indices = numlua.array({ { 1 } }, { dtype = "int64" })
  assert_error(function() numlua.put_along_axis({ { 1 } }, indices, 1, 1) end,
    "expects a tensor")
  assert_error(function()
    numlua.put_along_axis(values,
      numlua.array({ 1, 2 }, { dtype = "int64" }), 1, 2)
  end, "same rank")
  assert_error(function()
    numlua.put_along_axis(values,
      numlua.array({ { 1, 2, 1 }, { 1, 2, 1 } }, { dtype = "int64" }), 1, 1)
  end, "does not match")
  assert_error(function()
    numlua.put_along_axis(values,
      numlua.array({ { 1, 2 }, { 1, 2 } }, { dtype = "int64" }),
      { 1, 2, 3 }, 2)
  end, "values must be scalar")
  assert_error(function()
    numlua.put_along_axis(values,
      numlua.array({ { 3 } }, { dtype = "int64" }), 1, 2)
  end, "out of range")
  assert_error(function() numlua.put_along_axis(values, indices, 1, 0) end,
    "axis")
end)
