--
-- Difference helpers: prepend/append and ediff1d.
--

local numlua = require("numlua")

test("diff supports prepend and append edges", function()
  local values = numlua.array({ 1, 2, 3 })
  assert_tensor(numlua.diff(values, 1, 1, 0), { 1, 1, 1 })
  assert_tensor(numlua.diff(values, 1, 1, nil, 10), { 1, 1, 7 })
  assert_tensor(numlua.diff(values, 2, 1, 0, 10), { 0, 0, 6 })
  assert_tensor(numlua.diff(values, 1, 1, { 0 }, { 10 }), { 1, 1, 1, 7 })
  assert_tensor(numlua.diff(values, 1, 1,
    numlua.array({ -1 }), numlua.array({ 4 })), { 2, 1, 1, 1 })
end)

test("diff prepends and appends along N-D axes", function()
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.diff(matrix, 1, 1, 0),
    { { 1, 2 }, { 2, 2 } })
  assert_tensor(numlua.diff(matrix, 1, 2, 0),
    { { 1, 1 }, { 3, 1 } })
  assert_tensor(numlua.diff(matrix, 1, 2, nil, 9),
    { { 1, 7 }, { 1, 5 } })
  assert_tensor(numlua.diff(matrix, 1, 1,
    numlua.array({ { 0, 0 } }), numlua.array({ { 9, 9 } })),
    { { 1, 2 }, { 2, 2 }, { 6, 5 } })
end)

test("diff validates prepend and append shapes", function()
  local matrix = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function() numlua.diff(numlua.array(1)) end,
    "at least a 1-D")
  assert_error(function() numlua.diff(matrix, 1, 3, 0) end, "axis")
  assert_error(function() numlua.diff(matrix, 1, 1, { 0 }) end,
    "must have rank")
  assert_error(function() numlua.diff(matrix, 1, 1, { 0, 0, 0 }) end,
    "rank")
  assert_error(function() numlua.diff(matrix, 1, 2, { { 0, 0 } }) end,
    "shape does not match")
  assert_error(function() numlua.diff(matrix, 1, 1, true) end,
    "must be a tensor")
  assert_error(function() numlua.diff(matrix, 1, 1, { "x" }) end,
    "must be numeric")
end)

test("ediff1d flattens and adds begin/end values", function()
  assert_tensor(numlua.ediff1d(numlua.array({ { 1, 2 }, { 4, 7 } })),
    { 1, 2, 3 })
  assert_tensor(numlua.ediff1d(numlua.array({ { 1, 2 }, { 4, 7 } }), 99, 0),
    { 0, 1, 2, 3, 99 })
  assert_tensor(numlua.ediff1d(numlua.array({ 1, 2, 4 }),
    numlua.array({ 8, 9 }), numlua.array({ -1 })),
    { -1, 1, 2, 8, 9 })
  assert_tensor(numlua.ediff1d(numlua.array({ 1, 2, 4 }), { 7 }), { 1, 2, 7 })
  assert_tensor(numlua.ediff1d(numlua.array({ 5 }), 9, 0), { 0, 9 })
  assert_tensor(numlua.ediff1d(numlua.zeros(0), 1, 0), { 0, 1 })
end)

test("ediff1d validates its edges", function()
  assert_error(function() numlua.ediff1d({ 1, 2 }) end, "expects a tensor")
  assert_error(function() numlua.ediff1d(numlua.array({ 1, 2 }), true) end,
    "to_end must be")
  assert_error(function() numlua.ediff1d(numlua.array({ 1, 2 }),
    { "x" }) end, "must be numeric")
end)
