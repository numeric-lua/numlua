--
-- NumPy-parity meshgrid coordinate construction.
--

local numlua = require("numlua")

test("meshgrid defaults to xy indexing", function()
  local x = numlua.array({ 1, 2, 3 })
  local y = numlua.array({ 10, 20 })
  local grids = numlua.meshgrid(x, y)
  assert_eq(#grids, 2)

  assert_shape(grids[1], { 2, 3 })
  assert_shape(grids[2], { 2, 3 })
  assert_tensor(grids[1], {
    { 1, 2, 3 },
    { 1, 2, 3 },
  })
  assert_tensor(grids[2], {
    { 10, 10, 10 },
    { 20, 20, 20 },
  })
end)

test("meshgrid supports ij indexing and sparse output", function()
  local x = numlua.array({ 1, 2, 3 })
  local y = numlua.array({ 10, 20 })
  local grids = numlua.meshgrid(x, y, { indexing = "ij" })
  assert_shape(grids[1], { 3, 2 })
  assert_shape(grids[2], { 3, 2 })
  assert_tensor(grids[1], {
    { 1, 1 },
    { 2, 2 },
    { 3, 3 },
  })
  assert_tensor(grids[2], {
    { 10, 20 },
    { 10, 20 },
    { 10, 20 },
  })

  local sparse = numlua.meshgrid(x, y, {
    indexing = "ij",
    sparse = true,
  })
  assert_shape(sparse[1], { 3, 1 })
  assert_shape(sparse[2], { 1, 2 })
  assert_tensor(sparse[1], { { 1 }, { 2 }, { 3 } })
  assert_tensor(sparse[2], { { 10, 20 } })
end)

test("meshgrid orders three inputs with xy indexing", function()
  local grids = numlua.meshgrid(
    numlua.array({ 1, 2 }),
    numlua.array({ 10, 20, 30 }),
    numlua.array({ 100, 200, 300, 400 }))
  for index = 1, 3 do assert_shape(grids[index], { 3, 2, 4 }) end

  assert_eq(grids[1]:get(1, 1, 1), 1)
  assert_eq(grids[1]:get(1, 2, 1), 2)
  assert_eq(grids[1]:get(2, 1, 3), 1)
  assert_eq(grids[2]:get(1, 1, 1), 10)
  assert_eq(grids[2]:get(2, 1, 1), 20)
  assert_eq(grids[2]:get(3, 2, 4), 30)
  assert_eq(grids[3]:get(1, 1, 1), 100)
  assert_eq(grids[3]:get(1, 1, 2), 200)
  assert_eq(grids[3]:get(2, 2, 4), 400)
end)

test("meshgrid accepts Lua tables, one input, and preserves dtype", function()
  local one = numlua.meshgrid({ 4, 5, 6 })
  assert_eq(#one, 1)
  assert_shape(one[1], { 3 })
  assert_tensor(one[1], { 4, 5, 6 })

  local ints = numlua.meshgrid(
    numlua.array({ 1, 2 }, { dtype = "int64" }),
    numlua.array({ 3, 4 }, { dtype = "int64" }))
  assert_dtype(ints[1], "int64")
  assert_dtype(ints[2], "int64")
  assert_tensor(ints[1], { { 1, 2 }, { 1, 2 } })
end)

test("meshgrid copies by default and can return expanded views", function()
  local x = numlua.array({ 1, 2 })
  local y = numlua.array({ 3, 4 })
  local grids = numlua.meshgrid(x, y)
  grids[1]:fill_(0)
  assert_tensor(x, { 1, 2 })
  assert_tensor(y, { 3, 4 })

  local views = numlua.meshgrid(x, y, { copy = false })
  assert_tensor(views[1], { { 1, 2 }, { 1, 2 } })
  assert_tensor(views[2], { { 3, 3 }, { 4, 4 } })
end)

test("meshgrid validates inputs and options", function()
  assert_error(function() numlua.meshgrid() end, "at least one 1-D input")
  assert_error(function()
    numlua.meshgrid(numlua.array({ { 1, 2 } }))
  end, "must be 1-D")
  assert_error(function() numlua.meshgrid({ { 1, 2 } }) end, "must be 1-D")
  assert_error(function() numlua.meshgrid(7) end, "must be a 1-D tensor or table")
  assert_error(function()
    numlua.meshgrid(numlua.array({ 1 }), { indexing = "diagonal" })
  end, "indexing must be 'xy' or 'ij'")
  assert_error(function()
    numlua.meshgrid(numlua.array({ 1 }), { sparse = 1 })
  end, "sparse must be a boolean")
  assert_error(function()
    numlua.meshgrid(numlua.array({ 1 }), { copy = "yes" })
  end, "copy must be a boolean")
  assert_error(function()
    numlua.meshgrid(numlua.array({ 1 }), { unknown = true })
  end, "unknown option")
end)
