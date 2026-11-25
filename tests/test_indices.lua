--
-- Index-grid and triangular/diagonal index constructors.
--

local numlua = require("numlua")

test("indices builds 1-based coordinate grids", function()
  local grid = numlua.indices({ 2, 3 })
  assert_shape(grid, { 2, 2, 3 })
  assert_dtype(grid, "int64")
  assert_tensor(grid:narrow(1, 1, 1):squeeze(1),
    { { 1, 1, 1 }, { 2, 2, 2 } })
  assert_tensor(grid:narrow(1, 2, 1):squeeze(1),
    { { 1, 2, 3 }, { 1, 2, 3 } })

  local cube = numlua.indices({ 2, 2, 2 })
  assert_shape(cube, { 3, 2, 2, 2 })
  assert_eq(cube:get(1, 2, 1, 1), 2)
  assert_eq(cube:get(2, 1, 2, 1), 2)
  assert_eq(cube:get(3, 1, 1, 2), 2)
  assert_dtype(numlua.indices({ 2 }, { dtype = "int32" }), "int32")
end)

test("diag_indices returns diagonal coordinate vectors", function()
  local row, column = numlua.diag_indices(3)
  assert_tensor(row, { 1, 2, 3 })
  assert_tensor(column, { 1, 2, 3 })
  local a, b, c = numlua.diag_indices(2, 3)
  assert_tensor(a, { 1, 2 })
  assert_tensor(b, { 1, 2 })
  assert_tensor(c, { 1, 2 })
end)

test("tril_indices and triu_indices match triangular masks", function()
  local rows, columns = numlua.tril_indices(3)
  assert_tensor(rows, { 1, 2, 2, 3, 3, 3 })
  assert_tensor(columns, { 1, 1, 2, 1, 2, 3 })

  local upper_rows, upper_columns = numlua.triu_indices(3)
  assert_tensor(upper_rows, { 1, 1, 1, 2, 2, 3 })
  assert_tensor(upper_columns, { 1, 2, 3, 2, 3, 3 })

  local offset_rows, offset_columns = numlua.tril_indices(3, 1)
  assert_tensor(offset_rows, { 1, 1, 2, 2, 2, 3, 3, 3 })
  assert_tensor(offset_columns, { 1, 2, 1, 2, 3, 1, 2, 3 })
  local upper_offset_rows, upper_offset_columns = numlua.triu_indices(3, -1)
  assert_tensor(upper_offset_rows, { 1, 1, 1, 2, 2, 2, 3, 3 })
  assert_tensor(upper_offset_columns, { 1, 2, 3, 1, 2, 3, 2, 3 })
end)

test("index constructors accept rectangular shapes and dtypes", function()
  local rows, columns = numlua.tril_indices(2, 0, 3)
  assert_tensor(rows, { 1, 2, 2 })
  assert_tensor(columns, { 1, 1, 2 })
  local upper_rows, upper_columns = numlua.triu_indices(2, 0, 3)
  assert_tensor(upper_rows, { 1, 1, 1, 2, 2 })
  assert_tensor(upper_columns, { 1, 2, 3, 2, 3 })
  local dtype_rows = numlua.tril_indices(2, 0, 2, { dtype = "int32" })
  assert_dtype(dtype_rows, "int32")
end)

test("index constructors validate their arguments", function()
  assert_error(function() numlua.indices({}) end, "non-empty")
  assert_shape(numlua.indices({ 2, 0 }), { 2, 2, 0 })
  assert_error(function() numlua.indices({ 2.5 }) end, "non-negative")
  assert_error(function() numlua.indices({ 2 }, { dtype = "bool" }) end,
    "integer")
  local empty_diag = numlua.diag_indices(0)
  assert_shape(empty_diag, { 0 })
  assert_error(function() numlua.diag_indices(2, 0) end, "positive")
  assert_error(function() numlua.tril_indices(-1) end, "non-negative")
  assert_error(function() numlua.tril_indices(2, 1.5) end, "k")
  local empty_tril = numlua.tril_indices(2, 0, 0)
  assert_shape(empty_tril, { 0 })
  assert_error(function() numlua.triu_indices(2, 0, 2, { dtype = "bool" })
    end, "integer")
end)
