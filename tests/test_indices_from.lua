--
-- *_indices_from constructors.
--

local numlua = require("numlua")

test("tril_indices_from and triu_indices_from follow the input shape", function()
  local matrix = numlua.zeros(3, 4)
  local rows, columns = numlua.tril_indices_from(matrix)
  local expected_rows, expected_columns = numlua.tril_indices(3, 0, 4)
  assert_tensor(rows, expected_rows:totable())
  assert_tensor(columns, expected_columns:totable())

  local upper_rows, upper_columns = numlua.triu_indices_from(matrix, 1)
  local expected_upper_rows, expected_upper_columns = numlua.triu_indices(3, 1, 4)
  assert_tensor(upper_rows, expected_upper_rows:totable())
  assert_tensor(upper_columns, expected_upper_columns:totable())
end)

test("diag_indices_from uses the smallest dimension", function()
  local a, b = numlua.diag_indices_from(numlua.zeros(3, 4))
  assert_tensor(a, { 1, 2, 3 })
  assert_tensor(b, { 1, 2, 3 })
  local x, y, z = numlua.diag_indices_from(numlua.zeros(2, 3, 4))
  assert_tensor(x, { 1, 2 })
  assert_tensor(y, { 1, 2 })
  assert_tensor(z, { 1, 2 })
end)

test("indices_from helpers validate rank and input type", function()
  assert_error(function() numlua.tril_indices_from(numlua.zeros(2)) end,
    "2-D")
  assert_error(function() numlua.triu_indices_from({ { 1 } }) end,
    "expects a tensor")
  assert_error(function() numlua.diag_indices_from(numlua.zeros(2)) end,
    "at least a 2-D")
  assert_error(function() numlua.tril_indices_from(numlua.zeros(2, 2, 2))
    end, "2-D")
end)
