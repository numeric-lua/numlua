--
-- mask_indices callback helper.
--

local numlua = require("numlua")

test("mask_indices derives coordinates from a mask callback", function()
  local rows, columns = numlua.mask_indices(3, function(mask, k)
    return numlua.triu(mask, k):ne(0)
  end)
  local expected_rows, expected_columns = numlua.triu_indices(3)
  assert_tensor(rows, expected_rows:totable())
  assert_tensor(columns, expected_columns:totable())

  local lower_rows, lower_columns = numlua.mask_indices(3, function(mask, k)
    return numlua.tril(mask, k):ne(0)
  end, -1)
  local expected_lower_rows, expected_lower_columns = numlua.tril_indices(3, -1)
  assert_tensor(lower_rows, expected_lower_rows:totable())
  assert_tensor(lower_columns, expected_lower_columns:totable())
end)

test("mask_indices validates n, callback, and mask shape", function()
  assert_error(function()
    numlua.mask_indices(-1, function(mask) return mask end)
  end, "non-negative")
  assert_error(function() numlua.mask_indices(2, nil) end, "mask function")
  assert_error(function()
    numlua.mask_indices(2, function() return numlua.zeros(1, 2) end)
  end, "n by n")
  assert_error(function()
    numlua.mask_indices(2, function() return true end)
  end, "tensor or table")
end)
