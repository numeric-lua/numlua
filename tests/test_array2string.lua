--
-- array2string/array_str formatting.
--

local numlua = require("numlua")

test("array2string formats integers, floats, and booleans", function()
  assert_eq(numlua.array2string(numlua.array({ 1, 2, 3 },
    { dtype = "int64" })), "[1 2 3]")
  assert_eq(numlua.array2string(numlua.array({ 1, 2, 3 },
    { dtype = "int64" }), { separator = ", " }), "[1, 2, 3]")
  assert_eq(numlua.array2string(numlua.array({ { 1, 2 }, { 3, 4 } },
    { dtype = "int64" })), "[[1 2]\n[3 4]]")
  assert_eq(numlua.array2string(numlua.array({ 1.2345 }, { dtype = "float64" }),
    { precision = 2 }), "[1.2]")
  assert_eq(numlua.array2string(numlua.array({ true, false })),
    "[True False]")
  assert_eq(numlua.array2string(numlua.array(7),
    { prefix = "<", suffix = ">" }), "<7>")
  assert_eq(numlua.array_str(numlua.array({ 1, 2 })), "[1 2]")
end)

test("array2string validates options", function()
  assert_error(function() numlua.array2string({ 1 }) end, "expects a tensor")
  assert_error(function()
    numlua.array2string(numlua.array(1), { precision = -1 })
  end, "precision")
  assert_error(function()
    numlua.array2string(numlua.array(1), { separator = 1 })
  end, "strings")
  assert_error(function()
    numlua.array2string(numlua.array(1), { bogus = true })
  end, "unknown option")
end)
