--
-- Logarithmically spaced constructors.
--

local numlua = require("numlua")

test("logspace builds base-10 powers with float32 defaults", function()
  local values = numlua.logspace(0, 2, 3)
  assert_shape(values, { 3 })
  assert_dtype(values, "float32")
  assert_near(values:totable(), { 1, 10, 100 }, 1e-5)
  local default = numlua.logspace(0, 2)
  assert_shape(default, { 50 })
  assert_near(default:get(1), 1, 1e-6)
  assert_near(default:get(50), 100, 1e-4)
end)

test("logspace accepts a base, dtype, and endpoint option", function()
  assert_near(numlua.logspace(0, 3, 4, 2):totable(), { 1, 2, 4, 8 }, 1e-5)
  assert_dtype(numlua.logspace(0, 2, 3, 10, { dtype = "float64" }),
    "float64")
  assert_near(numlua.logspace(0, 2, 3, 10, { dtype = "float64" }):totable(),
    { 1, 10, 100 }, 1e-12)
  assert_near(numlua.logspace(0, 2, 3, 10, { endpoint = false }):totable(),
    { 1, 10 ^ (2 / 3), 10 ^ (4 / 3) }, 1e-5)
  assert_near(numlua.logspace(0, 2, { dtype = "float64" }):get(1), 1, 1e-12)
  assert_dtype(numlua.logspace(0, 2, { dtype = "float64" }), "float64")
end)

test("logspace handles zero and one point, and integer casts", function()
  assert_shape(numlua.logspace(0, 2, 0), { 0 })
  assert_eq(numlua.logspace(0, 2, 0):numel(), 0)
  assert_shape(numlua.logspace(0, 2, 1), { 1 })
  assert_near(numlua.logspace(0, 2, 1):get(1), 1, 1e-6)
  assert_near(numlua.logspace(0, 2, 1, 10, { endpoint = false }):get(1),
    1, 1e-6)
  local ints = numlua.logspace(0, 2, 3, 10, { dtype = "int64" })
  assert_dtype(ints, "int64")
  assert_tensor(ints, { 1, 10, 100 })
end)

test("geomspace builds geometric sequences with a common sign", function()
  assert_near(numlua.geomspace(1, 1000, 4):totable(), { 1, 10, 100, 1000 },
    1e-5)
  assert_dtype(numlua.geomspace(1, 1000, 4), "float32")
  assert_near(numlua.geomspace(-1, -100, 3):totable(), { -1, -10, -100 },
    1e-5)
  assert_near(numlua.geomspace(1, 1000, 4, { endpoint = false }):totable(),
    { 1, 10 ^ (3 / 4), 10 ^ (6 / 4), 10 ^ (9 / 4) }, 1e-5)
  assert_near(numlua.geomspace(2, 8, 2, { dtype = "float64" }):totable(),
    { 2, 8 }, 1e-12)
  assert_shape(numlua.geomspace(1, 10, 0), { 0 })
  assert_near(numlua.geomspace(1, 10, 1):get(1), 1, 1e-6)
end)

test("logspace and geomspace validate their arguments", function()
  assert_error(function() numlua.logspace("0", 2) end, "start")
  assert_error(function() numlua.logspace(0, 2, -1) end, "num")
  assert_error(function() numlua.logspace(0, 2, 2.5) end, "num")
  assert_error(function() numlua.logspace(0, 2, 3, 0) end, "base")
  assert_error(function() numlua.logspace(0, 2, 3, -2) end, "base")
  assert_error(function() numlua.logspace(0, 2, 3, 10, { base = 2 }) end,
    "unknown option")
  assert_error(function() numlua.logspace(0, 2, 3, 10,
    { endpoint = 1 }) end, "endpoint")
  assert_error(function() numlua.logspace(0, 2, 3, 10,
    { dtype = "bool" }) end, "bool")
  assert_error(function() numlua.logspace(0, math.huge) end, "finite")

  assert_error(function() numlua.geomspace(0, 10) end, "non-zero")
  assert_error(function() numlua.geomspace(1, -10) end, "same sign")
  assert_error(function() numlua.geomspace(-1, 10) end, "same sign")
  assert_error(function() numlua.geomspace(1, 10, 3, { dtype = "bool" }) end,
    "bool")
  assert_error(function() numlua.geomspace(1, 10, 3, { nope = true }) end,
    "unknown option")
  assert_error(function() numlua.geomspace(1, 10, 3, 2) end, "options")
end)
