--
-- Real/complex/scalar/Fortran predicates.
--

local numlua = require("numlua")

test("array predicates classify values", function()
  assert_eq(numlua.isrealobj(numlua.array({ 1, 2 })), true)
  assert_eq(numlua.iscomplexobj(numlua.array({ 1, 2 })), false)
  assert_eq(numlua.isrealobj(3), true)
  assert_eq(numlua.iscomplexobj(3), false)
  assert_eq(numlua.isscalar(3), true)
  assert_eq(numlua.isscalar(true), true)
  assert_eq(numlua.isscalar(numlua.array(3)), false)
  assert_eq(numlua.isscalar({ 1 }), false)
  assert_eq(numlua.isfortran(numlua.zeros(2, 2)), false)
  assert_error(function() numlua.isfortran({ 1 }) end, "expects a tensor")
end)
