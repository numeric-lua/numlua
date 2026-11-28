--
-- einsum contractions, permutations, and reductions.
--

local numlua = require("numlua")

test("einsum handles matmul, dot, and elementwise products", function()
  local a = numlua.array({ { 1, 2 }, { 3, 4 } })
  local b = numlua.array({ { 5, 6 }, { 7, 8 } })
  assert_tensor(numlua.einsum("ij,jk->ik", a, b), a:matmul(b):totable())
  assert_tensor(numlua.einsum("ij,ij->ij", a, b), a:mul(b):totable())
  assert_near(numlua.einsum("i,i->",
    numlua.array({ 1, 2 }), numlua.array({ 3, 4 })):item(), 11, 1e-6)
  assert_tensor(numlua.einsum("ij,kj->ik", a, b),
    a:matmul(b:t()):totable())
end)

test("einsum handles transposes and reductions", function()
  local a = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_tensor(numlua.einsum("ij->ji", a), a:t():totable())
  assert_tensor(numlua.einsum("ij->i", a), { 3, 7 })
  assert_tensor(numlua.einsum("ij->j", a), { 4, 6 })
end)

test("einsum validates subscripts", function()
  local a = numlua.array({ { 1, 2 }, { 3, 4 } })
  local b = numlua.array({ { 1, 2 }, { 3, 4 } })
  assert_error(function() numlua.einsum("ij,jk", a, b) end, "->")
  assert_error(function() numlua.einsum("ii,jk->ik", a, b) end, "repeated")
  assert_error(function() numlua.einsum("ij,jk->iz", a, b) end,
    "output label")
  assert_error(function() numlua.einsum("ij,jk->ik", a, { { 1 } }) end,
    "second operand")
end)
