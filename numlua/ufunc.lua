--
-- numlua.ufunc: NumPy-shaped elementwise arithmetic and transcendental helpers.
--
-- Every function delegates to a luatorch/ATen elementwise method.  There is no
-- tensor implementation or broadcast layer here: ATen performs both shape and
-- dtype promotion for the delegated operation.
--
local torch = require("torch")

local M = {}

local function check_operand(value, name, side)
  if torch.is_tensor(value) or type(value) == "number" then
    return value
  end
  error(string.format(
    "numlua.%s expects a tensor or number for the %s operand, got %s",
    name, side, type(value)), 3)
end

-- A Lua scalar has no dtype.  Match luatorch's scalar-path behavior when it is
-- possible: floating inputs keep their dtype, integral scalars keep an integral
-- input dtype, and fractional scalars use luatorch's float32 default.  This is
-- used only for the operations whose luatorch method has no scalar overload
-- (maximum/minimum) and for a scalar on the left of power.
local function scalar_tensor(value, like)
  local dtype
  if like:is_floating_point() then
    dtype = like:dtype()
  elseif value % 1 == 0 then
    -- ATen treats bool arithmetic with a number as integral, not boolean.
    dtype = like:dtype() == "bool" and "int64" or like:dtype()
  else
    dtype = "float32"
  end
  local scalar = torch.tensor(value)
  if scalar:dtype() ~= dtype then
    scalar = scalar:to(dtype)
  end
  return scalar
end

local BINARY_METHODS = {
  add = "add",
  subtract = "sub",
  multiply = "mul",
  divide = "div",
  power = "pow",
  maximum = "maximum",
  minimum = "minimum",
  remainder = "remainder",
}

local function binary(left, right, name)
  check_operand(left, name, "left")
  check_operand(right, name, "right")
  local method = BINARY_METHODS[name]

  if torch.is_tensor(left) then
    if (method == "maximum" or method == "minimum") and
        type(right) == "number" then
      return left[method](left, scalar_tensor(right, left))
    end
    -- add/sub/mul/div/pow have scalar overloads; maximum/minimum receive a
    -- tensor here.  Broadcasting is ATen's in every case.
    return left[method](left, right)
  end

  if torch.is_tensor(right) then
    if method == "add" or method == "mul" then
      return right[method](right, left)
    elseif method == "sub" then
      return right:neg():add(left)
    elseif method == "div" then
      return right:reciprocal():mul(left)
    end
    local scalar = scalar_tensor(left, right)
    return scalar[method](scalar, right)
  end

  -- Two Lua scalars become 0-D float32 tensors, like numlua.array(number).
  local a = torch.tensor(left)
  local b = torch.tensor(right)
  return a[method](a, b)
end

local UNARY_METHODS = {
  negative = "neg",
  abs = "abs",
  sqrt = "sqrt",
  exp = "exp",
  log = "log",
  log1p = "log1p",
  sin = "sin",
  cos = "cos",
  tanh = "tanh",
  square = "square",
  sign = "sign",
  floor = "floor",
  ceil = "ceil",
  trunc = "trunc",
  round = "round",
  reciprocal = "reciprocal",
  isinf = "isinf",
  isfinite = "isfinite",
  isposinf = "isposinf",
  isneginf = "isneginf",
  logical_not = "logical_not",
}

local function unary(value, name)
  local method = UNARY_METHODS[name]
  if torch.is_tensor(value) then
    return value[method](value)
  end
  if type(value) == "number" then
    local tensor = torch.tensor(value)
    return tensor[method](tensor)
  end
  error(string.format(
    "numlua.%s expects a tensor or number, got %s", name, type(value)), 3)
end

for name in pairs(BINARY_METHODS) do
  M[name] = function(left, right)
    return binary(left, right, name)
  end
end

for name in pairs(UNARY_METHODS) do
  M[name] = function(value)
    return unary(value, name)
  end
end

-- Logical operators are the one group that accepts boolean Lua scalars as
-- well as numbers and tensors.  Non-boolean operands are truth-tested before
-- the ATen kernel, matching NumPy's logical ufuncs.
local function logical_operand(value, name, side)
  if torch.is_tensor(value) then
    if value:dtype() == "bool" then return value end
    return value:ne(0)
  end
  if type(value) == "number" then
    return torch.tensor(value ~= 0)
  end
  if type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.%s expects a tensor, number, or boolean for the %s operand, " ..
    "got %s", name, side, type(value)), 3)
end

local function logical_binary(left, right, name, method)
  local a = logical_operand(left, name, "left")
  local b = logical_operand(right, name, "right")
  return a[method](a, b)
end

M.logical_and = function(left, right)
  return logical_binary(left, right, "logical_and", "logical_and")
end

M.logical_or = function(left, right)
  return logical_binary(left, right, "logical_or", "logical_or")
end

M.logical_xor = function(left, right)
  return logical_binary(left, right, "logical_xor", "ne")
end

M.logical_not = function(value)
  return logical_operand(value, "logical_not", "operand"):logical_not()
end

-- hypot has no scalar overload in luatorch, so materialize a missing tensor
-- operand before delegating to ATen.
M.hypot = function(left, right)
  check_operand(left, "hypot", "left")
  check_operand(right, "hypot", "right")
  if torch.is_tensor(left) then
    local other = torch.is_tensor(right) and right or scalar_tensor(right, left)
    return left:hypot(other)
  elseif torch.is_tensor(right) then
    local other = scalar_tensor(left, right)
    return other:hypot(right)
  end
  return torch.tensor(left):hypot(torch.tensor(right))
end

M.mod = M.remainder

return M
