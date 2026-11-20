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
  sin = "sin",
  cos = "cos",
  tanh = "tanh",
  square = "square",
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

return M
