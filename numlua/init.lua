--
-- numlua / init.lua
--
-- NumPy-shaped array API for LuaJIT, implemented as a thin facade over
-- luatorch tensors.  There is no second tensor engine here: array-producing
-- functions return luatorch Tensors and delegate the heavy lifting to one.
--
-- Conventions:
--   * dimensions / axes / indices are 1-based (the numeric-lua convention)
--   * reductions return luatorch Tensors, never bare Lua numbers
--   * all-axis reductions return 0-dimensional tensors
--   * dtypes are luatorch dtype names or luatorch dtype ids
--
-- See README.md and docs/api.md for the supported surface and limitations.
--

local torch = require("torch")
local npy = require("numlua.npy")
local text = require("numlua.text")
local linalg = require("numlua.linalg")
local manipulation = require("numlua.manipulation")
local ufunc = require("numlua.ufunc")
local npz = require("numlua.npz")

local numlua = {}

numlua.VERSION = "0.8.0"

-- ---------------------------------------------------------------------------
-- validation helpers
-- ---------------------------------------------------------------------------

local function check_tensor(value, name)
  if not torch.is_tensor(value) then
    error(string.format("numlua.%s expects a tensor, got %s",
      name, type(value)), 3)
  end
  return value
end

local function check_axis(axis, name)
  if axis == nil then
    return nil
  end
  if type(axis) ~= "number" or axis % 1 ~= 0 then
    error(string.format("numlua.%s: axis must be an integer or nil, got %s",
      name, tostring(axis)), 3)
  end
  if axis == 0 then
    error(string.format(
      "numlua.%s: axes are 1-based; axis 0 is invalid (use axis=1 for " ..
      "the first axis)", name), 3)
  end
  return axis
end

local function check_keepdims(keepdims, name)
  if keepdims ~= nil and type(keepdims) ~= "boolean" then
    error(string.format(
      "numlua.%s: keepdims must be a boolean or nil, got %s",
      name, type(keepdims)), 3)
  end
end

local function check_tensor_list(tensors, name)
  if type(tensors) ~= "table" or #tensors == 0 then
    error(string.format(
      "numlua.%s expects a non-empty table of tensors", name), 3)
  end
  for i = 1, #tensors do
    if not torch.is_tensor(tensors[i]) then
      error(string.format(
        "numlua.%s: element %d is not a tensor (got %s)",
        name, i, type(tensors[i])), 3)
    end
  end
end

-- ---------------------------------------------------------------------------
-- constructors and coercion
-- ---------------------------------------------------------------------------

-- Convert a nested Lua table (or scalar) to a luatorch Tensor.  Numeric Lua
-- tables become float32 unless opts.dtype is supplied, because Lua has no
-- syntactic distinction between integers and floats.
function numlua.array(value, opts)
  if opts ~= nil and type(opts) ~= "table" then
    error("numlua.array: options must be a table or nil", 2)
  end
  return torch.tensor(value, opts)
end

-- Like array(), but an existing Tensor is returned unchanged unless a dtype
-- is requested.
function numlua.asarray(value, opts)
  if opts ~= nil and type(opts) ~= "table" then
    error("numlua.asarray: options must be a table or nil", 2)
  end
  if torch.is_tensor(value) then
    if opts ~= nil and opts.dtype ~= nil then
      return value:to(opts.dtype)
    end
    return value
  end
  return torch.tensor(value, opts)
end

function numlua.zeros(...)
  return torch.zeros(...)
end

function numlua.ones(...)
  return torch.ones(...)
end

-- full(shape, fill_value[, opts]), where shape is a table or a single integer.
function numlua.full(shape, fill_value, opts)
  if type(shape) ~= "table" and type(shape) ~= "number" then
    error("numlua.full expects a shape table or a dimension", 2)
  end
  if type(fill_value) ~= "number" then
    error("numlua.full expects a numeric fill value, got " ..
      type(fill_value), 2)
  end
  if opts ~= nil and type(opts) ~= "table" then
    error("numlua.full: options must be a table or nil", 2)
  end
  local dims = type(shape) == "table" and shape or { shape }
  if #dims == 0 then
    error("numlua.full expects a non-empty shape", 2)
  end
  if opts ~= nil then
    return torch.full(dims, fill_value, opts)
  end
  return torch.full(dims, fill_value)
end

function numlua.eye(n, opts)
  return torch.eye(n, opts)
end

-- arange([start,] stop[, step][, opts]) follows luatorch: integer bounds infer
-- int64, while non-integer bounds infer float32.
function numlua.arange(...)
  return torch.arange(...)
end

-- linspace(start, stop[, num][, opts]); num defaults to NumPy's 50.
function numlua.linspace(start, stop, num, opts)
  if type(num) == "table" and opts == nil then
    opts = num
    num = nil
  end
  num = num or 50
  if opts ~= nil and type(opts) ~= "table" then
    error("numlua.linspace: options must be a table or nil", 2)
  end
  if opts ~= nil then
    return torch.linspace(start, stop, num, opts)
  end
  return torch.linspace(start, stop, num)
end

-- ---------------------------------------------------------------------------
-- shape
-- ---------------------------------------------------------------------------

function numlua.shape(t)
  check_tensor(t, "shape")
  return t:sizes()
end

function numlua.reshape(t, ...)
  check_tensor(t, "reshape")
  if select("#", ...) == 0 or (...) == nil then
    error("numlua.reshape expects a shape (table or variadic dimensions)", 2)
  end
  return t:reshape(...)
end

-- transpose(t[, axes]) reverses all axes by default, like NumPy.  axes is an
-- optional table of 1-based dimensions.
function numlua.transpose(t, axes)
  check_tensor(t, "transpose")
  local ndim = t:dim()
  if axes == nil then
    if ndim <= 1 then
      return t
    end
    local order = {}
    for i = 1, ndim do
      order[i] = ndim - i + 1
    end
    return t:permute(order)
  end
  if type(axes) ~= "table" then
    error("numlua.transpose: axes must be a table or nil", 2)
  end
  if #axes ~= ndim then
    error(string.format(
      "numlua.transpose expects %d axes, got %d", ndim, #axes), 2)
  end
  local seen = {}
  for i = 1, #axes do
    local axis = axes[i]
    if type(axis) ~= "number" or axis % 1 ~= 0 then
      error("numlua.transpose: axes must contain integers", 2)
    end
    if axis == 0 then
      error("numlua.transpose: axes are 1-based; axis 0 is invalid", 2)
    end
    local normalized = axis > 0 and axis or ndim + axis + 1
    if normalized < 1 or normalized > ndim then
      error(string.format(
        "numlua.transpose: axis %d is out of range for a %d-D tensor",
        axis, ndim), 2)
    end
    if seen[normalized] then
      error("numlua.transpose: axes must be unique", 2)
    end
    seen[normalized] = true
  end
  return t:permute(axes)
end

-- ravel returns a C-order (contiguous) 1-D view/copy.
function numlua.ravel(t)
  check_tensor(t, "ravel")
  return t:reshape({ t:numel() })
end

function numlua.squeeze(t, axis)
  check_tensor(t, "squeeze")
  if axis == nil then
    return t:squeeze()
  end
  check_axis(axis, "squeeze")
  return t:squeeze(axis)
end

function numlua.unsqueeze(t, axis)
  check_tensor(t, "unsqueeze")
  if axis == nil then
    error("numlua.unsqueeze expects an axis", 2)
  end
  check_axis(axis, "unsqueeze")
  return t:unsqueeze(axis)
end

-- ---------------------------------------------------------------------------
-- combining
-- ---------------------------------------------------------------------------

-- concatenate({t1, t2, ...}[, axis]) uses the first axis by default.
function numlua.concatenate(tensors, axis)
  check_tensor_list(tensors, "concatenate")
  axis = axis or 1
  check_axis(axis, "concatenate")
  return torch.cat(tensors, axis)
end

-- stack({t1, t2, ...}[, axis]) inserts a new axis; default is the first.
function numlua.stack(tensors, axis)
  check_tensor_list(tensors, "stack")
  axis = axis or 1
  check_axis(axis, "stack")
  return torch.stack(tensors, axis)
end

-- ---------------------------------------------------------------------------
-- reductions
-- ---------------------------------------------------------------------------

function numlua.sum(t, axis, keepdims)
  check_tensor(t, "sum")
  check_axis(axis, "sum")
  check_keepdims(keepdims, "sum")
  return t:sum(axis, keepdims)
end

-- ATen refuses to infer an output dtype for mean() on integer/bool tensors.
-- Follow NumPy's convention there: promote non-floating input to float64,
-- then reduce.  Floating inputs keep their dtype.
function numlua.mean(t, axis, keepdims)
  check_tensor(t, "mean")
  check_axis(axis, "mean")
  check_keepdims(keepdims, "mean")
  if not t:is_floating_point() then
    t = t:to("float64")
  end
  return t:mean(axis, keepdims)
end

function numlua.max(t, axis, keepdims)
  check_tensor(t, "max")
  check_axis(axis, "max")
  check_keepdims(keepdims, "max")
  -- luatorch returns (values, indices) for a dimensional max; numlua's max
  -- returns values only.  Use numlua.argmax/argmin for indices.
  return (t:max(axis, keepdims))
end

function numlua.min(t, axis, keepdims)
  check_tensor(t, "min")
  check_axis(axis, "min")
  check_keepdims(keepdims, "min")
  return (t:min(axis, keepdims))
end

-- argmax and argmin return 1-based flat indices when axis is nil, matching
-- luatorch.  Both delegate the reduction kernel to luatorch.
function numlua.median(t, axis, keepdims)
  check_tensor(t, "median")
  check_axis(axis, "median")
  check_keepdims(keepdims, "median")
  local value = t:median(axis, keepdims)
  return value
end

local function quantile_scalar(sorted, q, axis, keepdims)
  local n = axis == nil and sorted:numel() or sorted:size(axis)
  if n == 0 then error("numlua.quantile: empty input", 2) end
  local position = q * (n - 1) + 1
  local lower = math.floor(position)
  local upper = math.min(n, lower + 1)
  local fraction = position - lower
  if axis == nil then
    local a = sorted:get(lower)
    local b = sorted:get(upper)
    return a + (b - a) * fraction
  end
  local a = sorted:narrow(axis, lower, 1)
  local b = sorted:narrow(axis, upper, 1)
  local value = a:mul(1 - fraction):add(b:mul(fraction))
  if not keepdims then value = value:squeeze(axis) end
  return value
end

function numlua.quantile(t, q, axis, keepdims)
  check_tensor(t, "quantile")
  check_axis(axis, "quantile")
  check_keepdims(keepdims, "quantile")
  if type(q) == "number" then
    if q ~= q or q < 0 or q > 1 then
      error("numlua.quantile: q must be in [0, 1]", 2)
    end
    local sorted = axis == nil and t:reshape(-1):sort() or t:sort(axis)
    return quantile_scalar(sorted, q, axis, keepdims)
  end
  if type(q) ~= "table" or #q == 0 then
    error("numlua.quantile: q must be a number or non-empty table", 2)
  end
  for _, value in ipairs(q) do
    if type(value) ~= "number" or value ~= value or value < 0 or value > 1 then
      error("numlua.quantile: every q entry must be in [0, 1]", 2)
    end
  end
  local sorted = axis == nil and t:reshape(-1):sort() or t:sort(axis)
  if axis == nil then
    local values = {}
    for i, value in ipairs(q) do values[i] = quantile_scalar(sorted, value) end
    local dtype = t:is_floating_point() and t:dtype() or "float64"
    return torch.tensor(values, { dtype = dtype })
  end
  local parts = {}
  for i, value in ipairs(q) do
    parts[i] = quantile_scalar(sorted, value, axis, keepdims)
  end
  return torch.stack(parts, 1)
end
local function search_one(values, target, side)
  local low, high = 1, #values + 1
  while low < high do
    local mid = math.floor((low + high) / 2)
    if values[mid] < target or (side == "right" and values[mid] == target) then
      low = mid + 1
    else
      high = mid
    end
  end
  return low
end

function numlua.searchsorted(sorted, values, side)
  check_tensor(sorted, "searchsorted")
  if sorted:dim() ~= 1 then
    error("numlua.searchsorted expects a 1-D sorted tensor", 2)
  end
  side = side or "left"
  if side ~= "left" and side ~= "right" then
    error("numlua.searchsorted side must be 'left' or 'right'", 2)
  end
  local haystack = sorted:totable()
  if type(values) == "number" then
    return search_one(haystack, values, side)
  end
  if torch.is_tensor(values) then values = values:reshape(-1):totable() end
  if type(values) ~= "table" then
    error("numlua.searchsorted values must be a number, table, or tensor", 2)
  end
  local result = {}
  for i = 1, #values do result[i] = search_one(haystack, values[i], side) end
  return torch.tensor(result, { dtype = "int64" })
end

function numlua.bincount(input, weights, minlength)
  check_tensor(input, "bincount")
  return torch.bincount(input, weights, minlength)
end

function numlua.flatnonzero(t)
  check_tensor(t, "flatnonzero")
  local coords = t:reshape(-1):nonzero()
  if coords:numel() == 0 then return coords:reshape(-1) end
  return coords:narrow(2, 1, 1):squeeze(2)
end

function numlua.percentile(t, q, axis, keepdims)
  if type(q) == "table" then
    local scaled = {}
    for i, value in ipairs(q) do
      if type(value) ~= "number" or value < 0 or value > 100 then
        error("numlua.percentile: q entries must be in [0, 100]", 2)
      end
      scaled[i] = value / 100
    end
    return numlua.quantile(t, scaled, axis, keepdims)
  end
  if type(q) ~= "number" or q < 0 or q > 100 then
    error("numlua.percentile: q must be in [0, 100]", 2)
  end
  return numlua.quantile(t, q / 100, axis, keepdims)
end

function numlua.compress(condition, t, axis)
  check_tensor(condition, "compress condition")
  check_tensor(t, "compress")
  if condition:dtype() ~= "bool" then
    error("numlua.compress expects a boolean condition tensor", 2)
  end
  if axis ~= nil then
    error("numlua.compress with an axis is not implemented yet", 2)
  end
  if condition:numel() ~= t:numel() then
    error("numlua.compress condition must match the input size", 2)
  end
  return t:reshape(-1):masked_select(condition:reshape(-1))
end

function numlua.argmax(t, axis, keepdims)
  check_tensor(t, "argmax")
  check_axis(axis, "argmax")
  check_keepdims(keepdims, "argmax")
  return t:argmax(axis, keepdims)
end

function numlua.argmin(t, axis, keepdims)
  check_tensor(t, "argmin")
  check_axis(axis, "argmin")
  check_keepdims(keepdims, "argmin")
  return t:argmin(axis, keepdims)
end

-- ---------------------------------------------------------------------------
-- predicates and selection
-- ---------------------------------------------------------------------------

-- where() accepts scalar or tensor branches and scalar or tensor conditions.
-- Scalar operands become 0-D luatorch tensors so ATen performs all broadcasting.
local function as_condition(value)
  if torch.is_tensor(value) then
    if value:dtype() == "bool" then
      return value
    end
    return value:ne(0)
  end
  if type(value) == "number" then
    return torch.tensor(value):ne(0)
  end
  if type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.where expects a tensor condition, number, or boolean, got %s",
    type(value)), 3)
end

local function as_where_operand(value, name)
  if torch.is_tensor(value) then
    return value
  end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.where expects a tensor or number for %s, got %s",
    name, type(value)), 3)
end

function numlua.where(condition, x, y)
  return torch.where(as_condition(condition),
    as_where_operand(x, "x"), as_where_operand(y, "y"))
end

local function check_clip_bound(value, name)
  if type(value) == "number" or torch.is_tensor(value) then
    return value
  end
  error(string.format(
    "numlua.clip: %s must be a number or tensor, got %s",
    name, type(value)), 3)
end

-- clip() uses luatorch's scalar clamp when both bounds are numbers.  Tensor
-- bounds are combined with elementwise maximum/minimum so ATen handles their
-- broadcasting instead of numlua growing a broadcast layer.
function numlua.clip(t, minimum, maximum)
  check_tensor(t, "clip")
  if minimum == nil or maximum == nil then
    error("numlua.clip expects minimum and maximum bounds", 2)
  end
  check_clip_bound(minimum, "minimum")
  check_clip_bound(maximum, "maximum")
  if type(minimum) == "number" and type(maximum) == "number" then
    return t:clamp(minimum, maximum)
  end
  local lo = torch.is_tensor(minimum) and minimum or torch.tensor(minimum)
  local hi = torch.is_tensor(maximum) and maximum or torch.tensor(maximum)
  return t:maximum(lo):minimum(hi)
end

-- IEEE NaN is the only value not equal to itself, so luatorch's comparison
-- kernel already provides isnan without a new native primitive.
function numlua.isnan(t)
  check_tensor(t, "isnan")
  return t:ne(t)
end

local function logical_reduction(t, axis, keepdims, name)
  check_tensor(t, name)
  check_axis(axis, name)
  check_keepdims(keepdims, name)
  local result = t[name](t, axis, keepdims)
  -- luatorch ignores keepdim for all-axis all()/any(); reshape the 0-D result
  -- to NumPy's all-ones shape.  This is metadata only, not a kernel.
  if axis == nil and keepdims and t:dim() > 0 then
    local shape = {}
    for i = 1, t:dim() do
      shape[i] = 1
    end
    result = result:reshape(shape)
  end
  return result
end

function numlua.any(t, axis, keepdims)
  return logical_reduction(t, axis, keepdims, "any")
end

function numlua.all(t, axis, keepdims)
  return logical_reduction(t, axis, keepdims, "all")
end

-- A boolean nonzero mask summed along the axis is an int64 count; ATen's bool
-- sum reduction supplies the count kernel.
function numlua.count_nonzero(t, axis)
  check_tensor(t, "count_nonzero")
  check_axis(axis, "count_nonzero")
  return t:ne(0):sum(axis)
end

-- ---------------------------------------------------------------------------
-- linear algebra
-- ---------------------------------------------------------------------------

-- dot is the 1-D inner product.  Matrix/batched products live in matmul().
function numlua.dot(a, b)
  check_tensor(a, "dot")
  check_tensor(b, "dot")
  if a:dim() ~= 1 or b:dim() ~= 1 then
    error("numlua.dot expects two 1-D tensors; use numlua.matmul for " ..
      "matrix and batched products", 2)
  end
  return a:dot(b)
end

-- matmul delegates to luatorch/ATen, including batched and broadcasted forms.
function numlua.matmul(a, b)
  check_tensor(a, "matmul")
  check_tensor(b, "matmul")
  return a:matmul(b)
end

-- ---------------------------------------------------------------------------
-- dtype
-- ---------------------------------------------------------------------------

function numlua.dtype(t)
  check_tensor(t, "dtype")
  return t:dtype()
end

local DTYPE_RANK = {
  bool = 1, uint8 = 2, int8 = 3, int16 = 4, int32 = 5, int64 = 6,
  float16 = 7, float32 = 8, float64 = 9,
}

local function dtype_name(value, name)
  if torch.is_tensor(value) then return value:dtype() end
  if type(value) == "string" then
    if DTYPE_RANK[value] == nil then
      error("numlua." .. name .. ": unknown dtype '" .. value .. "'", 2)
    end
    return value
  end
  if type(value) == "number" then
    local resolved = torch.dtype_name(value)
    if resolved == nil then
      error("numlua." .. name .. ": unknown dtype id " .. tostring(value), 2)
    end
    return resolved
  end
  error("numlua." .. name .. " expects a dtype name, dtype id, or tensor", 2)
end

-- A deliberately small safe-promotion table for the dtypes the facade
-- exposes.  It is documented in docs/api.md; it is not a claim of complete
-- NumPy type-promotion parity.
function numlua.promote_types(left, right)
  left = dtype_name(left, "promote_types")
  right = dtype_name(right, "promote_types")
  if left == right then return left end
  local function integral(name) return not name:match("float") end
  if (left == "float32" and integral(right) and DTYPE_RANK[right] >= 5) or
      (right == "float32" and integral(left) and DTYPE_RANK[left] >= 5) then
    return "float64"
  end
  local rank = math.max(DTYPE_RANK[left], DTYPE_RANK[right])
  for name, value in pairs(DTYPE_RANK) do
    if value == rank then return name end
  end
  return "float64"
end

function numlua.result_type(...)
  local count = select("#", ...)
  if count == 0 then error("numlua.result_type expects at least one dtype", 2) end
  local result = dtype_name((select(1, ...)), "result_type")
  for index = 2, count do
    result = numlua.promote_types(result, (select(index, ...)))
  end
  return result
end

function numlua.can_cast(from, to)
  from = dtype_name(from, "can_cast")
  to = dtype_name(to, "can_cast")
  if from == to then return true end
  if DTYPE_RANK[from] > DTYPE_RANK[to] then return false end
  if from:match("float") and not to:match("float") then return false end
  if from ~= "bool" and to == "bool" then return false end
  return true
end

function numlua.astype(t, dtype)
  check_tensor(t, "astype")
  if dtype == nil then
    error("numlua.astype expects a dtype name or dtype id", 2)
  end
  return t:to(dtype)
end

-- ---------------------------------------------------------------------------
-- NumPy .npy I/O
-- ---------------------------------------------------------------------------

numlua.save = npy.save
numlua.load = npy.load
numlua.savez = npz.savez
numlua.loadz = npz.loadz

-- ---------------------------------------------------------------------------
-- whitespace / CSV text I/O
-- ---------------------------------------------------------------------------

numlua.loadtxt = text.loadtxt
numlua.savetxt = text.savetxt

local function export(module)
  for name, fn in pairs(module) do
    if numlua[name] ~= nil then
      error("duplicate numlua export: " .. name)
    end
    numlua[name] = fn
  end
end

export(manipulation)
export(ufunc)

numlua.linalg = linalg

return numlua
