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

local numlua = {}

numlua.VERSION = "0.1.0"

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
  -- returns values only.  Use numlua.argmax for indices.
  return (t:max(axis, keepdims))
end

function numlua.min(t, axis, keepdims)
  check_tensor(t, "min")
  check_axis(axis, "min")
  check_keepdims(keepdims, "min")
  return (t:min(axis, keepdims))
end

-- argmax returns 1-based flat indices when axis is nil, matching luatorch.
function numlua.argmax(t, axis, keepdims)
  check_tensor(t, "argmax")
  check_axis(axis, "argmax")
  check_keepdims(keepdims, "argmax")
  return t:argmax(axis, keepdims)
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

return numlua
