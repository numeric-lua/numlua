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

local unpack = table.unpack or unpack

local numlua = {}

numlua.VERSION = "0.26.0"

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

-- identity(n[, opts]) is NumPy's alias for a square eye().  Validate the
-- dimension so a typo cannot silently create an empty or fractional matrix.
function numlua.identity(n, opts)
  if type(n) ~= "number" or n % 1 ~= 0 or n < 0 then
    error("numlua.identity: n must be a non-negative integer", 2)
  end
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

-- Shared validation for the logarithmic constructors.  The options table is
-- deliberately small: `endpoint` (default true) and `dtype` (default
-- float32), matching the rest of numlua's constructors.  NumPy's `axis`
-- argument is not meaningful for a fresh 1-D array and is not accepted.
local SPACE_OPTIONS = { dtype = true, endpoint = true }

local function check_finite_number(value, name, label)
  if type(value) ~= "number" or value ~= value or
      value == math.huge or value == -math.huge then
    error(string.format(
      "numlua.%s: %s must be a finite number, got %s",
      name, label, tostring(value)), 3)
  end
end

local function space_options(opts, name)
  if opts == nil then return true, "float32" end
  if type(opts) ~= "table" then
    error(string.format(
      "numlua.%s: options must be a table or nil", name), 3)
  end
  for key in pairs(opts) do
    if not SPACE_OPTIONS[key] then
      error(string.format(
        "numlua.%s: unknown option '%s'", name, tostring(key)), 3)
    end
  end
  if opts.endpoint ~= nil and type(opts.endpoint) ~= "boolean" then
    error(string.format(
      "numlua.%s: endpoint must be a boolean or nil", name), 3)
  end
  local dtype = opts.dtype or "float32"
  if type(dtype) ~= "string" and type(dtype) ~= "number" then
    error(string.format(
      "numlua.%s: dtype must be a name, dtype id, or nil", name), 3)
  end
  local ok, probe = pcall(torch.zeros, 0, { dtype = dtype })
  if not ok then
    error(string.format(
      "numlua.%s: invalid dtype: %s", name, tostring(probe)), 3)
  end
  if probe:dtype() == "bool" then
    error(string.format(
      "numlua.%s: bool is not a supported output dtype", name), 3)
  end
  return opts.endpoint ~= false, probe:dtype()
end

local function space_count(num, name)
  num = num == nil and 50 or num
  if type(num) ~= "number" or num % 1 ~= 0 or num < 0 then
    error(string.format(
      "numlua.%s: num must be a non-negative integer, got %s",
      name, tostring(num)), 3)
  end
  return num
end

-- Both logspace and geomspace work on base-10 exponents.  Computing the
-- exponents in float64 keeps a float32 result as accurate as NumPy's
-- compute-in-double-then-cast behavior.  `endpoint = false` takes the first
-- `num` points of a `num + 1`-point double-precision interval.
local function space_exponents(start, stop, num, endpoint)
  local count = endpoint and num or (num + 1)
  local exponents = torch.linspace(start, stop, count, { dtype = "float64" })
  if not endpoint then
    exponents = exponents:narrow(1, 1, num)
  end
  return exponents
end

-- logspace(start, stop[, num[, base[, opts]]]); opts accepts endpoint/dtype.
function numlua.logspace(start, stop, num, base, opts)
  if type(num) == "table" and base == nil and opts == nil then
    opts, num = num, nil
  end
  if type(base) == "table" and opts == nil then
    opts, base = base, nil
  end
  if opts ~= nil and type(opts) ~= "table" then
    error("numlua.logspace: options must be a table or nil", 2)
  end
  local endpoint, dtype = space_options(opts, "logspace")
  num = space_count(num, "logspace")
  check_finite_number(start, "logspace", "start")
  check_finite_number(stop, "logspace", "stop")
  base = base == nil and 10 or base
  check_finite_number(base, "logspace", "base")
  if base <= 0 then
    error("numlua.logspace: base must be positive", 2)
  end
  if num == 0 then
    return torch.zeros(0, { dtype = dtype })
  end
  local exponents = space_exponents(start, stop, num, endpoint)
  local base_tensor = torch.tensor(base, { dtype = "float64" })
  return base_tensor:pow(exponents):to(dtype)
end

-- geomspace(start, stop[, num[, opts]]); negative endpoints keep their common
-- sign, matching NumPy.  Zero endpoints are invalid because a geometric
-- sequence cannot include zero.
function numlua.geomspace(start, stop, num, opts)
  if type(num) == "table" and opts == nil then
    opts, num = num, nil
  end
  if opts ~= nil and type(opts) ~= "table" then
    error("numlua.geomspace: options must be a table or nil", 2)
  end
  local endpoint, dtype = space_options(opts, "geomspace")
  num = space_count(num, "geomspace")
  check_finite_number(start, "geomspace", "start")
  check_finite_number(stop, "geomspace", "stop")
  if start == 0 or stop == 0 then
    error("numlua.geomspace: start and stop must be non-zero", 2)
  end
  if (start < 0) ~= (stop < 0) then
    error("numlua.geomspace: start and stop must have the same sign", 2)
  end
  if num == 0 then
    return torch.zeros(0, { dtype = dtype })
  end
  local sign = start < 0 and -1 or 1
  local log = math.log(10)
  local exponents = space_exponents(math.log(math.abs(start)) / log,
    math.log(math.abs(stop)) / log, num, endpoint)
  local base_tensor = torch.tensor(10, { dtype = "float64" })
  local values = base_tensor:pow(exponents):to(dtype)
  if sign < 0 then
    values = values:mul(-1)
  end
  return values
end

-- ---------------------------------------------------------------------------
-- *_like constructors
-- ---------------------------------------------------------------------------

local LIKE_OPTIONS = { dtype = true, shape = true }

-- Validate the small, NumPy-shaped options table shared by the *_like
-- constructors.  `shape` is an optional NumPy 2.x override: a dimension or a
-- table (including an empty table for 0-D).  A nil shape uses the prototype's
-- shape.  `dtype` is passed to luatorch as a name or dtype id.
local function like_options(opts, name)
  if opts ~= nil and type(opts) ~= "table" then
    error(string.format(
      "numlua.%s: options must be a table or nil", name), 3)
  end
  opts = opts or {}
  for key in pairs(opts) do
    if not LIKE_OPTIONS[key] then
      error(string.format(
        "numlua.%s: unknown option '%s'", name, tostring(key)), 3)
    end
  end
  local dtype = opts.dtype
  if dtype ~= nil and type(dtype) ~= "string" and type(dtype) ~= "number" then
    error(string.format(
      "numlua.%s: dtype must be a name, dtype id, or nil", name), 3)
  end
  local shape = opts.shape
  if shape ~= nil then
    if type(shape) == "number" then
      shape = { shape }
    elseif type(shape) ~= "table" then
      error(string.format(
        "numlua.%s: shape must be a dimension or a table", name), 3)
    else
      local copy = {}
      for index = 1, #shape do copy[index] = shape[index] end
      shape = copy
    end
    for index = 1, #shape do
      local dimension = shape[index]
      if type(dimension) ~= "number" or dimension ~= dimension or
          dimension % 1 ~= 0 or dimension < 0 or
          dimension == math.huge or dimension == -math.huge then
        error(string.format(
          "numlua.%s: shape entries must be non-negative integers", name), 3)
      end
    end
  end
  return shape, dtype
end

local function like_alloc(proto, opts, name, kind, value)
  check_tensor(proto, name)
  local shape, dtype = like_options(opts, name)
  local dtype_opts = { dtype = dtype or proto:dtype() }
  if shape == nil then
    if kind == "zeros" then return torch.zeros_like(proto, dtype_opts) end
    if kind == "ones" then return torch.ones_like(proto, dtype_opts) end
    if kind == "full" then return torch.full_like(proto, value, dtype_opts) end
    if proto:dim() == 0 then
      return torch.empty({ 1 }, dtype_opts):reshape({})
    end
    return torch.empty(proto:sizes(), dtype_opts)
  end

  local scalar = #shape == 0
  local target = scalar and { 1 } or shape
  local result
  if kind == "zeros" then
    result = torch.zeros(target, dtype_opts)
  elseif kind == "ones" then
    result = torch.ones(target, dtype_opts)
  elseif kind == "full" then
    result = torch.full(target, value, dtype_opts)
  else
    result = torch.empty(target, dtype_opts)
  end
  if scalar then result = result:reshape({}) end
  return result
end

-- zeros_like / ones_like / empty_like / full_like follow np.*_like: the
-- prototype supplies shape and dtype unless overridden.  `empty_like` leaves
-- the new storage uninitialized; the other three initialize every element.
function numlua.zeros_like(t, opts)
  return like_alloc(t, opts, "zeros_like", "zeros")
end

function numlua.ones_like(t, opts)
  return like_alloc(t, opts, "ones_like", "ones")
end

function numlua.empty_like(t, opts)
  return like_alloc(t, opts, "empty_like", "empty")
end

function numlua.full_like(t, fill_value, opts)
  if type(fill_value) ~= "number" then
    error("numlua.full_like expects a numeric fill value, got " ..
      type(fill_value), 2)
  end
  return like_alloc(t, opts, "full_like", "full", fill_value)
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

-- average follows np.average: an optional axis, optional weights (1-D along
-- the axis or the input's full shape), and a `returned` flag that also gives
-- the sum of weights.  The arithmetic delegates to ATen.
function numlua.average(t, axis, weights, returned)
  check_tensor(t, "average")
  axis = check_axis(axis, "average")
  if returned ~= nil and type(returned) ~= "boolean" then
    error("numlua.average: returned must be a boolean or nil", 2)
  end
  local dim
  if axis ~= nil then
    local ndim = t:dim()
    dim = axis > 0 and axis or ndim + axis + 1
    if dim < 1 or dim > ndim then
      error(string.format(
        "numlua.average: axis %d is out of range for a %d-D tensor",
        axis, ndim), 2)
    end
  end

  if weights == nil then
    local value = numlua.mean(t, axis)
    if not returned then return value end
    local count
    if t:dim() == 0 then
      count = torch.tensor(1, { dtype = "float64" })
    else
      count = torch.ones(t:sizes(), { dtype = "float64" }):sum(axis)
    end
    return value, count
  end

  if not torch.is_tensor(weights) then
    if type(weights) == "table" then
      local ok, converted = pcall(torch.tensor, weights)
      if not ok then
        error("numlua.average: weights are not numeric: " ..
          tostring(converted), 2)
      end
      weights = converted
    else
      error("numlua.average expects weights to be a tensor, table, or nil", 2)
    end
  end

  if dim == nil then
    for index = 1, t:dim() do
      if weights:dim() ~= t:dim() or weights:size(index) ~= t:size(index) then
        error("numlua.average: weights must match the input shape when " ..
          "axis is omitted", 2)
      end
    end
  else
    if weights:dim() == 1 then
      if weights:size(1) ~= t:size(dim) then
        error(string.format(
          "numlua.average: weights have %d elements but axis %d has size %d",
          weights:size(1), dim, t:size(dim)), 2)
      end
      local shape = {}
      for index = 1, t:dim() do shape[index] = 1 end
      shape[dim] = t:size(dim)
      weights = weights:reshape(shape):expand(t:sizes())
    elseif weights:dim() == t:dim() then
      for index = 1, t:dim() do
        if weights:size(index) ~= t:size(index) then
          error("numlua.average: weights shape does not match the input", 2)
        end
      end
    else
      error("numlua.average: weights must be 1-D or match the input rank", 2)
    end
  end

  local denominator = weights:sum(dim)
  if denominator:eq(0):any():item() then
    error("numlua.average: sum of weights must not be zero", 2)
  end
  local value = t:mul(weights):sum(dim):div(denominator)
  if returned then return value, denominator end
  return value
end

-- prod follows the same axis/keepdims rules as sum.  luatorch/ATen performs
-- the reduction, including the empty-reduction identity of 1.
function numlua.prod(t, axis, keepdims)
  check_tensor(t, "prod")
  check_axis(axis, "prod")
  check_keepdims(keepdims, "prod")
  return t:prod(axis, keepdims)
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

-- NumPy's peak-to-peak range is max - min over the same axis.  bool input is
-- cast to int64 first because ATen (like NumPy's subtraction ufunc) does not
-- define bool subtraction; the result is an int64 tensor.
function numlua.ptp(t, axis, keepdims)
  check_tensor(t, "ptp")
  check_axis(axis, "ptp")
  check_keepdims(keepdims, "ptp")
  if t:dtype() == "bool" then
    t = t:to("int64")
  end
  return numlua.max(t, axis, keepdims):sub(numlua.min(t, axis, keepdims))
end

-- Number of elements reduced along one axis.  ddof denominators need this;
-- ATen's var/std only expose the unbiased 0/1 flag, so other ddof values are
-- handled by the sum-of-squared-deviations formula below.
local function reduction_count(t, axis, name)
  if axis == nil then
    return t:numel()
  end
  local ndim = t:dim()
  local normalized = axis > 0 and axis or ndim + axis + 1
  if normalized < 1 or normalized > ndim then
    error(string.format(
      "numlua.%s: axis %d is out of range for a %d-D tensor",
      name, axis, ndim), 3)
  end
  return t:size(normalized)
end

local function check_ddof(ddof, name)
  if ddof == nil then
    return 0
  end
  if type(ddof) ~= "number" or ddof ~= ddof or
      ddof == math.huge or ddof == -math.huge then
    error(string.format(
      "numlua.%s: ddof must be a finite number or nil, got %s",
      name, tostring(ddof)), 3)
  end
  return ddof
end

-- NumPy's var/std default to ddof = 0 (population variance), unlike
-- luatorch's unbiased default.  Integer and bool input promotes to float64,
-- matching mean().
local function variance(t, axis, keepdims, ddof, name)
  ddof = check_ddof(ddof, name)
  if not t:is_floating_point() then
    t = t:to("float64")
  end
  if ddof == 0 or ddof == 1 then
    return t:var(axis, ddof == 1, keepdims)
  end
  local count = reduction_count(t, axis, name)
  local mean = t:mean(axis, true)
  local difference = t:sub(mean)
  return difference:mul(difference):sum(axis, keepdims):div(count - ddof)
end

function numlua.var(t, axis, keepdims, ddof)
  check_tensor(t, "var")
  check_axis(axis, "var")
  check_keepdims(keepdims, "var")
  return variance(t, axis, keepdims, ddof, "var")
end

function numlua.std(t, axis, keepdims, ddof)
  check_tensor(t, "std")
  check_axis(axis, "std")
  check_keepdims(keepdims, "std")
  return variance(t, axis, keepdims, ddof, "std"):sqrt()
end

-- NaN-aware reductions.  IEEE NaN is the only value not equal to itself, so
-- the comparison kernel supplies the mask without a native predicate.  These
-- follow NumPy's ignore-NaN semantics: nansum treats NaN as zero, the other
-- reductions count only valid values, and an all-NaN slice yields NaN.
local function nan_mask(t)
  return t:ne(t)
end

local function valid_nan_count(t, axis, keepdims)
  return nan_mask(t):logical_not():to("float64"):sum(axis, keepdims)
end

function numlua.nansum(t, axis, keepdims)
  check_tensor(t, "nansum")
  check_axis(axis, "nansum")
  check_keepdims(keepdims, "nansum")
  if not t:is_floating_point() then
    return numlua.sum(t, axis, keepdims)
  end
  return t:masked_fill(nan_mask(t), 0):sum(axis, keepdims)
end

-- nanprod treats every NaN as the multiplicative identity, so all-NaN slices
-- produce 1, matching NumPy.  Non-floating input has no NaNs and uses prod.
function numlua.nanprod(t, axis, keepdims)
  check_tensor(t, "nanprod")
  check_axis(axis, "nanprod")
  check_keepdims(keepdims, "nanprod")
  if not t:is_floating_point() then
    return numlua.prod(t, axis, keepdims)
  end
  return t:masked_fill(nan_mask(t), 1):prod(axis, keepdims)
end

function numlua.nanmean(t, axis, keepdims)
  check_tensor(t, "nanmean")
  check_axis(axis, "nanmean")
  check_keepdims(keepdims, "nanmean")
  if not t:is_floating_point() then
    return numlua.mean(t, axis, keepdims)
  end
  local total = t:masked_fill(nan_mask(t), 0):sum(axis, keepdims)
  local count = valid_nan_count(t, axis, keepdims):to(t:dtype())
  return total:div(count)
end

local function nan_extremum(t, axis, keepdims, minimum, name)
  check_tensor(t, name)
  check_axis(axis, name)
  check_keepdims(keepdims, name)
  if not t:is_floating_point() then
    return minimum and numlua.min(t, axis, keepdims)
      or numlua.max(t, axis, keepdims)
  end
  local fill = minimum and math.huge or -math.huge
  local cleaned = t:masked_fill(nan_mask(t), fill)
  local result = minimum and cleaned:min(axis, keepdims)
    or cleaned:max(axis, keepdims)
  local empty = valid_nan_count(t, axis, keepdims):eq(0)
  return result:masked_fill(empty, 0 / 0)
end

function numlua.nanmin(t, axis, keepdims)
  return nan_extremum(t, axis, keepdims, true, "nanmin")
end

function numlua.nanmax(t, axis, keepdims)
  return nan_extremum(t, axis, keepdims, false, "nanmax")
end

local function nan_variance(t, axis, keepdims, ddof, name)
  ddof = check_ddof(ddof, name)
  if not t:is_floating_point() then
    t = t:to("float64")
  end
  local missing = nan_mask(t)
  local valid = missing:logical_not()
  local mean = numlua.nanmean(t, axis, true)
  local difference = t:sub(mean)
  local total = difference:mul(difference):masked_fill(missing, 0)
    :sum(axis, keepdims)
  local count = valid:to("float64"):sum(axis, keepdims):to(t:dtype())
  local result = total:div(count:sub(ddof))
  return result:masked_fill(count:eq(0), 0 / 0)
end

function numlua.nanvar(t, axis, keepdims, ddof)
  check_tensor(t, "nanvar")
  check_axis(axis, "nanvar")
  check_keepdims(keepdims, "nanvar")
  return nan_variance(t, axis, keepdims, ddof, "nanvar")
end

function numlua.nanstd(t, axis, keepdims, ddof)
  check_tensor(t, "nanstd")
  check_axis(axis, "nanstd")
  check_keepdims(keepdims, "nanstd")
  return nan_variance(t, axis, keepdims, ddof, "nanstd"):sqrt()
end

-- nan_to_num replaces NaN and infinities with finite values.  luatorch/ATen
-- supplies NumPy's dtype maxima when no replacement is given.
function numlua.nan_to_num(t, nan, posinf, neginf)
  check_tensor(t, "nan_to_num")
  for _, entry in ipairs({
    { nan, "nan" }, { posinf, "posinf" }, { neginf, "neginf" },
  }) do
    local value, label = entry[1], entry[2]
    if value ~= nil and type(value) ~= "number" then
      error(string.format(
        "numlua.nan_to_num: %s must be a number or nil, got %s",
        label, type(value)), 2)
    end
  end
  return t:nan_to_num(nan, posinf, neginf)
end

-- NaN-aware cumulative reductions treat NaN as the sum/product identity and
-- otherwise delegate the cumulative kernel to ATen.
function numlua.nancumsum(t, axis)
  check_tensor(t, "nancumsum")
  check_axis(axis, "nancumsum")
  if not t:is_floating_point() then
    return numlua.cumsum(t, axis)
  end
  return numlua.cumsum(t:masked_fill(nan_mask(t), 0), axis)
end

function numlua.nancumprod(t, axis)
  check_tensor(t, "nancumprod")
  check_axis(axis, "nancumprod")
  if not t:is_floating_point() then
    return numlua.cumprod(t, axis)
  end
  return numlua.cumprod(t:masked_fill(nan_mask(t), 1), axis)
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

-- trapezoid integrates along one axis with the trapezoidal rule, following
-- np.trapezoid (formerly np.trapz).  `x` may be a 1-D coordinate vector for
-- the selected axis or an array with the input's shape; when it is omitted,
-- `dx` supplies the uniform spacing.  The default axis is the last one.
local function trapezoid_impl(t, x, dx, axis_value, name)
  check_tensor(t, name)
  check_axis(axis_value, name)
  local ndim = t:dim()
  if ndim == 0 then
    error(string.format("numlua.%s expects at least a 1-D tensor", name), 3)
  end
  local axis = axis_value == nil and ndim
    or (axis_value > 0 and axis_value or ndim + axis_value + 1)
  if axis < 1 or axis > ndim then
    error(string.format(
      "numlua.%s: axis %d is out of range for a %d-D tensor",
      name, axis_value, ndim), 3)
  end
  local size = t:size(axis)
  if size <= 1 then
    -- An empty or single-point integration interval has zero area.
    return t:sum(axis):mul(0)
  end

  local left = t:narrow(axis, 1, size - 1)
  local right = t:narrow(axis, 2, size - 1)
  local average = left:add(right):mul(0.5)

  if x == nil then
    dx = dx == nil and 1 or dx
    if type(dx) ~= "number" then
      error(string.format(
        "numlua.%s: dx must be a number or nil", name), 3)
    end
    return average:mul(dx):sum(axis)
  end

  if not torch.is_tensor(x) then
    if type(x) == "table" then
      x = torch.tensor(x)
    else
      error(string.format(
        "numlua.%s: x must be a tensor, table, or nil", name), 3)
    end
  end
  if x:dim() == 1 then
    if x:size(1) ~= size then
      error(string.format(
        "numlua.%s: x has %d elements but axis %d has size %d",
        name, x:size(1), axis, size), 3)
    end
    local shape = {}
    for i = 1, ndim do shape[i] = 1 end
    shape[axis] = size
    x = x:reshape(shape):expand(t:sizes())
  else
    if x:dim() ~= ndim then
      error(string.format(
        "numlua.%s: x must be 1-D or match the input rank", name), 3)
    end
    for i = 1, ndim do
      if x:size(i) ~= t:size(i) then
        error(string.format(
          "numlua.%s: x shape does not match the input", name), 3)
      end
    end
  end

  local widths = x:narrow(axis, 2, size - 1):sub(
    x:narrow(axis, 1, size - 1))
  return average:mul(widths):sum(axis)
end

function numlua.trapezoid(t, x, dx, axis)
  return trapezoid_impl(t, x, dx, axis, "trapezoid")
end

-- NumPy kept np.trapz as the historical spelling; expose the same behavior.
function numlua.trapz(t, x, dx, axis)
  return trapezoid_impl(t, x, dx, axis, "trapz")
end

-- unwrap removes phase jumps larger than `discont` along one axis by adding a
-- multiple of `period`, following np.unwrap.  Integer input is promoted to
-- float64; the default period is 2*pi and the default discont is period/2.
function numlua.unwrap(t, discont, axis_value, period)
  check_tensor(t, "unwrap")
  check_axis(axis_value, "unwrap")
  period = period == nil and (2 * math.pi) or period
  if type(period) ~= "number" or period ~= period or
      period == math.huge or period == -math.huge or period <= 0 then
    error("numlua.unwrap: period must be a positive finite number", 2)
  end
  discont = discont == nil and (period / 2) or discont
  if type(discont) ~= "number" or discont ~= discont or
      discont == math.huge or discont == -math.huge or discont < 0 then
    error("numlua.unwrap: discont must be a finite non-negative number", 2)
  end

  if not t:is_floating_point() then
    t = t:to("float64")
  end
  local ndim = t:dim()
  if ndim == 0 then
    error("numlua.unwrap expects at least a 1-D tensor", 2)
  end
  local axis = axis_value == nil and ndim
    or (axis_value > 0 and axis_value or ndim + axis_value + 1)
  if axis < 1 or axis > ndim then
    error(string.format(
      "numlua.unwrap: axis %d is out of range for a %d-D tensor",
      axis_value, ndim), 2)
  end
  local size = t:size(axis)
  if size <= 1 then
    return t:clone()
  end

  local difference = numlua.diff(t, 1, axis)
  local half = period / 2
  local wrapped = difference:add(half):remainder(period):sub(half)
  -- NumPy keeps positive jumps that land exactly on the half-period boundary.
  local boundary = wrapped:eq(-half):logical_and(difference:gt(0))
  wrapped = wrapped:masked_fill(boundary, 0)
  local correction = wrapped:sub(difference)
  correction = correction:masked_fill(difference:abs():lt(discont), 0)

  local cumulative = numlua.cumsum(correction, axis)
  local result = t:clone()
  result:narrow(axis, 2, size - 1):add_(cumulative)
  return result
end

local function interp_vector(value, name, label)
  local result
  if torch.is_tensor(value) then
    result = value:to("float64")
  elseif type(value) == "table" then
    result = torch.tensor(value, { dtype = "float64" })
  else
    error(string.format(
      "numlua.interp: %s must be a 1-D tensor or table, got %s",
      label, type(value)), 2)
  end
  if result:dim() ~= 1 then
    error("numlua.interp: " .. label .. " must be 1-D", 2)
  end
  return result
end

-- interp performs linear interpolation of fp(x) over the increasing xp grid.
-- x may be a scalar or any-rank tensor/table; the result is float64 and keeps
-- x's shape.  Out-of-range values use `left`/`right`, defaulting to fp[1] and
-- fp[-1].
function numlua.interp(x, xp, fp, left, right)
  xp = interp_vector(xp, "interp", "xp")
  fp = interp_vector(fp, "interp", "fp")
  local count = xp:size(1)
  if count < 2 then
    error("numlua.interp: xp must contain at least two points", 2)
  end
  if fp:size(1) ~= count then
    error(string.format(
      "numlua.interp: xp has %d points but fp has %d",
      count, fp:size(1)), 2)
  end
  if not numlua.diff(xp):gt(0):all():item() then
    error("numlua.interp: xp must be strictly increasing", 2)
  end
  if left ~= nil and type(left) ~= "number" then
    error("numlua.interp: left must be a number or nil", 2)
  end
  if right ~= nil and type(right) ~= "number" then
    error("numlua.interp: right must be a number or nil", 2)
  end

  local original_shape
  local flat_x
  if torch.is_tensor(x) then
    original_shape = x:sizes()
    flat_x = x:to("float64"):reshape(-1)
  elseif type(x) == "table" then
    local tensor = torch.tensor(x, { dtype = "float64" })
    original_shape = tensor:sizes()
    flat_x = tensor:reshape(-1)
  elseif type(x) == "number" then
    original_shape = {}
    flat_x = torch.tensor(x, { dtype = "float64" }):reshape(-1)
  else
    error("numlua.interp: x must be a number, tensor, or table, got " ..
      type(x), 2)
  end

  local indices = numlua.searchsorted(xp, flat_x, "left")
  local lower = indices:clamp(2, count):sub(1)
  local upper = lower:add(1)
  local x0 = xp:index_select(1, lower)
  local x1 = xp:index_select(1, upper)
  local f0 = fp:index_select(1, lower)
  local f1 = fp:index_select(1, upper)
  local ratio = flat_x:sub(x0):div(x1:sub(x0))
  local values = f0:add(ratio:mul(f1:sub(f0)))
  values = values:masked_fill(flat_x:lt(xp:get(1)),
    left == nil and fp:get(1) or left)
  values = values:masked_fill(flat_x:gt(xp:get(count)),
    right == nil and fp:get(count) or right)
  values = values:masked_fill(flat_x:ne(flat_x), 0 / 0)
  return values:reshape(original_shape)
end

function numlua.flatnonzero(t)
  check_tensor(t, "flatnonzero")
  local coords = t:reshape(-1):nonzero()
  if coords:numel() == 0 then return coords:reshape(-1) end
  return coords:narrow(2, 1, 1):squeeze(2)
end

local UNIQUE_OPTIONS = {
  return_index = true,
  return_inverse = true,
  return_counts = true,
}

-- unique(t[, options]) flattens the input and returns the sorted unique
-- values, optionally followed by first-occurrence indices, the inverse
-- mapping, and counts in NumPy's order.  luatorch supplies the sort/unique
-- kernel; numlua only derives return_index from the inverse mapping.
function numlua.unique(t, options)
  check_tensor(t, "unique")
  if options ~= nil and type(options) ~= "table" then
    error("numlua.unique: options must be a table or nil", 2)
  end
  options = options or {}
  for key in pairs(options) do
    if not UNIQUE_OPTIONS[key] then
      error("numlua.unique: unknown option '" .. tostring(key) .. "'", 2)
    end
  end
  for key in pairs(UNIQUE_OPTIONS) do
    if options[key] ~= nil and type(options[key]) ~= "boolean" then
      error("numlua.unique: " .. key .. " must be a boolean or nil", 2)
    end
  end

  local want_index = options.return_index == true
  local want_inverse = options.return_inverse == true
  local want_counts = options.return_counts == true
  local flat = t:reshape({ t:numel() })
  local values, inverse, counts = flat:unique(
    true, want_index or want_inverse, want_counts)

  local index
  if want_index then
    local first = {}
    local mapping = inverse:reshape(-1):totable()
    for position = 1, #mapping do
      local slot = mapping[position]
      if first[slot] == nil then first[slot] = position end
    end
    index = torch.tensor(first, { dtype = "int64" })
  end
  if want_inverse then
    inverse = inverse:reshape(t:dim() == 0 and { 1 } or t:sizes())
  end

  if not (want_index or want_inverse or want_counts) then
    return values
  end
  local result = { values }
  if want_index then result[#result + 1] = index end
  if want_inverse then result[#result + 1] = inverse end
  if want_counts then result[#result + 1] = counts end
  return unpack(result, 1, #result)
end

-- nonzero/argwhere use luatorch's C-order coordinate kernel.  luatorch returns
-- a 1-D empty tensor for an empty selection, so normalize that to NumPy's
-- (0, ndim) coordinate shape before splitting or returning it.
local function nonzero_coordinates(t, name)
  check_tensor(t, name)
  if t:dim() == 0 then
    error(string.format(
      "numlua.%s expects at least a 1-D tensor (use numlua.flatnonzero " ..
      "for 0-D input)", name), 3)
  end
  local coords = t:nonzero()
  if coords:numel() == 0 then
    return torch.empty({ 0, t:dim() }, { dtype = "int64" })
  end
  return coords
end

-- nonzero returns one 1-based coordinate tensor per input axis, like NumPy's
-- tuple of index arrays.  The variable number of results follows the input
-- rank; assign one variable per axis or collect the results with a helper.
function numlua.nonzero(t)
  local coords = nonzero_coordinates(t, "nonzero")
  local result = {}
  for dim = 1, t:dim() do
    result[dim] = coords:narrow(2, dim, 1):squeeze(2)
  end
  return unpack(result, 1, #result)
end

-- argwhere returns an (N, ndim) int64 tensor of 1-based coordinates.
function numlua.argwhere(t)
  return nonzero_coordinates(t, "argwhere")
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

local function compress_condition(value)
  if torch.is_tensor(value) then
    if value:dtype() == "bool" then return value end
    return value:ne(0)
  end
  if type(value) == "table" then
    local condition = torch.tensor(value)
    if condition:dtype() == "bool" then return condition end
    return condition:ne(0)
  end
  error("numlua.compress expects a boolean tensor or table, got " ..
    type(value), 2)
end

-- compress(condition, t[, axis]) follows np.compress.  With an axis, the
-- 1-D condition selects slices along that axis; without one it selects from
-- the flattened input.  As in NumPy, a condition shorter than the selected
-- axis is used as a prefix, while a longer one is an error.  The result owns
-- independent storage.
function numlua.compress(condition, t, axis)
  condition = compress_condition(condition)
  check_tensor(t, "compress")
  if axis == nil then
    local flat_condition = condition:reshape(-1)
    local flat_input = t:reshape(-1)
    if flat_condition:numel() > flat_input:numel() then
      error(string.format(
        "numlua.compress: condition has %d elements but the input has " ..
        "only %d", flat_condition:numel(), flat_input:numel()), 2)
    end
    if flat_condition:numel() < flat_input:numel() then
      flat_input = flat_input:narrow(1, 1, flat_condition:numel())
    end
    return flat_input:masked_select(flat_condition)
  end

  check_axis(axis, "compress")
  if condition:dim() ~= 1 then
    error("numlua.compress with an axis expects a 1-D condition", 2)
  end
  local ndim = t:dim()
  local normalized = axis > 0 and axis or ndim + axis + 1
  if normalized < 1 or normalized > ndim then
    error(string.format(
      "numlua.compress: axis %d is out of range for a %d-D tensor",
      axis, ndim), 2)
  end
  local size = t:size(normalized)
  if condition:numel() > size then
    error(string.format(
      "numlua.compress: condition has %d elements but axis %d has size %d",
      condition:numel(), normalized, size), 2)
  end
  local positions = torch.arange(1, condition:numel() + 1,
    { dtype = "int64" })
  local selected = positions:masked_select(condition)
  return t:index_select(normalized, selected)
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

-- nanargmin/nanargmax ignore NaNs and error on an all-NaN slice, matching
-- NumPy.  Filling with the opposite infinity turns the reduction into the
-- ordinary ATen argmin/argmax after the valid-value check.
local function nan_arg_extremum(t, axis, keepdims, minimum, name)
  check_tensor(t, name)
  check_axis(axis, name)
  check_keepdims(keepdims, name)
  if not t:is_floating_point() then
    return minimum and numlua.argmin(t, axis, keepdims)
      or numlua.argmax(t, axis, keepdims)
  end
  local missing = nan_mask(t)
  local count = missing:logical_not():sum(axis, keepdims)
  if count:eq(0):any():item() then
    error(string.format("numlua.%s: all-NaN slice encountered", name), 2)
  end
  local fill = minimum and math.huge or -math.huge
  local cleaned = t:masked_fill(missing, fill)
  return minimum and cleaned:argmin(axis, keepdims)
    or cleaned:argmax(axis, keepdims)
end

function numlua.nanargmin(t, axis, keepdims)
  return nan_arg_extremum(t, axis, keepdims, true, "nanargmin")
end

function numlua.nanargmax(t, axis, keepdims)
  return nan_arg_extremum(t, axis, keepdims, false, "nanargmax")
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

-- select() mirrors np.select: conditions are tested in order, the first true
-- condition wins, and `default` (0 when omitted) is used where every condition
-- is false.  Conditions and choices accept tensors or Lua scalars, like
-- where(); ATen performs the broadcasting and dtype promotion.
local function select_condition(value, index)
  if torch.is_tensor(value) or type(value) == "number" or
      type(value) == "boolean" then
    return as_condition(value)
  end
  error(string.format(
    "numlua.select: condition %d must be a tensor, number, or boolean, " ..
    "got %s", index, type(value)), 3)
end

local function select_choice(value, index)
  if torch.is_tensor(value) then
    return value
  end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.select: choice %d must be a tensor, number, or boolean, got %s",
    index, type(value)), 3)
end

local function select_default(value)
  if value == nil then
    return torch.tensor(0)
  end
  if torch.is_tensor(value) then
    return value
  end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.select: default must be a tensor, number, or boolean, got %s",
    type(value)), 3)
end

function numlua.select(condlist, choicelist, default)
  if type(condlist) ~= "table" or #condlist == 0 then
    error("numlua.select expects a non-empty table of conditions", 2)
  end
  if type(choicelist) ~= "table" or #choicelist == 0 then
    error("numlua.select expects a non-empty table of choices", 2)
  end
  if #condlist ~= #choicelist then
    error(string.format(
      "numlua.select: condition and choice lists must have the same length " ..
      "(got %d conditions and %d choices)", #condlist, #choicelist), 2)
  end

  local conditions, choices = {}, {}
  for index = 1, #condlist do
    conditions[index] = select_condition(condlist[index], index)
    choices[index] = select_choice(choicelist[index], index)
  end
  local result = select_default(default)

  -- Folding from the last condition to the first makes the first matching
  -- condition win, matching np.select instead of the last.
  for index = #conditions, 1, -1 do
    local ok, selected = pcall(torch.where, conditions[index],
      choices[index], result)
    if not ok then
      error("numlua.select: condition " .. index ..
        " and its choice do not broadcast with the result: " ..
        tostring(selected), 3)
    end
    result = selected
  end
  return result
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

-- ---------------------------------------------------------------------------
-- closeness predicates
-- ---------------------------------------------------------------------------

local DEFAULT_RTOL = 1e-5
local DEFAULT_ATOL = 1e-8

local function closeness_operand(value, name, argument)
  if torch.is_tensor(value) then return value end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.%s expects tensors or numbers for %s, got %s",
    name, argument, type(value)), 3)
end

local function closeness_options(rtol, atol, equal_nan, name)
  if rtol == nil then rtol = DEFAULT_RTOL end
  if atol == nil then atol = DEFAULT_ATOL end
  if type(rtol) ~= "number" or rtol ~= rtol or rtol < 0 or
      rtol == math.huge then
    error("numlua." .. name .. ": rtol must be a finite non-negative " ..
      "number", 3)
  end
  if type(atol) ~= "number" or atol ~= atol or atol < 0 or
      atol == math.huge then
    error("numlua." .. name .. ": atol must be a finite non-negative " ..
      "number", 3)
  end
  if equal_nan ~= nil and type(equal_nan) ~= "boolean" then
    error("numlua." .. name .. ": equal_nan must be a boolean or nil", 3)
  end
  return rtol, atol, equal_nan == true
end

-- isclose only needs a bool result, so mixed or integral operands are cast
-- through float64 before applying NumPy's formula.  Keeping both operands in
-- their floating dtype preserves float32/float16 precision otherwise.
local function closeness_operands(a, b, name)
  a = closeness_operand(a, name, "a")
  b = closeness_operand(b, name, "b")
  if not a:is_floating_point() or not b:is_floating_point() or
      a:dtype() ~= b:dtype() then
    a, b = a:to("float64"), b:to("float64")
  end
  return a, b
end

-- isclose(a, b[, rtol[, atol[, equal_nan]]]) returns a bool tensor.  The
-- formula is NumPy's array closeness rule, with equal infinities treated as
-- close and finite/infinite pairs never close.  NaN is close only when
-- equal_nan is true.
function numlua.isclose(a, b, rtol, atol, equal_nan)
  local relative, absolute, nan_equal = closeness_options(
    rtol, atol, equal_nan, "isclose")
  a, b = closeness_operands(a, b, "isclose")

  local difference = a:sub(b):abs()
  local tolerance = b:abs():mul(relative):add(absolute)
  local finite = a:isinf():logical_not():logical_and(
    b:isinf():logical_not())
  local close = difference:le(tolerance):logical_and(finite)
  close = close:logical_or(a:eq(b))
  if nan_equal then
    close = close:logical_or(a:ne(a):logical_and(b:ne(b)))
  end
  return close
end

-- allclose returns a Lua boolean, matching np.allclose; it is a scalar
-- predicate rather than a tensor reduction.
function numlua.allclose(a, b, rtol, atol, equal_nan)
  return numlua.isclose(a, b, rtol, atol, equal_nan):all():item() ~= false
end

local function equality_operand(value, name, argument)
  if torch.is_tensor(value) then return value end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  error(string.format(
    "numlua.%s expects tensors or numbers for %s, got %s",
    name, argument, type(value)), 3)
end

local function check_equal_nan(value, name)
  if value ~= nil and type(value) ~= "boolean" then
    error("numlua." .. name .. ": equal_nan must be a boolean or nil", 3)
  end
  return value == true
end

-- array_equal compares shapes exactly and returns a Lua boolean.  NaN is not
-- equal to itself unless equal_nan is true, matching np.array_equal.
function numlua.array_equal(a, b, equal_nan)
  local nan_equal = check_equal_nan(equal_nan, "array_equal")
  a = equality_operand(a, "array_equal", "a")
  b = equality_operand(b, "array_equal", "b")
  local left, right = a:sizes(), b:sizes()
  if #left ~= #right then return false end
  for i = 1, #left do
    if left[i] ~= right[i] then return false end
  end
  local equal = a:eq(b)
  if nan_equal then
    equal = equal:logical_or(a:ne(a):logical_and(b:ne(b)))
  end
  return equal:all():item() ~= false
end

-- array_equiv allows broadcasting, so a scalar compares equal to a matching
-- array.  Incompatible shapes return false rather than raising, as in NumPy.
function numlua.array_equiv(a, b)
  a = equality_operand(a, "array_equiv", "a")
  b = equality_operand(b, "array_equiv", "b")
  local ok, result = pcall(function()
    return a:eq(b):all():item() ~= false
  end)
  if not ok then return false end
  return result
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

-- dot follows NumPy's dot semantics: a 1-D inner product for two vectors and
-- the usual last-axis/second-to-last-axis sum-product otherwise.  ATen's
-- matmul already implements exactly those shape rules, including batches.
function numlua.dot(a, b)
  check_tensor(a, "dot")
  check_tensor(b, "dot")
  if a:dim() == 0 or b:dim() == 0 then
    error("numlua.dot expects at least 1-D tensors", 2)
  end
  return a:matmul(b)
end

-- inner sums the product over the last axis of each operand and keeps the
-- leading dimensions, matching np.inner.  The reshape/mul/sum path delegates
-- the arithmetic to ATen and has no separate broadcast layer.
function numlua.inner(a, b)
  check_tensor(a, "inner")
  check_tensor(b, "inner")
  if a:dim() == 0 or b:dim() == 0 then
    error("numlua.inner expects at least 1-D tensors", 2)
  end
  local length = a:size(a:dim())
  if b:size(b:dim()) ~= length then
    error(string.format(
      "numlua.inner: last-axis sizes differ (%d and %d)",
      length, b:size(b:dim())), 2)
  end
  local a_rows, b_rows = 1, 1
  for i = 1, a:dim() - 1 do a_rows = a_rows * a:size(i) end
  for i = 1, b:dim() - 1 do b_rows = b_rows * b:size(i) end
  local left = a:reshape(a_rows, length)
  local right = b:reshape(b_rows, length)
  local products = left:unsqueeze(2):mul(right:unsqueeze(1)):sum(3)
  local shape = {}
  for i = 1, a:dim() - 1 do
    shape[#shape + 1] = a:size(i)
  end
  for i = 1, b:dim() - 1 do
    shape[#shape + 1] = b:size(i)
  end
  if #shape == 0 then
    return products:reshape({})
  end
  return products:reshape(shape)
end

-- outer flattens both operands in C order and returns their outer product,
-- following np.outer.  luatorch's outer kernel supplies the multiplication.
function numlua.outer(a, b)
  check_tensor(a, "outer")
  check_tensor(b, "outer")
  return a:reshape(-1):outer(b:reshape(-1))
end

-- cross computes the 3-vector cross product over a 1-based axis, defaulting to
-- the last axis like np.cross.  ATen broadcasts batch dimensions; both inputs
-- must have the same rank and the selected axis must have length 3.
function numlua.cross(a, b, axis)
  check_tensor(a, "cross")
  check_tensor(b, "cross")
  local ndim = a:dim()
  if ndim == 0 or b:dim() ~= ndim then
    error("numlua.cross expects tensors with the same non-zero rank", 2)
  end
  local normalized
  if axis == nil then
    normalized = ndim
  else
    check_axis(axis, "cross")
    normalized = axis > 0 and axis or ndim + axis + 1
    if normalized < 1 or normalized > ndim then
      error(string.format(
        "numlua.cross: axis %d is out of range for a %d-D tensor",
        axis, ndim), 2)
    end
  end
  if a:size(normalized) ~= 3 or b:size(normalized) ~= 3 then
    error("numlua.cross expects the selected axis to have length 3", 2)
  end
  return torch.cross(a, b, normalized)
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
numlua.fromfile = npy.fromfile
numlua.tofile = npy.tofile
numlua.savez = npz.savez
numlua.savez_compressed = npz.savez_compressed
numlua.loadz = npz.loadz

-- ---------------------------------------------------------------------------
-- whitespace / CSV text I/O
-- ---------------------------------------------------------------------------

numlua.loadtxt = text.loadtxt
numlua.savetxt = text.savetxt
numlua.genfromtxt = text.genfromtxt

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
