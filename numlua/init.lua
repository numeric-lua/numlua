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

numlua.VERSION = "0.20.0"

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
