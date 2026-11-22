--
-- numlua.manipulation: NumPy-shaped layout and selection helpers.
--
-- These functions are views or ATen operations; no kernels live here.
--
local torch = require("torch")

local unpack = table.unpack or unpack

local M = {}

local function tensor(value, name)
  if not torch.is_tensor(value) then
    error(string.format("numlua.%s expects a tensor, got %s",
      name, type(value)), 3)
  end
  return value
end

local function axis(value, ndim, name)
  if type(value) ~= "number" or value % 1 ~= 0 or value == 0 then
    error(string.format("numlua.%s: axis must be a non-zero integer", name), 3)
  end
  local normalized = value > 0 and value or ndim + value + 1
  if normalized < 1 or normalized > ndim then
    error(string.format("numlua.%s: axis %d is out of range for %d-D input",
      name, value, ndim), 3)
  end
  return normalized
end

local function long_tensor(values)
  return torch.tensor(values, { dtype = "int64" })
end

local function shape_list(value, name, required)
  if value == nil and not required then return nil end
  if type(value) ~= "table" or #value == 0 then
    error(string.format("numlua.%s expects a non-empty shape table", name), 3)
  end
  local result = {}
  for index = 1, #value do
    local dimension = value[index]
    if type(dimension) ~= "number" or dimension % 1 ~= 0 or dimension < 0 then
      error(string.format("numlua.%s: shape entries must be non-negative integers",
        name), 3)
    end
    result[index] = dimension
  end
  return result
end

local function broadcast_shape(shapes)
  local dimensions = 0
  for _, shape in ipairs(shapes) do dimensions = math.max(dimensions, #shape) end
  local result = {}
  for index = 1, dimensions do result[index] = 1 end
  for _, shape in ipairs(shapes) do
    local offset = dimensions - #shape
    for index = 1, #shape do
      local target = offset + index
      local current, incoming = result[target], shape[index]
      if current ~= incoming then
        -- Right-aligned broadcasting: a size-1 axis can grow to any size,
        -- including 0.  A zero-size axis therefore wins over 1, as in NumPy.
        if current == 1 then
          result[target] = incoming
        elseif incoming == 1 then
          result[target] = current
        else
          error(string.format(
            "numlua.broadcast_arrays: incompatible shapes at dimension %d " ..
            "(%d vs %d)", target, current, incoming), 3)
        end
      end
    end
  end
  return result
end

function M.broadcast_to(t, shape)
  tensor(t, "broadcast_to")
  shape = shape_list(shape, "broadcast_to", true)
  return t:expand(shape)
end

function M.broadcast_arrays(...)
  local inputs = { ... }
  if #inputs == 0 then
    error("numlua.broadcast_arrays expects at least one tensor", 2)
  end
  local shapes = {}
  for index = 1, #inputs do
    tensor(inputs[index], "broadcast_arrays")
    shapes[index] = inputs[index]:sizes()
  end
  local target = broadcast_shape(shapes)
  local result = {}
  for index = 1, #inputs do result[index] = inputs[index]:expand(target) end
  return (table.unpack or unpack)(result)
end

-- Normalize an axis or table of axes for moveaxis.  Negative values count from
-- the end and duplicates are rejected, matching NumPy.
local function axis_list(value, ndim, name)
  local values = type(value) == "table" and value or { value }
  if #values == 0 then
    error(string.format("numlua.%s expects at least one axis", name), 3)
  end
  local result, seen = {}, {}
  for index = 1, #values do
    local normalized = axis(values[index], ndim, name)
    if seen[normalized] then
      error(string.format("numlua.%s: axes must be unique", name), 3)
    end
    seen[normalized] = true
    result[index] = normalized
  end
  return result
end

-- swapaxes and moveaxis are metadata-only permutations: the result shares
-- storage with the input.
function M.swapaxes(t, axis1, axis2)
  tensor(t, "swapaxes")
  local ndim = t:dim()
  local first = axis(axis1, ndim, "swapaxes")
  local second = axis(axis2, ndim, "swapaxes")
  if first == second then return t end
  return t:transpose(first, second)
end

function M.moveaxis(t, source, destination)
  tensor(t, "moveaxis")
  local ndim = t:dim()
  local sources = axis_list(source, ndim, "moveaxis")
  local destinations = axis_list(destination, ndim, "moveaxis")
  if #sources ~= #destinations then
    error("numlua.moveaxis: source and destination must have the same " ..
      "number of axes", 2)
  end

  local moved = {}
  for index = 1, #sources do moved[sources[index]] = true end
  local order = {}
  for index = 1, ndim do
    if not moved[index] then order[#order + 1] = index end
  end
  local placements = {}
  for index = 1, #sources do
    placements[index] = {
      destination = destinations[index],
      source = sources[index],
    }
  end
  table.sort(placements, function(left, right)
    return left.destination < right.destination
  end)
  for index = 1, #placements do
    local placement = placements[index]
    table.insert(order, placement.destination, placement.source)
  end
  return t:permute(order)
end

local function atleast_one(t, rank, name)
  tensor(t, name)
  local ndim = t:dim()
  if ndim >= rank then return t end
  if rank == 1 then return t:unsqueeze(1) end
  if rank == 2 then
    if ndim == 0 then return t:unsqueeze(1):unsqueeze(1) end
    return t:unsqueeze(1)
  end
  if ndim == 0 then
    return t:unsqueeze(1):unsqueeze(1):unsqueeze(1)
  elseif ndim == 1 then
    return t:unsqueeze(1):unsqueeze(3)
  end
  return t:unsqueeze(3)
end

-- Like broadcast_arrays, atleast_* accept one or more tensors and return one
-- result per input.  Promotion only adds size-1 dimensions, so it is a view.
local function atleast(rank, name, ...)
  local inputs = { ... }
  if #inputs == 0 then
    error(string.format("numlua.%s expects at least one tensor", name), 2)
  end
  local results = {}
  for index = 1, #inputs do
    results[index] = atleast_one(inputs[index], rank, name)
  end
  return (table.unpack or unpack)(results)
end

function M.atleast_1d(...) return atleast(1, "atleast_1d", ...) end
function M.atleast_2d(...) return atleast(2, "atleast_2d", ...) end
function M.atleast_3d(...) return atleast(3, "atleast_3d", ...) end

function M.tile(t, reps)
  tensor(t, "tile")
  if type(reps) == "number" then reps = { reps } end
  if type(reps) ~= "table" or #reps == 0 then
    error("numlua.tile expects a repetition table or number", 2)
  end
  local shape = t:sizes()
  local ndim = #shape
  local target = {}
  if #reps < ndim then
    for index = 1, ndim - #reps do target[index] = 1 end
  end
  for index = 1, #reps do
    local count = reps[index]
    if type(count) ~= "number" or count % 1 ~= 0 or count < 1 then
      error("numlua.tile: repetitions must be positive integers", 2)
    end
    target[#target + 1] = count
  end
  local result_shape = {}
  local result_dims = math.max(ndim, #target)
  local padded = {}
  for index = 1, result_dims - ndim do padded[index] = 1 end
  for index = 1, ndim do padded[#padded + 1] = shape[index] end
  local view = t
  if result_dims > ndim then view = t:reshape(padded) end
  for dim = 1, result_dims do
    result_shape[dim] = padded[dim] * target[dim]
    if target[dim] > 1 then
      local indices = {}
      for block = 1, target[dim] do
        for value = 1, padded[dim] do indices[#indices + 1] = value end
      end
      view = view:index_select(dim, long_tensor(indices))
    end
  end
  return view
end

-- resize returns a new tensor with the requested shape.  It follows NumPy's
-- ndarray.resize data rule: keep the C-order elements that fit, zero-fill any
-- new space, and drop surplus elements.  Unlike np.resize, it never tiles the
-- input, and unlike ndarray.resize, it does not mutate the input.
function M.resize(t, ...)
  tensor(t, "resize")
  local shape
  if select("#", ...) == 1 and type((...)) == "table" then
    shape = shape_list((...), "resize", true)
  else
    shape = shape_list({ ... }, "resize", true)
  end

  local target_count = 1
  for index = 1, #shape do target_count = target_count * shape[index] end
  local dtype = t:dtype()
  if target_count == 0 then
    return torch.zeros(shape, { dtype = dtype })
  end

  local old_count = t:numel()
  local keep = math.min(old_count, target_count)
  local result
  if keep > 0 then
    result = t:reshape({ old_count }):narrow(1, 1, keep):clone()
  else
    result = torch.zeros({ 0 }, { dtype = dtype })
  end
  if keep < target_count then
    result = torch.cat({ result,
      torch.zeros({ target_count - keep }, { dtype = dtype }) }, 1)
  end
  return result:reshape(shape)
end

M["repeat"] = function(t, repeats, axis_value)
  tensor(t, "repeat")
  if type(repeats) == "number" then
    if repeats % 1 ~= 0 or repeats < 1 then
      error("numlua.repeat: repeats must be a positive integer", 2)
    end
    if axis_value == nil then
      local flat = t:reshape({ t:numel() })
      local indices = {}
      for index = 1, flat:numel() do
        for _ = 1, repeats do indices[#indices + 1] = index end
      end
      return flat:index_select(1, long_tensor(indices))
    end
    local dim = axis(axis_value, t:dim(), "repeat")
    local indices = {}
    for index = 1, t:size(dim) do
      for _ = 1, repeats do indices[#indices + 1] = index end
    end
    return t:index_select(dim, long_tensor(indices))
  end
  if type(repeats) ~= "table" then
    error("numlua.repeat: repeats must be a number or integer table", 2)
  end
  if axis_value == nil then
    t = t:reshape(-1)
    axis_value = 1
  end
  local dim = axis(axis_value, t:dim(), "repeat")
  if #repeats ~= t:size(dim) then
    error("numlua.repeat: repeats must have one entry per element on the axis", 2)
  end
  local indices = {}
  for index = 1, #repeats do
    local count = repeats[index]
    if type(count) ~= "number" or count % 1 ~= 0 or count < 1 then
      error("numlua.repeat: repeats must be positive integers", 2)
    end
    for _ = 1, count do indices[#indices + 1] = index end
  end
  return t:index_select(dim, long_tensor(indices))
end

function M.flip(t, axis_value)
  tensor(t, "flip")
  local axes = type(axis_value) == "table" and axis_value or { axis_value }
  local result = t
  for _, value in ipairs(axes) do
    result = result:flip(axis(value, t:dim(), "flip"))
  end
  return result
end

local function roll_axis(t, shift, dim)
  local size = t:size(dim)
  if size == 0 then return t:clone() end
  shift = shift % size
  if shift == 0 then return t:clone() end
  local left = t:narrow(dim, size - shift + 1, shift)
  local right = t:narrow(dim, 1, size - shift)
  return torch.cat({ left, right }, dim)
end

function M.roll(t, shift, axis_value)
  tensor(t, "roll")
  if type(shift) ~= "number" or shift % 1 ~= 0 then
    error("numlua.roll: shift must be an integer", 2)
  end
  if axis_value == nil then
    local flat = t:reshape({ t:numel() })
    return roll_axis(flat, shift, 1):reshape(t:sizes())
  end
  return roll_axis(t, shift, axis(axis_value, t:dim(), "roll"))
end

local function pad_entry(pair)
  if type(pair) ~= "table" then
    error("numlua.pad: each pad_width pair must be a table", 2)
  end
  local before, after
  if #pair == 1 then
    before, after = pair[1], pair[1]
  elseif #pair == 2 then
    before, after = pair[1], pair[2]
  else
    error("numlua.pad: each pad_width pair must have one or two entries", 2)
  end
  if type(before) ~= "number" or type(after) ~= "number" or
      before % 1 ~= 0 or after % 1 ~= 0 or before < 0 or after < 0 then
    error("numlua.pad: each pad_width pair must be non-negative integers", 2)
  end
  return { before, after }
end

local function pad_pair(pad_width, ndim)
  local function broadcast(before, after)
    if type(before) ~= "number" or type(after) ~= "number" or
        before % 1 ~= 0 or after % 1 ~= 0 or before < 0 or after < 0 then
      error("numlua.pad: pad_width must be non-negative integers", 2)
    end
    local result = {}
    for index = 1, ndim do result[index] = { before, after } end
    return result
  end
  if type(pad_width) == "number" then
    return broadcast(pad_width, pad_width)
  end
  if type(pad_width) ~= "table" or #pad_width == 0 then
    error("numlua.pad: pad_width must be a number or non-empty table", 2)
  end
  if type(pad_width[1]) == "number" then
    if #pad_width == 1 then
      return broadcast(pad_width[1], pad_width[1])
    end
    if #pad_width == 2 and type(pad_width[2]) == "number" then
      return broadcast(pad_width[1], pad_width[2])
    end
    error("numlua.pad: pad_width must be a number or table of pairs", 2)
  end

  local first = pad_entry(pad_width[1])
  if #pad_width == 1 then
    return broadcast(first[1], first[2])
  end
  if #pad_width ~= ndim then
    error("numlua.pad: per-axis pad_width must have one pair per dimension", 2)
  end
  local result = { first }
  for index = 2, ndim do result[index] = pad_entry(pad_width[index]) end
  return result
end

-- Normalize one per-axis fill-value entry: a number fills both sides, a one-
-- or two-entry table gives the before/after values separately.  The label
-- keeps the diagnostic specific to the mode being configured.
local function pad_value_entry(value, label)
  if type(value) == "number" then
    return { value, value }
  end
  if type(value) ~= "table" then
    error(string.format(
      "numlua.pad: %s entries must be numbers or pairs", label), 2)
  end
  local before, after
  if #value == 1 then
    before, after = value[1], value[1]
  elseif #value == 2 then
    before, after = value[1], value[2]
  else
    error(string.format(
      "numlua.pad: %s pairs must have one or two entries", label), 2)
  end
  if type(before) ~= "number" or type(after) ~= "number" then
    error(string.format("numlua.pad: %ss must be numeric", label), 2)
  end
  return { before, after }
end

-- Normalize an optional per-axis value argument using NumPy's common
-- broadcasting forms: a number fills every side, a numeric pair gives the
-- before/after values for every axis, and a table with one entry per
-- dimension gives per-axis entries (each a number or a before/after pair).
-- `nil` returns `default` on every side.  This backs constant_values and
-- linear_ramp's end_values.
local function pad_value_pairs(value, ndim, label, default)
  local function broadcast(before, after)
    if type(before) ~= "number" or type(after) ~= "number" then
      error(string.format("numlua.pad: %ss must be numeric", label), 2)
    end
    local result = {}
    for index = 1, ndim do result[index] = { before, after } end
    return result
  end
  if value == nil then
    return broadcast(default, default)
  end
  if type(value) == "number" then
    return broadcast(value, value)
  end
  if type(value) ~= "table" or #value == 0 then
    error(string.format(
      "numlua.pad: %s must be a number or non-empty table", label), 2)
  end
  if type(value[1]) == "number" then
    if #value == 1 then
      return broadcast(value[1], value[1])
    end
    if #value == 2 then
      return broadcast(value[1], value[2])
    end
    error(string.format(
      "numlua.pad: %s must be a number, a pair, or per-axis entries",
      label), 2)
  end
  if #value ~= ndim then
    error(string.format(
      "numlua.pad: per-axis %ss must have one entry per dimension",
      label), 2)
  end
  local result = {}
  for index = 1, ndim do
    result[index] = pad_value_entry(value[index], label)
  end
  return result
end

-- Validate one stat_length entry.  NumPy accepts zero here but maximum and
-- minimum cannot reduce an empty chunk; requiring a positive integer keeps
-- every statistic mode well-defined.
local function pad_stat_entry(value)
  if type(value) == "number" then
    return { value, value }
  end
  if type(value) ~= "table" then
    error("numlua.pad: stat length entries must be positive integers", 2)
  end
  local before, after
  if #value == 1 then
    before, after = value[1], value[1]
  elseif #value == 2 then
    before, after = value[1], value[2]
  else
    error("numlua.pad: stat length pairs must have one or two entries", 2)
  end
  if type(before) ~= "number" or type(after) ~= "number" or
      before % 1 ~= 0 or after % 1 ~= 0 or before < 1 or after < 1 then
    error("numlua.pad: stat lengths must be positive integers", 2)
  end
  return { before, after }
end

-- Normalize stat_length using the same number/pair/per-axis forms as NumPy.
-- A nil entry means the whole axis, and lengths larger than the axis are
-- clamped by the caller.
local function pad_stat_lengths(value, ndim)
  local function broadcast(before, after)
    if type(before) ~= "number" or type(after) ~= "number" or
        before % 1 ~= 0 or after % 1 ~= 0 or before < 1 or after < 1 then
      error("numlua.pad: stat lengths must be positive integers", 2)
    end
    local result = {}
    for index = 1, ndim do result[index] = { before, after } end
    return result
  end
  if value == nil then
    local result = {}
    for index = 1, ndim do result[index] = { nil, nil } end
    return result
  end
  if type(value) == "number" then
    return broadcast(value, value)
  end
  if type(value) ~= "table" or #value == 0 then
    error("numlua.pad: stat length must be a positive integer or " ..
      "non-empty table", 2)
  end
  if type(value[1]) == "number" then
    if #value == 1 then
      return broadcast(value[1], value[1])
    end
    if #value == 2 then
      return broadcast(value[1], value[2])
    end
    error("numlua.pad: stat length must be a positive integer, a pair, " ..
      "or per-axis entries", 2)
  end
  if #value ~= ndim then
    error("numlua.pad: per-axis stat lengths must have one entry per " ..
      "dimension", 2)
  end
  local result = {}
  for index = 1, ndim do
    result[index] = pad_stat_entry(value[index])
  end
  return result
end

-- Round half to even, matching NumPy's np.round for integer linear_ramp
-- end_values.  Lua's math.floor handles negative values correctly here.
local function round_half_even(value)
  local floor = math.floor(value)
  local fraction = value - floor
  if fraction > 0.5 then
    return floor + 1
  end
  if fraction < 0.5 then
    return floor
  end
  return floor % 2 == 0 and floor or floor + 1
end

-- Build linear_ramp weights for one side of one axis.  The ramp excludes the
-- edge value: np.pad uses linspace(end_value, edge, width, endpoint=False),
-- then reverses the right ramp.
local function pad_ramp_weights(dim, width, ndim, reverse)
  local values = {}
  for index = 1, width do
    if reverse then
      values[index] = (width - index) / width
    else
      values[index] = (index - 1) / width
    end
  end
  local shape = {}
  for index = 1, ndim do shape[index] = 1 end
  shape[dim] = width
  return torch.tensor(values, { dtype = "float64" }):reshape(shape)
end

-- Build one ramp side from a size-1 edge slice.  Arithmetic is done in
-- float64 and cast back, so integer ramps truncate toward zero like NumPy's
-- linspace(..., dtype=int) and float16 does not lose the end value early.
local function pad_ramp_side(edge, end_value, dim, width, ndim, dtype, reverse)
  local weights = pad_ramp_weights(dim, width, ndim, reverse)
  local ramp = edge:to("float64"):mul(weights)
  ramp = ramp:add(weights:mul(-1):add(1):mul(end_value))
  return ramp:to(dtype)
end

-- Reduce an edge chunk along one axis for the maximum, mean, median, and
-- minimum pad modes.  Bool is not a valid reduction dtype in ATen, so those
-- statistics are computed through uint8 (max/min) or float64 (mean/median)
-- and cast back; NumPy likewise casts the statistic to the input dtype.
local function pad_stat_value(chunk, dim, mode, dtype)
  local value
  if mode == "maximum" or mode == "minimum" then
    local source = dtype == "bool" and chunk:to("uint8") or chunk
    if mode == "maximum" then
      value = source:max(dim, true)
    else
      value = source:min(dim, true)
    end
    return value:to(dtype)
  end
  local source = chunk:is_floating_point() and chunk or chunk:to("float64")
  if mode == "mean" then
    value = source:mean(dim, true)
  else
    value = source:median(dim)
  end
  -- NumPy rounds integer statistics (np.round, half to even) before casting;
  -- bool is not an integer dtype there, so nonzero fractions stay true.
  if dtype ~= "bool" and not dtype:match("float") then
    value = value:round()
  end
  return value:to(dtype)
end

-- Build the source indices for one axis.  edge clamps to the first/last
-- element, reflect mirrors without repeating the edge, symmetric mirrors with
-- the edge included, and wrap repeats the axis cyclically.
local function pad_indices(size, before, after, mode)
  local total = before + size + after
  local indices = {}
  if mode == "edge" then
    for index = 1, before do indices[index] = 1 end
    for index = 1, size do indices[before + index] = index end
    for index = 1, after do indices[before + size + index] = size end
  elseif mode == "reflect" then
    if size == 1 then
      for index = 1, total do indices[index] = 1 end
    else
      local period = 2 * (size - 1)
      for index = 1, total do
        local position = (index - before - 1) % period
        if position < size then
          indices[index] = position + 1
        else
          indices[index] = period - position + 1
        end
      end
    end
  elseif mode == "symmetric" then
    local period = 2 * size
    for index = 1, total do
      local position = (index - before - 1) % period
      if position < size then
        indices[index] = position + 1
      else
        indices[index] = period - position
      end
    end
  elseif mode == "wrap" then
    for index = 1, total do
      indices[index] = ((index - before - 1) % size) + 1
    end
  end
  return long_tensor(indices)
end

local PAD_STAT_MODES = {
  maximum = true,
  mean = true,
  median = true,
  minimum = true,
}

local PAD_MODES = {
  constant = true,
  edge = true,
  linear_ramp = true,
  maximum = true,
  mean = true,
  median = true,
  minimum = true,
  reflect = true,
  symmetric = true,
  wrap = true,
}

local PAD_MODE_LIST = "'constant', 'edge', 'linear_ramp', 'maximum', " ..
  "'mean', 'median', 'minimum', 'reflect', 'symmetric', or 'wrap'"

function M.pad(t, pad_width, mode, value)
  tensor(t, "pad")
  mode = mode or "constant"
  if not PAD_MODES[mode] then
    error(string.format("numlua.pad: invalid mode '%s'; expected %s",
      tostring(mode), PAD_MODE_LIST), 2)
  end
  local ndim = t:dim()
  local dtype = t:dtype()
  local pairs = pad_pair(pad_width, ndim)

  -- `value` is mode-specific, following NumPy's keyword names: the default
  -- fill for constant, end_values for linear_ramp, and stat_length for the
  -- four statistic modes.  The other modes take no value.
  local constants, end_values, stat_lengths
  if mode == "constant" then
    constants = pad_value_pairs(value, ndim, "constant value", 0)
  elseif mode == "linear_ramp" then
    end_values = pad_value_pairs(value, ndim, "end value", 0)
    if dtype ~= "bool" and not dtype:match("float") then
      for dim = 1, ndim do
        end_values[dim][1] = round_half_even(end_values[dim][1])
        end_values[dim][2] = round_half_even(end_values[dim][2])
      end
    end
  elseif PAD_STAT_MODES[mode] then
    stat_lengths = pad_stat_lengths(value, ndim)
  elseif value ~= nil then
    error(string.format(
      "numlua.pad: mode '%s' does not accept a value", mode), 2)
  end

  local result = t:clone()

  if mode == "constant" then
    -- The historical scalar form keeps working; tables extend it to NumPy's
    -- per-axis constant_values, including separate before/after values.
    for dim = ndim, 1, -1 do
      local before, after = pairs[dim][1], pairs[dim][2]
      if before > 0 or after > 0 then
        local shape = result:sizes()
        local parts = {}
        if before > 0 then
          local fill_shape = { unpack(shape) }
          fill_shape[dim] = before
          parts[#parts + 1] = torch.full(fill_shape, constants[dim][1],
            { dtype = dtype })
        end
        parts[#parts + 1] = result
        if after > 0 then
          local fill_shape = { unpack(shape) }
          fill_shape[dim] = after
          parts[#parts + 1] = torch.full(fill_shape, constants[dim][2],
            { dtype = dtype })
        end
        result = torch.cat(parts, dim)
      end
    end
    return result
  end

  local shape = t:sizes()
  for dim = 1, ndim do
    if shape[dim] == 0 and (pairs[dim][1] > 0 or pairs[dim][2] > 0) then
      error(string.format(
        "numlua.pad: can't extend empty axis %d using mode '%s'; use " ..
        "'constant'", dim, mode), 2)
    end
  end

  -- Axes are padded in order, as in NumPy: later statistics and ramps see the
  -- values added to earlier axes, which is what makes the corners consistent.
  for dim = 1, ndim do
    local before, after = pairs[dim][1], pairs[dim][2]
    if before > 0 or after > 0 then
      if mode == "linear_ramp" then
        local size = result:size(dim)
        local parts = {}
        if before > 0 then
          parts[#parts + 1] = pad_ramp_side(result:narrow(dim, 1, 1),
            end_values[dim][1], dim, before, ndim, dtype, false)
        end
        parts[#parts + 1] = result
        if after > 0 then
          parts[#parts + 1] = pad_ramp_side(result:narrow(dim, size, 1),
            end_values[dim][2], dim, after, ndim, dtype, true)
        end
        result = torch.cat(parts, dim)
      elseif PAD_STAT_MODES[mode] then
        local size = result:size(dim)
        local lengths = stat_lengths[dim]
        local left_length = math.min(size, lengths[1] or size)
        local right_length = math.min(size, lengths[2] or size)
        local left_stat = pad_stat_value(
          result:narrow(dim, 1, left_length), dim, mode, dtype)
        local right_stat = pad_stat_value(
          result:narrow(dim, size - right_length + 1, right_length), dim,
          mode, dtype)
        local parts = {}
        if before > 0 then
          local fill_shape = result:sizes()
          fill_shape[dim] = before
          parts[#parts + 1] = left_stat:expand(fill_shape)
        end
        parts[#parts + 1] = result
        if after > 0 then
          local fill_shape = result:sizes()
          fill_shape[dim] = after
          parts[#parts + 1] = right_stat:expand(fill_shape)
        end
        result = torch.cat(parts, dim)
      else
        result = result:index_select(dim,
          pad_indices(result:size(dim), before, after, mode))
      end
    end
  end
  return result
end

function M.diff(t, n, axis_value)
  tensor(t, "diff")
  n = n or 1
  if type(n) ~= "number" or n % 1 ~= 0 or n < 0 then
    error("numlua.diff: n must be a non-negative integer", 2)
  end
  local dim = axis_value == nil and t:dim() or axis(axis_value, t:dim(), "diff")
  local result = t
  for _ = 1, n do
    local size = result:size(dim)
    if size <= 1 then
      result = result:narrow(dim, 1, 0)
    else
      result = result:narrow(dim, 2, size - 1):sub(
        result:narrow(dim, 1, size - 1))
    end
  end
  return result
end

function M.cumsum(t, axis_value)
  tensor(t, "cumsum")
  if axis_value == nil then return t:reshape({ t:numel() }):cumsum(1) end
  return t:cumsum(axis(axis_value, t:dim(), "cumsum"))
end

function M.cumprod(t, axis_value)
  tensor(t, "cumprod")
  if axis_value == nil then return t:reshape({ t:numel() }):cumprod(1) end
  return t:cumprod(axis(axis_value, t:dim(), "cumprod"))
end

function M.argsort(t, axis_value, descending)
  tensor(t, "argsort")
  if t:dim() == 0 then
    error("numlua.argsort expects at least a 1-D tensor", 2)
  end
  if descending ~= nil and type(descending) ~= "boolean" then
    error("numlua.argsort: descending must be a boolean or nil", 2)
  end
  local dim = axis_value == nil and t:dim() or axis(axis_value, t:dim(), "argsort")
  local _, indices = t:sort(dim, descending == true)
  return indices
end

local TAKE_INDEX_DTYPES = {
  int64 = true,
  int32 = true,
  int16 = true,
  int8 = true,
  uint8 = true,
}

local function check_take_indices(indices)
  if not torch.is_tensor(indices) then
    error("numlua.take: indices must be an integer Tensor", 3)
  end
  local dtype = indices:dtype()
  if not TAKE_INDEX_DTYPES[dtype] then
    error("numlua.take: indices must be an integer Tensor, got " .. dtype, 3)
  end
end

-- Validate 1-based index values before index_select so errors name the
-- facade's convention and the offending index instead of ATen's 0-based
-- message.
local function check_take_range(indices, size, label)
  if indices:numel() == 0 then return end
  local lowest = indices:min():item()
  local highest = indices:max():item()
  if lowest < 1 or highest > size then
    local offending = lowest < 1 and lowest or highest
    error(string.format(
      "numlua.take: index %d is out of range for %s with size %d " ..
      "(indices are 1-based)", offending, label, size), 3)
  end
end

-- NumPy's take() shape rule: indices replace the selected axis (or the whole
-- flattened input when no axis is given), and an integer index removes it.
local function take_shape(input_shape, index_shape, dim)
  local target = {}
  if dim ~= nil then
    for index = 1, dim - 1 do
      target[#target + 1] = input_shape[index]
    end
  end
  for index = 1, #index_shape do
    target[#target + 1] = index_shape[index]
  end
  if dim ~= nil then
    for index = dim + 1, #input_shape do
      target[#target + 1] = input_shape[index]
    end
  end
  return target
end

function M.take(t, indices, axis_value)
  tensor(t, "take")
  check_take_indices(indices)
  local index_shape = indices:sizes()
  local flat_indices = indices:reshape({ indices:numel() })
  if axis_value == nil then
    if t:numel() == 0 and indices:numel() > 0 then
      error("numlua.take: cannot take from an empty tensor " ..
        "with non-empty indices", 2)
    end
    check_take_range(flat_indices, t:numel(), "the flattened input")
    local selected = t:reshape({ t:numel() }):index_select(1, flat_indices)
    return selected:reshape(take_shape(t:sizes(), index_shape, nil))
  end
  local normalized = axis(axis_value, t:dim(), "take")
  if t:size(normalized) == 0 and indices:numel() > 0 then
    error("numlua.take: cannot take from empty axis " .. normalized ..
      " with non-empty indices", 2)
  end
  check_take_range(flat_indices, t:size(normalized), "axis " .. normalized)
  local selected = t:index_select(normalized, flat_indices)
  return selected:reshape(take_shape(t:sizes(), index_shape, normalized))
end

-- take_along_axis follows NumPy's shape rule: indices and input have the same
-- rank, non-axis dimensions broadcast in either direction (1 against N), and
-- the length along the axis comes from indices.  ATen's gather needs matching
-- non-axis sizes, so numlua expands both operands to the broadcast shape.
local function check_along_indices(indices)
  if not torch.is_tensor(indices) then
    error("numlua.take_along_axis: indices must be an integer Tensor", 3)
  end
  local dtype = indices:dtype()
  if not TAKE_INDEX_DTYPES[dtype] then
    error("numlua.take_along_axis: indices must be an integer Tensor, got " ..
      dtype, 3)
  end
end

-- Validate 1-based index values before gather so errors name the facade's
-- convention and the offending index instead of ATen's 0-based message.
local function check_along_range(indices, size, label)
  if indices:numel() == 0 then return end
  local lowest = indices:min():item()
  local highest = indices:max():item()
  if lowest < 1 or highest > size then
    local offending = lowest < 1 and lowest or highest
    error(string.format(
      "numlua.take_along_axis: index %d is out of range for %s with size %d " ..
      "(indices are 1-based)", offending, label, size), 3)
  end
end

function M.take_along_axis(t, indices, axis_value)
  tensor(t, "take_along_axis")
  check_along_indices(indices)
  if axis_value == nil then
    -- NumPy flattens both operands when axis is None and requires a 1-D
    -- index tensor.
    if indices:dim() ~= 1 then
      error("numlua.take_along_axis: when axis is omitted, indices must be " ..
        "1-D", 2)
    end
    check_along_range(indices, t:numel(), "the flattened input")
    return t:reshape({ t:numel() }):index_select(1, indices)
  end

  local ndim = t:dim()
  local dim = axis(axis_value, ndim, "take_along_axis")
  if indices:dim() ~= ndim then
    error(string.format(
      "numlua.take_along_axis: indices must have the same rank as the input " ..
      "(got %d for a %d-D input)", indices:dim(), ndim), 2)
  end

  local input_shape = t:sizes()
  local index_shape = indices:sizes()
  local input_target = {}
  local index_target = {}
  for index = 1, ndim do
    if index == dim then
      input_target[index] = input_shape[index]
      index_target[index] = index_shape[index]
    else
      local current, incoming = input_shape[index], index_shape[index]
      local target
      if current == incoming then
        target = current
      elseif current == 1 then
        target = incoming
      elseif incoming == 1 then
        target = current
      else
        error(string.format(
          "numlua.take_along_axis: shapes do not broadcast on axis %d " ..
          "(input %d, indices %d)", index, current, incoming), 2)
      end
      input_target[index] = target
      index_target[index] = target
    end
  end

  check_along_range(indices, input_shape[dim], "axis " .. dim)
  local input = t:expand(input_target)
  local index_tensor = indices:expand(index_target)
  return input:gather(dim, index_tensor)
end

-- Convert one callback result to a tensor and enforce NumPy's result-rank
-- rule: apply_along_axis callbacks may return a scalar or a 1-D array, never
-- a higher-rank array.  Lua numbers/tables are materialized through
-- torch.tensor, so a callback can stay in plain Lua.
local function apply_axis_result(value)
  if torch.is_tensor(value) then
    if value:dim() > 1 then
      error(string.format(
        "numlua.apply_along_axis: function must return a scalar or 1-D " ..
        "array, got a %d-D tensor", value:dim()), 3)
    end
    return value
  end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  if type(value) == "table" then
    local ok, result = pcall(torch.tensor, value)
    if not ok then
      error("numlua.apply_along_axis: function returned an invalid " ..
        "table: " .. tostring(result), 3)
    end
    if result:dim() > 1 then
      error(string.format(
        "numlua.apply_along_axis: function must return a scalar or 1-D " ..
        "array, got a %d-D tensor", result:dim()), 3)
    end
    return result
  end
  error("numlua.apply_along_axis: function must return a scalar, 1-D " ..
    "tensor, number, boolean, or table, got " .. type(value), 3)
end

-- apply_along_axis(func, axis, t[, ...]) mirrors np.apply_along_axis.  The
-- callback receives a 1-D view of each slice along `axis` and must return a
-- scalar or a 1-D tensor/table/number; any returned dimensions are appended
-- after the non-axis input dimensions, exactly as in NumPy.  Extra arguments
-- are forwarded to the callback.  Slices are visited in C order.
function M.apply_along_axis(func, axis_value, t, ...)
  tensor(t, "apply_along_axis")
  if type(func) ~= "function" then
    error("numlua.apply_along_axis expects a function as the first " ..
      "argument", 2)
  end
  local ndim = t:dim()
  if ndim == 0 then
    error("numlua.apply_along_axis expects at least a 1-D tensor", 2)
  end
  local dim = axis(axis_value, ndim, "apply_along_axis")

  -- Dimensions and sizes of the slice iteration space, i.e. everything except
  -- `axis`.  `outer_dims` is ascending for view selection; `coords` is indexed
  -- by position in that list.
  local outer_dims, outer_shape = {}, {}
  local count = 1
  for index = 1, ndim do
    if index ~= dim then
      outer_dims[#outer_dims + 1] = index
      outer_shape[#outer_dims] = t:size(index)
      count = count * t:size(index)
    end
  end
  if count == 0 then
    error("numlua.apply_along_axis: input has no slices along axis " ..
      dim .. " (a non-axis dimension is zero)", 2)
  end

  local args = { ... }
  local arg_count = select("#", ...)
  local coords = {}
  local out, out_shape
  local expected_dim, expected_length

  local function current_slice()
    local view = t
    -- Descending selection keeps the remaining dimension numbers valid.
    for index = #outer_dims, 1, -1 do
      view = view:select(outer_dims[index], coords[index])
    end
    return view
  end

  local function store(result)
    if #out_shape == 0 then
      return
    end
    local target = out
    for index = #coords, 1, -1 do
      target = target:select(index, coords[index])
    end
    if result:dim() == 0 then
      target:fill_(result:item())
    else
      target:copy_(result:detach())
    end
  end

  local function accept(result)
    if expected_dim == nil then
      expected_dim = result:dim()
      expected_length = expected_dim == 1 and result:size(1) or nil
      out_shape = {}
      for index = 1, #outer_shape do
        out_shape[index] = outer_shape[index]
      end
      if expected_dim == 1 then
        out_shape[#out_shape + 1] = expected_length
      end
      if #out_shape == 0 then
        -- A 1-D input with a scalar callback has no iteration axes left; the
        -- single result is the 0-D output.
        out = result:detach():clone()
      else
        out = torch.zeros(out_shape, { dtype = result:dtype() })
      end
    else
      if result:dim() ~= expected_dim then
        error(string.format(
          "numlua.apply_along_axis: function returned a %d-D result after " ..
          "a %d-D result", result:dim(), expected_dim), 2)
      end
      if expected_dim == 1 and result:size(1) ~= expected_length then
        error(string.format(
          "numlua.apply_along_axis: function returned a result of length " ..
          "%d after length %d", result:size(1), expected_length), 2)
      end
    end
    store(result)
  end

  local function visit(depth)
    if depth > #outer_shape then
      accept(apply_axis_result(func(current_slice(),
        unpack(args, 1, arg_count))))
      return
    end
    for index = 1, outer_shape[depth] do
      coords[depth] = index
      visit(depth + 1)
    end
  end
  visit(1)
  return out
end

-- Convert one apply_over_axes callback result to a tensor.  NumPy requires an
-- ndarray; numlua also accepts the scalar and Lua-table spellings that
-- apply_along_axis accepts, then applies the same rank rule.
local function apply_over_axes_result(value)
  if torch.is_tensor(value) then
    return value
  end
  if type(value) == "number" or type(value) == "boolean" then
    return torch.tensor(value)
  end
  if type(value) == "table" then
    local ok, result = pcall(torch.tensor, value)
    if not ok then
      error("numlua.apply_over_axes: function returned an invalid " ..
        "table: " .. tostring(result), 3)
    end
    return result
  end
  error("numlua.apply_over_axes: function must return a tensor, scalar, " ..
    "or table, got " .. type(value), 3)
end

-- apply_over_axes(func, t, axes[, ...]) mirrors np.apply_over_axes.  Axes are
-- applied in order to the current value and may repeat.  A callback result
-- that keeps the rank is used as-is (NumPy's keepdims-style callback); a
-- result that removed the selected axis is re-expanded at that 1-based
-- position, which keeps the rank stable across axes.  Any other rank change is
-- an error, as in NumPy.  Extra positional arguments are forwarded to the
-- callback, matching apply_along_axis.
function M.apply_over_axes(func, t, axes_value, ...)
  tensor(t, "apply_over_axes")
  if type(func) ~= "function" then
    error("numlua.apply_over_axes expects a function as the first " ..
      "argument", 2)
  end
  if t:dim() == 0 then
    error("numlua.apply_over_axes: function is not applied to a scalar " ..
      "(0-D) tensor", 2)
  end
  local axes = axes_value
  if type(axes) == "number" then
    axes = { axes }
  end
  if type(axes) ~= "table" then
    error("numlua.apply_over_axes expects an axis or a table of axes, got " ..
      type(axes_value), 2)
  end

  local args = { ... }
  local arg_count = select("#", ...)
  local value = t
  for index = 1, #axes do
    local ndim = value:dim()
    local dim = axis(axes[index], ndim, "apply_over_axes")
    local result = apply_over_axes_result(func(value, dim,
      unpack(args, 1, arg_count)))
    if result:dim() == ndim then
      value = result
    elseif result:dim() == ndim - 1 then
      value = result:unsqueeze(dim)
    else
      error(string.format(
        "numlua.apply_over_axes: function is not returning an array of the " ..
        "correct shape (got a %d-D result for %d-D input on axis %d; " ..
        "expected %d-D or %d-D)", result:dim(), ndim, dim, ndim, ndim - 1),
        2)
    end
  end
  return value
end

local PUT_MODES = {
  raise = true,
  wrap = true,
  clip = true,
}

-- Flatten a nested Lua table of put indices, accepting only integers.  The
-- facade is 1-based, so a value of 0 is never valid; each mode decides what
-- to do with indices outside 1 .. numel(t).
local function flatten_put_indices(value)
  local result = {}
  local function visit(item)
    if type(item) == "table" then
      for index = 1, #item do visit(item[index]) end
      return
    end
    if type(item) ~= "number" then
      error("numlua.put: indices must be integers", 0)
    end
    -- NaN and infinities have no meaningful wrap/clip result; reject them
    -- explicitly instead of letting a NaN target reach Tensor:set.
    if item ~= item or item == math.huge or item == -math.huge then
      error("numlua.put: indices must be finite integers", 0)
    end
    if item % 1 ~= 0 then
      error("numlua.put: indices must be integers", 0)
    end
    result[#result + 1] = item
  end
  visit(value)
  return result
end

local function flatten_put_values(value)
  local result = {}
  local function visit(item)
    if type(item) == "table" then
      for index = 1, #item do visit(item[index]) end
      return
    end
    if type(item) ~= "number" and type(item) ~= "boolean" then
      error("numlua.put: values must be numbers or booleans", 0)
    end
    result[#result + 1] = item
  end
  visit(value)
  return result
end

local function put_indices(value)
  if torch.is_tensor(value) then
    local dtype = value:dtype()
    if not TAKE_INDEX_DTYPES[dtype] then
      error("numlua.put: indices must be an integer Tensor, got " .. dtype, 2)
    end
    if value:numel() == 0 then return {} end
    return value:reshape({ value:numel() }):totable()
  end
  if type(value) == "number" then
    if value ~= value or value == math.huge or value == -math.huge then
      error("numlua.put: indices must be finite integers", 2)
    end
    if value % 1 ~= 0 then
      error("numlua.put: indices must be integers", 2)
    end
    return { value }
  end
  if type(value) == "table" then
    return flatten_put_indices(value)
  end
  error("numlua.put: indices must be an integer Tensor, number, or table", 2)
end

local function put_values(value)
  if torch.is_tensor(value) then
    if value:numel() == 0 then return {} end
    return value:reshape({ value:numel() }):totable()
  end
  if type(value) == "number" or type(value) == "boolean" then
    return { value }
  end
  if type(value) == "table" then
    return flatten_put_values(value)
  end
  error("numlua.put: values must be a Tensor, number, boolean, or table", 2)
end

-- Convert Lua values to what Tensor:set expects for the target dtype.
-- Integer targets follow a C-style cast (truncation toward zero) instead of
-- luatorch's floor, and booleans become 1/0 there.  A bool target truth-tests
-- numeric values, matching NumPy's cast to bool.
local function cast_put_values(values, dtype)
  local boolean_target = dtype == "bool"
  local integral_target = not boolean_target and not dtype:match("float")
  local result = {}
  for index = 1, #values do
    local value = values[index]
    if boolean_target then
      if type(value) == "boolean" then
        result[index] = value
      else
        result[index] = value ~= 0
      end
    elseif type(value) == "boolean" then
      result[index] = value and 1 or 0
    elseif integral_target and type(value) == "number" and value % 1 ~= 0 then
      result[index] = value >= 0 and math.floor(value) or math.ceil(value)
    else
      result[index] = value
    end
  end
  return result
end

-- np.put-shaped flat assignment.  numlua is 1-based, so the valid range is
-- 1 .. numel(t) and negative indices are out of range (wrap/clip handle them
-- with those 1-based bounds).  The input tensor is modified and returned, as
-- in NumPy; clone first when a copy is wanted.
function M.put(t, indices, values, mode)
  tensor(t, "put")
  mode = mode or "raise"
  if not PUT_MODES[mode] then
    error(string.format(
      "numlua.put: mode must be 'raise', 'wrap', or 'clip', got %s",
      tostring(mode)), 2)
  end
  local targets = put_indices(indices)
  local replacements = cast_put_values(put_values(values), t:dtype())
  local count = t:numel()

  if #targets == 0 then return t end
  if count == 0 then
    error("numlua.put: cannot put into an empty tensor " ..
      "with non-empty indices", 2)
  end
  if #replacements == 0 then
    error("numlua.put: values must not be empty when indices are " ..
      "non-empty", 2)
  end

  local flat = t:reshape({ count })
  for position = 1, #targets do
    local raw = targets[position]
    local target
    if mode == "raise" then
      if raw < 1 or raw > count then
        error(string.format(
          "numlua.put: index %d is out of range for a tensor with %d " ..
          "elements", raw, count), 2)
      end
      target = raw
    elseif mode == "wrap" then
      target = ((raw - 1) % count) + 1
    elseif raw < 1 then
      target = 1
    elseif raw > count then
      target = count
    else
      target = raw
    end
    flat:set(target, replacements[((position - 1) % #replacements) + 1])
  end

  -- A contiguous reshape above is a view, so the writes already landed in t.
  -- For non-contiguous inputs reshape materializes a copy; copy it back.
  if not t:is_contiguous() then
    t:copy_(flat:reshape(t:sizes()))
  end
  return t
end

M.repeat_ = M["repeat"]

return M
