--
-- numlua.manipulation: NumPy-shaped layout and selection helpers.
--
-- These functions are views or ATen operations; no kernels live here.
--
local torch = require("torch")

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
      if current ~= 1 and incoming ~= 1 and current ~= incoming then
        error(string.format(
          "numlua.broadcast_arrays: incompatible shapes at dimension %d " ..
          "(%d vs %d)", target, current, incoming), 3)
      end
      result[target] = math.max(current, incoming)
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

-- Build the source indices for one axis.  edge clamps to the first/last
-- element, reflect mirrors without repeating the edge, and symmetric mirrors
-- with the edge included.
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
  end
  return long_tensor(indices)
end

function M.pad(t, pad_width, mode, value)
  tensor(t, "pad")
  mode = mode or "constant"
  if mode ~= "constant" and mode ~= "edge" and mode ~= "reflect" and
      mode ~= "symmetric" then
    error(string.format("numlua.pad: invalid mode '%s'; expected 'constant', " ..
      "'edge', 'reflect', or 'symmetric'", tostring(mode)), 2)
  end
  local pairs = pad_pair(pad_width, t:dim())
  local result = t:clone()

  if mode == "constant" then
    value = value or 0
    if type(value) ~= "number" then
      error("numlua.pad: constant value must be numeric", 2)
    end
    for dim = t:dim(), 1, -1 do
      local before, after = pairs[dim][1], pairs[dim][2]
      if before > 0 or after > 0 then
        local shape = result:sizes()
        local parts = {}
        if before > 0 then
          local fill_shape = { unpack(shape) }
          fill_shape[dim] = before
          parts[#parts + 1] = torch.full(fill_shape, value,
            { dtype = result:dtype() })
        end
        parts[#parts + 1] = result
        if after > 0 then
          local fill_shape = { unpack(shape) }
          fill_shape[dim] = after
          parts[#parts + 1] = torch.full(fill_shape, value,
            { dtype = result:dtype() })
        end
        result = torch.cat(parts, dim)
      end
    end
    return result
  end

  local shape = t:sizes()
  for dim = 1, t:dim() do
    if shape[dim] == 0 and (pairs[dim][1] > 0 or pairs[dim][2] > 0) then
      error(string.format(
        "numlua.pad: can't extend empty axis %d using mode '%s'; use " ..
        "'constant'", dim, mode), 2)
    end
  end
  for dim = 1, t:dim() do
    local before, after = pairs[dim][1], pairs[dim][2]
    if before > 0 or after > 0 then
      result = result:index_select(dim,
        pad_indices(result:size(dim), before, after, mode))
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
  local dim = axis_value == nil and t:dim() or axis(axis_value, t:dim(), "argsort")
  local _, indices = t:sort(dim, descending == true)
  return indices
end

function M.take(t, indices, axis_value)
  tensor(t, "take")
  if not torch.is_tensor(indices) then
    error("numlua.take: indices must be an integer Tensor", 2)
  end
  if axis_value == nil then
    return t:reshape({ t:numel() }):index_select(1, indices)
  end
  return t:index_select(axis(axis_value, t:dim(), "take"), indices)
end

M.repeat_ = M["repeat"]

return M
