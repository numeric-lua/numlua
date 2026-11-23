--
-- numlua / npy.lua
--
-- Minimal, dependency-free NumPy .npy reader/writer.
--
-- Supported dtypes:
--   float16 -> '<f2'
--   float32 -> '<f4'
--   float64 -> '<f8'
--   int8    -> '|i1'
--   int16   -> '<i2'
--   int32   -> '<i4'
--   int64   -> '<i8'
--   uint8   -> '|u1'
--   bool    -> '|b1'
--
-- Arrays are written as NPY v1.0 in C order by default; pass
-- `{ version = 2 }` or `{ version = 3 }` to write v2.0/v3.0 instead.  The
-- reader accepts NPY v1.0, v2.0, and v3.0 headers with the dtypes above and
-- any 1-D/2-D/N-D C-order file; it rejects big-endian files, Fortran order,
-- unsupported versions, and unsupported descr values with clear errors.
-- v2.0 and v3.0 differ from v1.0 only in their 4-byte header-length field (and
-- v3.0's UTF-8 header encoding), which this reader already parses as byte
-- strings.  On little-endian hosts, native-endian '=' descr values are
-- accepted as aliases for the '<' spellings, matching NumPy's dtype parser.
-- Only little-endian hosts are supported (LuaJIT ffi.abi("le")).
--

local ffi = require("ffi")
local torch = require("torch")
local libmod = require("torch.lib")
local lib = libmod.lib

local npy = {}

local MAGIC = "\147NUMPY"
local DEFAULT_VERSION = 1
local SUPPORTED_VERSIONS = "1.0, 2.0, 3.0"
local VERSION_ALIASES = {
  ["1.0"] = 1,
  ["2.0"] = 2,
  ["3.0"] = 3,
}

local DTYPES = {
  [0] = { descr = "<f4", name = "float32" },
  [1] = { descr = "<f8", name = "float64" },
  [2] = { descr = "<i8", name = "int64" },
  [3] = { descr = "<i4", name = "int32" },
  [4] = { descr = "<i2", name = "int16" },
  [5] = { descr = "|i1", name = "int8" },
  [6] = { descr = "|u1", name = "uint8" },
  [7] = { descr = "|b1", name = "bool" },
  [8] = { descr = "<f2", name = "float16" },
}

local LOAD_DTYPES = {
  ["<f2"] = { dtype = 8, size = 2 },
  ["<f4"] = { dtype = 0, size = 4 },
  ["<f8"] = { dtype = 1, size = 8 },
  ["<i2"] = { dtype = 4, size = 2 },
  ["<i4"] = { dtype = 3, size = 4 },
  ["<i8"] = { dtype = 2, size = 8 },
  ["|i1"] = { dtype = 5, size = 1 },
  ["|u1"] = { dtype = 6, size = 1 },
  ["|b1"] = { dtype = 7, size = 1 },
  -- NumPy writes the byte-sized integer dtypes with '|', but accept the
  -- explicit little-endian spellings too.
  ["<i1"] = { dtype = 5, size = 1 },
  ["<u1"] = { dtype = 6, size = 1 },
  ["<b1"] = { dtype = 7, size = 1 },
}

local SUPPORTED_DESCR =
  "'<f2', '<f4', '<f8', '<i2', '<i4', '<i8', '|i1', '|u1', '|b1'"
local SUPPORTED_NAMES =
  "float16, float32, float64, int8, int16, int32, int64, uint8, bool"

local function host_is_little_endian()
  return ffi.abi("le")
end

local function last_error()
  return ffi.string(lib.lt_last_error())
end

-- NPY v2.0 and v3.0 store the header length as a little-endian uint32 instead
-- of v1.0's uint16.
local function read_u32(data, offset)
  return data:byte(offset) + data:byte(offset + 1) * 256 +
    data:byte(offset + 2) * 65536 + data:byte(offset + 3) * 16777216
end

local function write_u32(value)
  return string.char(value % 256, math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256, math.floor(value / 16777216) % 256)
end

-- Normalize an optional NPY header version.  Both the .npy writer and the
-- .npz member writer use this so the accepted spellings cannot drift apart.
-- `what` names the public function in error messages.
function npy.parse_version(requested, what)
  local version = requested
  if type(requested) == "string" then
    version = VERSION_ALIASES[requested]
  end
  if version ~= 1 and version ~= 2 and version ~= 3 then
    error(string.format(
      "%s: version must be 1, 2, or 3 (or '1.0', '2.0', '3.0'), got %s",
      what, tostring(requested)), 2)
  end
  return version
end

-- Validate the optional save() options table.  Only `version` is understood,
-- and unknown keys are rejected so a typo cannot silently write v1.0.
local function parse_save_options(options)
  if options == nil then return DEFAULT_VERSION end
  if type(options) ~= "table" then
    error("numlua.save: options must be a table or nil", 3)
  end
  for key in pairs(options) do
    if key ~= "version" then
      error("numlua.save: unknown option '" .. tostring(key) .. "'", 3)
    end
  end
  local requested = options.version
  if requested == nil then return DEFAULT_VERSION end
  return npy.parse_version(requested, "numlua.save")
end

local function shape_to_string(shape)
  if #shape == 0 then
    return "()"
  end
  local parts = {}
  for i = 1, #shape do
    parts[i] = tostring(shape[i])
  end
  if #shape == 1 then
    return "(" .. parts[1] .. ",)"
  end
  return "(" .. table.concat(parts, ", ") .. ")"
end

local function make_header(descr, shape, version)
  local dict = string.format(
    "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
    descr, shape_to_string(shape))
  -- NPY v1.0 pads the header with spaces, terminated by '\n', so that the
  -- complete prefix (magic + version + uint16 length + header) is a multiple
  -- of 64 bytes.  v2.0/v3.0 use a uint32 length, so their 12-byte prefix
  -- changes the padding but keeps the same 64-byte alignment rule.
  local prefix_size = version == 1 and 10 or 12
  local total_without_padding = prefix_size + #dict + 1
  local padding = (64 - (total_without_padding % 64)) % 64
  return dict .. string.rep(" ", padding) .. "\n"
end

local function extract_quoted(header, key)
  local single = header:match("'" .. key .. "'%s*:%s*'([^']*)'")
  if single ~= nil then
    return single
  end
  return header:match('"' .. key .. '"%s*:%s*"([^"]*)"')
end

local function parse_shape(header)
  local shape_text = header:match("'shape'%s*:%s*(%b())")
    or header:match('"shape"%s*:%s*(%b())')
  if shape_text == nil then
    error("numlua.load: malformed .npy header (missing 'shape')", 3)
  end
  local inner = shape_text:sub(2, -2):gsub("%s+", "")
  local dims = {}
  local rest = inner
  while rest ~= "" do
    local value, remainder = rest:match("^(%d+),?(.*)$")
    if value == nil then
      error("numlua.load: malformed .npy shape " .. shape_text, 3)
    end
    dims[#dims + 1] = tonumber(value)
    rest = remainder
  end
  return dims
end

local function read_file(path)
  local handle, open_error = io.open(path, "rb")
  if handle == nil then
    error("numlua.load: cannot open '" .. path .. "' for reading: " ..
      tostring(open_error), 3)
  end
  local data, read_error = handle:read("*a")
  local close_ok, close_error = handle:close()
  if data == nil then
    error("numlua.load: failed reading '" .. path .. "': " ..
      tostring(read_error), 3)
  end
  if not close_ok then
    error("numlua.load: failed closing '" .. path .. "': " ..
      tostring(close_error), 3)
  end
  return data
end

-- Read from a path or an already-open Lua file handle.  Caller-owned handles
-- are read from their current position and are never closed by numlua.
local function read_input(source)
  local kind = io.type(source)
  if kind == "file" then
    local data, read_error = source:read("*a")
    if data == nil then
      error("numlua.load: failed reading file handle: " ..
        tostring(read_error), 3)
    end
    return data
  end
  if kind == "closed file" then
    error("numlua.load: cannot read from a closed file handle", 3)
  end
  if type(source) ~= "string" or source == "" then
    error("numlua.load expects a non-empty file path or open file handle", 3)
  end
  return read_file(source)
end

-- Write a tensor to a path or open file handle as a C-order .npy file.  The
-- default is NPY v1.0; pass an options table with `version = 2`/`3` (or
-- "2.0"/"3.0") to write the wider-header versions.  Open handles are written
-- at their current position and are never closed by numlua.  Returns true.
function npy.save(path, tensor, options)
  local target_kind = io.type(path)
  if target_kind == "closed file" then
    error("numlua.save: cannot write to a closed file handle", 2)
  end
  if target_kind ~= "file" and (type(path) ~= "string" or path == "") then
    error("numlua.save expects a non-empty file path or open file handle", 2)
  end
  if not torch.is_tensor(tensor) then
    error("numlua.save expects a tensor, got " .. type(tensor), 2)
  end
  if not host_is_little_endian() then
    error("numlua.save: only little-endian hosts are supported", 2)
  end
  local version = parse_save_options(options)

  local dtype_id = tensor:dtype_id()
  local info = DTYPES[dtype_id]
  if info == nil then
    error(string.format(
      "numlua.save: unsupported dtype '%s' (supported: %s)",
      tensor:dtype(), SUPPORTED_NAMES), 2)
  end

  -- detach() keeps autograd metadata out of the file; contiguous() makes the
  -- raw buffer C-order regardless of the input's strides.
  local contiguous = tensor:detach():contiguous()
  local header = make_header(info.descr, contiguous:sizes(), version)
  if version == 1 and #header > 65535 then
    error("numlua.save: .npy v1.0 header exceeds 65535 bytes " ..
      "(pass { version = 2 } for a wider header)", 2)
  end
  local header_size = #header
  local prefix = MAGIC .. string.char(version, 0)
  if version == 1 then
    prefix = prefix .. string.char(header_size % 256,
      math.floor(header_size / 256))
  else
    prefix = prefix .. write_u32(header_size)
  end

  local handle, open_error
  local owns_handle = target_kind ~= "file"
  if owns_handle then
    handle, open_error = io.open(path, "wb")
    if handle == nil then
      error("numlua.save: cannot open '" .. path .. "' for writing: " ..
        tostring(open_error), 2)
    end
  else
    handle = path
  end
  local label = owns_handle and ("'" .. path .. "'") or "file handle"

  local ok, write_error = handle:write(prefix, header)
  if ok ~= nil then
    local nbytes = contiguous:nbytes()
    if nbytes > 0 then
      local pointer = lib.lt_data_ptr(contiguous.h)
      if pointer == nil then
        ok, write_error = nil, last_error()
      else
        ok, write_error = handle:write(ffi.string(pointer, nbytes))
      end
    end
  end
  local close_ok, close_error = true, nil
  if owns_handle then
    close_ok, close_error = handle:close()
  end

  if ok == nil then
    error("numlua.save: failed writing " .. label .. ": " ..
      tostring(write_error), 2)
  end
  if not close_ok then
    error("numlua.save: failed closing " .. label .. ": " ..
      tostring(close_error), 2)
  end
  return true
end

-- Load a NumPy .npy v1.0 file from a path or open file handle and return a
-- luatorch Tensor.  Caller-owned handles are read from their current position
-- and are never closed by numlua.
function npy.load(path)
  local target_kind = io.type(path)
  if target_kind == "closed file" then
    error("numlua.load: cannot read from a closed file handle", 2)
  end
  if target_kind ~= "file" and (type(path) ~= "string" or path == "") then
    error("numlua.load expects a non-empty file path or open file handle", 2)
  end
  if not host_is_little_endian() then
    error("numlua.load: only little-endian hosts are supported", 2)
  end

  local data = read_input(path)
  if #data < 10 then
    error("numlua.load: truncated .npy file (missing header)", 2)
  end
  if data:sub(1, 6) ~= MAGIC then
    error("numlua.load: not an NPY file (bad magic)", 2)
  end
  local major, minor = data:byte(7), data:byte(8)
  if minor ~= 0 or major < 1 or major > 3 then
    error(string.format(
      "numlua.load: unsupported NPY version %d.%d (supported: %s)",
      major, minor, SUPPORTED_VERSIONS), 2)
  end

  local prefix_size, header_size
  if major == 1 then
    prefix_size = 10
    header_size = data:byte(9) + data:byte(10) * 256
  else
    if #data < 12 then
      error("numlua.load: truncated .npy file (missing header)", 2)
    end
    prefix_size = 12
    header_size = read_u32(data, 9)
  end
  if header_size < 1 then
    error("numlua.load: malformed .npy header (zero length)", 2)
  end
  if #data < prefix_size + header_size then
    error("numlua.load: truncated .npy header", 2)
  end
  local header = data:sub(prefix_size + 1, prefix_size + header_size)
  if header:sub(-1) ~= "\n" then
    error("numlua.load: malformed .npy header (missing trailing newline)", 2)
  end

  local descr = extract_quoted(header, "descr")
  if descr == nil then
    error("numlua.load: malformed .npy header (missing 'descr')", 2)
  end
  if descr:sub(1, 1) == ">" then
    error("numlua.load: big-endian .npy files are not supported " ..
      "(descr '" .. descr .. "')", 2)
  end

  local fortran = header:match("'fortran_order'%s*:%s*(%a+)")
    or header:match('"fortran_order"%s*:%s*(%a+)')
  if fortran == nil then
    error("numlua.load: malformed .npy header (missing 'fortran_order')", 2)
  end
  if fortran ~= "False" then
    if fortran == "True" then
      error("numlua.load: fortran_order=True is not supported; save with " ..
        "C order", 2)
    end
    error("numlua.load: malformed .npy header (invalid fortran_order '" ..
      fortran .. "')", 2)
  end

  local info = LOAD_DTYPES[descr]
  if info == nil and descr:sub(1, 1) == "=" then
    -- '=' selects the host byte order.  Only little-endian hosts reach this
    -- point, so it is an alias for the '<' spelling, as in NumPy.
    info = LOAD_DTYPES["<" .. descr:sub(2)]
  end
  if info == nil then
    error("numlua.load: unsupported .npy descr '" .. descr ..
      "' (supported: " .. SUPPORTED_DESCR .. ")", 2)
  end

  local shape = parse_shape(header)
  local numel = 1
  for i = 1, #shape do
    numel = numel * shape[i]
  end
  local expected_bytes = numel * info.size
  local consumed = prefix_size + header_size + expected_bytes
  -- A caller-owned handle may contain another array after this one.  Read all
  -- available bytes to parse the header, then put back anything beyond this
  -- array so a later load() starts at the next NPY prefix.
  if target_kind == "file" and #data > consumed then
    local position, seek_error = path:seek("cur", -(#data - consumed))
    if position == nil then
      error("numlua.load: failed positioning file handle: " ..
        tostring(seek_error), 2)
    end
  end
  local payload
  if target_kind == "file" then
    payload = data:sub(prefix_size + header_size + 1, consumed)
  else
    payload = data:sub(prefix_size + header_size + 1)
  end
  if #payload ~= expected_bytes then
    error(string.format(
      "numlua.load: shape/data size mismatch: expected %d bytes, got %d",
      expected_bytes, #payload), 2)
  end

  local tensor
  if #shape == 0 then
    -- luatorch's public empty() constructors require at least one dimension,
    -- and its scalar constructors cannot create float16 tensors.  Allocate a
    -- one-element tensor of the right dtype and reshape it to 0-D instead.
    tensor = torch.empty({ 1 }, { dtype = info.dtype }):reshape({})
  else
    tensor = torch.empty(shape, { dtype = info.dtype })
  end

  if expected_bytes > 0 then
    local pointer = lib.lt_data_ptr(tensor.h)
    if pointer == nil then
      error("numlua.load: could not access tensor storage: " ..
        tostring(last_error()), 2)
    end
    ffi.copy(ffi.cast("void*", pointer), payload, expected_bytes)
  end
  return tensor
end

-- ---------------------------------------------------------------------------
-- raw binary array I/O (NumPy fromfile/tofile)
-- ---------------------------------------------------------------------------

local IO_DTYPE_BY_NAME = {}
for _, info in pairs(DTYPES) do
  IO_DTYPE_BY_NAME[info.name] = info
  IO_DTYPE_BY_NAME[info.descr] = info
end

local function io_dtype(value, name, default_id)
  if value == nil then
    local default = DTYPES[default_id]
    return { name = default.name, size = LOAD_DTYPES[default.descr].size }
  end
  local info
  if type(value) == "string" then
    info = IO_DTYPE_BY_NAME[value]
  elseif type(value) == "number" and value % 1 == 0 then
    info = DTYPES[value]
  end
  if info == nil then
    error(string.format(
      "numlua.%s: unsupported dtype %s (supported: %s)", name,
      type(value) == "string" and ("'" .. value .. "'") or tostring(value),
      SUPPORTED_NAMES), 3)
  end
  return { name = info.name, size = LOAD_DTYPES[info.descr].size }
end

local function io_options(options, name, allowed)
  if options ~= nil and type(options) ~= "table" then
    error("numlua." .. name .. ": options must be a table or nil", 3)
  end
  options = options or {}
  for key in pairs(options) do
    if not allowed[key] then
      error("numlua." .. name .. ": unknown option '" ..
        tostring(key) .. "'", 3)
    end
  end
  return options
end

local FROMFILE_OPTIONS = {
  dtype = true,
  count = true,
  offset = true,
  sep = true,
}

local TOFILE_OPTIONS = { sep = true }

local function check_sep(sep, name)
  if sep == nil then return end
  if type(sep) ~= "string" then
    error("numlua." .. name .. ": sep must be a string or nil", 3)
  end
  if sep ~= "" then
    error("numlua." .. name .. ": text-mode sep is not supported; use " ..
      "loadtxt/savetxt for text data", 3)
  end
end

local function check_count(value, name)
  if value == nil then return -1 end
  if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
      value < -1 or value == math.huge or value == -math.huge then
    error("numlua." .. name .. ": count must be an integer >= -1", 3)
  end
  return value
end

local function check_offset(value, name)
  if value == nil then return 0 end
  if type(value) ~= "number" or value ~= value or value % 1 ~= 0 or
      value < 0 or value == math.huge or value == -math.huge then
    error("numlua." .. name .. ": offset must be a non-negative integer", 3)
  end
  return value
end

local function raw_target(source, name)
  local kind = io.type(source)
  if kind == "closed file" then
    error("numlua." .. name .. ": cannot use a closed file handle", 3)
  end
  if kind ~= "file" and (type(source) ~= "string" or source == "") then
    error("numlua." .. name ..
      " expects a non-empty file path or open file handle", 3)
  end
  return kind == "file"
end

local function open_raw_source(source, name)
  local handle, open_error = io.open(source, "rb")
  if handle == nil then
    error("numlua." .. name .. ": cannot open '" .. source ..
      "' for reading: " .. tostring(open_error), 3)
  end
  return handle
end

local function skip_raw_bytes(handle, offset, name, label)
  if offset == 0 then return end
  local position, seek_error = handle:seek("cur", offset)
  if position ~= nil then return end
  local skipped, read_error = handle:read(offset)
  if skipped == nil or #skipped < offset then
    error(string.format(
      "numlua.%s: cannot skip %d byte(s) in %s: %s", name, offset, label,
      tostring(read_error or seek_error)), 3)
  end
end

local function read_raw_data(handle, count, size, name, label)
  if count < 0 then
    local data, read_error = handle:read("*a")
    if data == nil then
      error("numlua." .. name .. ": failed reading " .. label .. ": " ..
        tostring(read_error), 3)
    end
    if #data % size ~= 0 then
      error(string.format(
        "numlua.%s: %s contains %d byte(s), not a multiple of the %d-byte " ..
        "dtype size", name, label, #data, size), 3)
    end
    return data, #data / size
  end
  local expected = count * size
  if expected == 0 then return "", 0 end
  local data, read_error = handle:read(expected)
  if data == nil or #data < expected then
    error(string.format(
      "numlua.%s: truncated %s: expected %d byte(s), got %d", name, label,
      expected, data == nil and 0 or #data), 3)
  end
  return data, count
end

-- fromfile(path_or_file[, options]) reads raw little-endian elements as a 1-D
-- tensor.  Options are dtype (default float64), count (-1 for all), offset in
-- bytes, and sep (only the empty binary separator is supported).
function npy.fromfile(source, options)
  local is_handle = raw_target(source, "fromfile")
  if not host_is_little_endian() then
    error("numlua.fromfile: only little-endian hosts are supported", 2)
  end
  options = io_options(options, "fromfile", FROMFILE_OPTIONS)
  local info = io_dtype(options.dtype, "fromfile", 1)
  local count = check_count(options.count, "fromfile")
  local offset = check_offset(options.offset, "fromfile")
  check_sep(options.sep, "fromfile")

  local handle, owns_handle = source, false
  local label = "file handle"
  if not is_handle then
    handle = open_raw_source(source, "fromfile")
    owns_handle = true
    label = "'" .. source .. "'"
  end

  local ok, data_or_error, element_count = pcall(function()
    skip_raw_bytes(handle, offset, "fromfile", label)
    return read_raw_data(handle, count, info.size, "fromfile", label)
  end)
  local close_ok, close_error = true, nil
  if owns_handle then
    close_ok, close_error = handle:close()
  end
  if not ok then
    error(data_or_error, 0)
  end
  if not close_ok then
    error("numlua.fromfile: failed closing " .. label .. ": " ..
      tostring(close_error), 2)
  end

  local data = data_or_error
  local elements = element_count
  local tensor
  if elements == 0 then
    tensor = torch.empty({ 0 }, { dtype = info.name })
  else
    tensor = torch.empty({ elements }, { dtype = info.name })
    local pointer = lib.lt_data_ptr(tensor.h)
    if pointer == nil then
      error("numlua.fromfile: could not access tensor storage: " ..
        tostring(last_error()), 2)
    end
    ffi.copy(ffi.cast("void*", pointer), data, #data)
  end
  return tensor
end

-- tofile(path_or_file, tensor[, options]) writes the tensor's raw C-order
-- little-endian bytes.  The only option is sep, and only "" (binary) is
-- accepted.  Caller-owned handles are written at their current position and
-- are never closed.
function npy.tofile(target, tensor, options)
  local is_handle = raw_target(target, "tofile")
  if not torch.is_tensor(tensor) then
    error("numlua.tofile expects a tensor, got " .. type(tensor), 2)
  end
  if not host_is_little_endian() then
    error("numlua.tofile: only little-endian hosts are supported", 2)
  end
  options = io_options(options, "tofile", TOFILE_OPTIONS)
  check_sep(options.sep, "tofile")

  local info = DTYPES[tensor:dtype_id()]
  if info == nil then
    error(string.format(
      "numlua.tofile: unsupported dtype '%s' (supported: %s)",
      tensor:dtype(), SUPPORTED_NAMES), 2)
  end

  local handle, owns_handle = target, false
  local label = "file handle"
  if not is_handle then
    local open_error
    handle, open_error = io.open(target, "wb")
    if handle == nil then
      error("numlua.tofile: cannot open '" .. target ..
        "' for writing: " .. tostring(open_error), 2)
    end
    owns_handle = true
    label = "'" .. target .. "'"
  end

  -- detach() and contiguous() mirror save(): autograd metadata is not written
  -- and a non-contiguous input is serialized in C order.
  local contiguous = tensor:detach():contiguous()
  local ok, write_error = true, nil
  local nbytes = contiguous:nbytes()
  if nbytes > 0 then
    local pointer = lib.lt_data_ptr(contiguous.h)
    if pointer == nil then
      ok, write_error = nil, last_error()
    else
      ok, write_error = handle:write(ffi.string(pointer, nbytes))
    end
  end
  local close_ok, close_error = true, nil
  if owns_handle then
    close_ok, close_error = handle:close()
  end

  if ok == nil then
    error("numlua.tofile: failed writing " .. label .. ": " ..
      tostring(write_error), 2)
  end
  if not close_ok then
    error("numlua.tofile: failed closing " .. label .. ": " ..
      tostring(close_error), 2)
  end
  return true
end

return npy
