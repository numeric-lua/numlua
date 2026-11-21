--
-- numlua / npy.lua
--
-- Minimal, dependency-free NumPy .npy v1.0 reader/writer.
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
-- Arrays are written in C order.  The reader accepts any 1-D/2-D/N-D C-order
-- file with the dtypes above and rejects big-endian files, Fortran order,
-- NPY versions other than 1.0, and unsupported descr values with clear
-- errors.  On little-endian hosts, native-endian '=' descr values are accepted
-- as aliases for the '<' spellings, matching NumPy's dtype parser.  Only
-- little-endian hosts are supported (LuaJIT ffi.abi("le")).
--

local ffi = require("ffi")
local torch = require("torch")
local libmod = require("torch.lib")
local lib = libmod.lib

local npy = {}

local MAGIC = "\147NUMPY"
local VERSION_MAJOR = 1
local VERSION_MINOR = 0

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

local function make_header(descr, shape)
  local dict = string.format(
    "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
    descr, shape_to_string(shape))
  -- NPY v1.0 pads the header with spaces, terminated by '\n', so that the
  -- complete prefix (magic + version + uint16 length + header) is a multiple
  -- of 64 bytes.
  local total_without_padding = 10 + #dict + 1
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

-- Write a tensor to path as an NPY v1.0 C-order file.  Returns true.
function npy.save(path, tensor)
  if type(path) ~= "string" or path == "" then
    error("numlua.save expects a non-empty file path", 2)
  end
  if not torch.is_tensor(tensor) then
    error("numlua.save expects a tensor, got " .. type(tensor), 2)
  end
  if not host_is_little_endian() then
    error("numlua.save: only little-endian hosts are supported", 2)
  end

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
  local header = make_header(info.descr, contiguous:sizes())
  if #header > 65535 then
    error("numlua.save: .npy v1.0 header exceeds 65535 bytes", 2)
  end
  local header_size = #header
  local prefix = MAGIC .. string.char(VERSION_MAJOR, VERSION_MINOR) ..
    string.char(header_size % 256, math.floor(header_size / 256))

  local handle, open_error = io.open(path, "wb")
  if handle == nil then
    error("numlua.save: cannot open '" .. path .. "' for writing: " ..
      tostring(open_error), 2)
  end

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
  local close_ok, close_error = handle:close()

  if ok == nil then
    error("numlua.save: failed writing '" .. path .. "': " ..
      tostring(write_error), 2)
  end
  if not close_ok then
    error("numlua.save: failed closing '" .. path .. "': " ..
      tostring(close_error), 2)
  end
  return true
end

-- Load a NumPy .npy v1.0 file and return a luatorch Tensor.
function npy.load(path)
  if type(path) ~= "string" or path == "" then
    error("numlua.load expects a non-empty file path", 2)
  end
  if not host_is_little_endian() then
    error("numlua.load: only little-endian hosts are supported", 2)
  end

  local data = read_file(path)
  if #data < 10 then
    error("numlua.load: truncated .npy file (missing header)", 2)
  end
  if data:sub(1, 6) ~= MAGIC then
    error("numlua.load: not an NPY file (bad magic)", 2)
  end
  local major, minor = data:byte(7), data:byte(8)
  if major ~= VERSION_MAJOR or minor ~= VERSION_MINOR then
    error(string.format(
      "numlua.load: unsupported NPY version %d.%d (only 1.0 is supported)",
      major, minor), 2)
  end

  local header_size = data:byte(9) + data:byte(10) * 256
  if header_size < 1 then
    error("numlua.load: malformed .npy header (zero length)", 2)
  end
  if #data < 10 + header_size then
    error("numlua.load: truncated .npy header", 2)
  end
  local header = data:sub(11, 10 + header_size)
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
  local payload = data:sub(10 + header_size + 1)
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

return npy
