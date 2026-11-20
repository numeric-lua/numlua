--
-- Minimal .npz archive support.
--
-- Reads and writes ZIP archives using the STORED method (no compression),
-- which is what NumPy's np.savez() uses.  Deflate-compressed entries (method
-- 8, as written by np.savez_compressed) are read when a system zlib library
-- can be loaded through LuaJIT FFI.  Stored archives remain dependency-free;
-- a deflate entry fails with a clear error when zlib is unavailable.
--
local bit = require("bit")
local ffi = require("ffi")
local torch = require("torch")
local npy = require("numlua.npy")

local M = {}

local function u16(value)
  return string.char(value % 256, math.floor(value / 256) % 256)
end

local function u32(value)
  return string.char(value % 256, math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256, math.floor(value / 16777216) % 256)
end

local function read_slice(data, offset, count, what)
  if type(offset) ~= "number" or offset < 1 or count < 0 or
      (count > 0 and offset + count - 1 > #data) then
    error("numlua.loadz: truncated archive while reading " .. what, 3)
  end
  if count == 0 then return "" end
  return data:sub(offset, offset + count - 1)
end

local function read_u16(data, offset, what)
  local bytes = read_slice(data, offset, 2, what or "central directory")
  return bytes:byte(1) + bytes:byte(2) * 256
end

local function read_u32(data, offset, what)
  local bytes = read_slice(data, offset, 4, what or "central directory")
  return bytes:byte(1) + bytes:byte(2) * 256 +
    bytes:byte(3) * 65536 + bytes:byte(4) * 16777216
end

local crc_table
local function make_crc_table()
  crc_table = {}
  for n = 0, 255 do
    local c = n
    for _ = 1, 8 do
      if bit.band(c, 1) ~= 0 then
        c = bit.bxor(0xedb88320, bit.rshift(c, 1))
      else
        c = bit.rshift(c, 1)
      end
    end
    crc_table[n] = c
  end
end

local function crc32(value)
  if crc_table == nil then make_crc_table() end
  local crc = 0xffffffff
  for index = 1, #value do
    local byte = value:byte(index)
    crc = bit.bxor(crc_table[bit.band(bit.bxor(crc, byte), 0xff)],
      bit.rshift(crc, 8))
  end
  crc = bit.bxor(crc, 0xffffffff)
  if crc < 0 then crc = crc + 4294967296 end
  return crc
end

-- ---------------------------------------------------------------------------
-- optional system zlib for deflate-compressed entries
-- ---------------------------------------------------------------------------

local ZLIB_CDEF = [[
typedef struct {
  const unsigned char *next_in;
  unsigned int avail_in;
  unsigned long total_in;
  unsigned char *next_out;
  unsigned int avail_out;
  unsigned long total_out;
  const char *msg;
  void *state;
  void *zalloc;
  void *zfree;
  void *opaque;
  int data_type;
  unsigned long adler;
  unsigned long reserved;
} numlua_z_stream;

int inflateInit2_(numlua_z_stream *strm, int windowBits,
  const char *version, int stream_size);
int inflate(numlua_z_stream *strm, int flush);
int inflateEnd(numlua_z_stream *strm);
const char *zlibVersion(void);
]]

local zlib_state -- nil = not checked, false = unavailable, table = loaded

local function zlib_disabled_by_environment()
  local value = os.getenv("NUMLUA_DISABLE_ZLIB")
  return value == "1" or value == "true" or value == "yes"
end

local function load_zlib()
  if zlib_state ~= nil then
    return zlib_state or nil
  end
  if zlib_disabled_by_environment() then
    zlib_state = false
    return nil
  end

  local candidates = {
    "z",
    "libz.so.1",
    "libz.so",
    "libz.dylib",
    "zlib1",
  }
  local loaded = false
  for index = 1, #candidates do
    local ok, handle = pcall(ffi.load, candidates[index])
    if ok then
      local defined = pcall(ffi.cdef, ZLIB_CDEF)
      if defined and pcall(function() return handle.zlibVersion() end) then
        loaded = handle
        break
      end
    end
  end

  zlib_state = loaded
  return zlib_state or nil
end

-- Public only as a small testing/diagnostic helper; the facade does not
-- expose it.  Deflate support is optional and must not be a hard dependency.
function M.zlib_available()
  return load_zlib() ~= nil
end

local ZLIB_ERROR_NAMES = {
  [2] = "Z_NEED_DICT",
  [-2] = "Z_STREAM_ERROR",
  [-3] = "Z_DATA_ERROR",
  [-4] = "Z_MEM_ERROR",
  [-5] = "Z_BUF_ERROR",
  [-6] = "Z_VERSION_ERROR",
}

local function zlib_error(code, stream)
  local message = ZLIB_ERROR_NAMES[code] or
    ("zlib error " .. tostring(code))
  if stream ~= nil and stream.msg ~= nil then
    local detail = ffi.string(stream.msg)
    if detail ~= "" then
      message = message .. ": " .. detail
    end
  end
  return message
end

-- Inflate a raw DEFLATE stream (as stored inside a ZIP method-8 entry) into a
-- string of exactly `expected` bytes.  Returns nil plus a diagnostic on error.
local function inflate_raw(compressed, expected)
  local zlib = load_zlib()
  if zlib == nil then
    return nil, "zlib unavailable"
  end

  local stream = ffi.new("numlua_z_stream")
  local out_size = expected + 1
  local out = ffi.new("unsigned char[?]", out_size)
  stream.next_in = ffi.cast("const unsigned char *", compressed)
  stream.avail_in = #compressed
  stream.next_out = out
  stream.avail_out = out_size

  local code = zlib.inflateInit2_(stream, -15, zlib.zlibVersion(),
    ffi.sizeof(stream))
  if code ~= 0 then
    return nil, zlib_error(code, stream)
  end

  code = zlib.inflate(stream, 4) -- Z_FINISH
  if code ~= 1 then -- Z_STREAM_END
    local message = zlib_error(code, stream)
    zlib.inflateEnd(stream)
    return nil, message
  end
  local end_code = zlib.inflateEnd(stream)
  if end_code ~= 0 then
    return nil, zlib_error(end_code, nil)
  end
  if tonumber(stream.total_out) ~= expected then
    return nil, string.format(
      "inflated size mismatch: expected %d bytes, got %d",
      expected, tonumber(stream.total_out))
  end
  if stream.avail_in ~= 0 then
    return nil, "compressed data has trailing bytes"
  end
  return ffi.string(out, expected)
end

-- ---------------------------------------------------------------------------
-- archive writing (always stored, no zlib dependency)
-- ---------------------------------------------------------------------------

local function check_archive(path, tensor, name)
  if type(path) ~= "string" or path == "" then
    error("numlua." .. name .. " expects a non-empty path", 2)
  end
  if type(tensor) ~= "table" then
    error("numlua." .. name .. " expects a table of tensors", 2)
  end
  for key, value in pairs(tensor) do
    if type(key) ~= "string" or key == "" then
      error("numlua." .. name .. ": array names must be non-empty strings", 2)
    end
    if not torch.is_tensor(value) then
      error("numlua." .. name .. ": " .. key .. " is not a Tensor", 2)
    end
  end
end

function M.savez(path, arrays)
  check_archive(path, arrays, "savez")
  local names = {}
  for name in pairs(arrays) do names[#names + 1] = name end
  table.sort(names)
  local local_parts, central_parts = {}, {}
  local offset = 0
  for _, name in ipairs(names) do
    local file_name = name:match("%.npy$") and name or name .. ".npy"
    local temp = os.tmpname()
    npy.save(temp, arrays[name])
    local handle = assert(io.open(temp, "rb"))
    local payload = assert(handle:read("*a"))
    handle:close()
    os.remove(temp)
    local crc = crc32(payload)
    local local_header = "PK\003\004" .. u16(20) .. u16(0) .. u16(0) ..
      u16(0) .. u16(0) .. u32(crc) .. u32(#payload) .. u32(#payload) ..
      u16(#file_name) .. u16(0) .. file_name
    local_parts[#local_parts + 1] = local_header
    local_parts[#local_parts + 1] = payload
    central_parts[#central_parts + 1] = "PK\001\002" .. u16(20) ..
      u16(20) .. u16(0) .. u16(0) .. u16(0) .. u16(0) .. u32(crc) ..
      u32(#payload) .. u32(#payload) .. u16(#file_name) .. u16(0) ..
      u16(0) .. u16(0) .. u16(0) .. u32(0) .. u32(offset) .. file_name
    offset = offset + #local_header + #payload
  end
  local central = table.concat(central_parts)
  local body = table.concat(local_parts) .. central
  local eocd = "PK\005\006" .. u16(0) .. u16(0) .. u16(#names) ..
    u16(#names) .. u32(#central) .. u32(#body - #central) .. u16(0)
  local handle, open_error = io.open(path, "wb")
  if handle == nil then
    error("numlua.savez: cannot open '" .. path .. "': " .. tostring(open_error), 2)
  end
  local ok, write_error = handle:write(body, eocd)
  local close_ok, close_error = handle:close()
  if ok == nil then
    error("numlua.savez: write failed: " .. tostring(write_error), 2)
  end
  if not close_ok then
    error("numlua.savez: close failed: " .. tostring(close_error), 2)
  end
  return true
end

-- ---------------------------------------------------------------------------
-- archive reading
-- ---------------------------------------------------------------------------

local function find_eocd(data)
  if #data < 22 then return nil end
  local lowest = math.max(1, #data - 65557)
  for position = #data - 21, lowest, -1 do
    if data:sub(position, position + 3) == "PK\005\006" then
      local comment_length = data:byte(position + 20) +
        data:byte(position + 21) * 256
      if position + 21 + comment_length <= #data then
        return position
      end
    end
  end
  return nil
end

local function member_bytes(data, name, central_start, method, flags,
    crc, compressed_size, uncompressed_size, local_offset)
  local local_header = local_offset + 1
  if local_offset < 0 or local_header + 30 > central_start then
    error("numlua.loadz: truncated local header for '" .. name .. "'", 3)
  end
  if data:sub(local_header, local_header + 3) ~= "PK\003\004" then
    error("numlua.loadz: corrupt local header for '" .. name .. "'", 3)
  end

  local local_flags = read_u16(data, local_header + 6,
    "local header for '" .. name .. "'")
  local local_method = read_u16(data, local_header + 8,
    "local header for '" .. name .. "'")
  local local_crc = read_u32(data, local_header + 14,
    "local header for '" .. name .. "'")
  local local_compressed_size = read_u32(data, local_header + 18,
    "local header for '" .. name .. "'")
  local local_uncompressed_size = read_u32(data, local_header + 22,
    "local header for '" .. name .. "'")
  local local_name_length = read_u16(data, local_header + 26,
    "local header for '" .. name .. "'")
  local local_extra_length = read_u16(data, local_header + 28,
    "local header for '" .. name .. "'")

  local local_name = read_slice(data, local_header + 30,
    local_name_length, "local header name for '" .. name .. "'")
  if local_name ~= name then
    error("numlua.loadz: corrupt local header for '" .. name ..
      "' (member name mismatch)", 3)
  end
  if local_method ~= method then
    error("numlua.loadz: corrupt local header for '" .. name ..
      "' (compression method mismatch)", 3)
  end
  if bit.band(local_flags, 1) ~= 0 then
    error("numlua.loadz: encrypted ZIP entries are not supported ('" ..
      name .. "')", 3)
  end

  local descriptor = bit.band(flags, 8) ~= 0
  if not descriptor then
    if local_crc ~= crc then
      error("numlua.loadz: CRC mismatch between local and central " ..
        "headers for '" .. name .. "'", 3)
    end
    if local_compressed_size ~= compressed_size or
        local_uncompressed_size ~= uncompressed_size then
      error("numlua.loadz: local header size mismatch for '" ..
        name .. "'", 3)
    end
  end

  local payload_start = local_header + 30 + local_name_length +
    local_extra_length
  if payload_start + compressed_size > central_start then
    error("numlua.loadz: truncated local data for '" .. name .. "'", 3)
  end
  return read_slice(data, payload_start, compressed_size,
    "local data for '" .. name .. "'")
end

function M.loadz(path)
  if type(path) ~= "string" or path == "" then
    error("numlua.loadz expects a non-empty path", 2)
  end
  local handle, open_error = io.open(path, "rb")
  if handle == nil then
    error("numlua.loadz: cannot open '" .. path .. "': " ..
      tostring(open_error), 2)
  end
  local data = handle:read("*a")
  local close_ok, close_error = handle:close()
  if data == nil then
    error("numlua.loadz: failed reading '" .. path .. "'", 2)
  end
  if not close_ok then
    error("numlua.loadz: cannot close '" .. path .. "': " ..
      tostring(close_error), 2)
  end

  local eocd = find_eocd(data)
  if eocd == nil then
    if data:sub(1, 4) == "PK\003\004" or
        data:find("PK\001\002", 1, true) ~= nil then
      error("numlua.loadz: truncated ZIP archive (missing EOCD)", 2)
    end
    error("numlua.loadz: not a ZIP archive (no EOCD)", 2)
  end

  local disk_number = read_u16(data, eocd + 4, "EOCD")
  local central_disk = read_u16(data, eocd + 6, "EOCD")
  local entries_on_disk = read_u16(data, eocd + 8, "EOCD")
  local entries = read_u16(data, eocd + 10, "EOCD")
  local central_size = read_u32(data, eocd + 12, "EOCD")
  local central_offset = read_u32(data, eocd + 16, "EOCD")
  local comment_length = read_u16(data, eocd + 20, "EOCD")

  if eocd + 21 + comment_length > #data then
    error("numlua.loadz: truncated ZIP archive (incomplete EOCD comment)", 2)
  end
  if entries == 0xffff or entries_on_disk == 0xffff or
      central_size == 0xffffffff or central_offset == 0xffffffff then
    error("numlua.loadz: ZIP64 archives are not supported", 2)
  end
  if disk_number ~= 0 or central_disk ~= 0 or
      entries_on_disk ~= entries then
    error("numlua.loadz: multi-disk ZIP archives are not supported", 2)
  end

  local central_end = central_offset + central_size
  if central_end > #data or central_end >= eocd then
    error("numlua.loadz: corrupt central directory", 2)
  end

  local position = central_offset + 1
  local result = {}
  for _ = 1, entries do
    if position + 45 > central_end then
      error("numlua.loadz: truncated central directory", 2)
    end
    if data:sub(position, position + 3) ~= "PK\001\002" then
      error("numlua.loadz: corrupt central directory", 2)
    end

    local flags = read_u16(data, position + 8, "central directory")
    local method = read_u16(data, position + 10, "central directory")
    local crc = read_u32(data, position + 16, "central directory")
    local compressed_size = read_u32(data, position + 20,
      "central directory")
    local uncompressed_size = read_u32(data, position + 24,
      "central directory")
    local name_length = read_u16(data, position + 28, "central directory")
    local extra_length = read_u16(data, position + 30, "central directory")
    local member_comment_length = read_u16(data, position + 32,
      "central directory")
    local start_disk = read_u16(data, position + 34, "central directory")
    local local_offset = read_u32(data, position + 42, "central directory")

    local record_end = position + 45 + name_length + extra_length +
      member_comment_length
    if record_end > central_end then
      error("numlua.loadz: truncated central directory", 2)
    end
    local name = read_slice(data, position + 46, name_length,
      "central directory name")
    if name == "" then
      error("numlua.loadz: corrupt central directory (empty name)", 2)
    end
    if bit.band(flags, 1) ~= 0 then
      error("numlua.loadz: encrypted ZIP entries are not supported ('" ..
        name .. "')", 2)
    end
    if start_disk ~= 0 then
      error("numlua.loadz: multi-disk ZIP archives are not supported", 2)
    end
    if method ~= 0 and method ~= 8 then
      error(string.format(
        "numlua.loadz: unsupported compression method %d for '%s' " ..
        "(only stored and deflate are supported)", method, name), 2)
    end
    if method == 8 and load_zlib() == nil then
      error("numlua.loadz: deflate-compressed entry '" .. name ..
        "' requires a system zlib library; none could be loaded through " ..
        "LuaJIT FFI", 2)
    end

    local payload = member_bytes(data, name, central_offset + 1, method,
      flags, crc, compressed_size, uncompressed_size, local_offset)

    local raw
    if method == 0 then
      if compressed_size ~= uncompressed_size then
        error("numlua.loadz: stored entry '" .. name ..
          "' has mismatched compressed and uncompressed sizes", 2)
      end
      raw = payload
    else
      local inflated, inflate_error = inflate_raw(payload, uncompressed_size)
      if inflated == nil then
        error("numlua.loadz: cannot inflate entry '" .. name .. "': " ..
          tostring(inflate_error), 2)
      end
      raw = inflated
    end

    local actual_crc = crc32(raw)
    if actual_crc ~= crc then
      error(string.format(
        "numlua.loadz: CRC mismatch for entry '%s' (expected %s, got %s)",
        name, tostring(crc), tostring(actual_crc)), 2)
    end

    local temp = os.tmpname()
    local out, out_error = io.open(temp, "wb")
    if out == nil then
      error("numlua.loadz: cannot create temporary file: " ..
        tostring(out_error), 2)
    end
    local wrote, write_error = out:write(raw)
    local closed, close_write_error = out:close()
    if wrote == nil then
      os.remove(temp)
      error("numlua.loadz: cannot write temporary member '" .. name ..
        "': " .. tostring(write_error), 2)
    end
    if not closed then
      os.remove(temp)
      error("numlua.loadz: cannot close temporary member '" .. name ..
        "': " .. tostring(close_write_error), 2)
    end

    local ok, tensor = pcall(npy.load, temp)
    local removed, remove_error = os.remove(temp)
    if not ok then
      error("numlua.loadz: invalid .npy member '" .. name .. "': " ..
        tostring(tensor), 2)
    end
    if not removed then
      error("numlua.loadz: cannot remove temporary file: " ..
        tostring(remove_error), 2)
    end

    local key = name:gsub("%.npy$", "")
    result[key] = tensor
    position = record_end + 1
  end

  if position - 1 ~= central_end then
    error("numlua.loadz: corrupt central directory " ..
      "(entry count/size mismatch)", 2)
  end
  return result
end

return M
