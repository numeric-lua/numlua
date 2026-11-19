--
-- Minimal .npz archive support.
--
-- Reads and writes ZIP archives using the STORED method (no compression),
-- which is what NumPy's np.savez() uses.  Deflate-compressed archives are
-- rejected with a clear error rather than silently corrupted.
--
local bit = require("bit")
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

local function read_u16(data, offset)
  return data:byte(offset) + data:byte(offset + 1) * 256
end

local function read_u32(data, offset)
  return data:byte(offset) + data:byte(offset + 1) * 256 +
    data:byte(offset + 2) * 65536 + data:byte(offset + 3) * 16777216
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
  return bit.bxor(crc, 0xffffffff)
end

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

local function rfind(data, needle, init)
  init = init or #data
  for position = init, 1, -1 do
    if data:sub(position, position + #needle - 1) == needle then
      return position
    end
  end
  return nil
end

local function find_eocd(data)
  local limit = math.max(1, #data - 65557)
  local position = #data
  while position >= limit do
    local found = rfind(data, "PK\005\006", position)
    if found == nil then return nil end
    if found + 21 <= #data then return found end
    position = found - 1
  end
  return nil
end

function M.loadz(path)
  if type(path) ~= "string" or path == "" then
    error("numlua.loadz expects a non-empty path", 2)
  end
  local handle, open_error = io.open(path, "rb")
  if handle == nil then
    error("numlua.loadz: cannot open '" .. path .. "': " .. tostring(open_error), 2)
  end
  local data = handle:read("*a")
  handle:close()
  local eocd = find_eocd(data)
  if eocd == nil then error("numlua.loadz: not a ZIP archive (no EOCD)", 2) end
  local entries = read_u16(data, eocd + 10)
  local central_offset = read_u32(data, eocd + 16)
  local position = central_offset + 1
  local result = {}
  for _ = 1, entries do
    if data:sub(position, position + 3) ~= "PK\001\002" then
      error("numlua.loadz: corrupt central directory", 2)
    end
    local method = read_u16(data, position + 10)
    local name_length = read_u16(data, position + 28)
    local extra_length = read_u16(data, position + 30)
    local comment_length = read_u16(data, position + 32)
    local local_offset = read_u32(data, position + 42)
    local name = data:sub(position + 46, position + 45 + name_length)
    if method ~= 0 then
      error("numlua.loadz: compressed entry '" .. name ..
        "' is unsupported (only stored ZIP entries)", 2)
    end
    local local_header = local_offset + 1
    if data:sub(local_header, local_header + 3) ~= "PK\003\004" then
      error("numlua.loadz: corrupt local header for '" .. name .. "'", 2)
    end
    local local_name = read_u16(data, local_header + 26)
    local local_extra = read_u16(data, local_header + 28)
    local payload_start = local_header + 30 + local_name + local_extra
    local payload = data:sub(payload_start,
      payload_start + read_u32(data, position + 24) - 1)
    local temp = os.tmpname()
    local out = assert(io.open(temp, "wb"))
    out:write(payload)
    out:close()
    local key = name:gsub("%.npy$", "")
    result[key] = npy.load(temp)
    os.remove(temp)
    position = position + 46 + name_length + extra_length + comment_length
  end
  return result
end

return M
