--
-- Generate the small NumPy-format fixture corpus used by the test suite.
--
-- The corpus is written under tests/.tmp/ at test time instead of being
-- committed, so no generated binary artefacts enter git.  The bytes are
-- produced here directly (NPY v1.0 header + little-endian payload, stored ZIP
-- for NPZ) rather than through numlua.save/savez, so the reader is exercised
-- against independently constructed files.
--
local bit = require("bit")
local ffi = require("ffi")

local M = {}

local function u16(value)
  return string.char(value % 256, math.floor(value / 256) % 256)
end

local function u32(value)
  return string.char(value % 256, math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256, math.floor(value / 16777216) % 256)
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

local function shape_text(shape)
  if #shape == 0 then return "()" end
  local parts = {}
  for index = 1, #shape do parts[index] = tostring(shape[index]) end
  if #shape == 1 then return "(" .. parts[1] .. ",)" end
  return "(" .. table.concat(parts, ", ") .. ")"
end

local function npy_bytes(descr, shape, payload)
  local dict = string.format(
    "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
    descr, shape_text(shape))
  local total_without_padding = 10 + #dict + 1
  local padding = (64 - (total_without_padding % 64)) % 64
  local header = dict .. string.rep(" ", padding) .. "\n"
  return "\147NUMPY" .. string.char(1, 0) ..
    string.char(#header % 256, math.floor(#header / 256)) ..
    header .. (payload or "")
end

local function encode_f32(values)
  local buffer = ffi.new("float[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values * 4)
end

local function encode_f64(values)
  local buffer = ffi.new("double[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values * 8)
end

local function encode_i64(values)
  local buffer = ffi.new("int64_t[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values * 8)
end

local function encode_bool(values)
  local parts = {}
  for index = 1, #values do parts[index] = values[index] and "\1" or "\0" end
  return table.concat(parts)
end

local function write_file(path, data)
  local handle, open_error = io.open(path, "wb")
  if handle == nil then
    error("fixture generator: cannot open '" .. path .. "': " ..
      tostring(open_error), 2)
  end
  local ok, write_error = handle:write(data)
  local close_ok, close_error = handle:close()
  if ok == nil then
    error("fixture generator: write failed for '" .. path .. "': " ..
      tostring(write_error), 2)
  end
  if not close_ok then
    error("fixture generator: close failed for '" .. path .. "': " ..
      tostring(close_error), 2)
  end
end

-- Write a stored (uncompressed) ZIP archive, matching np.savez's default.
local function write_npz(path, members)
  local local_parts, central_parts = {}, {}
  local offset = 0
  for _, member in ipairs(members) do
    local payload = member.payload
    local name = member.name
    local crc = crc32(payload)
    local local_header = "PK\003\004" .. u16(20) .. u16(0) .. u16(0) ..
      u16(0) .. u16(0) .. u32(crc) .. u32(#payload) .. u32(#payload) ..
      u16(#name) .. u16(0) .. name
    local_parts[#local_parts + 1] = local_header
    local_parts[#local_parts + 1] = payload
    central_parts[#central_parts + 1] = "PK\001\002" .. u16(20) ..
      u16(20) .. u16(0) .. u16(0) .. u16(0) .. u16(0) .. u32(crc) ..
      u32(#payload) .. u32(#payload) .. u16(#name) .. u16(0) ..
      u16(0) .. u16(0) .. u16(0) .. u32(0) .. u32(offset) .. name
    offset = offset + #local_header + #payload
  end
  local central = table.concat(central_parts)
  local body = table.concat(local_parts) .. central
  local eocd = "PK\005\006" .. u16(0) .. u16(0) .. u16(#members) ..
    u16(#members) .. u32(#central) .. u32(#body - #central) .. u16(0)
  write_file(path, body .. eocd)
end

-- Generate the corpus into `dir`.  Returns a manifest table:
--   { npy = { name -> path }, npz = path, big_endian = path }
function M.generate(dir)
  if type(dir) ~= "string" or dir == "" then
    error("fixture generator expects a non-empty output directory", 2)
  end
  if not ffi.abi("le") then
    error("fixture generator: little-endian host required", 2)
  end

  local npy = {}
  local npz_members = {}

  local function add(name, descr, shape, payload, in_npz)
    local path = dir .. "/generated-" .. name .. ".npy"
    local bytes = npy_bytes(descr, shape, payload)
    write_file(path, bytes)
    npy[name] = path
    if in_npz then
      npz_members[#npz_members + 1] = { name = name .. ".npy", payload = bytes }
    end
  end

  add("float32-3d", "<f4", { 2, 2, 2 },
    encode_f32({ 1, 2, 3, 4, 5, 6, 7, 8 }), false)
  add("float32-2x2", "<f4", { 2, 2 },
    encode_f32({ 1.5, -2.25, 0, 3.5 }), true)
  add("float64-2x3", "<f8", { 2, 3 },
    encode_f64({ math.pi, -0.125, 1e300, -1e-300, 2.5, 42 }), false)
  add("int64-1d", "<i8", { 4 },
    encode_i64({ -3, 0, 7, 42 }), true)
  add("bool-2x2", "|b1", { 2, 2 },
    encode_bool({ true, false, true, true }), true)
  add("empty-2x0", "<f4", { 2, 0 }, "", true)
  add("empty-0x3", "<i8", { 0, 3 }, "", false)
  add("scalar-float64", "<f8", {}, encode_f64({ 1.25 }), true)
  add("scalar-bool", "|b1", {}, encode_bool({ true }), false)

  -- A deliberately big-endian file: the reader must reject it before copying
  -- any payload.
  local big_endian = dir .. "/generated-big-endian-f4.npy"
  write_file(big_endian, npy_bytes(">f4", { 2 }, string.rep("\0", 8)))

  local npz = dir .. "/generated-corpus.npz"
  write_npz(npz, npz_members)

  return { npy = npy, npz = npz, big_endian = big_endian }
end

return M
