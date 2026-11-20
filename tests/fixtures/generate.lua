--
-- Generate the small NumPy-format fixture corpus used by the test suite.
--
-- The corpus is written under tests/.tmp/ at test time instead of being
-- committed, so no generated binary artefacts enter git.  The bytes are
-- produced here directly (NPY v1.0 header + little-endian payload, ZIP for
-- NPZ) rather than through numlua.save/savez, so the reader is exercised
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
  crc = bit.bxor(crc, 0xffffffff)
  if crc < 0 then crc = crc + 4294967296 end
  return crc
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

-- Convert a Lua number to IEEE 754 binary16 bits (round to nearest-even).
local function float_to_half(value)
  local float = ffi.new("float[1]", value)
  local bits = ffi.cast("uint32_t *", float)[0]
  local sign = bit.rshift(bit.band(bits, 0x80000000), 16)
  local exponent = bit.band(bit.rshift(bits, 23), 0xff)
  local mantissa = bit.band(bits, 0x7fffff)

  if exponent == 0xff then
    if mantissa == 0 then return bit.bor(sign, 0x7c00) end
    return bit.bor(sign, 0x7e00)
  end

  local half_exponent = exponent - 112
  if half_exponent >= 31 then
    return bit.bor(sign, 0x7c00)
  end
  if half_exponent <= 0 then
    if half_exponent < -10 then return sign end
    mantissa = bit.bor(mantissa, 0x800000)
    local shift = 14 - half_exponent
    local half_mantissa = bit.rshift(mantissa, shift)
    local round_bit = bit.band(bit.rshift(mantissa, shift - 1), 1)
    local sticky = bit.band(mantissa, bit.lshift(1, shift - 1) - 1) ~= 0
    if round_bit == 1 and (sticky or bit.band(half_mantissa, 1) == 1) then
      half_mantissa = half_mantissa + 1
    end
    return bit.bor(sign, half_mantissa)
  end

  local half_mantissa = bit.rshift(mantissa, 13)
  local round_bit = bit.band(bit.rshift(mantissa, 12), 1)
  local sticky = bit.band(mantissa, 0xfff) ~= 0
  if round_bit == 1 and (sticky or bit.band(half_mantissa, 1) == 1) then
    half_mantissa = half_mantissa + 1
  end
  if half_mantissa >= 1024 then
    half_mantissa = 0
    half_exponent = half_exponent + 1
    if half_exponent >= 31 then
      return bit.bor(sign, 0x7c00)
    end
  end
  return bit.bor(sign, bit.lshift(half_exponent, 10) + half_mantissa)
end

local function encode_f16(values)
  local buffer = ffi.new("uint16_t[?]", math.max(#values, 1))
  for index = 1, #values do
    buffer[index - 1] = float_to_half(values[index])
  end
  return ffi.string(buffer, #values * 2)
end

local function encode_i64(values)
  local buffer = ffi.new("int64_t[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values * 8)
end

local function encode_i32(values)
  local buffer = ffi.new("int32_t[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values * 4)
end

local function encode_i16(values)
  local buffer = ffi.new("int16_t[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values * 2)
end

local function encode_i8(values)
  local buffer = ffi.new("int8_t[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values)
end

local function encode_u8(values)
  local buffer = ffi.new("uint8_t[?]", math.max(#values, 1))
  for index = 1, #values do buffer[index - 1] = values[index] end
  return ffi.string(buffer, #values)
end

local function encode_bool(values)
  local parts = {}
  for index = 1, #values do parts[index] = values[index] and "\1" or "\0" end
  return table.concat(parts)
end

-- A valid raw DEFLATE stream made entirely of stored (type 00) blocks.  This
-- needs no zlib at generation time while still exercising the method-8 path.
local function raw_deflate_stored(payload)
  if #payload == 0 then
    return "\1\0\0\255\255"
  end
  local parts = {}
  local position = 1
  while position <= #payload do
    local length = math.min(65535, #payload - position + 1)
    local final = position + length - 1 == #payload
    parts[#parts + 1] = (final and "\1" or "\0") .. u16(length) ..
      u16(bit.band(bit.bnot(length), 0xffff)) ..
      payload:sub(position, position + length - 1)
    position = position + length
  end
  return table.concat(parts)
end

local generator_zlib_state
local function load_generator_zlib()
  if generator_zlib_state ~= nil then
    return generator_zlib_state or nil
  end
  local cdef = [[
unsigned long compress2(unsigned char *dest, unsigned long *destLen,
  const unsigned char *source, unsigned long sourceLen, int level);
unsigned long compressBound(unsigned long sourceLen);
]]
  local loaded = false
  for _, name in ipairs({ "z", "libz.so.1", "libz.so", "libz.dylib" }) do
    local ok, handle = pcall(ffi.load, name)
    if ok then
      if pcall(ffi.cdef, cdef) and
          pcall(function() return handle.compressBound(1) end) then
        loaded = handle
        break
      end
    end
  end
  generator_zlib_state = loaded
  return loaded or nil
end

-- Produce a real compressed raw DEFLATE stream when zlib is available, and
-- fall back to valid stored DEFLATE blocks when it is not.
local function compress_deflate(payload)
  local zlib = load_generator_zlib()
  if zlib == nil then
    return raw_deflate_stored(payload)
  end
  local bound = tonumber(zlib.compressBound(#payload))
  local out = ffi.new("unsigned char[?]", math.max(bound, 1))
  local out_length = ffi.new("unsigned long[1]", math.max(bound, 1))
  local code = zlib.compress2(out, out_length,
    ffi.cast("const unsigned char *", payload), #payload, 6)
  if code ~= 0 then
    return raw_deflate_stored(payload)
  end
  local wrapped = ffi.string(out, tonumber(out_length[0]))
  -- compress2 emits a zlib wrapper (2-byte header, 4-byte Adler-32 trailer);
  -- ZIP method 8 stores the raw DEFLATE stream in between.
  if #wrapped >= 6 and bit.band(wrapped:byte(1), 0x0f) == 8 and
      (wrapped:byte(1) * 256 + wrapped:byte(2)) % 31 == 0 then
    return wrapped:sub(3, -5)
  end
  return raw_deflate_stored(payload)
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

-- Write a ZIP archive.  Members default to the stored method; set
-- member.method = 8 to write a deflate-compressed entry.
local function write_npz(path, members)
  local local_parts, central_parts = {}, {}
  local offset = 0
  for _, member in ipairs(members) do
    local payload = member.payload
    local name = member.name
    local method = member.method or 0
    local stored = payload
    if method == 8 then
      stored = member.compressed or compress_deflate(payload)
    elseif method ~= 0 then
      error("fixture generator: unsupported ZIP method " .. tostring(method), 2)
    end
    local crc = crc32(payload)
    local local_header = "PK\003\004" .. u16(20) .. u16(0) .. u16(method) ..
      u16(0) .. u16(0) .. u32(crc) .. u32(#stored) .. u32(#payload) ..
      u16(#name) .. u16(0) .. name
    local_parts[#local_parts + 1] = local_header
    local_parts[#local_parts + 1] = stored
    central_parts[#central_parts + 1] = "PK\001\002" .. u16(20) ..
      u16(20) .. u16(0) .. u16(method) .. u16(0) .. u16(0) .. u32(crc) ..
      u32(#stored) .. u32(#payload) .. u16(#name) .. u16(0) ..
      u16(0) .. u16(0) .. u16(0) .. u32(0) .. u32(offset) .. name
    offset = offset + #local_header + #stored
  end
  local central = table.concat(central_parts)
  local body = table.concat(local_parts) .. central
  local eocd = "PK\005\006" .. u16(0) .. u16(0) .. u16(#members) ..
    u16(#members) .. u32(#central) .. u32(#body - #central) .. u16(0)
  write_file(path, body .. eocd)
end

M.write_npz = write_npz
M.npy_bytes = npy_bytes
M.raw_deflate_stored = raw_deflate_stored

-- Generate the corpus into `dir`.  Returns a manifest table:
--   { npy = { name -> path }, npz = path, npz_deflate = path,
--     big_endian = path }
function M.generate(dir)
  if type(dir) ~= "string" or dir == "" then
    error("fixture generator expects a non-empty output directory", 2)
  end
  if not ffi.abi("le") then
    error("fixture generator: little-endian host required", 2)
  end

  local npy = {}
  local stored_members = {}
  local deflate_members = {}

  local function add(name, descr, shape, payload, in_stored, in_deflate)
    local path = dir .. "/generated-" .. name .. ".npy"
    local bytes = npy_bytes(descr, shape, payload)
    write_file(path, bytes)
    npy[name] = path
    if in_stored then
      stored_members[#stored_members + 1] = {
        name = name .. ".npy", payload = bytes,
      }
    end
    if in_deflate then
      deflate_members[#deflate_members + 1] = {
        name = name .. ".npy", payload = bytes, method = 8,
      }
    end
  end

  add("float32-3d", "<f4", { 2, 2, 2 },
    encode_f32({ 1, 2, 3, 4, 5, 6, 7, 8 }), false, false)
  add("float32-2x2", "<f4", { 2, 2 },
    encode_f32({ 1.5, -2.25, 0, 3.5 }), true, true)
  add("float64-2x3", "<f8", { 2, 3 },
    encode_f64({ math.pi, -0.125, 1e300, -1e-300, 2.5, 42 }), false, false)
  add("int64-1d", "<i8", { 4 },
    encode_i64({ -3, 0, 7, 42 }), true, true)
  add("bool-2x2", "|b1", { 2, 2 },
    encode_bool({ true, false, true, true }), true, false)
  add("empty-2x0", "<f4", { 2, 0 }, "", true, false)
  add("empty-0x3", "<i8", { 0, 3 }, "", false, false)
  add("scalar-float64", "<f8", {}, encode_f64({ 1.25 }), true, true)
  add("scalar-bool", "|b1", {}, encode_bool({ true }), false, false)

  add("float16-1d", "<f2", { 4 },
    encode_f16({ 1.5, -2.25, 0, 3.25 }), true, true)
  add("float16-scalar", "<f2", {}, encode_f16({ -0.125 }), false, true)
  add("int8-1d", "|i1", { 5 },
    encode_i8({ -128, -3, 0, 7, 127 }), true, true)
  add("int16-2x2", "<i2", { 2, 2 },
    encode_i16({ -32768, -3, 0, 32767 }), true, true)
  add("int32-1d", "<i4", { 5 },
    encode_i32({ -2147483648, -3, 0, 7, 2147483647 }), true, true)
  add("uint8-2x2", "|u1", { 2, 2 },
    encode_u8({ 0, 1, 127, 255 }), true, true)

  -- A deliberately big-endian file: the reader must reject it before copying
  -- any payload.
  local big_endian = dir .. "/generated-big-endian-f4.npy"
  write_file(big_endian, npy_bytes(">f4", { 2 }, string.rep("\0", 8)))

  local npz = dir .. "/generated-corpus.npz"
  write_npz(npz, stored_members)

  local npz_deflate = dir .. "/generated-deflate.npz"
  write_npz(npz_deflate, deflate_members)

  return {
    npy = npy,
    npz = npz,
    npz_deflate = npz_deflate,
    big_endian = big_endian,
  }
end

return M
