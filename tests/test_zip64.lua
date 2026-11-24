--
-- ZIP64 .npz reader coverage.
--
-- NumPy switches to ZIP64 records when a member or the archive exceeds the
-- classic ZIP32 field widths.  The fixtures are built in Lua so the suite
-- stays dependency-free and does not commit generated archives.
--

local bit = require("bit")
local numlua = require("numlua")

local function u16(value)
  return string.char(value % 256, math.floor(value / 256) % 256)
end

local function u32(value)
  return string.char(value % 256, math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256, math.floor(value / 16777216) % 256)
end

local function u64(value)
  local low = value % 4294967296
  local high = math.floor(value / 4294967296)
  return u32(low) .. u32(high)
end

local function crc32(value)
  local table_ = {}
  for n = 0, 255 do
    local c = n
    for _ = 1, 8 do
      if bit.band(c, 1) ~= 0 then
        c = bit.bxor(0xedb88320, bit.rshift(c, 1))
      else
        c = bit.rshift(c, 1)
      end
    end
    table_[n] = c
  end
  local crc = 0xffffffff
  for index = 1, #value do
    local byte = value:byte(index)
    crc = bit.bxor(table_[bit.band(bit.bxor(crc, byte), 0xff)],
      bit.rshift(crc, 8))
  end
  crc = bit.bxor(crc, 0xffffffff)
  if crc < 0 then crc = crc + 4294967296 end
  return crc
end

local function path(name)
  return TMPDIR .. "/zip64-" .. name
end

local function write_bytes(file, bytes)
  local handle = assert(io.open(file, "wb"))
  assert(handle:write(bytes))
  assert(handle:close())
end

local function read_bytes(file)
  local handle = assert(io.open(file, "rb"))
  local bytes = assert(handle:read("*a"))
  assert(handle:close())
  return bytes
end

local function npy_payload(tensor)
  local file = path("member-source.npy")
  numlua.save(file, tensor)
  local bytes = read_bytes(file)
  os.remove(file)
  return bytes
end

-- Build a one-member ZIP64 archive.  Both the local and central headers use
-- saturated 32-bit sizes and an id-0x0001 extra field; the archive also has a
-- ZIP64 EOCD record and locator with the classic EOCD sentinels.
local function zip64_archive(name, payload)
  local crc = crc32(payload)
  local size = #payload
  local local_extra = u16(0x0001) .. u16(16) .. u64(size) .. u64(size)
  local local_header = "PK\003\004" .. u16(45) .. u16(0) .. u16(0) ..
    u16(0) .. u16(0) .. u32(crc) .. u32(0xffffffff) .. u32(0xffffffff) ..
    u16(#name) .. u16(#local_extra) .. name .. local_extra

  local central_extra = u16(0x0001) .. u16(24) .. u64(size) .. u64(size) ..
    u64(0)
  local central_offset = #local_header + #payload
  local central_header = "PK\001\002" .. u16(45) .. u16(45) .. u16(0) ..
    u16(0) .. u16(0) .. u16(0) .. u32(crc) .. u32(0xffffffff) ..
    u32(0xffffffff) .. u16(#name) .. u16(#central_extra) .. u16(0) ..
    u16(0) .. u16(0) .. u32(0) .. u32(0xffffffff) .. name .. central_extra

  local zip64_eocd_offset = central_offset + #central_header
  local zip64_eocd = "PK\006\006" .. u64(44) .. u16(45) .. u16(45) ..
    u32(0) .. u32(0) .. u64(1) .. u64(1) .. u64(#central_header) ..
    u64(central_offset)
  local locator = "PK\006\007" .. u32(0) .. u64(zip64_eocd_offset) .. u32(1)
  local eocd = "PK\005\006" .. u16(0) .. u16(0) .. u16(0xffff) ..
    u16(0xffff) .. u32(0xffffffff) .. u32(0xffffffff) .. u16(0)

  return local_header .. payload .. central_header .. zip64_eocd .. locator ..
    eocd
end

test("loadz reads ZIP64 archives", function()
  local expected = { { 1.5, 2.5 }, { 3.5, 4.5 } }
  local tensor = numlua.array(expected, { dtype = "float64" })
  local file = path("valid.npz")
  write_bytes(file, zip64_archive("values.npy", npy_payload(tensor)))
  local loaded = numlua.loadz(file)
  assert_tensor(loaded.values, expected, 0, "ZIP64 member values")
  assert_dtype(loaded.values, "float64", "ZIP64 member dtype")
  os.remove(file)
end)

test("loadz reads ZIP64 archives with archives larger than 65535 bytes", function()
  local tensor = numlua.arange(70000, { dtype = "int64" })
  local file = path("large.npz")
  write_bytes(file, zip64_archive("large.npy", npy_payload(tensor)))
  local loaded = numlua.loadz(file)
  assert_eq(loaded.large:numel(), 70000, "large ZIP64 member length")
  assert_eq(loaded.large:size(1), 70000, "large ZIP64 member shape")
  assert_eq(loaded.large:get(69999), 69998, "large ZIP64 member tail")
  os.remove(file)
end)

test("loadz rejects ZIP64 sentinels without a locator", function()
  local tensor = numlua.array({ 1, 2, 3 })
  local bytes = zip64_archive("values.npy", npy_payload(tensor))
  local eocd = assert(bytes:find("PK\005\006", 1, true))
  -- Keep only the classic EOCD, dropping the ZIP64 record and locator.
  local classic = bytes:sub(1, eocd - 21) .. bytes:sub(eocd)
  local file = path("missing-locator.npz")
  write_bytes(file, classic)
  assert_error(function() numlua.loadz(file) end, "missing its locator")
  os.remove(file)
end)

test("loadz rejects corrupt ZIP64 records", function()
  local tensor = numlua.array({ 1, 2, 3 })
  local bytes = zip64_archive("values.npy", npy_payload(tensor))
  local locator = assert(bytes:find("PK\006\007", 1, true))
  -- Point the locator at the archive body instead of the ZIP64 EOCD record.
  local broken = bytes:sub(1, locator + 7) .. u64(0) ..
    bytes:sub(locator + 16)
  local file = path("bad-record.npz")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "corrupt ZIP64")
  os.remove(file)
end)

test("loadz rejects truncated ZIP64 extra fields", function()
  local tensor = numlua.array({ 1, 2, 3 })
  local bytes = zip64_archive("values.npy", npy_payload(tensor))
  local central = assert(bytes:find("PK\001\002", 1, true))
  local name_length = bytes:byte(central + 28) + bytes:byte(central + 29) * 256
  local extra = central + 46 + name_length - 1
  -- Shrink the ZIP64 extra field's declared payload from 24 bytes to 16.
  local broken = bytes:sub(1, extra + 2) .. u16(16) .. bytes:sub(extra + 5)
  local file = path("truncated-extra.npz")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "truncated ZIP64 extra")
  os.remove(file)
end)
