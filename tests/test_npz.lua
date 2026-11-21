local numlua = require("numlua")
local npz = require("numlua.npz")
local generator = require("tests.fixtures.generate")

local counter = 0
local function path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".npz"
end

local function read_bytes(file)
  local handle = assert(io.open(file, "rb"))
  local data = handle:read("*a")
  handle:close()
  return data
end

local function write_bytes(file, data)
  local handle = assert(io.open(file, "wb"))
  assert(handle:write(data))
  handle:close()
end

local function u16(value)
  return string.char(value % 256, math.floor(value / 256) % 256)
end

local function u32(value)
  return string.char(value % 256, math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256, math.floor(value / 16777216) % 256)
end

local function u16_at(data, offset)
  local low, high = data:byte(offset, offset + 1)
  return low + high * 256
end

local function u32_at(data, offset)
  local a, b, c, d = data:byte(offset, offset + 3)
  return a + b * 256 + c * 65536 + d * 16777216
end

local function patch(data, offset, replacement)
  return data:sub(1, offset - 1) .. replacement ..
    data:sub(offset + #replacement)
end

local function valid_archive(name)
  local file = path(name)
  assert_true(numlua.savez(file, { x = numlua.array({ 1, 2, 3 }) }))
  return file, read_bytes(file)
end

local function file_exists(file)
  local handle = io.open(file, "rb")
  if handle == nil then return false end
  handle:close()
  return true
end

-- Lower the internal ZIP32 writer limits for one test, restoring them even
-- when an assertion fails.  This keeps overflow tests fast and avoids
-- allocating multi-gigabyte arrays.
local function with_writer_limits(overrides, fn)
  local limits = npz._writer_limits
  local saved = {}
  for key, value in pairs(overrides) do
    saved[key] = limits[key]
    limits[key] = value
  end
  local ok, err = pcall(fn)
  for key in pairs(saved) do
    limits[key] = saved[key]
  end
  if not ok then error(err, 0) end
end

local function corpus()
  return generator.generate(TMPDIR)
end

test("savez/loadz round-trip several arrays", function()
  local file = path("roundtrip")
  local half = numlua.array({ 1.5, -2.25, 0, 3.25 }):to("float16")
  assert_true(numlua.savez(file, {
    weights = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } }),
    labels = numlua.array({ 1, 0, 1 }, { dtype = "int64" }),
    mask = numlua.array({ true, false }),
    half = half,
    bytes = numlua.array({ -128, 0, 127 }, { dtype = "int8" }),
    shorts = numlua.array({ -32768, 0, 32767 }, { dtype = "int16" }),
    ints = numlua.array({ -2147483648, 0, 2147483647 },
      { dtype = "int32" }),
    unsigned = numlua.array({ 0, 127, 255 }, { dtype = "uint8" }),
  }))
  local loaded = numlua.loadz(file)
  assert_shape(loaded.weights, { 2, 2 })
  assert_tensor(loaded.weights, { { 1.5, 2.5 }, { 3.5, 4.5 } })
  assert_dtype(loaded.labels, "int64")
  assert_tensor(loaded.labels, { 1, 0, 1 })
  assert_tensor(loaded.mask, { true, false })
  assert_dtype(loaded.half, "float16")
  assert_tensor(loaded.half, { 1.5, -2.25, 0, 3.25 })
  assert_dtype(loaded.bytes, "int8")
  assert_tensor(loaded.bytes, { -128, 0, 127 })
  assert_dtype(loaded.shorts, "int16")
  assert_tensor(loaded.shorts, { -32768, 0, 32767 })
  assert_dtype(loaded.ints, "int32")
  assert_tensor(loaded.ints, { -2147483648, 0, 2147483647 })
  assert_dtype(loaded.unsigned, "uint8")
  assert_tensor(loaded.unsigned, { 0, 127, 255 })
  os.remove(file)
end)

test("savez/loadz preserve C order for non-contiguous tensors", function()
  local base = numlua.reshape(
    numlua.arange(24, { dtype = "int32" }), { 2, 3, 4 })
  local source = numlua.transpose(base, { 3, 1, 2 })
  assert_false(source:is_contiguous())
  assert_shape(source, { 4, 2, 3 })

  local file = path("non-contiguous")
  assert_true(numlua.savez(file, { view = source }))
  local loaded = numlua.loadz(file)
  assert_dtype(loaded.view, "int32")
  assert_shape(loaded.view, { 4, 2, 3 })
  assert_true(loaded.view:is_contiguous())
  assert_true(loaded.view:equal(source:contiguous()))
  os.remove(file)
end)

test("savez_compressed writes method-8 entries and round-trips arrays",
    function()
  if not npz.zlib_available() then
    assert_error(function()
      numlua.savez_compressed(path("compressed-unavailable"), {
        values = numlua.array({ 1, 2, 3 }),
      })
    end, "requires a system zlib")
    return
  end

  local file = path("compressed")
  local view = numlua.transpose(numlua.reshape(
    numlua.arange(24, { dtype = "int32" }), { 2, 3, 4 }), { 3, 1, 2 })
  local arrays = {
    values = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } }),
    big = numlua.zeros(32, 32),
    empty = numlua.zeros(0),
    scalar = numlua.array(1.25),
    mask = numlua.array({ true, false, true }),
    view = view,
  }
  assert_true(numlua.savez_compressed(file, arrays))

  local bytes = read_bytes(file)
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  assert_eq(u16_at(bytes, local_header + 8), 8)

  local stored_file = path("compressed-compare")
  assert_true(numlua.savez(stored_file, arrays))
  assert_true(#bytes < #read_bytes(stored_file))

  local loaded = numlua.loadz(file)
  assert_shape(loaded.values, { 2, 2 })
  assert_tensor(loaded.values, { { 1.5, 2.5 }, { 3.5, 4.5 } })
  assert_shape(loaded.big, { 32, 32 })
  assert_eq(loaded.big:sum():item(), 0)
  assert_shape(loaded.empty, { 0 })
  assert_eq(loaded.empty:numel(), 0)
  assert_eq(loaded.scalar:dim(), 0)
  assert_eq(loaded.scalar:item(), 1.25)
  assert_dtype(loaded.mask, "bool")
  assert_tensor(loaded.mask, { true, false, true })
  assert_dtype(loaded.view, "int32")
  assert_shape(loaded.view, { 4, 2, 3 })
  assert_true(loaded.view:equal(arrays.view))
  os.remove(file)
  os.remove(stored_file)
end)

test("savez_compressed errors or falls back without zlib", function()
  local file = path("compressed-fallback")
  if npz.zlib_available() then
    -- The fallback option is harmless when zlib is present: entries stay
    -- compressed and no fallback signal is returned.
    local ok, method = numlua.savez_compressed(file, {
      x = numlua.arange(3),
    }, { fallback = "stored" })
    assert_true(ok)
    assert_eq(method, nil)
    local bytes = read_bytes(file)
    local header = assert(bytes:find("PK\003\004", 1, true))
    assert_eq(u16_at(bytes, header + 8), 8)
    os.remove(file)
    return
  end

  assert_error(function()
    numlua.savez_compressed(file, { x = numlua.arange(3) })
  end, "requires a system zlib")
  assert_true(io.open(file, "rb") == nil)

  local ok, method = numlua.savez_compressed(file, {
    x = numlua.arange(3),
  }, { fallback = "stored" })
  assert_true(ok)
  assert_eq(method, "stored")
  local bytes = read_bytes(file)
  local header = assert(bytes:find("PK\003\004", 1, true))
  assert_eq(u16_at(bytes, header + 8), 0)
  local loaded = numlua.loadz(file)
  assert_tensor(loaded.x, { 0, 1, 2 })
  os.remove(file)
end)

test("loadz rejects implausible deflate expansion before allocating",
    function()
  if not npz.zlib_available() then return end
  local file = path("inflate-ratio")
  assert_true(numlua.savez_compressed(file, {
    x = numlua.array({ 1, 2, 3 }),
  }))
  local bytes = read_bytes(file)
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  local central = assert(bytes:find("PK\001\002", 1, true))
  local actual = u32_at(bytes, local_header + 22)
  assert_true(actual > 0)

  -- A declared size far past deflate's 1032:1 bound is rejected without
  -- allocating the declared buffer.
  local implausible = 64 * 1024 * 1024
  local broken = patch(bytes, local_header + 22, u32(implausible))
  broken = patch(broken, central + 24, u32(implausible))
  local broken_file = path("inflate-ratio-broken")
  write_bytes(broken_file, broken)
  assert_error(function() numlua.loadz(broken_file) end, "implausible")

  -- A merely wrong size inside the plausible bound reaches zlib and fails
  -- with the size check rather than the guard, so valid sizes are not
  -- rejected just because the archive is corrupt.
  local off_by_one = patch(bytes, local_header + 22, u32(actual + 1))
  off_by_one = patch(off_by_one, central + 24, u32(actual + 1))
  local off_file = path("inflate-ratio-off-by-one")
  write_bytes(off_file, off_by_one)
  assert_error(function() numlua.loadz(off_file) end,
    "inflated size mismatch")

  os.remove(file)
  os.remove(broken_file)
  os.remove(off_file)
end)

test("loadz accepts highly compressible deflate members", function()
  if not npz.zlib_available() then return end
  local file = path("compressible")
  local zeros = numlua.zeros(1024 * 1024, { dtype = "int8" })
  assert_true(numlua.savez_compressed(file, { zeros = zeros }))

  local bytes = read_bytes(file)
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  assert_eq(u16_at(bytes, local_header + 8), 8)
  local compressed = u32_at(bytes, local_header + 18)
  local uncompressed = u32_at(bytes, local_header + 22)
  assert_true(uncompressed > compressed * 100,
    "test input is not highly compressible")

  -- The expansion-ratio guard must not reject a legitimate high-ratio stream.
  local loaded = numlua.loadz(file)
  assert_shape(loaded.zeros, { 1024 * 1024 })
  assert_eq(loaded.zeros:numel(), 1024 * 1024)
  os.remove(file)
end)

test("zlib-disabled errors name the environment switch", function()
  if npz.zlib_available() then return end
  local value = os.getenv("NUMLUA_DISABLE_ZLIB")
  if value ~= "1" and value ~= "true" and value ~= "yes" then return end

  local file = path("disabled-diagnostic")
  assert_error(function()
    numlua.savez_compressed(file, { x = numlua.arange(3) })
  end, "NUMLUA_DISABLE_ZLIB")

  local files = corpus()
  assert_error(function() numlua.loadz(files.npz_deflate) end,
    "NUMLUA_DISABLE_ZLIB")
end)

test("savez_compressed validates options and arrays", function()
  local file = path("compressed-invalid")
  assert_error(function()
    numlua.savez_compressed(file, { x = numlua.zeros(2) }, true)
  end, "options table")
  assert_error(function()
    numlua.savez_compressed(file, { x = numlua.zeros(2) },
      { fallback = "deflate" })
  end, "fallback must be 'stored'")
  assert_error(function()
    numlua.savez_compressed(file, { x = 1 })
  end, "is not a Tensor")
  assert_error(function()
    numlua.savez_compressed("", { x = numlua.zeros(2) })
  end, "non-empty path")
end)

test("loadz reads generated deflate entries or fails clearly without zlib",
    function()
  local files = corpus()
  if npz.zlib_available() then
    local loaded = numlua.loadz(files.npz_deflate)

    assert_dtype(loaded["float32-2x2"], "float32")
    assert_tensor(loaded["float32-2x2"], { { 1.5, -2.25 }, { 0, 3.5 } })

    assert_dtype(loaded["float16-1d"], "float16")
    assert_tensor(loaded["float16-1d"], { 1.5, -2.25, 0, 3.25 })

    assert_dtype(loaded["float16-scalar"], "float16")
    assert_eq(loaded["float16-scalar"]:dim(), 0)
    assert_eq(loaded["float16-scalar"]:item(), -0.125)

    assert_dtype(loaded["int8-1d"], "int8")
    assert_tensor(loaded["int8-1d"], { -128, -3, 0, 7, 127 })

    assert_dtype(loaded["int16-2x2"], "int16")
    assert_tensor(loaded["int16-2x2"], { { -32768, -3 }, { 0, 32767 } })

    assert_dtype(loaded["int32-1d"], "int32")
    assert_tensor(loaded["int32-1d"],
      { -2147483648, -3, 0, 7, 2147483647 })

    assert_dtype(loaded["uint8-2x2"], "uint8")
    assert_tensor(loaded["uint8-2x2"], { { 0, 1 }, { 127, 255 } })

    assert_dtype(loaded["int64-1d"], "int64")
    assert_tensor(loaded["int64-1d"], { -3, 0, 7, 42 })

    assert_dtype(loaded["scalar-float64"], "float64")
    assert_eq(loaded["scalar-float64"]:item(), 1.25)
  else
    assert_error(function() numlua.loadz(files.npz_deflate) end,
      "requires a system zlib")
  end
end)

test("loadz rejects non-ZIP input and unsupported compression methods",
    function()
  local file = path("plain")
  write_bytes(file, "not a zip")
  assert_error(function() numlua.loadz(file) end, "not a ZIP")
  os.remove(file)

  local valid, bytes = valid_archive("method")
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  local central = assert(bytes:find("PK\001\002", 1, true))
  bytes = patch(bytes, local_header + 8, u16(12))
  bytes = patch(bytes, central + 10, u16(12))
  local file2 = path("method-unsupported")
  write_bytes(file2, bytes)
  assert_error(function() numlua.loadz(file2) end,
    "unsupported compression method 12")
  os.remove(valid)
  os.remove(file2)
end)

test("loadz rejects corrupt central directories", function()
  local valid, bytes = valid_archive("central-signature")
  local central = assert(bytes:find("PK\001\002", 1, true))
  local broken = patch(bytes, central, "PK\001\003")
  local file = path("bad-central")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "corrupt central directory")
  os.remove(file)

  valid, bytes = valid_archive("central-offset")
  local eocd = assert(bytes:find("PK\005\006", 1, true))
  broken = patch(bytes, eocd + 16, u32(#bytes + 100))
  file = path("bad-central-offset")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "corrupt central directory")
  os.remove(file)

  valid, bytes = valid_archive("central-truncated")
  eocd = assert(bytes:find("PK\005\006", 1, true))
  broken = patch(bytes, eocd + 12, u32(4))
  file = path("truncated-central")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "truncated central directory")
  os.remove(file)
  os.remove(valid)
end)

test("loadz rejects CRC mismatches", function()
  local valid, bytes = valid_archive("crc-central")
  local central = assert(bytes:find("PK\001\002", 1, true))
  local broken = patch(bytes, central + 16, u32(0))
  local file = path("bad-crc-central")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "CRC mismatch")
  os.remove(file)

  valid, bytes = valid_archive("crc-local")
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  broken = patch(bytes, local_header + 14, u32(0))
  file = path("bad-crc-local")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "CRC mismatch")
  os.remove(file)
  os.remove(valid)
end)

test("loadz rejects truncated archives", function()
  local valid, bytes = valid_archive("truncated")
  local file = path("truncated-eocd")
  write_bytes(file, bytes:sub(1, #bytes - 10))
  assert_error(function() numlua.loadz(file) end, "truncated ZIP archive")
  os.remove(file)

  local local_header = assert(bytes:find("PK\003\004", 1, true))
  local central = assert(bytes:find("PK\001\002", 1, true))
  local broken = patch(bytes, local_header + 18, u32(100000))
  broken = patch(broken, central + 20, u32(100000))
  file = path("truncated-local-data")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "truncated local data")
  os.remove(file)
  os.remove(valid)
end)

test("loadz rejects encrypted, multi-disk, and ZIP64 archives", function()
  local valid, bytes = valid_archive("encrypted")
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  local central = assert(bytes:find("PK\001\002", 1, true))
  local broken = patch(bytes, local_header + 6, u16(1))
  broken = patch(broken, central + 8, u16(1))
  local file = path("encrypted")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "encrypted ZIP entries")
  os.remove(file)
  os.remove(valid)

  -- A non-zero EOCD disk number rejects a split archive before any member is
  -- read.
  valid, bytes = valid_archive("multi-disk-eocd")
  local eocd = assert(bytes:find("PK\005\006", 1, true))
  broken = patch(bytes, eocd + 4, u16(1))
  file = path("multi-disk-eocd")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "multi-disk ZIP")
  os.remove(file)
  os.remove(valid)

  -- A member that claims a non-zero start disk is also multi-disk.
  valid, bytes = valid_archive("multi-disk-entry")
  central = assert(bytes:find("PK\001\002", 1, true))
  broken = patch(bytes, central + 34, u16(1))
  file = path("multi-disk-entry")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "multi-disk ZIP")
  os.remove(file)
  os.remove(valid)

  -- 0xffff in the EOCD entry count is the ZIP64 sentinel.
  valid, bytes = valid_archive("zip64")
  eocd = assert(bytes:find("PK\005\006", 1, true))
  broken = patch(bytes, eocd + 10, u16(0xffff))
  file = path("zip64")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "ZIP64")
  os.remove(file)
  os.remove(valid)
end)

test("loadz validates local headers against the central directory", function()
  local valid, bytes = valid_archive("local-method")
  local local_header = assert(bytes:find("PK\003\004", 1, true))
  local broken = patch(bytes, local_header + 8, u16(8))
  local file = path("local-method")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end,
    "compression method mismatch")
  os.remove(file)
  os.remove(valid)

  valid, bytes = valid_archive("local-name")
  local_header = assert(bytes:find("PK\003\004", 1, true))
  broken = patch(bytes, local_header + 30, "y")
  file = path("local-name")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end, "member name mismatch")
  os.remove(file)
  os.remove(valid)

  valid, bytes = valid_archive("local-size")
  local_header = assert(bytes:find("PK\003\004", 1, true))
  broken = patch(bytes, local_header + 18, u32(999))
  file = path("local-size")
  write_bytes(file, broken)
  assert_error(function() numlua.loadz(file) end,
    "local header size mismatch")
  os.remove(file)
  os.remove(valid)
end)

test("savez enforces the ZIP32 entry-count limit", function()
  local file = path("entry-limit")
  with_writer_limits({ max_entries = 2 }, function()
    assert_error(function()
      numlua.savez(file, {
        a = numlua.array(1),
        b = numlua.array(2),
        c = numlua.array(3),
      })
    end, "at most 2 entries")
  end)
  assert_false(file_exists(file))
end)

test("savez enforces ZIP32 member-name and size limits", function()
  local long_name = string.rep("x", 65536)
  local file = path("name-limit")
  assert_error(function()
    numlua.savez(file, { [long_name] = numlua.array(1) })
  end, "ZIP32 names are limited to 65535 bytes")
  assert_false(file_exists(file))

  -- A scalar .npy member is 132 bytes, so a 100-byte limit rejects the
  -- uncompressed payload before any output file is opened.
  file = path("entry-size-limit")
  with_writer_limits({ max_u32 = 100 }, function()
    assert_error(function()
      numlua.savez(file, { x = numlua.array(1) })
    end, "ZIP32 entries are limited to 100 bytes")
  end)
  assert_false(file_exists(file))

  if npz.zlib_available() then
    -- With compression enabled the deflated size is checked first; a one-byte
    -- limit exercises that branch.
    file = path("compressed-size-limit")
    with_writer_limits({ max_u32 = 1 }, function()
      assert_error(function()
        numlua.savez_compressed(file, { x = numlua.array(1) })
      end, "compressed entry 'x'")
    end)
    assert_false(file_exists(file))
  end
end)

test("savez enforces the ZIP32 archive-offset limit", function()
  local file = path("offset-limit")
  -- Each scalar member occupies 163 local bytes (30-byte header, 1-byte name,
  -- 132-byte .npy payload); the second pushes the central-directory offset
  -- past 270 while both individual payloads remain under the limit.
  with_writer_limits({ max_u32 = 270 }, function()
    assert_error(function()
      numlua.savez(file, {
        a = numlua.array(1),
        b = numlua.array(2),
      })
    end, "central-directory offsets are limited to 270 bytes")
  end)
  assert_false(file_exists(file))
end)

test("savez accepts archives exactly at the ZIP32 entry limit", function()
  local file = path("entry-limit-ok")
  with_writer_limits({ max_entries = 2 }, function()
    assert_true(numlua.savez(file, {
      a = numlua.array(1),
      b = numlua.array(2),
    }))
  end)
  local loaded = numlua.loadz(file)
  assert_eq(loaded.a:item(), 1)
  assert_eq(loaded.b:item(), 2)
  os.remove(file)
end)

test("savez writes an empty archive that loadz reads as empty", function()
  local file = path("empty-archive")
  assert_true(numlua.savez(file, {}))
  local bytes = read_bytes(file)
  assert_eq(#bytes, 22)
  assert_eq(bytes:sub(1, 4), "PK\005\006")
  local loaded = numlua.loadz(file)
  assert_eq(next(loaded), nil)
  os.remove(file)
end)

test("savez preserves keys that already end in .npy", function()
  local file = path("npy-suffix")
  assert_true(numlua.savez(file, {
    ["weights.npy"] = numlua.array({ 1.5, -2.5 }),
  }))
  local bytes = read_bytes(file)
  assert_true(bytes:find("weights.npy", 1, true) ~= nil)
  assert_true(bytes:find("weights.npy.npy", 1, true) == nil)

  local loaded = numlua.loadz(file)
  assert_eq(loaded["weights.npy"], nil)
  assert_tensor(loaded.weights, { 1.5, -2.5 })
  os.remove(file)
end)
