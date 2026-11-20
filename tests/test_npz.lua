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

local function patch(data, offset, replacement)
  return data:sub(1, offset - 1) .. replacement ..
    data:sub(offset + #replacement)
end

local function valid_archive(name)
  local file = path(name)
  assert_true(numlua.savez(file, { x = numlua.array({ 1, 2, 3 }) }))
  return file, read_bytes(file)
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
