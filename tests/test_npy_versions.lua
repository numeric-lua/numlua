--
-- NPY header-version conformance tests.
--
-- These tests inspect the bytes written by numlua.save independently of the
-- reader: magic, version fields, header-length width, 64-byte alignment, and
-- the C-order payload.  They also exercise the full version/dtype matrix so a
-- regression in one writer path is visible without a NumPy installation.
--

local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/npy-version-" .. name .. "-" .. counter .. ".npy"
end

local function read_bytes(file)
  local handle = assert(io.open(file, "rb"))
  local data = handle:read("*a")
  handle:close()
  return data
end

local function file_exists(file)
  local handle = io.open(file, "rb")
  if handle == nil then return false end
  handle:close()
  return true
end

local function u16_at(data, offset)
  local a, b = data:byte(offset, offset + 1)
  return a + b * 256
end

local function u32_at(data, offset)
  local a, b, c, d = data:byte(offset, offset + 3)
  return a + b * 256 + c * 65536 + d * 16777216
end

local function save_bytes(tensor, options)
  local file = tmp_path("bytes")
  assert_true(numlua.save(file, tensor, options))
  local bytes = read_bytes(file)
  os.remove(file)
  return bytes
end

-- Split a .npy byte string into its version prefix, header, and payload.
local function split_npy(data)
  assert_eq(data:sub(1, 6), "\147NUMPY", "bad NPY magic")
  local major, minor = data:byte(7), data:byte(8)
  local prefix_size, header_size
  if major == 1 then
    prefix_size = 10
    header_size = u16_at(data, 9)
  else
    prefix_size = 12
    header_size = u32_at(data, 9)
  end
  assert_true(header_size >= 1, "empty NPY header")
  assert_true(#data >= prefix_size + header_size, "truncated NPY header")
  return {
    major = major,
    minor = minor,
    prefix_size = prefix_size,
    header_size = header_size,
    header = data:sub(prefix_size + 1, prefix_size + header_size),
    payload = data:sub(prefix_size + header_size + 1),
  }
end

local function assert_header(parsed, descr, shape_text)
  assert_eq(parsed.header:sub(-1), "\n", "header must end with a newline")
  assert_eq((parsed.prefix_size + parsed.header_size) % 64, 0,
    "prefix + header must be 64-byte aligned")
  assert_true(parsed.header:find("'descr': '" .. descr .. "'", 1, true)
    ~= nil, "header lacks descr " .. descr)
  assert_true(parsed.header:find("'fortran_order': False", 1, true) ~= nil,
    "header must declare C order")
  assert_true(parsed.header:find("'shape': " .. shape_text, 1, true) ~= nil,
    "header lacks shape " .. shape_text)
  for index = 1, #parsed.header do
    assert_true(parsed.header:byte(index) < 128,
      "writer headers are ASCII and UTF-8 compatible")
  end
end

test("writer emits v1.0, v2.0, and v3.0 headers with the right width", function()
  local tensor = numlua.array({ { 1.5, -2.25, 0 }, { 3.5, 4.5, -0.5 } })
  local payloads = {}
  local encoded = {}

  for version = 1, 3 do
    local data = save_bytes(tensor, { version = version })
    local parsed = split_npy(data)
    assert_eq(parsed.major, version)
    assert_eq(parsed.minor, 0)
    assert_eq(parsed.prefix_size, version == 1 and 10 or 12)
    assert_header(parsed, "<f4", "(2, 3)")
    assert_eq(#data, parsed.prefix_size + parsed.header_size + tensor:nbytes())
    assert_eq(#parsed.payload, tensor:nbytes())
    payloads[version] = parsed.payload
    encoded[version] = data

    local roundtrip_file = tmp_path("roundtrip")
    local roundtrip_handle = assert(io.open(roundtrip_file, "wb"))
    assert(roundtrip_handle:write(data))
    roundtrip_handle:close()
    local loaded = numlua.load(roundtrip_file)
    assert_true(loaded:equal(tensor))
    os.remove(roundtrip_file)
  end

  -- The payload and everything after the 8-byte magic/version prefix are
  -- version-independent for an ASCII header; v2.0 and v3.0 share the uint32
  -- prefix, while v1.0's uint16 prefix changes only the padding.
  assert_eq(payloads[1], payloads[2])
  assert_eq(payloads[2], payloads[3])
  assert_eq(encoded[2]:sub(9), encoded[3]:sub(9))

  local v1 = split_npy(encoded[1])
  assert_eq(v1.header_size, u16_at(encoded[1], 9))
  assert_true(v1.header_size <= 65535)
  local v2 = split_npy(encoded[2])
  assert_eq(v2.header_size, u32_at(encoded[2], 9))
  local v3 = split_npy(encoded[3])
  assert_eq(v3.header_size, u32_at(encoded[3], 9))
end)

test("default save stays byte-identical to v1.0", function()
  local tensor = numlua.array({ 1, 2, 3, 4 }, { dtype = "int32" })
  local default = save_bytes(tensor)
  assert_eq(default, save_bytes(tensor, { version = 1 }))
  assert_eq(default, save_bytes(tensor, { version = "1.0" }))

  local parsed = split_npy(default)
  assert_eq(parsed.major, 1)
  assert_eq(parsed.minor, 0)
  assert_eq(parsed.prefix_size, 10)
  assert_header(parsed, "<i4", "(4,)")
  assert_eq(#parsed.payload, 16)

  -- Numeric and dotted-string versions produce the same bytes.
  assert_eq(save_bytes(tensor, { version = 2 }),
    save_bytes(tensor, { version = "2.0" }))
  assert_eq(save_bytes(tensor, { version = 3 }),
    save_bytes(tensor, { version = "3.0" }))
end)

test("every dtype round-trips in every header version", function()
  local cases = {
    { dtype = "float16", values = { 1.5, -2.25, 0, 3.25 }, descr = "<f2",
      size = 2 },
    { dtype = "float32", values = { 1.5, -2.25, 0, 3.25 }, descr = "<f4",
      size = 4 },
    { dtype = "float64", values = { math.pi, -0.125, 1e300, -1e-300 },
      descr = "<f8", size = 8 },
    { dtype = "int8", values = { -128, -1, 0, 127 }, descr = "|i1",
      size = 1 },
    { dtype = "int16", values = { -32768, -1, 0, 32767 }, descr = "<i2",
      size = 2 },
    { dtype = "int32", values = { -2147483648, -1, 0, 2147483647 },
      descr = "<i4", size = 4 },
    { dtype = "int64", values = { -3, -1, 0, 42 }, descr = "<i8", size = 8 },
    { dtype = "uint8", values = { 0, 1, 127, 255 }, descr = "|u1",
      size = 1 },
    { dtype = "bool", values = { true, false, true, false }, descr = "|b1",
      size = 1 },
  }

  for _, case in ipairs(cases) do
    local source
    if case.dtype == "float16" then
      source = numlua.array(case.values,
        { dtype = "float32" }):to("float16")
    else
      source = numlua.array(case.values, { dtype = case.dtype })
    end
    for version = 1, 3 do
      local file = tmp_path(case.dtype .. "-v" .. version)
      assert_true(numlua.save(file, source, { version = version }))
      local parsed = split_npy(read_bytes(file))
      assert_eq(parsed.major, version)
      assert_header(parsed, case.descr, "(4,)")
      assert_eq(#parsed.payload, 4 * case.size)
      local loaded = numlua.load(file)
      assert_dtype(loaded, case.dtype)
      assert_shape(loaded, { 4 })
      assert_true(loaded:equal(source))
      os.remove(file)
    end
  end
end)

test("0-D, empty, and non-contiguous tensors round-trip in every version", function()
  local cases = {
    {
      name = "scalar",
      tensor = numlua.array(7, { dtype = "int64" }),
      shape = {},
      values = 7,
    },
    {
      name = "empty",
      tensor = numlua.zeros({ 2, 0 }),
      shape = { 2, 0 },
      numel = 0,
    },
    {
      name = "transposed",
      tensor = numlua.transpose(numlua.array({
        { 1, 2, 3 },
        { 4, 5, 6 },
      })),
      shape = { 3, 2 },
      values = { { 1, 4 }, { 2, 5 }, { 3, 6 } },
    },
  }

  for _, case in ipairs(cases) do
    for version = 1, 3 do
      local file = tmp_path(case.name .. "-v" .. version)
      assert_true(numlua.save(file, case.tensor, { version = version }))
      local parsed = split_npy(read_bytes(file))
      assert_eq(parsed.major, version)
      assert_eq(#parsed.payload, case.tensor:nbytes())
      local loaded = numlua.load(file)
      assert_dtype(loaded, case.tensor:dtype())
      assert_shape(loaded, case.shape)
      assert_true(loaded:equal(case.tensor))
      os.remove(file)
    end
  end
end)

test("version options are rejected before any file is written", function()
  local tensor = numlua.array({ 1, 2, 3 })
  local file = tmp_path("invalid")
  local bad_options = {
    "2",
    2,
    { version = 0 },
    { version = 4 },
    { version = 1.5 },
    { version = "2" },
    { version = "4.0" },
    { version = true },
    { compress = true },
  }
  for _, options in ipairs(bad_options) do
    assert_error(function() numlua.save(file, tensor, options) end)
    assert_false(file_exists(file),
      "a rejected version option must not create " .. file)
  end
end)
