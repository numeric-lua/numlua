local numlua = require("numlua")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".npy"
end

local function read_bytes(path)
  local handle = assert(io.open(path, "rb"))
  local data = handle:read("*a")
  handle:close()
  return data
end

local function write_bytes(path, data)
  local handle = assert(io.open(path, "wb"))
  assert(handle:write(data))
  handle:close()
end

local function u32(value)
  return string.char(value % 256, math.floor(value / 256) % 256,
    math.floor(value / 65536) % 256, math.floor(value / 16777216) % 256)
end

local function make_header(descr, fortran, shape_text, prefix_size)
  local dict = string.format(
    "{'descr': '%s', 'fortran_order': %s, 'shape': %s, }",
    descr, fortran, shape_text)
  local total_without_padding = (prefix_size or 10) + #dict + 1
  local padding = (64 - (total_without_padding % 64)) % 64
  return dict .. string.rep(" ", padding) .. "\n"
end

-- Craft a .npy file with an arbitrary header.  Used only to test rejection
-- paths; valid files are produced by numlua.save.
local function craft(path, options)
  local descr = options.descr or "<f4"
  local fortran = options.fortran or "False"
  local shape_text = options.shape_text or "(2, 3)"
  local version = options.version or { 1, 0 }
  local prefix_size = version[1] == 1 and 10 or 12
  local header = make_header(descr, fortran, shape_text, prefix_size)
  if options.strip_newline then
    header = header:sub(1, -2)
  end
  local magic = options.magic or "\147NUMPY"
  local payload = options.payload or string.rep("\0", options.payload_size or 0)
  local prefix = magic .. string.char(version[1], version[2])
  if version[1] == 1 then
    prefix = prefix .. string.char(#header % 256, math.floor(#header / 256))
  else
    prefix = prefix .. u32(#header)
  end
  write_bytes(path, prefix .. header .. payload)
  return path
end

-- Rewrite a saved file's descr byte-order marker to '=' (native endian).
-- The replacement keeps the header length identical.
local function use_native_endian_descr(path)
  local data = read_bytes(path)
  local header_size = data:byte(9) + data:byte(10) * 256
  local header = data:sub(11, 10 + header_size)
  local patched, replacements = header:gsub("('descr': ')[<|]", "%1=", 1)
  assert_eq(replacements, 1)
  write_bytes(path, data:sub(1, 10) .. patched ..
    data:sub(11 + header_size))
end

-- Rewrite a saved v1.0 file with a v2.0/v3.0 version and 4-byte header-length
-- field.  The header is re-padded for the wider prefix, and the payload is
-- preserved byte for byte.
local function rewrite_with_version(source, target, major, minor)
  local data = read_bytes(source)
  local header_size = data:byte(9) + data:byte(10) * 256
  local header = data:sub(11, 10 + header_size)
  local payload = data:sub(11 + header_size)
  local dict = assert(header:match("^(.-)%s*\n$"))
  local total_without_padding = 12 + #dict + 1
  local padding = (64 - (total_without_padding % 64)) % 64
  local new_header = dict .. string.rep(" ", padding) .. "\n"
  write_bytes(target, "\147NUMPY" .. string.char(major, minor) ..
    u32(#new_header) .. new_header .. payload)
end

test("save/load round-trip for every supported dtype and shape", function()
  local cases = {
    {
      dtype = "float32",
      values = { 1.5, -2.25, 0, 3.5, -4.75, 5 },
    },
    {
      dtype = "float64",
      values = { math.pi, -0.125, 1e300, -1e-300, 2.5, 42 },
    },
    {
      dtype = "int64",
      values = { -3, 0, 7, 42, -100, 9 },
    },
    {
      dtype = "bool",
      values = { true, false, true, true, false, true },
    },
    {
      dtype = "float16",
      values = { 1.5, -2.25, 0, 3.25, 0.5, -0.125 },
    },
    {
      dtype = "int8",
      values = { -128, -3, 0, 7, 42, 127 },
    },
    {
      dtype = "int16",
      values = { -32768, -3, 0, 7, 42, 32767 },
    },
    {
      dtype = "int32",
      values = { -2147483648, -3, 0, 7, 42, 2147483647 },
    },
    {
      dtype = "uint8",
      values = { 0, 1, 7, 42, 127, 255 },
    },
  }
  local shapes = { { 6 }, { 2, 3 }, { 1, 2, 3 } }

  for _, case in ipairs(cases) do
    for _, shape in ipairs(shapes) do
      -- luatorch cannot build a float16 tensor directly from Lua numbers.
      local source
      if case.dtype == "float16" then
        source = numlua.reshape(
          numlua.array(case.values, { dtype = "float32" }):to("float16"),
          shape)
      else
        source = numlua.reshape(
          numlua.array(case.values, { dtype = case.dtype }), shape)
      end
      local path = tmp_path("roundtrip-" .. case.dtype .. "-" .. #shape)
      assert_true(numlua.save(path, source))
      local loaded = numlua.load(path)
      assert_dtype(loaded, case.dtype)
      assert_shape(loaded, shape)
      assert_tensor(loaded, source:totable(), 0)
      assert_true(loaded:equal(source))
      os.remove(path)
    end
  end
end)

test("saved headers are valid NPY v1.0 headers", function()
  local cases = {
    { shape = { 6 }, shape_text = "(6,)" },
    { shape = { 2, 3 }, shape_text = "(2, 3)" },
    { shape = { 1, 2, 3 }, shape_text = "(1, 2, 3)" },
  }
  for _, case in ipairs(cases) do
    local tensor = numlua.reshape(numlua.arange(6), case.shape)
    local path = tmp_path("header")
    numlua.save(path, tensor)
    local data = read_bytes(path)

    assert_eq(data:sub(1, 6), "\147NUMPY")
    assert_eq(data:byte(7), 1)
    assert_eq(data:byte(8), 0)
    local header_size = data:byte(9) + data:byte(10) * 256
    assert_eq((10 + header_size) % 64, 0, "header must be 64-byte aligned")
    local header = data:sub(11, 10 + header_size)
    assert_eq(header:sub(-1), "\n")
    assert_true(header:find("'descr': '<i8'", 1, true) ~= nil)
    assert_true(header:find("'fortran_order': False", 1, true) ~= nil)
    assert_true(header:find("'shape': " .. case.shape_text, 1, true) ~= nil)
    assert_eq(#data, 10 + header_size + tensor:nbytes())
    os.remove(path)
  end
end)

test("bool files use the |b1 descr", function()
  local path = tmp_path("bool-header")
  numlua.save(path, numlua.array({ true, false }))
  local data = read_bytes(path)
  local header_size = data:byte(9) + data:byte(10) * 256
  local header = data:sub(11, 10 + header_size)
  assert_true(header:find("'descr': '|b1'", 1, true) ~= nil)
  os.remove(path)
end)

test("save writes non-contiguous tensors in C order", function()
  local source = numlua.transpose(numlua.array({
    { 1, 2, 3 },
    { 4, 5, 6 },
  }))
  assert_false(source:is_contiguous())
  local path = tmp_path("non-contiguous")
  numlua.save(path, source)
  local loaded = numlua.load(path)
  assert_shape(loaded, { 3, 2 })
  assert_tensor(loaded, { { 1, 4 }, { 2, 5 }, { 3, 6 } })
  os.remove(path)

  local cube = numlua.reshape(
    numlua.arange(24, { dtype = "int32" }), { 2, 3, 4 })
  local permuted = numlua.transpose(cube, { 3, 1, 2 })
  assert_false(permuted:is_contiguous())
  path = tmp_path("non-contiguous-3d")
  numlua.save(path, permuted)
  loaded = numlua.load(path)
  assert_dtype(loaded, "int32")
  assert_shape(loaded, { 4, 2, 3 })
  assert_true(loaded:equal(permuted:contiguous()))
  os.remove(path)
end)

test("save/load handles empty and scalar arrays", function()
  local empty = numlua.zeros({ 2, 0 })
  local empty_path = tmp_path("empty")
  numlua.save(empty_path, empty)
  local loaded_empty = numlua.load(empty_path)
  assert_shape(loaded_empty, { 2, 0 })
  assert_eq(loaded_empty:numel(), 0)
  os.remove(empty_path)

  local scalar = numlua.array(7, { dtype = "int64" })
  local scalar_path = tmp_path("scalar")
  numlua.save(scalar_path, scalar)
  local loaded_scalar = numlua.load(scalar_path)
  assert_eq(loaded_scalar:dim(), 0)
  assert_dtype(loaded_scalar, "int64")
  assert_eq(loaded_scalar:item(), 7)
  os.remove(scalar_path)

  local half_scalar = numlua.array(-0.125):to("float16")
  local half_path = tmp_path("scalar-float16")
  numlua.save(half_path, half_scalar)
  local loaded_half = numlua.load(half_path)
  assert_eq(loaded_half:dim(), 0)
  assert_dtype(loaded_half, "float16")
  assert_eq(loaded_half:item(), -0.125)
  os.remove(half_path)
end)

test("load rejects big-endian files", function()
  local path = craft(tmp_path("big-endian"), {
    descr = ">f4",
    payload_size = 24,
  })
  assert_error(function() numlua.load(path) end, "big-endian")
  os.remove(path)
end)

test("load rejects Fortran-order files", function()
  local path = craft(tmp_path("fortran"), {
    descr = "<f4",
    fortran = "True",
    payload_size = 24,
  })
  assert_error(function() numlua.load(path) end, "fortran_order=True")
  os.remove(path)
end)

test("load accepts native-endian '=' descr values", function()
  local cases = {
    { dtype = "float16", values = { 1.5, -2.25, 0, 3.25 } },
    { dtype = "float32", values = { 1.5, -2.25, 0, 3.25 } },
    { dtype = "float64", values = { math.pi, -0.125, 2.5, -42 } },
    { dtype = "int8", values = { -128, -1, 0, 127 } },
    { dtype = "int16", values = { -32768, -1, 0, 32767 } },
    { dtype = "int32", values = { -2147483648, -1, 0, 2147483647 } },
    { dtype = "int64", values = { -3, -1, 0, 42 } },
    { dtype = "uint8", values = { 0, 1, 127, 255 } },
    { dtype = "bool", values = { true, false, true, false } },
  }
  for _, case in ipairs(cases) do
    -- luatorch cannot build a float16 tensor directly from Lua numbers.
    local source
    if case.dtype == "float16" then
      source = numlua.array(case.values,
        { dtype = "float32" }):to("float16")
    else
      source = numlua.array(case.values, { dtype = case.dtype })
    end
    local path = tmp_path("native-endian-" .. case.dtype)
    assert_true(numlua.save(path, source))
    use_native_endian_descr(path)

    local data = read_bytes(path)
    local header_size = data:byte(9) + data:byte(10) * 256
    assert_true(data:sub(11, 10 + header_size):find(
      "'descr': '=", 1, true) ~= nil, "descr was not rewritten")

    local loaded = numlua.load(path)
    assert_dtype(loaded, case.dtype)
    assert_shape(loaded, { 4 })
    assert_tensor(loaded, source:totable(), 0)
    assert_true(loaded:equal(source))
    os.remove(path)
  end
end)

test("load reads NPY v2.0 and v3.0 headers", function()
  -- A v2.0 file with the same 2x3 float32 payload as a v1.0 save loads
  -- identically; the only format difference is the 4-byte header length.
  local crafted = craft(tmp_path("v2-crafted"), {
    version = { 2, 0 },
    payload = string.rep("\0\0\128\63", 6),
  })
  assert_tensor(numlua.load(crafted), { { 1, 1, 1 }, { 1, 1, 1 } })
  os.remove(crafted)

  local source = tmp_path("version-source")
  local tensor = numlua.array({ { 1.5, -2.25 }, { 0, 3.5 } })
  numlua.save(source, tensor)
  for _, version in ipairs({ { 2, 0 }, { 3, 0 } }) do
    local target = tmp_path("version-" .. version[1])
    rewrite_with_version(source, target, version[1], version[2])
    local data = read_bytes(target)
    assert_eq(data:byte(7), version[1])
    assert_eq(data:byte(8), version[2])
    local loaded = numlua.load(target)
    assert_dtype(loaded, "float32")
    assert_shape(loaded, { 2, 2 })
    assert_true(loaded:equal(tensor))
    os.remove(target)
  end
  os.remove(source)

  -- 0-D arrays take the wider-header path too.
  local scalar = numlua.array(7, { dtype = "int64" })
  local scalar_path = tmp_path("v3-scalar")
  numlua.save(scalar_path, scalar)
  local target = scalar_path .. ".v3"
  rewrite_with_version(scalar_path, target, 3, 0)
  local scalar_loaded = numlua.load(target)
  assert_eq(scalar_loaded:dim(), 0)
  assert_dtype(scalar_loaded, "int64")
  assert_eq(scalar_loaded:item(), 7)
  os.remove(scalar_path)
  os.remove(target)
end)

test("load rejects unsupported descr values", function()
  local path = craft(tmp_path("unsupported"), {
    descr = "<u4",
    payload_size = 24,
  })
  assert_error(function() numlua.load(path) end, "unsupported .npy descr")
  os.remove(path)

  -- '=' selects the host byte order, but it must not make an unsupported
  -- dtype acceptable; the original spelling appears in the error.
  local native_path = craft(tmp_path("unsupported-native"), {
    descr = "=u4",
    payload_size = 24,
  })
  assert_error(function() numlua.load(native_path) end,
    "unsupported .npy descr '=u4'")
  os.remove(native_path)
end)

test("load rejects malformed and non-NPY files", function()
  local bad_magic = tmp_path("bad-magic")
  write_bytes(bad_magic, "NOPE!!" .. string.rep("\0", 20))
  assert_error(function() numlua.load(bad_magic) end, "not an NPY file")
  os.remove(bad_magic)

  local version_path = craft(tmp_path("version"), {
    version = { 4, 0 },
    payload_size = 24,
  })
  assert_error(function() numlua.load(version_path) end,
    "unsupported NPY version 4.0")
  os.remove(version_path)

  -- A supported major version with a non-zero minor is still unknown.
  local minor_path = craft(tmp_path("version-minor"), {
    version = { 2, 1 },
    payload_size = 24,
  })
  assert_error(function() numlua.load(minor_path) end,
    "unsupported NPY version 2.1")
  os.remove(minor_path)

  -- v2.0/v3.0 truncation must fail before reading a partial header.
  local truncated_v2 = tmp_path("v2-truncated")
  write_bytes(truncated_v2, "\147NUMPY" .. string.char(2, 0) ..
    u32(1000) .. "{'descr': '<f4'")
  assert_error(function() numlua.load(truncated_v2) end,
    "truncated .npy header")
  os.remove(truncated_v2)

  local zero_v2 = tmp_path("v2-zero")
  write_bytes(zero_v2, "\147NUMPY" .. string.char(2, 0) .. u32(0))
  assert_error(function() numlua.load(zero_v2) end, "zero length")
  os.remove(zero_v2)

  local short_v2 = tmp_path("v2-short")
  write_bytes(short_v2, "\147NUMPY" .. string.char(2, 0) .. "\1\0")
  assert_error(function() numlua.load(short_v2) end, "missing header")
  os.remove(short_v2)

  local no_newline = craft(tmp_path("no-newline"), {
    strip_newline = true,
    payload_size = 24,
  })
  assert_error(function() numlua.load(no_newline) end,
    "missing trailing newline")
  os.remove(no_newline)

  local bad_shape = craft(tmp_path("bad-shape"), {
    shape_text = "(2, nope)",
    payload_size = 24,
  })
  assert_error(function() numlua.load(bad_shape) end, "malformed .npy shape")
  os.remove(bad_shape)

  local truncated = craft(tmp_path("truncated"), {
    descr = "<f4",
    payload = string.rep("\0", 4),
  })
  assert_error(function() numlua.load(truncated) end, "size mismatch")
  os.remove(truncated)
end)

test("save and load report file and dtype errors", function()
  local path = tmp_path("errors")
  local int32 = numlua.array({ 1, 2 }, { dtype = "int32" })
  assert_error(function() numlua.save(path, 1) end,
    "numlua.save expects a tensor")
  assert_error(function() numlua.save("", int32) end,
    "expects a non-empty file path")
  assert_error(function() numlua.save(nil, int32) end,
    "expects a non-empty file path")

  assert_error(function()
    numlua.load(TMPDIR .. "/definitely-missing.npy")
  end, "cannot open")
  assert_error(function() numlua.load(nil) end,
    "expects a non-empty file path")
end)
