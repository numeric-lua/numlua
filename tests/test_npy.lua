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

local function make_header(descr, fortran, shape_text)
  local dict = string.format(
    "{'descr': '%s', 'fortran_order': %s, 'shape': %s, }",
    descr, fortran, shape_text)
  local total_without_padding = 10 + #dict + 1
  local padding = (64 - (total_without_padding % 64)) % 64
  return dict .. string.rep(" ", padding) .. "\n"
end

-- Craft a .npy file with an arbitrary header.  Used only to test rejection
-- paths; valid files are produced by numlua.save.
local function craft(path, options)
  local descr = options.descr or "<f4"
  local fortran = options.fortran or "False"
  local shape_text = options.shape_text or "(2, 3)"
  local header = make_header(descr, fortran, shape_text)
  if options.strip_newline then
    header = header:sub(1, -2)
  end
  local version = options.version or { 1, 0 }
  local magic = options.magic or "\147NUMPY"
  local payload = options.payload or string.rep("\0", options.payload_size or 0)
  local prefix = magic .. string.char(version[1], version[2]) ..
    string.char(#header % 256, math.floor(#header / 256))
  write_bytes(path, prefix .. header .. payload)
  return path
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

test("load rejects unsupported descr values", function()
  local path = craft(tmp_path("unsupported"), {
    descr = "<u4",
    payload_size = 24,
  })
  assert_error(function() numlua.load(path) end, "unsupported .npy descr")
  os.remove(path)
end)

test("load rejects malformed and non-NPY files", function()
  local bad_magic = tmp_path("bad-magic")
  write_bytes(bad_magic, "NOPE!!" .. string.rep("\0", 20))
  assert_error(function() numlua.load(bad_magic) end, "not an NPY file")
  os.remove(bad_magic)

  local version_path = craft(tmp_path("version"), {
    version = { 2, 0 },
    payload_size = 24,
  })
  assert_error(function() numlua.load(version_path) end,
    "unsupported NPY version 2.0")
  os.remove(version_path)

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
