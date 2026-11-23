--
-- .npz file-object I/O parity: savez/savez_compressed/loadz accept handles.
--

local numlua = require("numlua")
local npz = require("numlua.npz")

local counter = 0
local function tmp_path(name)
  counter = counter + 1
  return TMPDIR .. "/" .. name .. "-" .. counter .. ".npz"
end

test("savez and loadz accept an open file handle", function()
  local path = tmp_path("npz-handle")
  local handle = assert(io.open(path, "w+b"))
  local arrays = {
    x = numlua.array({ { 1, 2 }, { 3, 4 } }, { dtype = "int64" }),
    y = numlua.array({ 1.5, -2.5 }, { dtype = "float64" }),
  }

  assert_true(numlua.savez(handle, arrays))
  assert_eq(io.type(handle), "file", "savez must not close a caller handle")
  assert_eq(handle:seek("set", 0), 0)

  local loaded = numlua.loadz(handle)
  assert_eq(io.type(handle), "file", "loadz must not close a caller handle")
  assert_eq(#loaded.x:sizes(), 2)
  assert_true(loaded.x:equal(arrays.x))
  assert_true(loaded.y:equal(arrays.y))

  assert_true(handle:close())
  os.remove(path)
end)

test("npz handles are used at their current position", function()
  local path = tmp_path("npz-offset")
  local handle = assert(io.open(path, "w+b"))
  assert(handle:write("prefix"))
  local arrays = { value = numlua.array({ 7, 8, 9 }) }
  assert_true(numlua.savez(handle, arrays))
  assert_eq(io.type(handle), "file")

  assert_eq(handle:seek("set", 6), 6)
  local loaded = numlua.loadz(handle)
  assert_true(loaded.value:equal(arrays.value))

  -- The prefix was not overwritten and the archive starts at byte 7.
  assert_eq(handle:seek("set", 0), 0)
  assert_eq(handle:read(6), "prefix")
  assert_true(handle:close())
  os.remove(path)
end)

test("savez_compressed accepts handles with or without zlib", function()
  local path = tmp_path("npz-compressed-handle")
  local handle = assert(io.open(path, "w+b"))
  local arrays = { z = numlua.array({ 4, 5, 6 }, { dtype = "float32" }) }

  if npz.zlib_available() then
    assert_true(numlua.savez_compressed(handle, arrays))
  else
    assert_error(function()
      numlua.savez_compressed(handle, arrays)
    end, "requires a system zlib")
    assert_true(numlua.savez_compressed(handle, arrays, { fallback = "stored" }))
    assert_eq(io.type(handle), "file")
  end

  assert_eq(handle:seek("set", 0), 0)
  local loaded = numlua.loadz(handle)
  assert_true(loaded.z:equal(arrays.z))
  assert_true(handle:close())
  os.remove(path)
end)

test("npz handle I/O validates targets and does not close on failure", function()
  local path = tmp_path("npz-handle-errors")
  local arrays = { x = numlua.array({ 1 }) }

  local closed = assert(io.open(path, "wb"))
  assert_true(closed:close())
  assert_error(function() numlua.savez(closed, arrays) end,
    "cannot use a closed file handle")
  assert_error(function() numlua.loadz(closed) end,
    "cannot use a closed file handle")

  assert_error(function() numlua.savez(nil, arrays) end,
    "non-empty path or open file handle")
  assert_error(function() numlua.loadz(nil) end,
    "non-empty path or open file handle")
  assert_error(function() numlua.savez("", arrays) end,
    "non-empty path or open file handle")

  local write_path = tmp_path("npz-readonly")
  assert_true(numlua.savez(write_path, arrays))
  local readonly = assert(io.open(write_path, "rb"))
  assert_error(function() numlua.savez(readonly, arrays) end,
    "write failed for file handle")
  assert_eq(io.type(readonly), "file")
  assert_true(readonly:close())
  os.remove(write_path)

  local malformed = assert(io.open(path, "w+b"))
  assert(malformed:write("not a zip archive"))
  assert_eq(malformed:seek("set", 0), 0)
  assert_error(function() numlua.loadz(malformed) end, "not a ZIP archive")
  assert_eq(io.type(malformed), "file")
  assert_true(malformed:close())
  os.remove(path)
end)
