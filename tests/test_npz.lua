local numlua = require("numlua")

local function path(name)
  return TMPDIR .. "/" .. name .. ".npz"
end

test("savez/loadz round-trip several arrays", function()
  local file = path("roundtrip")
  assert_true(numlua.savez(file, {
    weights = numlua.array({ { 1.5, 2.5 }, { 3.5, 4.5 } }),
    labels = numlua.array({ 1, 0, 1 }, { dtype = "int64" }),
    mask = numlua.array({ true, false }),
  }))
  local loaded = numlua.loadz(file)
  assert_shape(loaded.weights, { 2, 2 })
  assert_tensor(loaded.weights, { { 1.5, 2.5 }, { 3.5, 4.5 } })
  assert_dtype(loaded.labels, "int64")
  assert_tensor(loaded.labels, { 1, 0, 1 })
  assert_tensor(loaded.mask, { true, false })
  os.remove(file)
end)

test("loadz rejects non-ZIP and compressed entries", function()
  local file = path("plain")
  local handle = assert(io.open(file, "wb"))
  handle:write("not a zip")
  handle:close()
  assert_error(function() numlua.loadz(file) end, "not a ZIP")
  os.remove(file)

  local valid = path("valid")
  assert_true(numlua.savez(valid, { x = numlua.array({ 1 }) }))
  local handle2 = assert(io.open(valid, "rb"))
  local bytes = handle2:read("*a")
  handle2:close()
  local central = bytes:find("PK\001\002", 1, true)
  assert_true(central ~= nil)
  bytes = bytes:sub(1, central + 9) .. string.char(8, 0) ..
    bytes:sub(central + 12)
  local file2 = path("compressed")
  local out = assert(io.open(file2, "wb"))
  out:write(bytes)
  out:close()
  assert_error(function() numlua.loadz(file2) end, "compressed entry")
  os.remove(valid)
  os.remove(file2)
end)
