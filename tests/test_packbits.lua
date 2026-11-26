--
-- packbits/unpackbits bit packing.
--

local numlua = require("numlua")

test("packbits packs big- and little-endian bit orders", function()
  local bits = numlua.array({ 1, 0, 1, 0, 0, 0, 0, 1 }, { dtype = "uint8" })
  assert_tensor(numlua.packbits(bits), { 161 })
  assert_tensor(numlua.packbits(bits, "little"), { 133 })
  assert_tensor(numlua.packbits(numlua.array({ 1, 0, 0, 0, 0, 0, 0, 0, 1 })),
    { 128, 128 })
  assert_shape(numlua.packbits(numlua.zeros(0)), { 0 })
end)

test("unpackbits expands bytes, truncates, and pads", function()
  assert_tensor(numlua.unpackbits(numlua.array({ 161 }, { dtype = "uint8" })),
    { 1, 0, 1, 0, 0, 0, 0, 1 })
  assert_tensor(numlua.unpackbits(
    numlua.array({ 161 }, { dtype = "uint8" }), 4), { 1, 0, 1, 0 })
  assert_tensor(numlua.unpackbits(
    numlua.array({ 161 }, { dtype = "uint8" }), 10),
    { 1, 0, 1, 0, 0, 0, 0, 1, 0, 0 })
  assert_tensor(numlua.unpackbits(
    numlua.array({ 133 }, { dtype = "uint8" }), nil, "little"),
    { 1, 0, 1, 0, 0, 0, 0, 1 })
end)

test("packbits round-trips and validates inputs", function()
  local bits = numlua.array({ 1, 0, 1, 1, 0, 0, 0, 1, 1 }, { dtype = "uint8" })
  local packed = numlua.packbits(bits)
  assert_tensor(numlua.unpackbits(packed, bits:numel()), bits:totable())
  assert_error(function() numlua.packbits(bits, "middle") end, "bitorder")
  assert_error(function() numlua.unpackbits(
    numlua.array({ 256 }, { dtype = "int64" })) end, "0..255")
  assert_error(function() numlua.unpackbits(
    numlua.array({ 1 }, { dtype = "uint8" }), -1) end, "count")
end)
