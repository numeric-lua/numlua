--
-- load auto-detection of .npz archives.
--

local numlua = require("numlua")

test("load reads npz archives by extension and magic", function()
  local file = TMPDIR .. "/auto-load.npz"
  assert_true(numlua.savez(file, { x = numlua.array({ 1, 2 }) }))
  local loaded = numlua.load(file)
  assert_tensor(loaded.x, { 1, 2 })
  os.remove(file)

  local magic_file = TMPDIR .. "/auto-load.bin"
  assert_true(numlua.savez(magic_file, { y = numlua.array(3) }))
  local magic_loaded = numlua.load(magic_file)
  assert_eq(magic_loaded.y:item(), 3)
  os.remove(magic_file)
end)

test("load still reads npy files", function()
  local file = TMPDIR .. "/auto-load.npy"
  assert_true(numlua.save(file, numlua.array({ 4, 5 })))
  assert_tensor(numlua.load(file), { 4, 5 })
  os.remove(file)
end)
