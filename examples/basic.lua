--
-- Runnable numlua example.
--
-- From the repository root:
--   LUA_PATH="../../lua/?.lua;../../lua/?/init.lua;;" luajit examples/basic.lua
--
local script = arg and arg[0] or "examples/basic.lua"
local root = script:match("^(.*)[/\\]examples[/\\][^/\\]+$")
if root == nil or root == "" then
  root = "."
end
root = root:gsub("/+$", "")
if root == "" then
  root = "."
end
package.path = root .. "/?.lua;" .. root .. "/?/init.lua;" .. package.path

local torch = require("torch")
local numlua = require("numlua")

local function heading(text)
  print("\n== " .. text .. " ==")
end

heading("Construction")
local a = numlua.arange(1, 7):reshape(2, 3)
assert(torch.is_tensor(a))
assert(numlua.dtype(a) == "int64")
print(a:format())
print("shape = " .. table.concat(numlua.shape(a), " x ") .. ", dtype = " ..
  numlua.dtype(a))

heading("Shape and reductions")
print("transpose:\n" .. numlua.transpose(a):format())
print("column sums:\n" .. numlua.sum(a, 1):format())
print("mean (integer input promotes to float64): " ..
  tostring(numlua.mean(a):item()) .. " (" .. numlua.dtype(numlua.mean(a)) .. ")")
print("flat argmax is 1-based: " .. tostring(numlua.argmax(a):item()))
print("per-row argmax:\n" .. numlua.argmax(a, 2):format())
print("stacked rows shape: " ..
  table.concat(numlua.shape(numlua.concatenate({ a, a }, 1)), " x "))

heading("Matrix multiply")
local weights = numlua.array({
  { 1.0, -1.0 },
  { 0.5, 2.0 },
  { 3.0, 0.0 },
})
local product = numlua.matmul(numlua.astype(a, "float32"), weights)
assert(numlua.dtype(product) == "float32")
print(product:format())

heading(".npy round-trip")
local path = root .. "/examples/.basic-demo.npy"
assert(numlua.save(path, product))
local restored = numlua.load(path)
assert(restored:equal(product))
assert(numlua.dtype(restored) == numlua.dtype(product))
print("restored " .. tostring(restored:nbytes()) .. " payload bytes from " .. path)
os.remove(path)

print("\nnumlua example finished successfully.")
