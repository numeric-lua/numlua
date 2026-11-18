--
-- Runnable numlua example.
--
-- From the repository root:
--   LUA_PATH="../luatorch/lua/?.lua;../luatorch/lua/?/init.lua;;" luajit examples/basic.lua
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
package.path = table.concat({
  root .. "/?.lua",
  root .. "/?/init.lua",
  root .. "/../luatorch/lua/?.lua",
  root .. "/../luatorch/lua/?/init.lua",
  root .. "/../../luatorch/lua/?.lua",
  root .. "/../../luatorch/lua/?/init.lua",
  root .. "/../luatorch/lua/?.lua",
  root .. "/../luatorch/lua/?/init.lua",
  package.path,
}, ";")

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

heading("Linear algebra facade")
local la = numlua.linalg
local square = numlua.array({ { 4.0, 7.0 }, { 2.0, 6.0 } },
  { dtype = "float64" })
print("det = " .. tostring(la.det(square):item()))
print("norm = " .. tostring(la.norm(square):item()))
print("inverse:\n" .. la.inv(square):format())
local u, s, vh = la.svd(square, false)
print("svd shapes = " .. table.concat(numlua.shape(u), "x") .. ", " ..
  table.concat(numlua.shape(s), "x") .. ", " ..
  table.concat(numlua.shape(vh), "x"))

heading("Predicates and selection")
local values = numlua.array({ -2.0, -0.5, 0.0, 0.5, 2.0 })
print("clipped:\n" .. numlua.clip(values, -1, 1):format())
print("positive values selected:\n" ..
  numlua.where(values:gt(0), values, 0):format())
print("nonzeros = " .. tostring(numlua.count_nonzero(values):item()))
print("any positive = " .. tostring(numlua.any(values:gt(0)):item()))
print("all above -3 = " .. tostring(numlua.all(values:gt(-3)):item()))
print("isnan mask:\n" .. numlua.isnan(values):format())

heading(".npy round-trip")
local path = root .. "/examples/.basic-demo.npy"
assert(numlua.save(path, product))
local restored = numlua.load(path)
assert(restored:equal(product))
assert(numlua.dtype(restored) == numlua.dtype(product))
print("restored " .. tostring(restored:nbytes()) .. " payload bytes from " .. path)
os.remove(path)

heading("Text/CSV round-trip")
local text_path = root .. "/examples/.basic-demo.csv"
assert(numlua.savetxt(text_path, product, { delimiter = "," }))
local text_restored = numlua.loadtxt(text_path, { delimiter = "," })
assert(text_restored:equal(product))
print("restored " .. tostring(text_restored:numel()) ..
  " values from " .. text_path)
os.remove(text_path)

print("\nnumlua example finished successfully.")
