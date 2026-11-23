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

local split_rows = numlua.split(a, 2, 1)
local top, bottom = split_rows[1], split_rows[2]
print("split rows: " .. table.concat(numlua.shape(top), " x ") .. " and " ..
  table.concat(numlua.shape(bottom), " x "))
assert(top:equal(a:narrow(1, 1, 1)) and bottom:equal(a:narrow(1, 2, 1)))

print("diag:\n" .. numlua.diag(numlua.array({ 1, 2, 3 })):format())
print("tril:\n" .. numlua.tril(numlua.array({
  { 1, 2 },
  { 3, 4 },
})):format())
assert(numlua.diag(numlua.array({ 1, 1 })):equal(numlua.eye(2)))

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
print("isclose against itself: " ..
  tostring(numlua.allclose(values, values)))
print("array_equiv scalar: " .. tostring(numlua.array_equiv(1, numlua.ones(3))))
assert(numlua.array_equal(numlua.array({ 1, 2 }), numlua.array({ 1, 2 })))

local edited = numlua.arange(6)
numlua.put(edited, { 1, 6 }, { 10, 60 })
numlua.put(edited, { 9 }, -1, "wrap")
print("after put:\n" .. edited:format())
assert(edited:get(1) == 10 and edited:get(3) == -1 and edited:get(6) == 60)

local matrix = numlua.array({ { 10, 30, 20 }, { 60, 40, 50 } })
local sorted = numlua.take_along_axis(matrix, numlua.argsort(matrix, 2), 2)
print("argsort + take_along_axis:\n" .. sorted:format())
assert(sorted:equal(numlua.array({ { 10, 20, 30 }, { 40, 50, 60 } })))

print("sort values:\n" .. numlua.sort(matrix):format())
assert(numlua.sort(matrix):equal(sorted))

local unique_values, unique_counts = numlua.unique(
  numlua.array({ 3, 1, 3, 2 }), { return_counts = true })
print("unique values:\n" .. unique_values:format())
print("unique counts:\n" .. unique_counts:format())
assert(unique_values:equal(numlua.array({ 1, 2, 3 })))

local coordinates = numlua.argwhere(matrix:gt(25))
print("argwhere above 25:\n" .. coordinates:format())
assert(numlua.dtype(coordinates) == "int64")

heading("Elementwise ufuncs")
local roots = numlua.sqrt(numlua.array({ 1.0, 4.0, 9.0 }))
print("square roots:\n" .. roots:format())
print("scaled and shifted:\n" ..
  numlua.add(numlua.multiply(roots, 2), 1):format())
print("maximum with scalar 2.5:\n" .. numlua.maximum(roots, 2.5):format())
assert(numlua.dtype(numlua.sqrt(numlua.arange(1, 4))) == "float32")

heading(".npy round-trip")
local path = root .. "/examples/.basic-demo.npy"
assert(numlua.save(path, product))
local restored = numlua.load(path)
assert(restored:equal(product))
assert(numlua.dtype(restored) == numlua.dtype(product))
print("restored " .. tostring(restored:nbytes()) .. " payload bytes from " .. path)
os.remove(path)

local v3_path = root .. "/examples/.basic-demo-v3.npy"
assert(numlua.save(v3_path, product, { version = 3 }))
assert(numlua.load(v3_path):equal(product))
print("restored the same tensor from a v3.0 header")
os.remove(v3_path)

local handle = assert(io.open(path, "w+b"))
assert(numlua.save(handle, product))
handle:seek("set", 0)
assert(numlua.load(handle):equal(product))
assert(io.type(handle) == "file")
handle:close()
print("restored the same tensor through an open file handle")
os.remove(path)

heading("Raw binary round-trip")
local raw_path = root .. "/examples/.basic-demo.bin"
local raw_input = numlua.array({ 1, 2, 3 }, { dtype = "int32" })
assert(numlua.tofile(raw_path, raw_input))
local raw_restored = numlua.fromfile(raw_path, { dtype = "int32" })
assert(raw_restored:equal(raw_input))
print("restored " .. tostring(raw_restored:numel()) ..
  " int32 values from " .. raw_path)
os.remove(raw_path)

heading("Text/CSV round-trip")
local text_path = root .. "/examples/.basic-demo.csv"
assert(numlua.savetxt(text_path, product, { delimiter = "," }))
local text_restored = numlua.loadtxt(text_path, { delimiter = "," })
assert(text_restored:equal(product))
print("restored " .. tostring(text_restored:numel()) ..
  " values from " .. text_path)
os.remove(text_path)

print("\nnumlua example finished successfully.")
