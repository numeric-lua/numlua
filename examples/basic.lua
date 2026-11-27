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
local unpack = table.unpack or unpack

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
print("population std: " .. tostring(numlua.std(a):item()) ..
  " (" .. numlua.dtype(numlua.std(a)) .. ")")
print("NaN-aware mean: " ..
  tostring(numlua.nanmean(numlua.array({ 1.0, 0 / 0, 3.0 })):item()))
print("flat argmax is 1-based: " .. tostring(numlua.argmax(a):item()))
print("per-row argmax:\n" .. numlua.argmax(a, 2):format())
print("stacked rows shape: " ..
  table.concat(numlua.shape(numlua.concatenate({ a, a }, 1)), " x "))

local split_rows = numlua.split(a, 2, 1)
local top, bottom = split_rows[1], split_rows[2]
print("split rows: " .. table.concat(numlua.shape(top), " x ") .. " and " ..
  table.concat(numlua.shape(bottom), " x "))
assert(top:equal(a:narrow(1, 1, 1)) and bottom:equal(a:narrow(1, 2, 1)))
local stacked = numlua.vstack({ numlua.array({ 1, 2 }),
  numlua.array({ 3, 4 }) })
print("vstack:\n" .. stacked:format())
print("block:\n" .. numlua.block({
  { numlua.array({ { 1 } }), numlua.array({ { 2 } }) },
  { numlua.array({ { 3 } }), numlua.array({ { 4 } }) },
}):format())
print("append: " .. table.concat(numlua.append(numlua.array({ 1, 2 }),
  { 3, 4 }):totable(), ", "))
print("rollaxis shape: " ..
  table.concat(numlua.shape(numlua.rollaxis(
    numlua.arange(24):reshape(2, 3, 4), 2, 1)), " x "))

print("diag:\n" .. numlua.diag(numlua.array({ 1, 2, 3 })):format())
print("tril:\n" .. numlua.tril(numlua.array({
  { 1, 2 },
  { 3, 4 },
})):format())
assert(numlua.diag(numlua.array({ 1, 1 })):equal(numlua.eye(2)))

print("rot90:\n" .. numlua.rot90(a):format())
print("diagonal: " .. table.concat(numlua.diagonal(a):totable(), ", "))
print("trace = " .. tostring(numlua.trace(a):item()))
assert(numlua.trace(a):item() == 6)
local like = numlua.zeros_like(a, { dtype = "float64" })
assert(numlua.dtype(like) == "float64" and like:sum():item() == 0)

local grid_x, grid_y = unpack(numlua.meshgrid(
  numlua.array({ 1, 2 }), numlua.array({ 10, 20 }), { indexing = "ij" }))
print("meshgrid ij shapes: " .. table.concat(numlua.shape(grid_x), " x ") ..
  " and " .. table.concat(numlua.shape(grid_y), " x "))
assert(grid_x:get(2, 1) == 2 and grid_y:get(2, 1) == 10)

local gradient = numlua.gradient(numlua.array({ 1, 4, 9, 16 },
  { dtype = "float64" }))
print("gradient: " .. table.concat(gradient:totable(), ", "))
assert(gradient:get(1) == 3 and gradient:get(4) == 7)
local edge2 = numlua.gradient(numlua.array({ 0, 1, 4, 9 },
  { dtype = "float64" }), nil, nil, { edge_order = 2 })
print("gradient edge_order=2: " .. table.concat(edge2:totable(), ", "))
assert(edge2:get(1) == 0 and edge2:get(4) == 6)

print("product reduction: " .. tostring(numlua.prod(a):item()))
print("peak-to-peak: " .. tostring(numlua.ptp(a):item()))
assert(numlua.prod(a):item() == 720 and numlua.ptp(a):item() == 5)
print("average: " .. tostring(numlua.average(a):item()))
print("weighted average: " .. tostring(numlua.average(
  numlua.array({ 1, 2, 3 }), nil, { 1, 2, 3 }):item()))
local nan_values = numlua.array({ 1, 0 / 0, -2, 3 })
print("nanargmin/nanargmax: " ..
  tostring(numlua.nanargmin(nan_values):item()) .. "/" ..
  tostring(numlua.nanargmax(nan_values):item()))
print("nanmedian: " .. tostring(numlua.nanmedian(nan_values):item()))
print("nanquantile: " .. tostring(numlua.nanquantile(nan_values, 0.5):item()))
print("iqr: " .. tostring(numlua.iqr(numlua.array({ 1, 2, 3, 4 })):item()))
print("quantile lower: " .. tostring(numlua.quantile(
  numlua.array({ 1, 2, 3, 4 }), 0.5, nil, false, "lower")))
print("broadcast_shapes: " ..
  table.concat(numlua.broadcast_shapes({ 2, 1, 3 }, { 1, 4, 1 }), " x "))
print("diff with edges: " .. table.concat(numlua.diff(
  numlua.array({ 1, 2, 3 }), 1, 1, 0, 10):totable(), ", "))
print("ediff1d: " ..
  table.concat(numlua.ediff1d(numlua.array({ { 1, 2 }, { 4, 7 } }),
    99, 0):totable(), ", "))
assert(numlua.nanargmin(nan_values):item() == 3)
assert(numlua.nanargmax(nan_values):item() == 4)
print("covariance:\n" .. numlua.cov(numlua.array({
  { 1, 2, 3 }, { 2, 4, 6 },
})):format())
print("correlation:\n" .. numlua.corrcoef(numlua.array({
  { 1, 2, 3 }, { 2, 4, 6 },
})):format())
local fit = numlua.polyfit(numlua.array({ 0, 1, 2, 3 }),
  numlua.array({ 3, 5, 7, 9 }), 1)
print("polyfit: " .. table.concat(fit:totable(), ", "))
print("polyval at 2: " .. tostring(numlua.polyval(fit, 2):item()))
print("polyadd: " .. table.concat(numlua.polyadd({ 1, 2 }, { 1 }):totable(),
  ", "))
print("polymul: " .. table.concat(numlua.polymul({ 1, 1 }, { 1, -1 })
  :totable(), ", "))
print("polyder: " .. table.concat(numlua.polyder({ 1, 2, 3 }):totable(), ", "))
print("poly roots: " .. table.concat(numlua.poly({ 1, 2 }):totable(), ", "))
local counts, edges = numlua.histogram(numlua.array({ 0, 1, 2, 3, 4 }),
  { bins = 2 })
print("histogram counts: " .. table.concat(counts:totable(), ", ") ..
  " edges: " .. table.concat(edges:totable(), ", "))
local histogram2d = numlua.histogram2d(numlua.array({ 0.5, 1.5 }),
  numlua.array({ 0.5, 1.5 }), { bins = 2, range = { { 0, 2 }, { 0, 2 } } })
print("histogram2d:\n" .. histogram2d:format())
print("histogramdd:\n" .. numlua.histogramdd(numlua.array({
  { 0.5, 0.5 }, { 1.5, 1.5 },
}), { bins = 2, range = { { 0, 2 }, { 0, 2 } } }):format())
local gx, gy = numlua.mgrid({ 0, 2 }, { 0, 2 })
print("mgrid shape: " .. table.concat(numlua.shape(gx), " x ") .. "/" ..
  table.concat(numlua.shape(gy), " x "))
local ndindex = numlua.ndindex({ 2, 2 })
local first_coordinates = ndindex()
print("ndindex first: " .. table.concat(first_coordinates, ", "))
local counter = 0
print("fromiter: " .. table.concat(numlua.fromiter(function()
  counter = counter + 1
  if counter <= 3 then return counter end
end, "int64"):totable(), ", "))
print("packbits: " .. table.concat(numlua.packbits(
  numlua.array({ 1, 0, 1, 0, 0, 0, 0, 1 })):totable(), ", "))
print("ravel: " .. table.concat(numlua.ravel_multi_index(
  { { 1, 2 }, { 2, 3 } }, { 2, 3 }):totable(), ", "))
local row, column = numlua.unravel_index(
  numlua.array({ 2, 6 }, { dtype = "int64" }), { 2, 3 })
print("unravel: " .. table.concat(row:totable(), ", ") .. " / " ..
  table.concat(column:totable(), ", "))
print("logspace: " .. table.concat(numlua.logspace(0, 2, 3):totable(), ", "))
print("geomspace: " ..
  table.concat(numlua.geomspace(1, 1000, 4):totable(), ", "))
print("tri:\n" .. numlua.tri(3):format())
print("Vandermonde:\n" .. numlua.vander(numlua.array({ 1, 2, 3 })):format())
assert(numlua.identity(2):equal(numlua.eye(2)))
local compressed = numlua.compress({ true, false, true }, a, 2)
print("compress columns:\n" .. compressed:format())
assert(compressed:size(2) == 2 and compressed:get(1, 2) == 3)

print("trapezoid: " .. tostring(numlua.trapezoid(
  numlua.array({ 1, 2, 3 })):item()))
local dot_product = numlua.dot(a,
  numlua.array({ 1, 1, 1 }, { dtype = "int64" }))
print("dot with a vector: " .. table.concat(dot_product:totable(), ", "))
assert(dot_product:get(1) == 6 and dot_product:get(2) == 15)
local inner_product = numlua.inner(a, a)
print("inner product shape: " ..
  table.concat(numlua.shape(inner_product), " x "))
assert(inner_product:get(1, 1) == 14 and inner_product:get(2, 2) == 77)
local outer_product = numlua.outer(numlua.array({ 1, 2 }),
  numlua.array({ 3, 4 }))
print("outer product:\n" .. outer_product:format())
assert(outer_product:get(2, 1) == 6)
local cross_product = numlua.cross(numlua.array({ 1, 2, 3 }),
  numlua.array({ 4, 5, 6 }))
print("cross product: " .. table.concat(cross_product:totable(), ", "))
assert(cross_product:get(1) == -3 and cross_product:get(3) == -3)

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
print("choose: " .. table.concat(numlua.choose(
  numlua.array({ 0, 1, 0 }, { dtype = "int64" }), { 5, 7 }):totable(), ", "))
print("extract: " .. table.concat(numlua.extract(
  numlua.array({ true, false, true }, { dtype = "bool" }), values):totable(),
  ", "))
print("piecewise:\n" .. numlua.piecewise(values, {
  values:lt(0), values:gt(0),
}, { function(value) return numlua.negative(value) end,
     function(value) return value:mul(2) end }, 0):format())
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
print("msort:\n" .. numlua.msort(matrix):format())
print("partition:\n" .. numlua.partition(matrix, 1, 2):format())
assert(numlua.partition(matrix, 1, 2):equal(sorted))
print("indices shape: " ..
  table.concat(numlua.shape(numlua.indices({ 2, 3 })), " x "))
local lower_rows, lower_columns = numlua.tril_indices(3)
print("tril rows: " .. table.concat(lower_rows:totable(), ", ") ..
  " columns: " .. table.concat(lower_columns:totable(), ", "))
local from_rows, from_columns = numlua.tril_indices_from(matrix)
print("tril_indices_from shape: " ..
  tostring(from_rows:numel()) .. "/" .. tostring(from_columns:numel()))
local mask_rows, mask_columns = numlua.mask_indices(3, function(mask, k)
  return numlua.triu(mask, k):ne(0)
end)
print("mask_indices shape: " ..
  tostring(mask_rows:numel()) .. "/" .. tostring(mask_columns:numel()))
local negative_take = numlua.take(matrix,
  numlua.array({ -1 }, { dtype = "int64" }))
print("take last flat element: " .. tostring(negative_take:item()))
assert(negative_take:item() == 50)
local written = numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } })
numlua.put_along_axis(written,
  numlua.array({ { 1 }, { 3 } }, { dtype = "int64" }),
  numlua.array({ { 10 }, { 60 } }), 2)
print("put_along_axis:\n" .. written:format())
print("delete: " .. table.concat(numlua.delete(
  numlua.array({ { 1, 2, 3 }, { 4, 5, 6 } }), 2, 2):reshape(-1)
  :totable(), ", "))
print("insert: " .. table.concat(numlua.insert(
  numlua.array({ 1, 2, 3 }), 2, 9):totable(), ", "))
print("trim_zeros: " .. table.concat(numlua.trim_zeros(
  numlua.array({ 0, 1, 2, 0 })):totable(), ", "))

local unique_values, unique_counts = numlua.unique(
  numlua.array({ 3, 1, 3, 2 }), { return_counts = true })
print("unique values:\n" .. unique_values:format())
print("unique counts:\n" .. unique_counts:format())
assert(unique_values:equal(numlua.array({ 1, 2, 3 })))
print("intersect1d: " .. table.concat(numlua.intersect1d(
  { 3, 1, 2 }, { 2, 3, 4 }):totable(), ", "))
print("unique axis 1:\n" .. numlua.unique(
  numlua.array({ { 1, 2 }, { 1, 2 }, { 3, 4 } }), { axis = 1 }):format())

print("digitize: " .. table.concat(numlua.digitize(
  numlua.array({ 0, 2, 4, 6 }), numlua.array({ 1, 3, 5 })):totable(), ", "))
print("searchsorted with sorter: " .. table.concat(numlua.searchsorted(
  numlua.array({ 3, 1, 2 }), numlua.array({ 1, 2 }), "left",
  numlua.argsort(numlua.array({ 3, 1, 2 }))):totable(), ", "))
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
print("deg2rad: " .. table.concat(numlua.deg2rad(
  numlua.array({ 0, 180 })):totable(), ", "))
print("sinc: " .. table.concat(numlua.sinc(
  numlua.array({ 0, 1 })):totable(), ", "))

local special = numlua.array({ 1, math.huge, -math.huge, 0 / 0 })
print("isinf:\n" .. numlua.isinf(special):format())
print("isfinite:\n" .. numlua.isfinite(special):format())
print("floor/ceil/round:\n" .. numlua.floor(numlua.array({ 1.5, -1.5 }))
  :format() .. numlua.ceil(numlua.array({ 1.5, -1.5 })):format() ..
  numlua.round(numlua.array({ 1.5, 2.5 })):format())
print("logical_and:\n" ..
  numlua.logical_and(numlua.array({ 1, 0, 2 }), true):format())
print("nan_to_num:\n" ..
  numlua.nan_to_num(numlua.array({ 0 / 0, math.huge }), 7, 8, 9):format())
print("nancumsum:\n" ..
  numlua.nancumsum(numlua.array({ 1, 0 / 0, 3 })):format())
print("isposinf/isneginf:\n" ..
  numlua.isposinf(numlua.array({ math.huge, -math.huge })):format() ..
  numlua.isneginf(numlua.array({ math.huge, -math.huge })):format())
print("hypot: " .. table.concat(numlua.hypot(
  numlua.array({ 3, 5 }), numlua.array({ 4, 12 })):totable(), ", "))
print("remainder: " .. table.concat(numlua.remainder(
  numlua.array({ 5, -5 }), 2):totable(), ", "))
print("heaviside: " .. table.concat(numlua.heaviside(
  numlua.array({ -1, 0, 1 }), 0.5):totable(), ", "))
print("unwrap:\n" .. numlua.unwrap(numlua.array({ 0, 3.5, 7 })):format())
print("interp: " .. tostring(numlua.interp(2,
  numlua.array({ 1, 3, 5 }), numlua.array({ 10, 30, 50 })):item()))

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

local npz_path = root .. "/examples/.basic-demo.npz"
local npz_handle = assert(io.open(npz_path, "w+b"))
assert(numlua.savez(npz_handle, { block = product }))
npz_handle:seek("set", 0)
local archive = numlua.loadz(npz_handle)
assert(archive.block:equal(product))
assert(io.type(npz_handle) == "file")
npz_handle:close()
print("restored the same tensor from an .npz file handle")
os.remove(npz_path)

heading("Raw binary round-trip")
local raw_path = root .. "/examples/.basic-demo.bin"
local raw_input = numlua.array({ 1, 2, 3 }, { dtype = "int32" })
assert(numlua.tofile(raw_path, raw_input))
local raw_restored = numlua.fromfile(raw_path, { dtype = "int32" })
assert(raw_restored:equal(raw_input))
print("restored " .. tostring(raw_restored:numel()) ..
  " int32 values from " .. raw_path)
os.remove(raw_path)

local raw_text_path = root .. "/examples/.basic-demo-raw.txt"
assert(numlua.tofile(raw_text_path, raw_input, { sep = "," }))
local raw_text_restored = numlua.fromfile(raw_text_path,
  { dtype = "int32", sep = "," })
assert(raw_text_restored:equal(raw_input))
print("restored int32 values from comma-separated text")
os.remove(raw_text_path)

heading("Text/CSV round-trip")
local text_path = root .. "/examples/.basic-demo.csv"
assert(numlua.savetxt(text_path, product, { delimiter = "," }))
local text_restored = numlua.loadtxt(text_path, { delimiter = "," })
assert(text_restored:equal(product))
print("restored " .. tostring(text_restored:numel()) ..
  " values from " .. text_path)
os.remove(text_path)

local quoted_path = root .. "/examples/.basic-demo-quoted.csv"
local quoted_handle = assert(io.open(quoted_path, "w"))
assert(quoted_handle:write("\"1.5\",\"2.5\"\n\"3.5\",\"4.5\"\n"))
quoted_handle:close()
local quoted = numlua.loadtxt(quoted_path, {
  delimiter = ",",
  quotechar = "\"",
})
print("quoted CSV:\n" .. quoted:format())
assert(quoted:get(1, 1) == 1.5 and quoted:get(2, 2) == 4.5)
os.remove(quoted_path)

local bool_path = root .. "/examples/.basic-demo-bool.txt"
local bool_values = numlua.array({ { true, false }, { false, true } })
assert(numlua.savetxt(bool_path, bool_values))
local bool_loaded = numlua.loadtxt(bool_path, { dtype = "bool" })
print("bool text:\n" .. bool_loaded:format())
assert(bool_loaded:equal(bool_values))
os.remove(bool_path)

print("fromstring: " .. table.concat(numlua.fromstring(
  "1 2 3", "int64", nil, " "):totable(), ", "))
local genfromtxt_path = root .. "/examples/.basic-demo-missing.csv"
local missing_handle = assert(io.open(genfromtxt_path, "w"))
assert(missing_handle:write("1,NA\n2,3\n"))
missing_handle:close()
local filled = numlua.genfromtxt(genfromtxt_path, {
  delimiter = ",",
  missing_values = "NA",
  filling_values = -1,
})
print("genfromtxt filled missing value: " .. tostring(filled:get(1, 2)))
assert(filled:get(1, 2) == -1 and filled:get(2, 2) == 3)
os.remove(genfromtxt_path)

print("\nnumlua example finished successfully.")
