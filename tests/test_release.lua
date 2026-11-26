--
-- Release metadata consistency checks.
--
-- These tests keep numlua.VERSION, the packaged rockspec, the changelog, and
-- the LuaRocks module list from drifting apart between release cycles.
--

local numlua = require("numlua")

local root = TMPDIR:gsub("[/\\]tests[/\\]%.tmp$", "")
if root == TMPDIR or root == "" then root = "." end

local function read_file(path)
  local handle, open_error = io.open(path, "rb")
  if handle == nil then
    error("cannot open " .. path .. ": " .. tostring(open_error), 2)
  end
  local data = handle:read("*a")
  handle:close()
  return data
end

test("VERSION matches the rockspec and changelog", function()
  local version = numlua.VERSION
  local rockspec = read_file(root .. "/numlua-" .. version .. "-1.rockspec")
  assert_true(rockspec:find('version = "' .. version .. '-1"', 1, true) ~= nil,
    "rockspec version does not match numlua.VERSION")
  assert_true(rockspec:find('tag = "v' .. version .. '"', 1, true) ~= nil,
    "rockspec tag does not match numlua.VERSION")

  local changelog = read_file(root .. "/CHANGELOG.md")
  assert_true(changelog:find("## " .. version .. " —", 1, true) ~= nil,
    "CHANGELOG has no entry for " .. version)
end)

test("rockspec packages every numlua module", function()
  local rockspec = read_file(root .. "/numlua-" .. numlua.VERSION ..
    "-1.rockspec")
  local modules = {
    "numlua/init.lua",
    "numlua/npy.lua",
    "numlua/npz.lua",
    "numlua/text.lua",
    "numlua/linalg.lua",
    "numlua/manipulation.lua",
    "numlua/ufunc.lua",
  }
  for _, path in ipairs(modules) do
    read_file(root .. "/" .. path) -- the module file must exist
    assert_true(rockspec:find('"' .. path .. '"', 1, true) ~= nil,
      "rockspec does not package " .. path)
  end
end)

test("new public API and format support are documented", function()
  local api = read_file(root .. "/docs/api.md")
  for _, symbol in ipairs({ "numlua.put", "numlua.take_along_axis",
      "numlua.apply_along_axis", "numlua.apply_over_axes",
      "numlua.select", "numlua.sort", "numlua.unique", "numlua.nonzero",
      "numlua.argwhere", "numlua.split", "numlua.array_split",
      "numlua.hsplit", "numlua.vsplit", "numlua.dsplit", "numlua.diag",
      "numlua.diagflat", "numlua.tril", "numlua.triu", "numlua.isclose",
      "numlua.allclose", "numlua.array_equal", "numlua.array_equiv",
      "numlua.fromfile", "numlua.tofile", "numlua.rot90", "numlua.fliplr",
      "numlua.flipud", "numlua.diagonal", "numlua.trace",
      "numlua.zeros_like", "numlua.ones_like", "numlua.empty_like",
      "numlua.full_like", "numlua.var", "numlua.std", "numlua.nansum",
      "numlua.nanmean", "numlua.nanmin", "numlua.nanmax", "numlua.nanvar",
      "numlua.nanstd", "numlua.meshgrid", "numlua.gradient",
      "numlua.genfromtxt", "numlua.prod", "numlua.nanprod", "numlua.ptp",
      "numlua.logspace", "numlua.geomspace", "numlua.identity", "numlua.tri",
      "numlua.vander", "numlua.compress", "numlua.inner", "numlua.outer",
      "numlua.cross", "numlua.trapezoid", "numlua.trapz", "numlua.interp",
      "numlua.unwrap", "numlua.nan_to_num", "numlua.nancumsum",
      "numlua.nancumprod", "numlua.isinf", "numlua.isfinite",
      "numlua.sign", "numlua.floor", "numlua.ceil", "numlua.trunc",
      "numlua.round", "numlua.reciprocal", "numlua.log1p",
      "numlua.logical_and", "numlua.logical_or", "numlua.logical_xor",
      "numlua.logical_not", "numlua.broadcast_shapes", "numlua.ediff1d",
      "numlua.average", "numlua.nanargmin", "numlua.nanargmax",
      "numlua.polyval", "numlua.polyfit", "numlua.cov", "numlua.corrcoef",
      "numlua.histogram", "numlua.histogram_bin_edges",
      "numlua.ravel_multi_index", "numlua.unravel_index",
      "numlua.isposinf", "numlua.isneginf", "numlua.hypot",
      "numlua.remainder", "numlua.mod", "numlua.partition",
      "numlua.argpartition", "numlua.indices", "numlua.diag_indices",
      "numlua.tril_indices", "numlua.triu_indices", "numlua.digitize",
      "numlua.searchsorted", "numlua.put_along_axis", "numlua.delete",
      "numlua.insert", "numlua.choose", "numlua.extract",
      "numlua.trim_zeros", "numlua.nanmedian", "numlua.nanquantile",
      "numlua.nanpercentile", "numlua.histogram2d", "numlua.fromiter",
      "numlua.packbits", "numlua.unpackbits", "numlua.mask_indices",
      "numlua.poly", "numlua.polyadd", "numlua.polysub",
      "numlua.polymul", "numlua.fromstring", "numlua.intersect1d",
      "numlua.union1d", "numlua.setdiff1d", "numlua.setxor1d",
      "numlua.isin", "numlua.in1d",
      "numlua.polydiv", "numlua.polyder", "numlua.polyint",
      "numlua.deg2rad", "numlua.rad2deg", "numlua.radians",
      "numlua.degrees", "numlua.sinc", "numlua.heaviside", "numlua.iqr",
      "numlua.diag_indices_from",
      "numlua.tril_indices_from", "numlua.triu_indices_from",
      "numlua.vstack", "numlua.hstack", "numlua.dstack",
      "numlua.column_stack", "numlua.row_stack", "numlua.append",
      "numlua.rollaxis", "numlua.msort", "numlua.piecewise" }) do
    assert_true(api:find(symbol, 1, true) ~= nil,
      "docs/api.md does not document " .. symbol)
  end

  -- The reader gained v2.0/v3.0 support; the docs must say so.
  assert_true(api:find("v2.0", 1, true) ~= nil and
      api:find("v3.0", 1, true) ~= nil,
    "docs/api.md does not document NPY v2.0/v3.0 reading")
  assert_true(api:find("numlua.save(path_or_file, tensor[, options])", 1, true) ~= nil,
    "docs/api.md does not document save options")
  assert_true(api:find("open Lua file handle", 1, true) ~= nil,
    "docs/api.md does not document file-handle .npy I/O")
  assert_true(api:find("numlua.nonzero", 1, true) ~= nil and
      api:find("numlua.argwhere", 1, true) ~= nil,
    "docs/api.md does not document nonzero/argwhere")
  assert_true(api:find("numlua.unique", 1, true) ~= nil,
    "docs/api.md does not document unique")
  assert_true(api:find("numlua.fromfile", 1, true) ~= nil and
      api:find("numlua.tofile", 1, true) ~= nil,
    "docs/api.md does not document raw I/O")
  assert_true(api:find("numlua.savez(path_or_file", 1, true) ~= nil and
      api:find("numlua.loadz(path_or_file", 1, true) ~= nil,
    "docs/api.md does not document .npz file handles")
  assert_true(api:find("text mode", 1, true) ~= nil,
    "docs/api.md does not document raw text mode")
  assert_true(api:find("printf-style format", 1, true) ~= nil,
    "docs/api.md does not document the tofile text format")
  assert_true(api:find("numlua.array_equiv", 1, true) ~= nil,
    "docs/api.md does not document array_equiv")
  assert_true(api:find("version = 2", 1, true) ~= nil,
    "docs/api.md does not document writing NPY v2.0")
  assert_true(api:find("0-D index removes", 1, true) ~= nil,
    "docs/api.md does not document scalar take indices")
  assert_true(api:find("finite integers", 1, true) ~= nil,
    "docs/api.md does not document finite put indices")
  assert_true(api:find('"wrap"', 1, true) ~= nil,
    "docs/api.md does not document the pad wrap mode")
  assert_true(api:find("constant_values", 1, true) ~= nil,
    "docs/api.md does not document pad constant values")
  assert_true(api:find("linear_ramp", 1, true) ~= nil,
    "docs/api.md does not document the pad linear_ramp mode")
  assert_true(api:find("stat_length", 1, true) ~= nil,
    "docs/api.md does not document pad stat_length")
  assert_true(api:find("normalize to the same member name", 1, true) ~= nil,
    "docs/api.md does not document npz member-name collisions")
  assert_true(api:find("options.version", 1, true) ~= nil,
    "docs/api.md does not document the npz member version option")
  assert_true(api:find('-- "' .. numlua.VERSION .. '"', 1, true) ~= nil,
    "docs/api.md version example does not match numlua.VERSION")

  for _, option in ipairs({ "skiprows", "max_rows", "usecols", "converters",
      "unpack", "ndmin", "header", "footer", "comments", "newline",
      "quotechar" }) do
    assert_true(api:find("`" .. option .. "`", 1, true) ~= nil,
      "docs/api.md does not document the text option " .. option)
  end
  assert_true(api:find("per-column formats", 1, true) ~= nil,
    "docs/api.md does not document per-column fmt tables")
  for _, option in ipairs({ "missing_values", "filling_values",
      "skip_header", "skip_footer" }) do
    assert_true(api:find("`" .. option .. "`", 1, true) ~= nil,
      "docs/api.md does not document the genfromtxt option " .. option)
  end
  assert_true(api:find("NaN-aware", 1, true) ~= nil,
    "docs/api.md does not document NaN-aware reductions")
  assert_true(api:find("first-order edge", 1, true) ~= nil,
    "docs/api.md does not document gradient edge behavior")
  assert_true(api:find('indexing = "xy"', 1, true) ~= nil,
    "docs/api.md does not document meshgrid indexing")
  assert_true(api:find("ZIP64 archives are\nread", 1, true) ~= nil or
      api:find("ZIP64 archives are read", 1, true) ~= nil or
      api:find("ZIP64 end-of-central-directory", 1, true) ~= nil,
    "docs/api.md does not document ZIP64 reading")

  local readme = read_file(root .. "/README.md")
  assert_true(readme:find("Status: " .. numlua.VERSION, 1, true) ~= nil,
    "README.md status does not match numlua.VERSION")
  assert_true(readme:find("numlua-" .. numlua.VERSION .. "-1.rockspec",
    1, true) ~= nil, "README.md does not name the current rockspec")
  assert_true(readme:find("version = 3", 1, true) ~= nil,
    "README.md has no NPY v3.0 writing example")
  assert_true(readme:find("numlua.select", 1, true) ~= nil,
    "README.md does not demonstrate numlua.select")
  assert_true(readme:find("member NPY header version", 1, true) ~= nil,
    "README.md does not document npz member versions")
  assert_true(readme:find("constant_values", 1, true) ~= nil,
    "README.md does not document pad constant values")
  assert_true(readme:find("linear_ramp", 1, true) ~= nil,
    "README.md does not document the pad linear_ramp mode")
  assert_true(readme:find("silently overwriting a member", 1, true) ~= nil,
    "README.md does not document npz member-name collisions")
  assert_true(readme:find("numlua.apply_along_axis", 1, true) ~= nil,
    "README.md does not demonstrate numlua.apply_along_axis")
  assert_true(readme:find("numlua.apply_over_axes", 1, true) ~= nil,
    "README.md does not demonstrate numlua.apply_over_axes")
  assert_true(readme:find("numlua.sort", 1, true) ~= nil and
      readme:find("numlua.unique", 1, true) ~= nil,
    "README.md does not document sort/unique")
  assert_true(readme:find("numlua.argwhere", 1, true) ~= nil,
    "README.md does not document argwhere")
  assert_true(readme:find("converters", 1, true) ~= nil,
    "README.md does not document loadtxt converters")
  assert_true(readme:find("file handle", 1, true) ~= nil,
    "README.md does not document file handles")
  assert_true(readme:find("numlua.fromfile", 1, true) ~= nil and
      readme:find("numlua.tofile", 1, true) ~= nil,
    "README.md does not document raw I/O")
  assert_true(readme:find("savez(path_or_file", 1, true) ~= nil and
      readme:find("loadz(path_or_file", 1, true) ~= nil,
    "README.md does not document .npz file handles")
  assert_true(readme:find("printf-style", 1, true) ~= nil,
    "README.md does not document the tofile text format")
  assert_true(readme:find("numlua.rot90", 1, true) ~= nil and
      readme:find("numlua.diagonal", 1, true) ~= nil,
    "README.md does not document rotation/diagonal helpers")
  assert_true(readme:find("numlua.zeros_like", 1, true) ~= nil and
      readme:find("numlua.full_like", 1, true) ~= nil,
    "README.md does not document *_like constructors")
  assert_true(readme:find("numlua.split", 1, true) ~= nil and
      readme:find("numlua.array_split", 1, true) ~= nil,
    "README.md does not document the split family")
  assert_true(readme:find("numlua.isclose", 1, true) ~= nil and
      readme:find("numlua.array_equiv", 1, true) ~= nil,
    "README.md does not document closeness predicates")
  assert_true(readme:find("numlua.var", 1, true) ~= nil and
      readme:find("numlua.std", 1, true) ~= nil and
      readme:find("nanmean", 1, true) ~= nil,
    "README.md does not document variance/NaN reductions")
  assert_true(readme:find("numlua.meshgrid", 1, true) ~= nil,
    "README.md does not document meshgrid")
  assert_true(readme:find("numlua.gradient", 1, true) ~= nil,
    "README.md does not document gradient")
  assert_true(readme:find("numlua.genfromtxt", 1, true) ~= nil,
    "README.md does not document genfromtxt")
  assert_true(readme:find("numlua.prod", 1, true) ~= nil and
      readme:find("numlua.ptp", 1, true) ~= nil and
      readme:find("numlua.nanprod", 1, true) ~= nil,
    "README.md does not document product/ptp reductions")
  assert_true(readme:find("numlua.logspace", 1, true) ~= nil and
      readme:find("numlua.geomspace", 1, true) ~= nil,
    "README.md does not document logarithmic constructors")
  assert_true(readme:find("numlua.tri", 1, true) ~= nil and
      readme:find("numlua.vander", 1, true) ~= nil and
      readme:find("numlua.identity", 1, true) ~= nil,
    "README.md does not document the new constructors")
  assert_true(readme:find("numlua.compress", 1, true) ~= nil,
    "README.md does not document compress")
  assert_true(readme:find("ZIP64 archives are both read and written", 1, true) ~= nil,
    "README.md does not document ZIP64 reading")
  assert_true(readme:find("numlua.inner", 1, true) ~= nil and
      readme:find("numlua.outer", 1, true) ~= nil and
      readme:find("numlua.cross", 1, true) ~= nil,
    "README.md does not document inner/outer/cross")
  assert_true(readme:find("numlua.trapezoid", 1, true) ~= nil and
      readme:find("numlua.trapz", 1, true) ~= nil,
    "README.md does not document trapezoid/trapz")
  assert_true(readme:find("numlua.interp", 1, true) ~= nil and
      readme:find("numlua.unwrap", 1, true) ~= nil,
    "README.md does not document interp/unwrap")
  assert_true(readme:find("numlua.nan_to_num", 1, true) ~= nil and
      readme:find("numlua.nancumsum", 1, true) ~= nil and
      readme:find("numlua.nancumprod", 1, true) ~= nil,
    "README.md does not document NaN helpers")
  assert_true(readme:find("numlua.isinf", 1, true) ~= nil and
      readme:find("numlua.logical_and", 1, true) ~= nil and
      readme:find("numlua.round", 1, true) ~= nil,
    "README.md does not document the new elementwise ufuncs")
  assert_true(readme:find("numlua.broadcast_shapes", 1, true) ~= nil and
      readme:find("numlua.ediff1d", 1, true) ~= nil,
    "README.md does not document broadcast_shapes/ediff1d")
  assert_true(readme:find("numlua.average", 1, true) ~= nil and
      readme:find("numlua.nanargmin", 1, true) ~= nil and
      readme:find("numlua.nanargmax", 1, true) ~= nil,
    "README.md does not document average/nanarg")
  assert_true(readme:find("numlua.polyval", 1, true) ~= nil and
      readme:find("numlua.polyfit", 1, true) ~= nil and
      readme:find("numlua.cov", 1, true) ~= nil and
      readme:find("numlua.corrcoef", 1, true) ~= nil,
    "README.md does not document polynomial/covariance helpers")
  assert_true(readme:find("numlua.histogram", 1, true) ~= nil and
      readme:find("numlua.histogram_bin_edges", 1, true) ~= nil,
    "README.md does not document histogram helpers")
  assert_true(readme:find("numlua.ravel_multi_index", 1, true) ~= nil and
      readme:find("numlua.unravel_index", 1, true) ~= nil,
    "README.md does not document index conversion helpers")
  assert_true(readme:find("numlua.isposinf", 1, true) ~= nil and
      readme:find("numlua.hypot", 1, true) ~= nil and
      readme:find("numlua.remainder", 1, true) ~= nil,
    "README.md does not document the new math ufuncs")
  assert_true(readme:find("numlua.partition", 1, true) ~= nil and
      readme:find("numlua.argpartition", 1, true) ~= nil,
    "README.md does not document partition/argpartition")
  assert_true(readme:find("numlua.indices", 1, true) ~= nil and
      readme:find("numlua.tril_indices", 1, true) ~= nil and
      readme:find("numlua.triu_indices", 1, true) ~= nil,
    "README.md does not document index constructors")
  assert_true(readme:find("numlua.digitize", 1, true) ~= nil,
    "README.md does not document digitize")
  assert_true(readme:find("edge_order = 2", 1, true) ~= nil,
    "README.md does not document second-order gradient edges")
  assert_true(readme:find("Negative indices count from the end", 1, true) ~= nil,
    "README.md does not document negative take indices")
  assert_true(readme:find("numlua.vstack", 1, true) ~= nil and
      readme:find("numlua.append", 1, true) ~= nil,
    "README.md does not document stacking/append")
  assert_true(readme:find("numlua.rollaxis", 1, true) ~= nil and
      readme:find("numlua.msort", 1, true) ~= nil,
    "README.md does not document rollaxis/msort")
  assert_true(readme:find("numlua.piecewise", 1, true) ~= nil,
    "README.md does not document piecewise")
  assert_true(readme:find("switches to ZIP64", 1, true) ~= nil,
    "README.md does not document ZIP64 writing")
  assert_true(readme:find("numlua.put_along_axis", 1, true) ~= nil and
      readme:find("numlua.delete", 1, true) ~= nil,
    "README.md does not document put_along_axis/delete")
  assert_true(readme:find("numlua.nanmedian", 1, true) ~= nil,
    "README.md does not document nanmedian")
  assert_true(readme:find("numlua.diag_indices_from", 1, true) ~= nil and
      readme:find("numlua.tril_indices_from", 1, true) ~= nil,
    "README.md does not document indices_from helpers")
  assert_true(readme:find("numlua.unique` also accepts `{ axis", 1, true) ~= nil,
    "README.md does not document unique axis support")
  assert_true(readme:find("numlua.insert", 1, true) ~= nil and
      readme:find("numlua.choose", 1, true) ~= nil and
      readme:find("numlua.extract", 1, true) ~= nil and
      readme:find("numlua.trim_zeros", 1, true) ~= nil,
    "README.md does not document insert/choose/extract/trim_zeros")
  assert_true(readme:find("numlua.nanquantile", 1, true) ~= nil and
      readme:find("numlua.nanpercentile", 1, true) ~= nil,
    "README.md does not document nanquantile/nanpercentile")
  assert_true(readme:find("numlua.histogram2d", 1, true) ~= nil and
      readme:find("numlua.packbits", 1, true) ~= nil,
    "README.md does not document histogram2d/packbits")
  assert_true(readme:find("numlua.fromiter", 1, true) ~= nil and
      readme:find("numlua.mask_indices", 1, true) ~= nil,
    "README.md does not document fromiter/mask_indices")
  assert_true(readme:find("numlua.polyadd", 1, true) ~= nil and
      readme:find("numlua.polyder", 1, true) ~= nil,
    "README.md does not document polynomial arithmetic/calculus")
  assert_true(readme:find("numlua.deg2rad", 1, true) ~= nil and
      readme:find("numlua.sinc", 1, true) ~= nil and
      readme:find("numlua.heaviside", 1, true) ~= nil,
    "README.md does not document angle/sinc/heaviside")
  assert_true(readme:find("numlua.iqr", 1, true) ~= nil,
    "README.md does not document iqr")
  assert_true(readme:find("numlua.intersect1d", 1, true) ~= nil and
      readme:find("numlua.isin", 1, true) ~= nil,
    "README.md does not document set/membership helpers")
  assert_true(readme:find("numlua.fromstring", 1, true) ~= nil,
    "README.md does not document fromstring")
  assert_true(readme:find("numlua.poly` builds", 1, true) ~= nil,
    "README.md does not document poly roots")
  assert_true(readme:find("interpolation method", 1, true) ~= nil,
    "README.md does not document quantile methods")
  assert_true(readme:find("optional 1-based `sorter`", 1, true) ~= nil,
    "README.md does not document searchsorted sorters")
  assert_true(readme:find("comment markers inside", 1, true) ~= nil,
    "README.md does not document quote-aware comments")
  assert_true(readme:find("Bool tensors", 1, true) ~= nil,
    "README.md does not document bool text output")
  for _, option in ipairs({ "usecols", "ndmin", "header", "newline",
      "missing_values", "filling_values", "quotechar" }) do
    assert_true(readme:find(option, 1, true) ~= nil,
      "README.md does not mention the text option " .. option)
  end
end)

test("CI smoke job names the current rockspec and version", function()
  local workflow = read_file(root .. "/.github/workflows/ci.yml")
  assert_true(workflow:find("numlua-" .. numlua.VERSION .. "-1.rockspec",
    1, true) ~= nil,
    "CI smoke job does not name the current rockspec")
  assert_true(workflow:find('assert(n.VERSION=="' .. numlua.VERSION .. '")',
    1, true) ~= nil,
    "CI smoke job does not assert the current numlua.VERSION")
end)
