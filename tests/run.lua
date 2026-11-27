--
-- numlua test runner.
--
--   LUA_PATH="../../lua/?.lua;../../lua/?/init.lua;./?.lua;./?/init.lua;;" \
--     luajit tests/run.lua
--
-- Test files are plain scripts that call test("name", function() ... end).
-- No luarocks or other external test dependency is used.
--

local script = arg and arg[0] or "tests/run.lua"
local root = script:match("^(.*)[/\\]tests[/\\][^/\\]+$")
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
  root .. "/../../lua/?.lua",
  root .. "/../../lua/?/init.lua",
  package.path,
}, ";")

TMPDIR = root .. "/tests/.tmp"

local passed, failed = 0, 0
local failures = {}
local current_file = "?"

function test(name, fn)
  local ok, err = pcall(fn)
  if ok then
    passed = passed + 1
    io.write(".")
  else
    failed = failed + 1
    failures[#failures + 1] = string.format("[%s] %s\n  %s",
      current_file, name, tostring(err):gsub("\n", "\n  "))
    io.write("F")
  end
  io.flush()
end

local function describe(value)
  if type(value) == "string" then
    return string.format("%q", value)
  end
  if type(value) == "table" then
    local parts = {}
    for i = 1, #value do
      parts[i] = describe(value[i])
    end
    return "{" .. table.concat(parts, ", ") .. "}"
  end
  return tostring(value)
end

local function compare(actual, expected, tolerance, path)
  if type(expected) == "table" then
    if type(actual) ~= "table" then
      error(string.format("%s: expected a table, got %s",
        path, type(actual)), 0)
    end
    local expected_count = 0
    for key, value in pairs(expected) do
      expected_count = expected_count + 1
      compare(actual[key], value, tolerance,
        path .. "[" .. tostring(key) .. "]")
    end
    local actual_count = 0
    for _ in pairs(actual) do
      actual_count = actual_count + 1
    end
    if actual_count ~= expected_count then
      error(string.format("%s: expected %d values, got %d",
        path, expected_count, actual_count), 0)
    end
    return
  end
  if type(expected) == "number" and type(actual) == "number" then
    if math.abs(actual - expected) > tolerance then
      error(string.format("%s: expected %.12g, got %.12g (tolerance %.3g)",
        path, expected, actual, tolerance), 0)
    end
    return
  end
  if actual ~= expected then
    error(string.format("%s: expected %s, got %s",
      path, describe(expected), describe(actual)), 0)
  end
end

function assert_eq(actual, expected, message)
  local ok, err = pcall(compare, actual, expected, 0, "value")
  if not ok then
    error((message or "assert_eq failed") .. "\n  " .. tostring(err), 2)
  end
end

function assert_near(actual, expected, tolerance, message)
  tolerance = tolerance or 1e-5
  local ok, err = pcall(compare, actual, expected, tolerance, "value")
  if not ok then
    error((message or "assert_near failed") .. "\n  " .. tostring(err), 2)
  end
end

function assert_true(value, message)
  if not value then
    error(message or "expected a truthy value", 2)
  end
end

function assert_false(value, message)
  if value then
    error(message or "expected a falsy value", 2)
  end
end

function assert_error(fn, substring)
  local ok, err = pcall(fn)
  if ok then
    error("expected an error" ..
      (substring and (" containing '" .. substring .. "'") or ""), 2)
  end
  local message = tostring(err)
  if substring and not message:find(substring, 1, true) then
    error(string.format("error %q does not contain %q", message, substring), 2)
  end
end

function assert_tensor(actual, expected, tolerance, message)
  assert_true(require("torch").is_tensor(actual),
    message or "expected a luatorch Tensor")
  local ok, err = pcall(compare, actual:totable(), expected,
    tolerance or 1e-5, "tensor")
  if not ok then
    error((message or "assert_tensor failed") .. "\n  " .. tostring(err), 2)
  end
end

function assert_shape(tensor, expected, message)
  assert_eq(tensor:sizes(), expected, message or "unexpected tensor shape")
end

function assert_dtype(tensor, expected, message)
  assert_eq(tensor:dtype(), expected, message or "unexpected tensor dtype")
end

local files = {
  "test_constructors.lua",
  "test_mgrid.lua",
  "test_binary_repr.lua",
  "test_ndindex.lua",
  "test_ndenumerate.lua",
  "test_block.lua",
  "test_copyto_putmask.lua",
  "test_tensordot.lua",
  "test_fromiter.lua",
  "test_fromstring.lua",
  "test_genfromtxt_invalid.lua",
  "test_packbits.lua",
  "test_logspace.lua",
  "test_meshgrid.lua",
  "test_like.lua",
  "test_shape.lua",
  "test_rollaxis.lua",
  "test_stacking.lua",
  "test_split.lua",
  "test_diag.lua",
  "test_vander_tri.lua",
  "test_diagonal.lua",
  "test_reductions.lua",
  "test_product_ptp.lua",
  "test_trapezoid.lua",
  "test_interp.lua",
  "test_polynomial.lua",
  "test_poly_arith.lua",
  "test_poly_roots.lua",
  "test_poly_calc.lua",
  "test_statistics.lua",
  "test_average.lua",
  "test_cov.lua",
  "test_histogram.lua",
  "test_histogram2d.lua",
  "test_histogramdd.lua",
  "test_nan_reductions.lua",
  "test_nan_utils.lua",
  "test_nanmedian.lua",
  "test_nanquantile.lua",
  "test_nan_arg.lua",
  "test_predicates.lua",
  "test_close.lua",
  "test_select.lua",
  "test_piecewise.lua",
  "test_ufuncs.lua",
  "test_ufunc_extras.lua",
  "test_ufunc_more.lua",
  "test_angle_sinc.lua",
  "test_heaviside_iqr.lua",
  "test_manipulation.lua",
  "test_diff_edges.lua",
  "test_gradient.lua",
  "test_gradient_edge2.lua",
  "test_unwrap.lua",
  "test_pad.lua",
  "test_apply_along_axis.lua",
  "test_apply_over_axes.lua",
  "test_selection.lua",
  "test_put_along_axis.lua",
  "test_delete.lua",
  "test_insert.lua",
  "test_negative_indices.lua",
  "test_compress_axis.lua",
  "test_numpy_helpers.lua",
  "test_unique_axis.lua",
  "test_set_ops.lua",
  "test_isin.lua",
  "test_quantile_methods.lua",
  "test_choose.lua",
  "test_trim_extract.lua",
  "test_partition.lua",
  "test_indices.lua",
  "test_indices_from.lua",
  "test_mask_indices.lua",
  "test_digitize.lua",
  "test_searchsorted_sorter.lua",
  "test_index_conversion.lua",
  "test_layout_extensions.lua",
  "test_broadcast_shapes.lua",
  "test_rotate.lua",
  "test_npz.lua",
  "test_npz_handles.lua",
  "test_zip64.lua",
  "test_dtype_promotion.lua",
  "test_linalg.lua",
  "test_linalg_extra.lua",
  "test_linalg_more.lua",
  "test_linalg_facade.lua",
  "test_dot.lua",
  "test_inner_outer.lua",
  "test_cross.lua",
  "test_dtype.lua",
  "test_npy.lua",
  "test_npy_handles.lua",
  "test_raw_io.lua",
  "test_raw_text.lua",
  "test_npy_versions.lua",
  "test_npy_fixtures.lua",
  "test_generated_fixtures.lua",
  "test_txt.lua",
  "test_txt_handles.lua",
  "test_quotechar.lua",
  "test_bool_text.lua",
  "test_genfromtxt.lua",
  "test_errors.lua",
  "test_release.lua",
}

for _, file in ipairs(files) do
  local chunk, load_error = loadfile(root .. "/tests/" .. file)
  if chunk == nil then
    io.write("\n")
    io.stderr:write("cannot load " .. file .. ": " .. tostring(load_error) .. "\n")
    os.exit(2)
  end
  current_file = file
  chunk()
end

io.write("\n")
if failed > 0 then
  io.stderr:write("\n" .. table.concat(failures, "\n\n") .. "\n")
  io.stderr:write(string.format("\n%d passed, %d failed\n", passed, failed))
  os.exit(1)
end
print(string.format("%d tests passed", passed))
