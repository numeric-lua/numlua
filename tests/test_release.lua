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
      "numlua.fromfile", "numlua.tofile" }) do
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
    "docs/api.md does not document raw binary I/O")
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
      "unpack", "ndmin", "header", "footer", "comments", "newline" }) do
    assert_true(api:find("`" .. option .. "`", 1, true) ~= nil,
      "docs/api.md does not document the text option " .. option)
  end
  assert_true(api:find("per-column formats", 1, true) ~= nil,
    "docs/api.md does not document per-column fmt tables")

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
    "README.md does not document raw binary I/O")
  assert_true(readme:find("numlua.split", 1, true) ~= nil and
      readme:find("numlua.array_split", 1, true) ~= nil,
    "README.md does not document the split family")
  assert_true(readme:find("numlua.isclose", 1, true) ~= nil and
      readme:find("numlua.array_equiv", 1, true) ~= nil,
    "README.md does not document closeness predicates")
  for _, option in ipairs({ "usecols", "ndmin", "header", "newline" }) do
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
