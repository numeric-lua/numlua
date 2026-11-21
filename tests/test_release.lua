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

test("new public selection API is documented", function()
  local api = read_file(root .. "/docs/api.md")
  assert_true(api:find("numlua.put", 1, true) ~= nil,
    "docs/api.md does not document numlua.put")
end)
