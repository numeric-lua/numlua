-- numlua -> luaplotlib integration example.
--
--   LUA_PATH="../luaplotlib/?.lua;../luaplotlib/?/init.lua;;" luajit examples/plot.lua
--
-- Writes examples/output/numlua-sin.svg when luaplotlib is available.

local script = (arg and arg[0]) or "examples/plot.lua"
local dir = script:gsub("\\", "/"):match("^(.*)/[^/]+$") or "."
local root
if dir == "examples" or dir == "./examples" then
  root = "."
else
  root = dir
end
package.path = table.concat({
  root .. "/?.lua", root .. "/?/init.lua",
  root .. "/../luatorch/lua/?.lua", root .. "/../luatorch/lua/?/init.lua",
  package.path,
}, ";")

local numlua = require("numlua")
local ok, plt = pcall(require, "luaplotlib")
if not ok then
  io.stderr:write("luaplotlib is not on LUA_PATH; add ../luaplotlib to run this example\n")
  os.exit(1)
end

local x = numlua.linspace(0, 2 * math.pi, 120)
local y = x:sin()
local fig = plt.figure({ width = 800, height = 500 })
fig:plot(x:totable(), y:totable(), { label = "sin(x)" })
fig:title("numlua + luaplotlib")
fig:xlabel("x")
fig:ylabel("sin(x)")
fig:grid(true)
fig:legend()
os.execute("mkdir -p " .. root .. "/examples/output")
local path = root .. "/examples/output/numlua-sin.svg"
fig:savefig(path)
print("wrote " .. path)
