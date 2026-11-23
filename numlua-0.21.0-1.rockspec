package = "numlua"
version = "0.21.0-1"

source = {
  url = "git+https://github.com/numeric-lua/numlua.git",
  tag = "v0.21.0",
}

description = {
  summary = "NumPy-shaped arrays for LuaJIT",
  detailed = [[
    A thin, 1-based facade over luatorch tensors: constructors, shape and
    layout helpers, elementwise ufuncs, reductions, predicates, linear algebra,
    .npy/.npz and text I/O.  No second tensor engine.
  ]],
  homepage = "https://github.com/numeric-lua/numlua",
  license = "BSD-3-Clause",
}

dependencies = {
  "lua >= 5.1",
  "luatorch >= 0.8",
}

build = {
  type = "builtin",
  modules = {
    numlua = "numlua/init.lua",
    ["numlua.npy"] = "numlua/npy.lua",
    ["numlua.npz"] = "numlua/npz.lua",
    ["numlua.text"] = "numlua/text.lua",
    ["numlua.linalg"] = "numlua/linalg.lua",
    ["numlua.manipulation"] = "numlua/manipulation.lua",
    ["numlua.ufunc"] = "numlua/ufunc.lua",
  },
}
