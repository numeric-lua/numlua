--
-- numlua micro-benchmarks.
--
--   ./scripts/bench.sh [--samples N] [--warmup N]
--
-- Each row compares a numlua facade call with the equivalent direct luatorch
-- call (the ATen baseline), except for .npy I/O where the baseline is a raw
-- Lua/FFI read/write with the header work removed.  Wall-clock samples are
-- interleaved and the median is reported.  This is a reporting tool, not a
-- CI gate.
--
local root = (arg and arg[0] or "bench/bench.lua"):match("^(.*)/bench/[^/]+$")
  or "."
package.path = table.concat({
  root .. "/?.lua",
  root .. "/?/init.lua",
  root .. "/../luatorch/lua/?.lua",
  root .. "/../luatorch/lua/?/init.lua",
  package.path,
}, ";")

local ffi = require("ffi")
local torch = require("torch")
local numlua = require("numlua")
local lib = require("torch.lib").lib

ffi.cdef[[
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int clk_id, struct timespec* tp);
]]

local timespec = ffi.new("struct timespec")
local function now()
  if ffi.C.clock_gettime(1, timespec) == 0 then
    return tonumber(timespec.tv_sec) + tonumber(timespec.tv_nsec) * 1e-9
  end
  return os.clock()
end

local function median(values)
  table.sort(values)
  local count = #values
  if count % 2 == 1 then return values[(count + 1) / 2] end
  return (values[count / 2] + values[count / 2 + 1]) / 2
end

local sink
local function measure(fn, iterations)
  local started = now()
  for _ = 1, iterations do sink = fn() end
  return (now() - started) / iterations
end

local function measure_pair(fn, baseline, iterations, samples, warmup)
  for _ = 1, warmup do
    sink = fn()
    sink = baseline()
  end
  local facade, reference = {}, {}
  for sample = 1, samples do
    collectgarbage("collect")
    if sample % 2 == 1 then
      facade[sample] = measure(fn, iterations)
      collectgarbage("collect")
      reference[sample] = measure(baseline, iterations)
    else
      reference[sample] = measure(baseline, iterations)
      collectgarbage("collect")
      facade[sample] = measure(fn, iterations)
    end
  end
  return median(facade), median(reference)
end

local samples = 5
local warmup = 2
local index = 1
while arg ~= nil and index <= #arg do
  local option = arg[index]
  if option == "--samples" then
    index = index + 1
    samples = tonumber(arg[index])
  elseif option == "--warmup" then
    index = index + 1
    warmup = tonumber(arg[index])
  elseif option == "--help" or option == "-h" then
    print("usage: luajit bench/bench.lua [--samples N] [--warmup N]")
    os.exit(0)
  else
    io.stderr:write("unknown benchmark option: " .. tostring(option) .. "\n")
    os.exit(2)
  end
  index = index + 1
end
if not samples or samples < 1 or samples % 1 ~= 0 or
    not warmup or warmup < 0 or warmup % 1 ~= 0 then
  io.stderr:write("--samples must be a positive integer and --warmup " ..
    "a non-negative integer\n")
  os.exit(2)
end

-- Keep threading fixed so the numbers describe this host and not a changing
-- OpenMP scheduling decision.
torch.set_num_threads(1)
torch.manual_seed(42)

local results = {}
local function bench(group, name, iterations, fn, baseline, baseline_name, note)
  local facade_seconds, baseline_seconds = measure_pair(
    fn, baseline, iterations, samples, warmup)
  results[#results + 1] = {
    group = group,
    name = name,
    iterations = iterations,
    facade = facade_seconds,
    baseline = baseline_seconds,
    baseline_name = baseline_name,
    note = note or "",
  }
end

-- ---------------------------------------------------------------------------
-- construction
-- ---------------------------------------------------------------------------

local side = 256
local nested = {}
for row = 1, side do
  local values = {}
  for column = 1, side do values[column] = row + column / 1024 end
  nested[row] = values
end

bench("construction", "array nested 256x256", 20,
  function() return numlua.array(nested, { dtype = "float32" }) end,
  function() return torch.tensor(nested, { dtype = "float32" }) end,
  "torch.tensor (ATen)", "same table-to-tensor path")

bench("construction", "zeros 1024x1024", 100,
  function() return numlua.zeros(1024, 1024) end,
  function() return torch.zeros(1024, 1024) end,
  "torch.zeros (ATen)", "4 MiB float32")

bench("construction", "arange 1M float32", 200,
  function() return numlua.arange(1000000, { dtype = "float32" }) end,
  function() return torch.arange(1000000, { dtype = "float32" }) end,
  "torch.arange (ATen)", "4 MiB float32")

-- ---------------------------------------------------------------------------
-- reductions
-- ---------------------------------------------------------------------------

local flat = torch.randn(4000000, { dtype = "float32" })
local matrix = torch.randn(1024, 1024, { dtype = "float32" })
local integers = torch.arange(1000000, { dtype = "int64" })

bench("reductions", "sum all 4M float32", 20,
  function() return numlua.sum(flat) end,
  function() return flat:sum() end,
  "Tensor:sum (ATen)", "0-D result")

bench("reductions", "sum axis=2 1024x1024", 30,
  function() return numlua.sum(matrix, 2) end,
  function() return matrix:sum(2) end,
  "Tensor:sum (ATen)", "row reduction, 1-based axis")

bench("reductions", "mean all 1M int64", 20,
  function() return numlua.mean(integers) end,
  function() return integers:to("float64"):mean() end,
  "to(float64):mean (ATen)", "numlua's documented int promotion")

bench("reductions", "max axis=2 1024x1024", 30,
  function() return numlua.max(matrix, 2) end,
  function() local values = matrix:max(2) return values end,
  "Tensor:max (ATen)", "ATen also computes indices; both discard them")

-- ---------------------------------------------------------------------------
-- broadcasting
-- ---------------------------------------------------------------------------

local row = torch.randn(1, 1024, { dtype = "float32" })
local column = torch.randn(1024, 1, { dtype = "float32" })
local mask = torch.rand(1024, 1024, { dtype = "float32" }):gt(0.5)

bench("broadcasting", "broadcast_to 1024x1024", 2000,
  function() return numlua.broadcast_to(row, { 1024, 1024 }) end,
  function() return row:expand(1024, 1024) end,
  "Tensor:expand (ATen)", "view, no data copy")

bench("broadcasting", "broadcast_arrays row+column", 1000,
  function()
    local a, b = numlua.broadcast_arrays(row, column)
    return a, b
  end,
  function()
    local a = row:expand(1024, 1024)
    local b = column:expand(1024, 1024)
    return a, b
  end,
  "manual expand pair (ATen)", "shape inference in Lua")

bench("broadcasting", "where broadcast 1024x1024", 20,
  function() return numlua.where(mask, row, column) end,
  function() return torch.where(mask, row, column) end,
  "torch.where (ATen)", "materialized 4 MiB result")

-- ---------------------------------------------------------------------------
-- layout
-- ---------------------------------------------------------------------------

local huge = torch.randn(2048, 2048, { dtype = "float32" })
local huge_flat = torch.randn(2048 * 2048, { dtype = "float32" })
local transposed = huge:permute(2, 1)

bench("layout", "transpose 2048x2048", 2000,
  function() return numlua.transpose(huge) end,
  function() return huge:permute(2, 1) end,
  "Tensor:permute (ATen)", "view, no data copy")

bench("layout", "reshape 4M contiguous", 2000,
  function() return numlua.reshape(huge_flat, { 2048, 2048 }) end,
  function() return huge_flat:reshape({ 2048, 2048 }) end,
  "Tensor:reshape (ATen)", "view, no data copy")

bench("layout", "ravel 4M contiguous", 2000,
  function() return numlua.ravel(huge) end,
  function() return huge:reshape({ huge:numel() }) end,
  "Tensor:reshape (ATen)", "C-order view of contiguous input")

bench("layout", "ravel 1024x1024 transposed", 20,
  function() return numlua.ravel(matrix:permute(2, 1)) end,
  function() return matrix:permute(2, 1):reshape({ matrix:numel() }) end,
  "permute + reshape (ATen)", "non-contiguous input must copy")

bench("layout", "tile 1024x1024 x2 rows", 10,
  function() return numlua.tile(matrix, { 2, 1 }) end,
  function() return torch.cat({ matrix, matrix }, 1) end,
  "torch.cat (ATen)", "numlua uses index_select")

bench("layout", "flip 2048x2048", 10,
  function() return numlua.flip(huge, 1) end,
  function() return huge:flip(1) end,
  "Tensor:flip (ATen)", "materialized 16 MiB result")

-- ---------------------------------------------------------------------------
-- .npy I/O
-- ---------------------------------------------------------------------------

local io_tensor = torch.randn(1024, 1024, { dtype = "float32" })
local npy_path = root .. "/tests/.tmp/numlua-bench.npy"
local raw_path = root .. "/tests/.tmp/numlua-bench-raw.npy"

local function make_header()
  local dict = string.format(
    "{'descr': '<f4', 'fortran_order': False, 'shape': (1024, 1024), }")
  local total_without_padding = 10 + #dict + 1
  local padding = (64 - (total_without_padding % 64)) % 64
  return dict .. string.rep(" ", padding) .. "\n"
end

local raw_header = make_header()
local raw_prefix = "\147NUMPY" .. string.char(1, 0) ..
  string.char(#raw_header % 256, math.floor(#raw_header / 256))
local payload_offset = 10 + #raw_header

local function raw_save()
  local handle = assert(io.open(raw_path, "wb"))
  assert(handle:write(raw_prefix, raw_header))
  local pointer = lib.lt_data_ptr(io_tensor.h)
  assert(handle:write(ffi.string(pointer, io_tensor:nbytes())))
  assert(handle:close())
  return true
end

local function raw_load()
  local handle = assert(io.open(raw_path, "rb"))
  local data = handle:read("*a")
  assert(handle:close())
  local tensor = torch.empty({ 1024, 1024 }, { dtype = "float32" })
  local pointer = lib.lt_data_ptr(tensor.h)
  ffi.copy(ffi.cast("void*", pointer),
    ffi.cast("const char*", data) + payload_offset, tensor:nbytes())
  return tensor
end

assert(numlua.save(npy_path, io_tensor))
assert(raw_save())

bench(".npy I/O", "save 1024x1024 float32", 5,
  function() return numlua.save(npy_path, io_tensor) end,
  raw_save,
  "raw Lua/FFI write", "baseline writes a precomputed header")

bench(".npy I/O", "load 1024x1024 float32", 5,
  function() return numlua.load(npy_path) end,
  raw_load,
  "raw Lua/FFI read", "baseline skips header parsing")

-- ---------------------------------------------------------------------------
-- report
-- ---------------------------------------------------------------------------

print("# numlua micro-benchmark results")
print()
print(string.format(
  "Wall-clock median of %d samples; %d warmup runs; ratio = numlua / " ..
  "baseline (lower is better).", samples, warmup))
print()
print("| Group | Benchmark | numlua ms | baseline ms | ratio | Baseline | Note |")
print("| --- | --- | ---: | ---: | ---: | --- | --- |")

local current_group = nil
for _, row_result in ipairs(results) do
  if row_result.group ~= current_group then
    current_group = row_result.group
  end
  print(string.format("| %s | `%s` | %.4f | %.4f | %.2fx | %s | %s |",
    row_result.group, row_result.name, row_result.facade * 1000,
    row_result.baseline * 1000, row_result.facade / row_result.baseline,
    row_result.baseline_name, row_result.note))
end

print()
print(string.format("torch %s (libtorch %s), LuaJIT %s, %d thread(s)",
  torch.VERSION, torch.version, jit.version, torch.get_num_threads()))
