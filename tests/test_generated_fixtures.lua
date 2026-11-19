local numlua = require("numlua")
local generator = require("tests.fixtures.generate")

local corpus
local function fixtures()
  if corpus == nil then
    corpus = generator.generate(TMPDIR)
  end
  return corpus
end

test("loads the generated .npy corpus across dtypes and shapes", function()
  local files = fixtures().npy

  local f32_3d = numlua.load(files["float32-3d"])
  assert_dtype(f32_3d, "float32")
  assert_shape(f32_3d, { 2, 2, 2 })
  assert_tensor(f32_3d, { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } })

  local f32_2d = numlua.load(files["float32-2x2"])
  assert_dtype(f32_2d, "float32")
  assert_shape(f32_2d, { 2, 2 })
  assert_tensor(f32_2d, { { 1.5, -2.25 }, { 0, 3.5 } })

  local f64 = numlua.load(files["float64-2x3"])
  assert_dtype(f64, "float64")
  assert_shape(f64, { 2, 3 })
  assert_near(f64:get(1, 1), math.pi, 1e-12)
  assert_near(f64:get(1, 2), -0.125, 0)
  assert_eq(f64:get(1, 3), 1e300)
  assert_eq(f64:get(2, 1), -1e-300)
  assert_eq(f64:get(2, 2), 2.5)
  assert_eq(f64:get(2, 3), 42)

  local i64 = numlua.load(files["int64-1d"])
  assert_dtype(i64, "int64")
  assert_tensor(i64, { -3, 0, 7, 42 })

  local bool = numlua.load(files["bool-2x2"])
  assert_dtype(bool, "bool")
  assert_tensor(bool, { { true, false }, { true, true } })

  local empty = numlua.load(files["empty-2x0"])
  assert_shape(empty, { 2, 0 })
  assert_eq(empty:numel(), 0)

  local empty_int = numlua.load(files["empty-0x3"])
  assert_dtype(empty_int, "int64")
  assert_shape(empty_int, { 0, 3 })
  assert_eq(empty_int:numel(), 0)

  local scalar_f64 = numlua.load(files["scalar-float64"])
  assert_dtype(scalar_f64, "float64")
  assert_eq(scalar_f64:dim(), 0)
  assert_eq(scalar_f64:item(), 1.25)

  local scalar_bool = numlua.load(files["scalar-bool"])
  assert_dtype(scalar_bool, "bool")
  assert_eq(scalar_bool:dim(), 0)
  assert_eq(scalar_bool:item(), true)
end)

test("loads the generated stored-ZIP .npz corpus", function()
  local loaded = numlua.loadz(fixtures().npz)

  assert_dtype(loaded["float32-2x2"], "float32")
  assert_tensor(loaded["float32-2x2"], { { 1.5, -2.25 }, { 0, 3.5 } })

  assert_dtype(loaded["int64-1d"], "int64")
  assert_tensor(loaded["int64-1d"], { -3, 0, 7, 42 })

  assert_dtype(loaded["bool-2x2"], "bool")
  assert_tensor(loaded["bool-2x2"], { { true, false }, { true, true } })

  assert_shape(loaded["empty-2x0"], { 2, 0 })
  assert_eq(loaded["empty-2x0"]:numel(), 0)

  assert_dtype(loaded["scalar-float64"], "float64")
  assert_eq(loaded["scalar-float64"]:item(), 1.25)
end)

test("rejects the generated big-endian .npy fixture", function()
  assert_error(function()
    numlua.load(fixtures().big_endian)
  end, "big-endian")
end)
