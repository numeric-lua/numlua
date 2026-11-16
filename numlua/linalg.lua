--
-- numlua.linalg: NumPy-shaped linear algebra names over luatorch tensors.
--
-- Dimensions are 1-based.  Every array result is a luatorch Tensor; scalar
-- results (det, norm) are 0-dimensional Tensors.  No kernels or dtype rules
-- live here.
--
local torch = require("torch")

local linalg = {}

local function tensor(value, name)
  if not torch.is_tensor(value) then
    error(string.format("numlua.linalg.%s expects a tensor, got %s",
      name, type(value)), 3)
  end
  return value
end

local function square(value, name)
  tensor(value, name)
  local sizes = value:sizes()
  if #sizes ~= 2 or sizes[1] ~= sizes[2] then
    error(string.format(
      "numlua.linalg.%s expects a square matrix, got %s",
      name, table.concat(sizes, "x")), 3)
  end
  return value
end

local function matrix(value, name)
  tensor(value, name)
  if value:dim() ~= 2 then
    error(string.format(
      "numlua.linalg.%s expects a matrix, got %d dimensions",
      name, value:dim()), 3)
  end
  return value
end

-- norm(t[, p[, dim[, keepdim]]]) delegates to Tensor:norm.  p defaults to 2;
-- dim is 1-based.  With no dim the result is a 0-D flattened/Frobenius norm.
function linalg.norm(t, p, dim, keepdim)
  tensor(t, "norm")
  if p ~= nil and type(p) ~= "number" then
    error("numlua.linalg.norm: p must be a number or nil", 2)
  end
  if dim ~= nil then
    if type(dim) ~= "number" or dim % 1 ~= 0 then
      error("numlua.linalg.norm: dim must be an integer or nil", 2)
    end
    if dim == 0 then
      error("numlua.linalg.norm: dimensions are 1-based; dim 0 is invalid", 2)
    end
  end
  if keepdim ~= nil and type(keepdim) ~= "boolean" then
    error("numlua.linalg.norm: keepdim must be a boolean or nil", 2)
  end
  return t:norm(p, dim, keepdim)
end

function linalg.inv(t)
  return square(t, "inv"):inverse()
end

function linalg.det(t)
  return square(t, "det"):det()
end

function linalg.solve(a, b)
  square(a, "solve")
  tensor(b, "solve")
  local n = a:size(1)
  local bdims = b:sizes()
  if (#bdims ~= 1 and #bdims ~= 2) or bdims[1] ~= n then
    error(string.format(
      "numlua.linalg.solve: b must be a vector or matrix with first " ..
      "dimension %d, got %s", n,
      #bdims > 0 and table.concat(bdims, "x") or "a scalar"), 2)
  end
  return a:solve(b)
end

-- svd(t[, full_matrices]) returns U, S, Vh.  full_matrices defaults to true,
-- following luatorch/NumPy.
function linalg.svd(t, full_matrices)
  matrix(t, "svd")
  if full_matrices ~= nil and type(full_matrices) ~= "boolean" then
    error("numlua.linalg.svd: full_matrices must be a boolean or nil", 2)
  end
  return t:svd(full_matrices)
end

-- qr(t[, mode]) returns Q, R.  mode is "reduced" (default), "complete", or
-- "r"; shape and mode errors are ATen's.
function linalg.qr(t, mode)
  matrix(t, "qr")
  if mode ~= nil and type(mode) ~= "string" then
    error("numlua.linalg.qr: mode must be a string or nil", 2)
  end
  return t:qr(mode)
end

return linalg
