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

-- pinv(t[, rcond]) is the Moore-Penrose pseudoinverse via SVD.  Singular
-- values at or below rcond * largest_singular_value are treated as zero.
function linalg.pinv(t, rcond)
  matrix(t, "pinv")
  rcond = rcond or 1e-15
  if type(rcond) ~= "number" or rcond < 0 then
    error("numlua.linalg.pinv: rcond must be a non-negative number", 2)
  end
  local u, s, vh = t:svd(false)
  local singular = s:reshape({ s:numel() })
  local count = singular:numel()
  local largest = singular:max():item()
  local cutoff = rcond * (largest > 0 and largest or 1)
  local inverse = torch.zeros(count, count, { dtype = t:dtype() })
  for index = 1, count do
    local value = singular:get(index)
    if value > cutoff then inverse:set(index, index, 1 / value) end
  end
  return vh:t():matmul(inverse):matmul(u:t())
end

-- lstsq(a, b[, rcond]) returns the minimum-norm least-squares solution using
-- the SVD pseudoinverse.  It is a convenience wrapper, not a LAPACK driver.
function linalg.lstsq(a, b, rcond)
  matrix(a, "lstsq")
  tensor(b, "lstsq")
  local bdims = b:sizes()
  if (#bdims ~= 1 and #bdims ~= 2) or bdims[1] ~= a:size(1) then
    error(string.format(
      "numlua.linalg.lstsq: b must be a vector or matrix with first " ..
      "dimension %d", a:size(1)), 2)
  end
  return linalg.pinv(a, rcond):matmul(b)
end

-- matrix_norm(t[, ord]) supports the Frobenius norm (default), the spectral
-- and smallest-singular-value norms (2 and -2), 1, infinity, and the nuclear
-- norm for a 2-D matrix.  ord=2, ord=-2, and "nuc" use the SVD.
function linalg.matrix_norm(t, ord)
  matrix(t, "matrix_norm")
  ord = ord or "fro"
  if ord == "fro" or ord == "f" then
    return t:norm(2)
  elseif ord == 1 then
    return t:abs():sum(1):max()
  elseif ord == math.huge then
    return t:abs():sum(2):max()
  elseif ord == -1 then
    return t:abs():sum(1):min()
  elseif ord == -math.huge then
    return t:abs():sum(2):min()
  elseif ord == 2 or ord == -2 or ord == "nuc" then
    local _, s = t:svd(false)
    if ord == "nuc" then return s:sum() end
    if ord == -2 then return s:min() end
    return s:max()
  end
  error("numlua.linalg.matrix_norm: unsupported ord " .. tostring(ord), 2)
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

function linalg.vdot(a, b)
  tensor(a, "vdot")
  tensor(b, "vdot")
  if a:numel() ~= b:numel() then
    error("numlua.linalg.vdot: operands must have the same number of " ..
      "elements", 2)
  end
  return a:reshape({ a:numel() }):dot(b:reshape({ b:numel() }))
end

function linalg.multi_dot(arrays)
  if type(arrays) ~= "table" or #arrays == 0 then
    error("numlua.linalg.multi_dot expects a non-empty table", 2)
  end
  local result = matrix(arrays[1], "multi_dot")
  for index = 2, #arrays do
    local next_matrix = matrix(arrays[index], "multi_dot")
    if result:size(2) ~= next_matrix:size(1) then
      error(string.format(
        "numlua.linalg.multi_dot: incompatible shapes %dx%d and %dx%d",
        result:size(1), result:size(2), next_matrix:size(1),
        next_matrix:size(2)), 2)
    end
    result = result:matmul(next_matrix)
  end
  return result
end

function linalg.matrix_rank(t, tolerance)
  matrix(t, "matrix_rank")
  if tolerance ~= nil and (type(tolerance) ~= "number" or tolerance < 0) then
    error("numlua.linalg.matrix_rank: tolerance must be non-negative or nil", 2)
  end
  local _, singular = t:svd(false)
  local values = singular:reshape(-1)
  if values:numel() == 0 then
    return torch.tensor(0, { dtype = "int64" })
  end
  local largest = values:max():item()
  local eps = (t:dtype() == "float16" or t:dtype() == "float32")
    and 1.1920929e-7 or 2.220446049250313e-16
  local cutoff = tolerance or
    largest * math.max(t:size(1), t:size(2)) * eps
  local rank = 0
  for index = 1, values:numel() do
    if values:get(index) > cutoff then rank = rank + 1 end
  end
  return torch.tensor(rank, { dtype = "int64" })
end

function linalg.slogdet(t)
  square(t, "slogdet")
  local value = t:det()
  if torch.is_tensor(value) then value = value:item() end
  if value == 0 then return 0, -math.huge end
  return value > 0 and 1 or -1, math.log(math.abs(value))
end

function linalg.kron(a, b)
  tensor(a, "kron")
  tensor(b, "kron")
  if a:dim() > 2 or b:dim() > 2 then
    error("numlua.linalg.kron supports 1-D and 2-D operands", 2)
  end
  if a:dim() == 1 and b:dim() == 1 then
    if a:numel() == 0 or b:numel() == 0 then
      return torch.empty({ 0 }, { dtype = b:dtype() })
    end
    local parts = {}
    for index = 1, a:numel() do
      parts[index] = b:mul(a:get(index))
    end
    return torch.cat(parts, 1)
  elseif a:dim() == 2 and b:dim() == 2 then
    local rows, columns = a:size(1) * b:size(1), a:size(2) * b:size(2)
    if rows == 0 or columns == 0 then
      return torch.empty({ rows, columns }, { dtype = b:dtype() })
    end
    local result_rows = {}
    for i = 1, a:size(1) do
      local pieces = {}
      for j = 1, a:size(2) do
        pieces[j] = b:mul(a:get(i, j))
      end
      result_rows[i] = torch.cat(pieces, 2)
    end
    return torch.cat(result_rows, 1)
  end
  error("numlua.linalg.kron requires both operands to have the same rank", 2)
end

return linalg
