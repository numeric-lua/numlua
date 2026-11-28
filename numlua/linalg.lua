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
  if #sizes < 2 or sizes[#sizes - 1] ~= sizes[#sizes] then
    error(string.format(
      "numlua.linalg.%s expects a square matrix, got %s",
      name, table.concat(sizes, "x")), 3)
  end
  return value
end

local function batch_matrix(value, name)
  tensor(value, name)
  if value:dim() < 2 then
    error(string.format(
      "numlua.linalg.%s expects a matrix, got %d dimensions",
      name, value:dim()), 3)
  end
  return value
end

local function batch_prefix_matches(a, b, count)
  for index = 1, count do
    if a:size(index) ~= b:size(index) then return false end
  end
  return true
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
  local ndim = a:dim()
  local n = a:size(ndim - 1)
  local batch_rank = ndim - 2
  local bdims = b:sizes()
  local valid = false
  if #bdims == 1 and bdims[1] == n then
    valid = true
  elseif #bdims == batch_rank + 1 and bdims[#bdims] == n and
      batch_prefix_matches(a, b, batch_rank) then
    valid = true
  elseif #bdims == batch_rank + 2 and bdims[#bdims - 1] == n and
      batch_prefix_matches(a, b, batch_rank) then
    valid = true
  end
  if not valid then
    error(string.format(
      "numlua.linalg.solve: incompatible right-hand side shape %s for " ..
      "%s (first dimension must match)", table.concat(bdims, "x"),
      table.concat(a:sizes(), "x")), 2)
  end
  return a:solve(b)
end

-- svd(t[, full_matrices]) returns U, S, Vh.  full_matrices defaults to true,
-- following luatorch/NumPy.
function linalg.svd(t, full_matrices)
  batch_matrix(t, "svd")
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
  batch_matrix(t, "qr")
  if mode ~= nil and type(mode) ~= "string" then
    error("numlua.linalg.qr: mode must be a string or nil", 2)
  end
  return t:qr(mode)
end

local function product_shape(t, first, last)
  local product = 1
  for index = first, last do product = product * t:size(index) end
  return product
end

function linalg.tensorinv(t, ind)
  tensor(t, "tensorinv")
  ind = ind == nil and 2 or ind
  if type(ind) ~= "number" or ind % 1 ~= 0 or ind < 1 or ind >= t:dim() then
    error("numlua.linalg.tensorinv: ind must be an integer in 1..ndim-1", 2)
  end
  local left = product_shape(t, 1, ind)
  local right = product_shape(t, ind + 1, t:dim())
  if left ~= right then
    error("numlua.linalg.tensorinv: leading and trailing products must match", 2)
  end
  local inverse = linalg.pinv(t:reshape(left, right))
  local shape = {}
  for index = ind + 1, t:dim() do shape[#shape + 1] = t:size(index) end
  for index = 1, ind do shape[#shape + 1] = t:size(index) end
  return inverse:reshape(shape)
end

function linalg.tensorsolve(a, b, axes)
  tensor(a, "tensorsolve")
  tensor(b, "tensorsolve")
  if axes ~= nil then
    error("numlua.linalg.tensorsolve: axes is not supported", 2)
  end
  if a:dim() % 2 ~= 0 then
    error("numlua.linalg.tensorsolve expects an even-rank coefficient tensor", 2)
  end
  local half = math.floor(a:dim() / 2)
  local left, right = product_shape(a, 1, half),
    product_shape(a, half + 1, a:dim())
  if left ~= right then
    error("numlua.linalg.tensorsolve: coefficient halves must have equal " ..
      "products", 2)
  end
  if b:dim() ~= half then
    error("numlua.linalg.tensorsolve: b rank must match half the input rank", 2)
  end
  for index = 1, half do
    if b:size(index) ~= a:size(index) then
      error("numlua.linalg.tensorsolve: b shape does not match the " ..
        "coefficient tensor", 2)
    end
  end
  local solution = a:reshape(left, right):solve(b:reshape(left))
  local shape = {}
  for index = 1, half do shape[index] = a:size(index) end
  return solution:reshape(shape)
end

function linalg.lu(t)
  return square(t, "lu"):lu()
end

function linalg.lu_solve(p, l, u, b)
  square(p, "lu_solve")
  square(l, "lu_solve")
  square(u, "lu_solve")
  tensor(b, "lu_solve")
  local ndim = p:dim()
  local n = p:size(ndim - 1)
  local batch_rank = ndim - 2
  for index = 1, ndim do
    if l:size(index) ~= p:size(index) or u:size(index) ~= p:size(index) then
      error("numlua.linalg.lu_solve expects matching P, L, U factors", 2)
    end
  end
  local dims = b:sizes()
  local valid = false
  if #dims == 1 and dims[1] == n then
    valid = true
  elseif #dims == batch_rank + 1 and dims[#dims] == n and
      batch_prefix_matches(p, b, batch_rank) then
    valid = true
  elseif #dims == batch_rank + 2 and dims[#dims - 1] == n and
      batch_prefix_matches(p, b, batch_rank) then
    valid = true
  end
  if not valid then
    error(string.format(
      "numlua.linalg.lu_solve: incompatible right-hand side shape %s " ..
      "(first dimension must match)", table.concat(dims, "x")), 2)
  end
  local right = b
  local batched_vector = batch_rank > 0 and #dims == batch_rank + 1
  if batched_vector then right = b:unsqueeze(b:dim() + 1) end
  local pb = p:matmul(right)
  if batched_vector then pb = pb:squeeze(pb:dim()) end
  return u:solve(l:solve(pb))
end

function linalg.svdvals(t)
  local _, singular = batch_matrix(t, "svdvals"):svd(false)
  return singular
end

function linalg.vector_norm(t, ord, axis, keepdims)
  return linalg.norm(t, ord, axis, keepdims)
end

function linalg.cholesky(t)
  return square(t, "cholesky"):cholesky()
end

function linalg.eigh(t)
  return square(t, "eigh"):eigh()
end

function linalg.eigvalsh(t)
  local values = square(t, "eigvalsh"):eigh()
  return values
end

function linalg.matrix_power(t, exponent)
  square(t, "matrix_power")
  if type(exponent) ~= "number" or exponent % 1 ~= 0 then
    error("numlua.linalg.matrix_power: exponent must be an integer", 2)
  end
  local base = t
  if exponent < 0 then
    base = t:inverse()
    exponent = -exponent
  end
  local ndim = t:dim()
  local n = t:size(ndim - 1)
  local eye_shape, full = {}, {}
  for index = 1, ndim - 2 do
    eye_shape[index] = 1
    full[index] = t:size(index)
  end
  eye_shape[#eye_shape + 1] = n
  eye_shape[#eye_shape + 1] = n
  full[#full + 1] = n
  full[#full + 1] = n
  local result = torch.eye(n, { dtype = t:dtype() }):reshape(eye_shape)
    :expand(full)
  while exponent > 0 do
    if exponent % 2 == 1 then result = result:matmul(base) end
    exponent = math.floor(exponent / 2)
    if exponent > 0 then base = base:matmul(base) end
  end
  return result
end

function linalg.cond(t, p)
  square(t, "cond")
  p = p == nil and 2 or p
  local dtype = t:is_floating_point() and t:dtype() or "float64"
  local ndim = t:dim()
  if p == 2 or p == -2 then
    local _, singular = t:svd(false)
    local values = singular
    if values:numel() == 0 then
      return torch.tensor(0, { dtype = dtype })
    end
    local largest = values:max(values:dim())
    local smallest = values:min(values:dim())
    local ratio, zero
    if p == 2 then
      ratio, zero = largest:div(smallest), smallest:eq(0)
    else
      ratio, zero = smallest:div(largest), largest:eq(0)
    end
    local infinite = zero:mul(0):add(math.huge)
    return torch.where(zero, infinite, ratio):to(dtype)
  end
  if p ~= 1 and p ~= math.huge and p ~= "fro" and p ~= "f" then
    error("numlua.linalg.cond: unsupported p " .. tostring(p), 2)
  end
  local ok, inverse = pcall(function() return t:inverse() end)
  if not ok then
    return t:abs():sum(ndim):sum(ndim - 1):mul(0):add(math.huge)
      :to(dtype)
  end
  local function norm_value(value)
    if p == 1 then return value:abs():sum(ndim - 1):max(ndim - 1) end
    if p == math.huge then return value:abs():sum(ndim):max(ndim - 1) end
    return value:abs():pow(2):sum(ndim):sum(ndim - 1):sqrt()
  end
  return norm_value(t):mul(norm_value(inverse)):to(dtype)
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
  batch_matrix(t, "matrix_rank")
  if tolerance ~= nil and (type(tolerance) ~= "number" or tolerance < 0) then
    error("numlua.linalg.matrix_rank: tolerance must be non-negative or nil", 2)
  end
  local _, singular = t:svd(false)
  local values = singular
  local ndim = t:dim()
  if values:numel() == 0 then
    local shape = {}
    for index = 1, ndim - 2 do shape[index] = t:size(index) end
    if #shape == 0 then return torch.tensor(0, { dtype = "int64" }) end
    return torch.zeros(shape, { dtype = "int64" })
  end
  local largest = values:max(values:dim())
  local eps = (t:dtype() == "float16" or t:dtype() == "float32")
    and 1.1920929e-7 or 2.220446049250313e-16
  local cutoff
  if tolerance ~= nil then
    cutoff = tolerance
  else
    cutoff = largest:mul(math.max(t:size(ndim - 1), t:size(ndim)) * eps)
  end
  local expanded = type(cutoff) == "number" and cutoff
    or cutoff:unsqueeze(values:dim())
  local rank = values:gt(expanded):sum(values:dim())
  return rank:to("int64")
end

function linalg.slogdet(t)
  square(t, "slogdet")
  local value = t:det()
  if torch.is_tensor(value) and value:numel() > 1 then
    return value:sign(), value:abs():log()
  end
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
