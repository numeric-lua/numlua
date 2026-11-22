--
-- numlua / text.lua
--
-- Minimal, dependency-free whitespace/CSV numeric text I/O.
--
--   loadtxt(path, options)  -> luatorch Tensor (1-D or 2-D)
--   savetxt(path, tensor, options) -> true
--
-- Options:
--   delimiter  field separator.  loadtxt defaults to runs of whitespace;
--              savetxt defaults to a single space.
--   comments   loadtxt comment marker(s), as a string or table of strings.
--              Defaults to "#"; false or "" disables comment stripping.
--   dtype      loadtxt result dtype (default "float32").
--   fmt        savetxt printf-style format string or function.
--
-- Text files do not record shape or dtype.  A one-row or one-column file loads
-- as 1-D; files with two or more rows and columns load as 2-D.  bool is not a
-- supported text dtype; use an integral dtype and 0/1 values instead.
--

local torch = require("torch")

local text = {}

local DEFAULT_LOAD_DTYPE = "float32"
local DEFAULT_COMMENT = "#"
local DEFAULT_SAVE_DELIMITER = " "

local SUPPORTED_DTYPES = {
  float16 = true,
  float32 = true,
  float64 = true,
  int8 = true,
  int16 = true,
  int32 = true,
  int64 = true,
  uint8 = true,
}

local FLOAT_FORMATS = {
  float16 = "%.5g",
  float32 = "%.9g",
  float64 = "%.17g",
}

local LOAD_OPTIONS = {
  delimiter = true,
  comments = true,
  dtype = true,
  skiprows = true,
  max_rows = true,
}

local SAVE_OPTIONS = {
  delimiter = true,
  fmt = true,
}

local function check_options(options, name, allowed)
  if options ~= nil and type(options) ~= "table" then
    error(string.format("numlua.%s: options must be a table or nil", name), 3)
  end
  options = options or {}
  for key in pairs(options) do
    if not allowed[key] then
      error(string.format("numlua.%s: unknown option '%s'",
        name, tostring(key)), 3)
    end
  end
  return options
end

local function check_delimiter(delimiter, name)
  if delimiter ~= nil and (type(delimiter) ~= "string" or delimiter == "") then
    error(string.format(
      "numlua.%s: delimiter must be a non-empty string or nil", name), 3)
  end
  return delimiter
end

local function check_nonnegative_integer(value, option, name)
  if value == nil then
    return nil
  end
  if type(value) ~= "number" or value % 1 ~= 0 or value < 0 then
    error(string.format(
      "numlua.%s: %s must be a non-negative integer or nil, got %s",
      name, option, tostring(value)), 3)
  end
  return value
end

local function comment_markers(comments)
  if comments == nil then
    return { DEFAULT_COMMENT }
  end
  if comments == false or comments == "" then
    return {}
  end
  if type(comments) == "string" then
    return { comments }
  end
  if type(comments) == "table" then
    local markers = {}
    for i = 1, #comments do
      local marker = comments[i]
      if type(marker) ~= "string" or marker == "" then
        error("numlua.loadtxt: comments must contain non-empty strings", 3)
      end
      markers[#markers + 1] = marker
    end
    return markers
  end
  error("numlua.loadtxt: comments must be a string, a table of strings, " ..
    "false, or nil", 3)
end

local function check_load_dtype(dtype)
  if dtype == nil then
    return DEFAULT_LOAD_DTYPE
  end
  if type(dtype) ~= "string" and type(dtype) ~= "number" then
    error("numlua.loadtxt: dtype must be a luatorch dtype name or id", 3)
  end
  local ok, probe = pcall(torch.empty, { 0 }, { dtype = dtype })
  if not ok then
    error(string.format("numlua.loadtxt: unsupported dtype %s",
      type(dtype) == "string" and ("'" .. dtype .. "'") or tostring(dtype)), 3)
  end
  local name = probe:dtype()
  if not SUPPORTED_DTYPES[name] then
    error(string.format(
      "numlua.loadtxt: unsupported dtype '%s' (text I/O supports numeric " ..
      "dtypes, not bool)", name), 3)
  end
  return name
end

local function check_save_dtype(tensor)
  local name = tensor:dtype()
  if not SUPPORTED_DTYPES[name] then
    error(string.format(
      "numlua.savetxt: unsupported dtype '%s' (text I/O supports numeric " ..
      "dtypes, not bool)", name), 3)
  end
  return name
end

local function trim(value)
  return (value:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function read_file(path)
  local handle, open_error = io.open(path, "r")
  if handle == nil then
    error("numlua.loadtxt: cannot open '" .. path .. "' for reading: " ..
      tostring(open_error), 3)
  end
  local data, read_error = handle:read("*a")
  local close_ok, close_error = handle:close()
  if data == nil then
    error("numlua.loadtxt: failed reading '" .. path .. "': " ..
      tostring(read_error), 3)
  end
  if not close_ok then
    error("numlua.loadtxt: failed closing '" .. path .. "': " ..
      tostring(close_error), 3)
  end
  return data
end

local function strip_comments(line, markers)
  local first = nil
  for i = 1, #markers do
    local index = line:find(markers[i], 1, true)
    if index ~= nil and (first == nil or index < first) then
      first = index
    end
  end
  if first == nil then
    return line
  end
  return line:sub(1, first - 1)
end

local function split_fields(line, delimiter)
  local fields = {}
  if delimiter == nil then
    for field in line:gmatch("%S+") do
      fields[#fields + 1] = field
    end
    return fields
  end
  local start = 1
  while true do
    local index = line:find(delimiter, start, true)
    if index == nil then
      fields[#fields + 1] = trim(line:sub(start))
      return fields
    end
    fields[#fields + 1] = trim(line:sub(start, index - 1))
    start = index + #delimiter
  end
end

local function parse_number(field)
  local value = tonumber(field)
  if value ~= nil then
    return value
  end
  local lower = field:lower()
  if lower == "nan" or lower == "+nan" or lower == "-nan" then
    return 0 / 0
  end
  if lower == "inf" or lower == "+inf" or lower == "infinity"
    or lower == "+infinity" then
    return math.huge
  end
  if lower == "-inf" or lower == "-infinity" then
    return -math.huge
  end
  return nil
end

-- Read a whitespace- or delimiter-separated text file and return a 1-D/2-D
-- luatorch tensor.  Each non-comment line must have the same number of fields.
function text.loadtxt(path, options)
  if type(path) ~= "string" or path == "" then
    error("numlua.loadtxt expects a non-empty file path", 2)
  end
  options = check_options(options, "loadtxt", LOAD_OPTIONS)
  local delimiter = check_delimiter(options.delimiter, "loadtxt")
  local markers = comment_markers(options.comments)
  local dtype = check_load_dtype(options.dtype)
  local skiprows = check_nonnegative_integer(
    options.skiprows, "skiprows", "loadtxt")
  local max_rows = check_nonnegative_integer(
    options.max_rows, "max_rows", "loadtxt")

  local data = read_file(path)
  local rows = {}
  local columns = nil

  -- max_rows=0 reads no data rows at all, but still validates/open-reads the
  -- path and options first, like NumPy.
  if data ~= "" and max_rows ~= 0 then
    -- Split on CRLF and lone CR as well as LF, matching Python's splitlines
    -- and NumPy's text reader.  Numeric fields never contain a carriage
    -- return, so normalizing before splitting cannot merge or split values.
    data = data:gsub("\r\n", "\n"):gsub("\r", "\n")
    if data:sub(-1) ~= "\n" then
      data = data .. "\n"
    end
    local line_number = 0
    for raw in data:gmatch("([^\n]*)\n") do
      line_number = line_number + 1
      -- skiprows counts physical lines, including comments and blank lines,
      -- exactly like NumPy's skiprows.  max_rows below counts data rows only.
      if line_number > (skiprows or 0) then
        local line = strip_comments(raw, markers)
        if trim(line) ~= "" then
          local fields = split_fields(line, delimiter)
          if columns == nil then
            columns = #fields
          elseif #fields ~= columns then
            error(string.format(
              "numlua.loadtxt: line %d has %d column(s), expected %d",
              line_number, #fields, columns), 2)
          end
          local row = {}
          for column = 1, #fields do
            local value = parse_number(fields[column])
            if value == nil then
              error(string.format(
                "numlua.loadtxt: line %d: invalid number '%s' in column %d",
                line_number, fields[column], column), 2)
            end
            row[column] = value
          end
          rows[#rows + 1] = row
          if max_rows ~= nil and #rows >= max_rows then
            break
          end
        end
      end
    end
  end

  if #rows == 0 then
    return torch.empty({ 0 }, { dtype = dtype })
  end
  -- Like NumPy's default squeeze behavior, a one-row or one-column result is
  -- returned as a 1-D tensor.  Plain text has no way to distinguish a vector
  -- from a 1xN or Nx1 matrix.
  if columns == 1 or #rows == 1 then
    local values = {}
    for i = 1, #rows do
      for j = 1, columns do
        values[#values + 1] = rows[i][j]
      end
    end
    return torch.tensor(values, { dtype = dtype })
  end
  return torch.tensor(rows, { dtype = dtype })
end

local function format_value(value, dtype, fmt)
  if fmt ~= nil then
    local result
    if type(fmt) == "function" then
      local ok, out = pcall(fmt, value)
      if not ok then
        error("numlua.savetxt: fmt function failed: " .. tostring(out), 3)
      end
      result = out
    else
      local ok, out = pcall(string.format, fmt, value)
      if not ok then
        error(string.format(
          "numlua.savetxt: invalid fmt '%s' for value %s: %s",
          fmt, tostring(value), tostring(out)), 3)
      end
      result = out
    end
    if type(result) ~= "string" then
      error("numlua.savetxt: fmt must produce a string, got " ..
        type(result), 3)
    end
    return result
  end

  if FLOAT_FORMATS[dtype] then
    if value ~= value then
      return "nan"
    end
    if value == math.huge then
      return "inf"
    end
    if value == -math.huge then
      return "-inf"
    end
    return string.format(FLOAT_FORMATS[dtype], value)
  end
  return string.format("%d", math.floor(value))
end

local function write_row(handle, parts, delimiter)
  local ok, write_error = handle:write(table.concat(parts, delimiter), "\n")
  if ok == nil then
    return nil, write_error
  end
  return true
end

-- Write a 1-D/2-D tensor as numeric text.  Floating-point defaults use enough
-- digits to round-trip at the tensor's dtype; integral dtypes use base 10.
function text.savetxt(path, tensor, options)
  if type(path) ~= "string" or path == "" then
    error("numlua.savetxt expects a non-empty file path", 2)
  end
  if not torch.is_tensor(tensor) then
    error("numlua.savetxt expects a tensor, got " .. type(tensor), 2)
  end
  options = check_options(options, "savetxt", SAVE_OPTIONS)
  local delimiter = check_delimiter(options.delimiter, "savetxt")
    or DEFAULT_SAVE_DELIMITER
  local fmt = options.fmt
  if fmt ~= nil and type(fmt) ~= "string" and type(fmt) ~= "function" then
    error("numlua.savetxt: fmt must be a format string, function, or nil", 2)
  end

  local dim = tensor:dim()
  if dim ~= 1 and dim ~= 2 then
    error(string.format(
      "numlua.savetxt supports 1-D and 2-D arrays, got %d-D", dim), 2)
  end
  local dtype = check_save_dtype(tensor)

  local handle, open_error = io.open(path, "w")
  if handle == nil then
    error("numlua.savetxt: cannot open '" .. path .. "' for writing: " ..
      tostring(open_error), 2)
  end

  local sizes = tensor:sizes()
  local write_ok, write_error = true, nil
  local format_ok, format_error = pcall(function()
    if dim == 1 then
      if sizes[1] > 0 then
        local values = tensor:totable()
        local parts = {}
        for i = 1, sizes[1] do
          parts[i] = format_value(values[i], dtype, fmt)
        end
        write_ok, write_error = write_row(handle, parts, delimiter)
      end
    else
      local rows = tensor:totable()
      for i = 1, sizes[1] do
        local row = rows[i]
        local parts = {}
        for j = 1, sizes[2] do
          parts[j] = format_value(row[j], dtype, fmt)
        end
        if sizes[2] == 0 then
          local ok, err = handle:write("\n")
          write_ok, write_error = ok, err
        else
          write_ok, write_error = write_row(handle, parts, delimiter)
        end
        if write_ok == nil then
          break
        end
      end
    end
  end)
  local close_ok, close_error = handle:close()

  if not format_ok then
    error(format_error, 0)
  end
  if write_ok == nil then
    error("numlua.savetxt: failed writing '" .. path .. "': " ..
      tostring(write_error), 2)
  end
  if not close_ok then
    error("numlua.savetxt: failed closing '" .. path .. "': " ..
      tostring(close_error), 2)
  end
  return true
end

return text
