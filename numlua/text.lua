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
--              savetxt defaults to "# " for header/footer prefixes.
--   dtype      loadtxt result dtype (default "float32").
--   fmt        savetxt printf-style format string or function, or a table with
--              one format per column.
--   skiprows   loadtxt physical lines to skip before parsing (default 0).
--   max_rows   loadtxt data rows to read after skipping (default all).
--   usecols    loadtxt 1-based column number or table of column numbers.
--   unpack     loadtxt transpose a 2-D result.
--   ndmin      loadtxt minimum rank, 0/1/2 (default 0).
--   header     savetxt comment-prefixed text before the data.
--   footer     savetxt comment-prefixed text after the data.
--   newline    savetxt line terminator (default "\n").
--
-- Text files do not record shape or dtype.  A one-row or one-column file loads
-- as 1-D unless ndmin=2; files with two or more rows and columns load as 2-D.
-- bool is not a supported text dtype; use an integral dtype and 0/1 values.
--

local torch = require("torch")

local text = {}

local DEFAULT_LOAD_DTYPE = "float32"
local DEFAULT_COMMENT = "#"
local DEFAULT_SAVE_DELIMITER = " "
local DEFAULT_SAVE_NEWLINE = "\n"
local DEFAULT_SAVE_COMMENTS = "# "

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
  usecols = true,
  converters = true,
  unpack = true,
  ndmin = true,
}

local SAVE_OPTIONS = {
  delimiter = true,
  fmt = true,
  header = true,
  footer = true,
  comments = true,
  newline = true,
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

-- Normalize usecols to a list of 1-based column numbers.  The upper bound is
-- checked once the first data row establishes the total column count.
local function parse_usecols(value)
  if value == nil then
    return nil
  end
  local columns
  if type(value) == "number" then
    columns = { value }
  elseif type(value) == "table" then
    if #value == 0 then
      error("numlua.loadtxt: usecols must be a column number or a " ..
        "non-empty table of column numbers", 3)
    end
    columns = value
  else
    error("numlua.loadtxt: usecols must be a column number or a table of " ..
      "column numbers, got " .. type(value), 3)
  end
  local normalized = {}
  for i = 1, #columns do
    local column = columns[i]
    if type(column) ~= "number" or column % 1 ~= 0 then
      error("numlua.loadtxt: usecols entries must be integers, got " ..
        tostring(column), 3)
    end
    if column == 0 then
      error("numlua.loadtxt: usecols columns are 1-based; column 0 is " ..
        "invalid (use 1 for the first column)", 3)
    end
    normalized[i] = column
  end
  return normalized
end

local function check_unpack(value)
  if value ~= nil and type(value) ~= "boolean" then
    error("numlua.loadtxt: unpack must be a boolean or nil, got " ..
      tostring(value), 3)
  end
  return value == true
end

local function check_string(value, option, name)
  if value ~= nil and type(value) ~= "string" then
    error(string.format(
      "numlua.%s: %s must be a string or nil, got %s",
      name, option, type(value)), 3)
  end
  return value
end

local function check_ndmin(value)
  if value == nil then
    return 0
  end
  if type(value) ~= "number" or value % 1 ~= 0 or
      (value ~= 0 and value ~= 1 and value ~= 2) then
    error("numlua.loadtxt: ndmin must be 0, 1, or 2, got " ..
      tostring(value), 3)
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

-- Normalize the converters option.  A single function applies to every input
-- column; a table maps 1-based physical column numbers to functions.  Unlike
-- NumPy's 0-based dictionary keys, numlua keeps the facade's 1-based column
-- convention.  Bounds are checked once the first data row establishes the
-- file width.
local function parse_converters(value)
  if value == nil then
    return nil
  end
  if type(value) == "function" then
    return { all = value }
  end
  if type(value) ~= "table" then
    error("numlua.loadtxt: converters must be a function, a table, or nil", 3)
  end
  local mapping = {}
  for column, converter in pairs(value) do
    if type(column) ~= "number" or column % 1 ~= 0 then
      error("numlua.loadtxt: converter columns must be integers, got " ..
        tostring(column), 3)
    end
    if column == 0 then
      error("numlua.loadtxt: converter columns are 1-based; column 0 is " ..
        "invalid (use 1 for the first column)", 3)
    end
    if column < 0 then
      error("numlua.loadtxt: converter columns must be positive 1-based " ..
        "column numbers, got " .. tostring(column), 3)
    end
    if type(converter) ~= "function" then
      error(string.format(
        "numlua.loadtxt: converter for column %d must be a function, got %s",
        column, type(converter)), 3)
    end
    mapping[column] = converter
  end
  return { map = mapping }
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
  local usecols = parse_usecols(options.usecols)
  local converters = parse_converters(options.converters)
  local unpack = check_unpack(options.unpack)
  local ndmin = check_ndmin(options.ndmin)

  local data = read_file(path)
  local rows = {}
  local columns = nil
  local selected = nil

  -- Resolve usecols against the first row's total width.  Negative entries
  -- count from the end, and duplicates are preserved in the requested order.
  local function resolve_usecols()
    if usecols == nil then
      return
    end
    selected = {}
    for i = 1, #usecols do
      local requested = usecols[i]
      -- 1-based negative columns count from the end: -1 is the last column.
      local column = requested < 0 and requested + columns + 1 or requested
      if column < 1 or column > columns then
        error(string.format(
          "numlua.loadtxt: usecols index %d is out of range for %d " ..
          "column(s)", requested, columns), 3)
      end
      selected[i] = column
    end
  end

  -- Converter keys refer to physical input columns, not output positions, so
  -- they are validated against the full file width even when usecols is set.
  local function validate_converters()
    if converters == nil or converters.map == nil then
      return
    end
    for column in pairs(converters.map) do
      if column > columns then
        error(string.format(
          "numlua.loadtxt: converter column %d is out of range for %d " ..
          "column(s)", column, columns), 3)
      end
    end
  end

  -- Apply a per-column converter before the built-in numeric parser.  The
  -- callback receives the raw field string and must return one number; this
  -- lets callers handle sentinels such as "NA" without adding a text parser.
  local function parse_field(field, line_number, column)
    local converter = converters ~= nil and
      (converters.all or (converters.map and converters.map[column])) or nil
    if converter == nil then
      return parse_number(field)
    end
    local ok, value = pcall(converter, field)
    if not ok then
      error(string.format(
        "numlua.loadtxt: line %d: converter for column %d failed: %s",
        line_number, column, tostring(value)), 3)
    end
    if type(value) ~= "number" then
      error(string.format(
        "numlua.loadtxt: line %d: converter for column %d must return a " ..
        "number, got %s", line_number, column, type(value)), 3)
    end
    return value
  end

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
            resolve_usecols()
            validate_converters()
          elseif #fields ~= columns then
            error(string.format(
              "numlua.loadtxt: line %d has %d column(s), expected %d",
              line_number, #fields, columns), 2)
          end
          local row = {}
          if selected ~= nil then
            for output_column = 1, #selected do
              local source_column = selected[output_column]
              local value = parse_field(
                fields[source_column], line_number, source_column)
              if value == nil then
                error(string.format(
                  "numlua.loadtxt: line %d: invalid number '%s' in " ..
                  "column %d", line_number, fields[source_column],
                  source_column), 2)
              end
              row[output_column] = value
            end
          else
            for column = 1, #fields do
              local value = parse_field(fields[column], line_number, column)
              if value == nil then
                error(string.format(
                  "numlua.loadtxt: line %d: invalid number '%s' in " ..
                  "column %d", line_number, fields[column], column), 2)
              end
              row[column] = value
            end
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
    local empty = ndmin == 2 and torch.empty({ 0, 1 }, { dtype = dtype })
      or torch.empty({ 0 }, { dtype = dtype })
    if unpack and empty:dim() == 2 then
      empty = empty:transpose(1, 2)
    end
    return empty
  end
  local output_columns = selected ~= nil and #selected or columns
  -- ndmin=2 keeps the physical row/column shape, so a one-row file stays
  -- (1, N) and a one-column file stays (N, 1).  The default (ndmin=0) and
  -- ndmin=1 keep NumPy's squeeze behavior for a single row or column.
  local result
  if ndmin < 2 and (output_columns == 1 or #rows == 1) then
    local values = {}
    for i = 1, #rows do
      for j = 1, output_columns do
        values[#values + 1] = rows[i][j]
      end
    end
    result = torch.tensor(values, { dtype = dtype })
  else
    result = torch.tensor(rows, { dtype = dtype })
  end
  if unpack and result:dim() == 2 then
    result = result:transpose(1, 2)
  end
  return result
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

-- Validate the fmt option against the number of output columns.  A table
-- supplies one format string or function per column; a string/function
-- applies to every column.
local function check_fmt(fmt, columns)
  if fmt == nil then
    return nil
  end
  if type(fmt) == "string" or type(fmt) == "function" then
    return fmt
  end
  if type(fmt) ~= "table" then
    error("numlua.savetxt: fmt must be a format string, function, table, " ..
      "or nil", 2)
  end
  if #fmt ~= columns then
    error(string.format(
      "numlua.savetxt: fmt table has %d entries, expected %d",
      #fmt, columns), 2)
  end
  for i = 1, #fmt do
    local entry = fmt[i]
    if type(entry) ~= "string" and type(entry) ~= "function" then
      error(string.format(
        "numlua.savetxt: fmt table entry %d must be a format string or " ..
        "function, got %s", i, type(entry)), 2)
    end
  end
  return fmt
end

local function write_row(handle, parts, delimiter, newline)
  local ok, write_error = handle:write(
    table.concat(parts, delimiter), newline)
  if ok == nil then
    return nil, write_error
  end
  return true
end

-- Write a header or footer, prefixing every line with the comment marker and
-- terminating it with the configured newline, following np.savetxt.
local function write_annotation(handle, text, comments, newline)
  if text == nil or text == "" then
    return true
  end
  local marked = comments .. text:gsub("\n", function()
    return "\n" .. comments
  end)
  local ok, write_error = handle:write(marked, newline)
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
  local header = check_string(options.header, "header", "savetxt") or ""
  local footer = check_string(options.footer, "footer", "savetxt") or ""
  local comments = check_string(
    options.comments, "comments", "savetxt") or DEFAULT_SAVE_COMMENTS
  local newline = check_string(
    options.newline, "newline", "savetxt") or DEFAULT_SAVE_NEWLINE

  local dim = tensor:dim()
  if dim ~= 1 and dim ~= 2 then
    error(string.format(
      "numlua.savetxt supports 1-D and 2-D arrays, got %d-D", dim), 2)
  end
  local dtype = check_save_dtype(tensor)
  local sizes = tensor:sizes()
  local columns = dim == 1 and 1 or sizes[2]
  local fmt = check_fmt(options.fmt, columns)

  local handle, open_error = io.open(path, "w")
  if handle == nil then
    error("numlua.savetxt: cannot open '" .. path .. "' for writing: " ..
      tostring(open_error), 2)
  end

  local write_ok, write_error = true, nil
  local format_ok, format_error = pcall(function()
    write_ok, write_error = write_annotation(
      handle, header, comments, newline)
    if write_ok == nil then
      return
    end
    if dim == 1 then
      if sizes[1] > 0 then
        local values = tensor:totable()
        local parts = {}
        local column_fmt = type(fmt) == "table" and fmt[1] or fmt
        for i = 1, sizes[1] do
          parts[i] = format_value(values[i], dtype, column_fmt)
        end
        write_ok, write_error = write_row(
          handle, parts, delimiter, newline)
      end
    else
      local rows = tensor:totable()
      for i = 1, sizes[1] do
        local row = rows[i]
        local parts = {}
        for j = 1, sizes[2] do
          local column_fmt = type(fmt) == "table" and fmt[j] or fmt
          parts[j] = format_value(row[j], dtype, column_fmt)
        end
        if sizes[2] == 0 then
          local ok, err = handle:write(newline)
          write_ok, write_error = ok, err
        else
          write_ok, write_error = write_row(
            handle, parts, delimiter, newline)
        end
        if write_ok == nil then
          break
        end
      end
    end
    if write_ok ~= nil then
      write_ok, write_error = write_annotation(
        handle, footer, comments, newline)
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
