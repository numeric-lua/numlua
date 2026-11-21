local numlua = require("numlua")

test("new-style constructors validate their arguments", function()
  assert_error(function() numlua.array({ 1, 2 }, 5) end,
    "numlua.array: options must be a table")
  assert_error(function() numlua.asarray({ 1, 2 }, "float32") end,
    "numlua.asarray: options must be a table")
  assert_error(function() numlua.full("bad", 1) end,
    "numlua.full expects a shape table")
  assert_error(function() numlua.full({ 2, 2 }, true) end,
    "expects a numeric fill value")
  assert_error(function() numlua.full({}, 1) end,
    "expects a non-empty shape")
  assert_error(function() numlua.full({ 2, 2 }, 1, "bad") end,
    "options must be a table")
  assert_error(function() numlua.linspace(0, 1, 2, "bad") end,
    "options must be a table")
end)

test("shape and reduction helpers validate tensors", function()
  local cases = {
    { "shape", function() numlua.shape(1) end },
    { "reshape", function() numlua.reshape(1, 2) end },
    { "transpose", function() numlua.transpose(1) end },
    { "ravel", function() numlua.ravel(1) end },
    { "squeeze", function() numlua.squeeze("x") end },
    { "unsqueeze", function() numlua.unsqueeze(1, 1) end },
    { "sum", function() numlua.sum(1) end },
    { "mean", function() numlua.mean(1) end },
    { "max", function() numlua.max(1) end },
    { "min", function() numlua.min(1) end },
    { "argmax", function() numlua.argmax(1) end },
    { "argmin", function() numlua.argmin(1) end },
    { "dot", function() numlua.dot(1, 1) end },
    { "matmul", function() numlua.matmul(1, 1) end },
    { "dtype", function() numlua.dtype(1) end },
    { "astype", function() numlua.astype(1, "float32") end },
  }
  for _, case in ipairs(cases) do
    assert_error(case[2], "numlua." .. case[1] .. " expects a tensor")
  end
end)

test("combine helpers validate their inputs", function()
  assert_error(function() numlua.concatenate(nil, 1) end,
    "numlua.concatenate expects a non-empty table")
  assert_error(function() numlua.stack({}, 1) end,
    "numlua.stack expects a non-empty table")
  assert_error(function() numlua.concatenate({ numlua.zeros(2) }, 0) end,
    "axis 0 is invalid")
  assert_error(function() numlua.stack({ numlua.zeros(2) }, 0) end,
    "axis 0 is invalid")
  assert_error(function() numlua.stack({ numlua.zeros(2) }, 1.5) end,
    "axis must be an integer")
end)

test("reduce helpers reject non-integer axes", function()
  local t = numlua.zeros(2, 2)
  assert_error(function() numlua.sum(t, "last") end,
    "axis must be an integer")
  assert_error(function() numlua.max(t, 0.5) end,
    "axis must be an integer")
  assert_error(function() numlua.argmax(t, 0) end,
    "axis 0 is invalid")
  assert_error(function() numlua.argmin(t, 0) end,
    "axis 0 is invalid")
  assert_error(function() numlua.argmin(t, 1, "yes") end,
    "keepdims must be a boolean")
  assert_error(function() numlua.mean(t, 1, "yes") end,
    "keepdims must be a boolean")
end)

test("astype requires a dtype", function()
  local t = numlua.zeros(2)
  assert_error(function() numlua.astype(t) end,
    "numlua.astype expects a dtype")
end)
