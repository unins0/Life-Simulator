-- nn/cpu.lua — CPU forward backend.
--
-- Two execution modes:
--   * LuaJIT FFI: weights and the data scratch are 'double' cdata arrays.
--     The math is identical to the pure-Lua reference (fp64 throughout), so
--     outputs are BIT-IDENTICAL to reference.forward for fp32/double inputs.
--   * plain Lua (no ffi): falls back to nn/reference (same exact semantics).
-- Models are explicit objects with caller-owned output buffers; this module
-- never owns simulation state.

local HAS_FFI, ffi = pcall(function() return require('ffi') end)
if not HAS_FFI then ffi = nil end

local cpu = {}
cpu.name = 'cpu'
cpu.has_ffi = HAS_FFI

if not HAS_FFI then
    -- Pure-Lua fallback: identical semantics by construction.
    local reference = require('nn.reference')
    for k, v in pairs(reference) do cpu[k] = v end
    cpu.new_model = reference.new_model
    cpu.forward_into = reference.forward_into
    cpu.forward = reference.forward
    cpu.run_debug = reference.run_debug
    return cpu
end

-- Weight model: layers + node-major weights as a 0-based double cdata array.
local MT = {}
MT.__index = MT
cpu.MODEL_MT = MT

function cpu.new_model(layers)
    local m = setmetatable({ layers = layers, weights = nil }, MT)
    m.data = ffi.new('double[?]', 128) -- scratch (grown on demand)
    return m
end

function MT:load_weights(stream)
    local n = #stream
    local w = ffi.new('double[?]', n)
    for i = 1, n do w[i - 1] = stream[i] end
    self.weights = w
    return self
end

function MT:forward(inputs)
    return cpu.forward(self, inputs)
end

function MT:forward_into(inputs, out)
    return cpu.forward_into(self, inputs, out)
end

function MT:run_debug(inputs)
    return cpu.run_debug(self, inputs)
end

local function ensure_scratch(self, total)
    if self.capacity and self.capacity >= total then return end
    self.data = ffi.new('double[?]', total)
    self.capacity = total
end

-- forward_into: fp64 math on cdata (bit-identical to the Lua reference).
function cpu.forward_into(model, inputs, out)
    local layers = model.layers
    local w = model.weights
    if not w then
        error('model has no weights loaded', 2)
    end
    local len = #layers
    local idx = 0 -- 0-based cursor
    local offset = 0
    local total = 0
    for i = 1, len do total = total + layers[i] end
    ensure_scratch(model, total)
    local data = model.data
    ffi.fill(data, total * ffi.sizeof('double'))
    for i = 1, layers[1] do data[i - 1] = inputs[i] or 0 end

    for i = 2, len do
        local layer = layers[i]
        local prev = layers[i - 1]
        local next_offset = offset + prev
        for j = 1, prev do
            local value = data[j + offset - 1] + w[idx]
            if value <= w[idx + 1] then
                value = w[idx + 2]
            end
            for k = 1, layer do
                data[next_offset + k - 1] = data[next_offset + k - 1] + value * w[idx + k + 2]
            end
            idx = idx + layer + 3
        end
        offset = next_offset
    end

    for i = 1, layers[len] do
        local value = data[offset + i - 1] + w[idx]
        if value <= w[idx + 1] then
            value = w[idx + 2]
        end
        out[i] = value
        idx = idx + 3
    end
    return out
end

function cpu.forward(model, inputs)
    local layers = model.layers
    local out = {}
    for i = 1, layers[#layers] do out[i] = 0 end
    return cpu.forward_into(model, inputs, out)
end

-- run_debug: mirror of reference.run_debug (pre-activation + dead flags).
function cpu.run_debug(model, inputs)
    local layers = model.layers
    local w = model.weights
    local len = #layers
    local idx = 0
    local offset = 0
    local total = 0
    for i = 1, len do total = total + layers[i] end
    ensure_scratch(model, total)
    local data = model.data
    ffi.fill(data, total * ffi.sizeof('double'))
    for i = 1, layers[1] do data[i - 1] = inputs[i] or 0 end

    for i = 2, len do
        local layer = layers[i]
        local prev = layers[i - 1]
        local next_offset = offset + prev
        for j = 1, prev do
            local value = data[j + offset - 1] + w[idx]
            if value <= w[idx + 1] then
                value = w[idx + 2]
            end
            for k = 1, layer do
                data[next_offset + k - 1] = data[next_offset + k - 1] + value * w[idx + k + 2]
            end
            idx = idx + layer + 3
        end
        offset = next_offset
    end

    local outputs, pre, dead_flags = {}, {}, {}
    for i = 1, layers[len] do
        local bias = w[idx]
        local threshold = w[idx + 1]
        local dead = w[idx + 2]
        local value = data[offset + i - 1] + bias
        pre[i] = value
        if value <= threshold then
            dead_flags[i] = true
            outputs[i] = dead
        else
            dead_flags[i] = false
            outputs[i] = value
        end
        idx = idx + 3
    end
    return outputs, pre, dead_flags
end

return cpu
