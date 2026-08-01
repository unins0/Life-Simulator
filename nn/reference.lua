-- nn/reference.lua — pure-Lua exact activation reference (validation mode).
--
-- No FFI, no game modules. Forward semantics are the EXACT contract shared by
-- the sim (see ai_module.run): per node
--     value = previous + bias
--     if value <= threshold then value = dead_value end   -- inclusive '<='!
-- (NaN compares false, so NaN survives the dead-zone check and propagates.)
-- Models are explicit objects (layers + node-major weights); this module
-- never owns simulation state and never allocates caller-visible buffers.

local reference = {}
reference.name = 'reference'

function reference.new_model(layers)
    return setmetatable({
        layers = layers,
        weights = nil, -- node-major numeric array (per-network)
    }, reference.MODEL_MT)
end

reference.MODEL_MT = {
    __index = {
        load_weights = function(self, stream)
            self.weights = stream
            return self
        end,
        forward = function(self, inputs)
            return reference.forward(self, inputs)
        end,
        forward_into = function(self, inputs, out)
            return reference.forward_into(self, inputs, out)
        end,
        run_debug = function(self, inputs)
            return reference.run_debug(self, inputs)
        end,
    },
}

-- Precomputed node-major layout (kept in sync with format.decompose):
-- for each non-final node [bias, threshold, dead, weights...]; final nodes
-- [bias, threshold, dead].
function reference._prepare(model)
    local w = model.weights
    if not w then
        return nil, 'model has no weights loaded'
    end
    return model.layers, w
end

-- forward_into: exact activation semantics; writes into caller-owned `out`
-- (a 1-based table of size layers[#layers]). Returns out.
function reference.forward_into(model, inputs, out)
    local layers, w = reference._prepare(model)
    local len = #layers
    local idx = 1
    local data = reference._data
    local total = 0
    for i = 1, len do total = total + layers[i] end
    for i = 1, total do data[i] = nil end
    for i = 1, layers[1] do data[i] = inputs[i] or 0 end

    local offset = 0
    for i = 2, len do
        local layer = layers[i]
        local prev = layers[i - 1]
        local next_offset = offset + prev
        for j = 1, prev do
            local value = (data[j + offset] or 0) + (w[idx] or 0)
            if value <= (w[idx + 1] or 0) then
                value = w[idx + 2] or 0
            end
            for k = 1, layer do
                local ofs = next_offset + k
                data[ofs] = (data[ofs] or 0) + value * (w[idx + k + 2] or 0)
            end
            idx = idx + layer + 3
        end
        offset = next_offset
    end

    for i = 1, layers[len] do
        local value = (data[offset + i] or 0) + (w[idx] or 0)
        if value <= (w[idx + 1] or 0) then
            value = w[idx + 2] or 0
        end
        out[i] = value
        idx = idx + 3
    end
    return out
end

-- Shared scratch (module-level; results are always written to caller tables).
reference._data = {}

function reference.forward(model, inputs)
    local layers = model.layers
    local out = {}
    for i = 1, layers[#layers] do out[i] = 0 end
    return reference.forward_into(model, inputs, out)
end

-- run_debug: like forward but also returns the pre-activation value (value
-- after adding bias, before the dead-zone compare) and dead flags per output
-- node — used by the corpus recorder to classify threshold-crossing.
function reference.run_debug(model, inputs)
    local layers, w = reference._prepare(model)
    local len = #layers
    local idx = 1
    local data = reference._data
    local total = 0
    for i = 1, len do total = total + layers[i] end
    for i = 1, total do data[i] = nil end
    for i = 1, layers[1] do data[i] = inputs[i] or 0 end

    local offset = 0
    for i = 2, len do
        local layer = layers[i]
        local prev = layers[i - 1]
        local next_offset = offset + prev
        for j = 1, prev do
            local value = (data[j + offset] or 0) + (w[idx] or 0)
            if value <= (w[idx + 1] or 0) then
                value = w[idx + 2] or 0
            end
            for k = 1, layer do
                local ofs = next_offset + k
                data[ofs] = (data[ofs] or 0) + value * (w[idx + k + 2] or 0)
            end
            idx = idx + layer + 3
        end
        offset = next_offset
    end

    local outputs, pre, dead_flags = {}, {}, {}
    for i = 1, layers[len] do
        local bias = w[idx] or 0
        local threshold = w[idx + 1] or 0
        local dead = w[idx + 2] or 0
        local value = (data[offset + i] or 0) + bias
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

return reference
