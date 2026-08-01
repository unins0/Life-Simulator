-- nn/api.lua — public NN package surface (exposed by nn/init.lua).
--
--   local nn = require('nn')
--   local runtime, err = nn.new({ backend="auto", deterministic=false,
--                                  precision="fp8", topology=..., model=... })
--   runtime.backend  -- "cpu" | "gpu" (never "auto")
--   runtime:shutdown()
--   runtime:forward(network_id, weights, inputs)
--   runtime:forward_into(network_id, weights, inputs, out)
--   runtime:forward_batch(network_id, batch_items, in_desc, out_desc)
--   runtime:set_precision("fp8", { experimental = false })
--   runtime:capabilities()
-- Modules are re-exported for tests and tooling: errors, format, quantize,
-- corpus, serialize (nn.save / nn.load use plain io, no love.filesystem).

local M = {}

local runtime_mod = require('nn.runtime')
local errors = require('nn.errors')
local format = require('nn.format')
local quantize = require('nn.quantize')
local corpus = require('nn.corpus')
local serialize = require('nn.serialize')
local capabilities = require('nn.capabilities')
local reference = require('nn.reference')
local cpu = require('nn.cpu')

M.errors = errors
M.format = format
M.quantize = quantize
M.corpus = corpus
M.serialize = serialize
M.capabilities = capabilities
M.reference = reference
M.cpu = cpu
M.Runtime = runtime_mod
M.new = runtime_mod.new

-- nn.save(path, model, opts) -> true | nil, err
-- Serializes a model { networks = { <id> = { layers, weights } } } to .nnw
-- using plain io.
function M.save(path, model, opts)
    local bytes, err = serialize.model_to_nnw(model, opts)
    if not bytes then return nil, err end
    local f, ferr = io.open(path, 'wb')
    if not f then
        return nil, errors.new('INVALID_ARGUMENT',
            ('cannot open %q for writing: %s'):format(path, tostring(ferr)))
    end
    local ok, werr = f:write(bytes)
    f:close()
    if not ok then
        return nil, errors.new('INVALID_SERIALIZATION',
            ('write to %q failed: %s'):format(path, tostring(werr)))
    end
    return true
end

-- nn.load(path, runtime) -> model | nil, err
-- Reads a .nnw model back; when a runtime is given the topology is validated
-- against it (INVALID_TOPOLOGY on mismatch).
function M.load(path, runtime)
    local f, ferr = io.open(path, 'rb')
    if not f then
        return nil, errors.new('INVALID_SERIALIZATION',
            ('cannot open %q: %s'):format(path, tostring(ferr)))
    end
    local bytes = f:read('*a')
    f:close()
    local model, err = serialize.read(bytes)
    if not model then return nil, err end
    if runtime then
        for id, net in pairs(runtime.networks) do
            local m = model.networks[id]
            if not m then
                return nil, errors.new('INVALID_TOPOLOGY',
                    ('model has no network %q'):format(id), runtime.backend)
            end
            for i = 1, #net.layers do
                if m.layers[i] ~= net.layers[i] then
                    return nil, errors.new('INVALID_TOPOLOGY',
                        ('model topology for %q differs from the runtime'):format(id),
                        runtime.backend)
                end
            end
        end
    end
    return model
end

return M
