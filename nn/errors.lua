-- nn/errors.lua — structured error taxonomy for the standalone NN core.
--
-- Every error surfaced by the NN package is a plain table:
--   { class = <CLASS>, message = <string>, backend = "cpu"|"gpu", recoverable = false }
-- Constructors never raise (they build the table); callers decide whether to
-- raise. `is(err, class)` answers "is this a structured error of that class?".
-- The Vulkan modules reuse this module through pcall(require, 'nn.errors') —
-- keep the public surface (new / error / M[CLASS]) stable.

local M = {}

M.CLASSES = {
    INVALID_ARGUMENT            = 'INVALID_ARGUMENT',
    INVALID_TOPOLOGY            = 'INVALID_TOPOLOGY',
    INVALID_PRECISION           = 'INVALID_PRECISION',
    INVALID_FORMAT              = 'INVALID_FORMAT',
    INVALID_WEIGHTS             = 'INVALID_WEIGHTS',
    INVALID_SERIALIZATION       = 'INVALID_SERIALIZATION',
    UNSUPPORTED_PRECISION       = 'UNSUPPORTED_PRECISION',
    VULKAN_UNSUPPORTED_PLATFORM = 'VULKAN_UNSUPPORTED_PLATFORM',
    VULKAN_EXTENSION_MISSING    = 'VULKAN_EXTENSION_MISSING',
    VULKAN_INITIALIZATION_FAILED = 'VULKAN_INITIALIZATION_FAILED',
    VULKAN_PIPELINE_FAILED      = 'VULKAN_PIPELINE_FAILED',
    VULKAN_ARENA_EXHAUSTED      = 'VULKAN_ARENA_EXHAUSTED',
    DEVICE_LOST                 = 'DEVICE_LOST',
    WORKER_SHUTDOWN             = 'WORKER_SHUTDOWN',
    CORPUS_GATE_FAILED          = 'CORPUS_GATE_FAILED',
    ABI_MISMATCH                = 'ABI_MISMATCH',
}

function M.new(class, message, backend)
    return {
        class = class or 'INVALID_ARGUMENT',
        message = message or '',
        backend = backend or 'cpu',
        recoverable = false,
    }
end

-- Alias used by the vulkan compat shim (pcall(errors.error, class, message)).
M.error = M.new

-- Class-keyed constructors (vulkan.lua calls errors_mod[CLASS](message)).
for _class in pairs(M.CLASSES) do
    M[_class] = function(message, backend)
        return M.new(_class, message, backend)
    end
end

function M.is(err, class)
    return type(err) == 'table' and type(err.class) == 'string' and err.class == class
end

return M
