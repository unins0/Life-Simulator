-- nn/capabilities.lua — capability descriptors per backend.
--
-- CPU: every precision is quantizable in fp64 (CPU-authoritative); fp16
-- arithmetic is emulated (native_fp16_arithmetic=false). GPU: fp16 requires
-- the VK_KHR_shader_float16_int8 gate; fp8/fp4/fp2 are the packed payload
-- formats. block_sizes and supported_networks are fixed by the format spec.

local M = {}

M.BLOCK_SIZES = { 8, 16, 32, 64 }
M.SUPPORTED_NETWORKS = { 'seed', 'spore', 'sprout' }

function M.cpu_caps(deterministic)
    return {
        backend = 'cpu',
        deterministic = deterministic or false,
        precisions = { fp32 = true, fp16 = true, fp8 = true, fp4 = true, fp2 = true },
        native_fp16_arithmetic = false,
        packed_storage = true,
        max_batch_size = 1000000,
        max_arena_bytes = 0, -- no GPU arena on the CPU path
        block_sizes = M.BLOCK_SIZES,
        supported_networks = M.SUPPORTED_NETWORKS,
    }
end

function M.gpu_caps(vulkan_caps, deterministic)
    local ccaps = (vulkan_caps and vulkan_caps.caps) or {}
    local f16 = ccaps.f16_arithmetic or false
    return {
        backend = 'gpu',
        deterministic = deterministic or false,
        precisions = { fp32 = true, fp16 = f16, fp8 = true, fp4 = true, fp2 = true },
        native_fp16_arithmetic = f16,
        packed_storage = true,
        max_batch_size = 1000000,
        max_arena_bytes = 16 * 1024 * 1024, -- 4 arenas at 4 MiB each (default)
        block_sizes = M.BLOCK_SIZES,
        supported_networks = M.SUPPORTED_NETWORKS,
    }
end

function M.for_backend(backend, opts)
    opts = opts or {}
    local deterministic = opts.deterministic or false
    if backend == 'gpu' then
        local v
        local ok, mod = pcall(require, 'nn.vulkan')
        if ok and type(mod) == 'table' then v = mod end
        local vcaps
        if v and type(v.query_capabilities) == 'function' then
            local okc, c = pcall(v.query_capabilities)
            if okc and type(c) == 'table' then vcaps = c end
        end
        return M.gpu_caps(vcaps, deterministic)
    end
    return M.cpu_caps(deterministic)
end

return M
