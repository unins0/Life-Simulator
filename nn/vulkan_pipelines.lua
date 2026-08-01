-- nn/vulkan_pipelines.lua — pipeline cache, profile grouping, .spv loading.
--
-- Production profiles are UNIFORM PER-NETWORK: one format + one block size for
-- all layers of a network (Seed=fp8/b16, Spore=fp8/b16, Sprout=fp8/b16 by
-- default; per-network variation such as Spore=fp4/b16 is allowed post-gate).
-- One forward pass of one network = ONE dispatch. Different network profiles
-- get separate specialized pipeline submissions — there is NO runtime format
-- branch for specialized pipelines. Nonuniform per-layer maps are representable
-- in .nnw but runtime-REJECTED unless an explicit experimental per-layer
-- dispatch flag is set (see validate_uniform_profile).
--
-- Pipeline cache key = network profile (network_id, uniform format, uniform
-- block size, compute format, scale/offset storage types, quant ABI version).
-- Pipelines are cached and NEVER rebuilt per tick.

local vulkan = require('nn.vulkan')
local ffi = vulkan.ffi
local mkerr = vulkan.mkerr
local ERR = vulkan.ERRORS
local VK = vulkan.VK

local _M = {
    _VERSION = '1.0.0',
    name = 'vulkan_pipelines',
}

-- shader_abi version the module understands; pipelines built from a different
-- ABI are refused (cache key includes it). v2: per-cell config bases
-- (cfg[8 + cell*4 ..]) + the input-layer activation flag (cfg[5] bit 0).
_M.SHADER_ABI = 2
_M.SPV_DIR = 'nn/shaders/spv'

-- Payload formats -> specialization constant id (constant_id 0 in the shader).
_M.FORMAT_IDS = { fp8 = 0, fp4 = 1, fp2 = 2 }
-- Uniform block size -> block_log2 (constant_id 1).
_M.BLOCK_LOG2 = { [8] = 3, [16] = 4, [32] = 5, [64] = 6 }
-- Compute format -> constant_id 2 (informational in the packed shader).
_M.COMPUTE_FORMAT_IDS = { fp32 = 0, fp16 = 1 }
-- Which .spv a format compiles to (packed path is the production one).
_M.SHADER_FILES = {
    fp8 = 'forward_packed.spv',
    fp4 = 'forward_packed.spv',
    fp2 = 'forward_packed.spv',
    fp32 = 'forward_fp32.spv',
    fp16 = 'forward_fp16.spv',
}

-- profile_key(profile) — canonical, stable cache key for a network profile.
-- Pure-Lua (no ffi/loader needed); a test mirrors it for stability checks.
function _M.profile_key(profile)
    profile = profile or {}
    return table.concat({
        'vkabi', tostring(_M.SHADER_ABI),
        tostring(profile.network_id or ''),
        profile.format or 'fp8',
        tostring(profile.block_size or 0),
        profile.compute_format or 'fp32',
        profile.scale_type or 'fp32',
        profile.offset_type or 'fp32',
        tostring(profile.quant_abi or 1),
    }, '|')
end

-- validate_uniform_profile(layer_map, opts) -> {format=.., block_size=.., uniform=true} | nil, err
-- layer_map: { {format='fp8', block_size=16}, ... } per layer (topological order).
-- Nonuniform per-layer maps are runtime-REJECTED unless
-- opts.experimental_per_layer_dispatch is truthy.
function _M.validate_uniform_profile(layer_map, opts)
    opts = opts or {}
    if not layer_map or #layer_map == 0 then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'empty layer map (no layers to validate)')
    end
    local fmt = layer_map[1].format
    local block = layer_map[1].block_size
    for i = 2, #layer_map do
        if layer_map[i].format ~= fmt or layer_map[i].block_size ~= block then
            if not opts.experimental_per_layer_dispatch then
                return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED,
                    ('nonuniform per-layer map at layer %d (%s/%d vs %s/%d); runtime-REJECTED, ' ..
                     'set opts.experimental_per_layer_dispatch to allow per-layer dispatch')
                        :format(i, layer_map[i].format or '?', layer_map[i].block_size or 0, fmt, block))
            end
        end
    end
    return { format = fmt, block_size = block, uniform = true }
end

-- load_spv(path) -> (uint32[] words, byte_size) | nil, err
-- .spv files are build artifacts (see nn/shaders/spv/README.md). A missing or
-- malformed binary fails with a structured VULKAN_PIPELINE_FAILED, never a raise.
function _M.load_spv(path)
    if not ffi then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'ffi unavailable; cannot load .spv')
    end
    local f = io.open(path, 'rb')
    if not f then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'spv not found: ' .. tostring(path))
    end
    local data = f:read('*a')
    f:close()
    if #data == 0 or #data % 4 ~= 0 then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'invalid spv (size not a multiple of 4): ' .. path)
    end
    local words = ffi.new('uint32_t[?]', math.floor(#data / 4))
    ffi.copy(words, data, #data)
    return words, #data
end

-- build_pipeline(ctx, profile, opts) -> pipe | nil, err
-- opts.spv_dir overrides the .spv directory. Creates shader module (with
-- specialization constants FORMAT / BLOCK_LOG2 / COMPUTE_FORMAT), descriptor
-- set layout (bindings 0..6, storage buffers), pipeline layout and the compute
-- pipeline. The shader module is released once the pipeline is baked.
function _M.build_pipeline(ctx, profile, opts)
    opts = opts or {}
    if not ctx then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'no vulkan context (device not initialized)')
    end
    if not ffi then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'ffi unavailable')
    end
    local fn = ctx.fn.device

    local spv_name = _M.SHADER_FILES[profile.format] or 'forward_packed.spv'
    local spv_path = (opts.spv_dir or _M.SPV_DIR) .. '/' .. spv_name
    local words, spv_size = _M.load_spv(spv_path)
    if not words then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED,
            ('failed to load shader for format %q: %s'):format(profile.format, spv_size))
    end

    -- Shader module.
    local smci = ffi.new('VkShaderModuleCreateInfo')
    smci.sType = VK.STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    smci.codeSize = spv_size
    smci.pCode = words
    local smod = ffi.new('VkShaderModule[1]')
    local res = fn.createShaderModule(ctx.device, smci, nil, smod)
    if res ~= VK.SUCCESS then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkCreateShaderModule failed: ' .. res)
    end

    -- Specialization constants: format, block_log2, compute format.
    local block_log2 = _M.BLOCK_LOG2[profile.block_size] or 3
    local spec_data = ffi.new('uint32_t[3]',
        _M.FORMAT_IDS[profile.format] or 0,
        block_log2,
        _M.COMPUTE_FORMAT_IDS[profile.compute_format] or 0)
    local entries = ffi.new('VkSpecializationMapEntry[3]')
    entries[0].constantID = 0; entries[0].offset = 0; entries[0].size = 4
    entries[1].constantID = 1; entries[1].offset = 4; entries[1].size = 4
    entries[2].constantID = 2; entries[2].offset = 8; entries[2].size = 4
    local spec = ffi.new('VkSpecializationInfo')
    spec.mapEntryCount = 3
    spec.pMapEntries = entries
    spec.dataSize = 12
    spec.pData = spec_data

    -- Compute stage.
    local stage = ffi.new('VkPipelineShaderStageCreateInfo')
    stage.sType = VK.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
    stage.stage = VK.SHADER_STAGE_COMPUTE_BIT
    stage.module = smod[0]
    stage.pName = vulkan.cstr('main')
    stage.pSpecializationInfo = spec

    -- Descriptor set layout: 7 storage-buffer bindings, compute stage.
    local bindings = ffi.new('VkDescriptorSetLayoutBinding[7]')
    for i = 0, 6 do
        bindings[i].binding = i
        bindings[i].descriptorType = VK.DESCRIPTOR_TYPE_STORAGE_BUFFER
        bindings[i].descriptorCount = 1
        bindings[i].stageFlags = VK.SHADER_STAGE_COMPUTE_BIT
        bindings[i].pImmutableSamplers = nil
    end
    local dslci = ffi.new('VkDescriptorSetLayoutCreateInfo')
    dslci.sType = VK.STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO
    dslci.bindingCount = 7
    dslci.pBindings = bindings
    local dsl = ffi.new('VkDescriptorSetLayout[1]')
    res = fn.createDescriptorSetLayout(ctx.device, dslci, nil, dsl)
    if res ~= VK.SUCCESS then
        fn.destroyShaderModule(ctx.device, smod[0], nil)
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkCreateDescriptorSetLayout failed: ' .. res)
    end

    -- Pipeline layout (no push constants in the skeleton; vkCmdPushConstants is
    -- bound by nn/vulkan.lua but unused here).
    local plci = ffi.new('VkPipelineLayoutCreateInfo')
    plci.sType = VK.STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    plci.setLayoutCount = 1
    plci.pSetLayouts = dsl
    local pl = ffi.new('VkPipelineLayout[1]')
    res = fn.createPipelineLayout(ctx.device, plci, nil, pl)
    if res ~= VK.SUCCESS then
        fn.destroyDescriptorSetLayout(ctx.device, dsl[0], nil)
        fn.destroyShaderModule(ctx.device, smod[0], nil)
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkCreatePipelineLayout failed: ' .. res)
    end

    -- Compute pipeline.
    local cpi = ffi.new('VkComputePipelineCreateInfo')
    cpi.sType = VK.STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
    cpi.stage = stage
    cpi.layout = pl[0]
    local pipe = ffi.new('VkPipeline[1]')
    res = fn.createComputePipelines(ctx.device, 0, 1, cpi, nil, pipe)
    if res ~= VK.SUCCESS then
        fn.destroyPipelineLayout(ctx.device, pl[0], nil)
        fn.destroyDescriptorSetLayout(ctx.device, dsl[0], nil)
        fn.destroyShaderModule(ctx.device, smod[0], nil)
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkCreateComputePipelines failed: ' .. res)
    end
    fn.destroyShaderModule(ctx.device, smod[0], nil) -- baked into the pipeline

    return {
        pipeline = pipe[0],
        layout = pl[0],
        set_layout = dsl[0],
        shader_module = 0,
        key = _M.profile_key(profile),
        profile = profile,
        spv = spv_path,
    }
end

-- ------------------------------------------------------------- PipelineCache --
-- Caches pipelines by profile key. NEVER rebuilds per tick: get() returns the
-- cached pipeline for a profile already built.
local PipelineCache = {}
PipelineCache.__index = PipelineCache

function PipelineCache.new(ctx, opts)
    opts = opts or {}
    return setmetatable({
        ctx = ctx,
        opts = opts,
        by_key = {},
        order = {},
        pool = 0,
        pool_size = 0,
    }, PipelineCache)
end
_M.PipelineCache = PipelineCache

function PipelineCache:get(profile)
    local key = _M.profile_key(profile)
    local cached = self.by_key[key]
    if cached then
        return cached, nil, true
    end
    local pipe, err = _M.build_pipeline(self.ctx, profile, self.opts)
    if not pipe then
        return nil, err, false
    end
    self.by_key[key] = pipe
    self.order[#self.order + 1] = key
    return pipe, nil, false
end

-- ensure_pool(): a descriptor pool large enough for live sets. Called lazily.
function PipelineCache:ensure_pool()
    if self.pool ~= 0 then return self.pool, nil end
    local fn = self.ctx.fn.device
    local ps = ffi.new('VkDescriptorPoolSize[1]')
    ps[0].type = VK.DESCRIPTOR_TYPE_STORAGE_BUFFER
    ps[0].descriptorCount = 1024
    local dpci = ffi.new('VkDescriptorPoolCreateInfo')
    dpci.sType = VK.STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
    dpci.maxSets = 256
    dpci.poolSizeCount = 1
    dpci.pPoolSizes = ps
    local pool = ffi.new('VkDescriptorPool[1]')
    local res = fn.createDescriptorPool(self.ctx.device, dpci, nil, pool)
    if res ~= VK.SUCCESS then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkCreateDescriptorPool failed: ' .. res)
    end
    self.pool = pool[0]
    self.pool_size = 1024
    return pool[0], nil
end

-- create_descriptor_set(ctx, pipe, cache, buffers) -> set | nil, err
-- buffers: { input, payload, scale, offset, special, output, config } VkBuffer
-- handles (from the arenas). Allocates ONE set and writes all 7 storage-buffer
-- descriptors. Callers SHOULD reuse the set across dispatches (updating
-- bindings via update_descriptor_set) so the descriptor pool never exhausts:
-- the pool is bounded and a fresh set per dispatch leaks it.
function _M.create_descriptor_set(ctx, pipe, cache, buffers)
    local fn = ctx.fn.device
    local pool, perr = cache:ensure_pool()
    if not pool then return nil, perr end

    local dsai = ffi.new('VkDescriptorSetAllocateInfo')
    dsai.sType = VK.STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO
    dsai.descriptorPool = pool
    dsai.descriptorSetCount = 1
    local layout = ffi.new('VkDescriptorSetLayout[1]', pipe.set_layout)
    dsai.pSetLayouts = layout
    local set = ffi.new('VkDescriptorSet[1]')
    local res = fn.allocateDescriptorSets(ctx.device, dsai, set)
    if res ~= VK.SUCCESS then
        return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkAllocateDescriptorSets failed: ' .. res)
    end
    local ok, werr = _M.update_descriptor_set(ctx, set[0], buffers)
    if not ok then return nil, werr end
    return set[0]
end

-- update_descriptor_set(ctx, set, buffers) -> true | nil, err
-- Rebinds the 7 storage-buffer descriptors of an existing set. Cheap; used to
-- reuse one set per pipeline across dispatches when a worker buffer grew or
-- the arenas reallocated. Safe because the worker waits for its fence before
-- the next dispatch (never updates a set still in flight).
function _M.update_descriptor_set(ctx, set, buffers)
    local fn = ctx.fn.device
    local writes = ffi.new('VkWriteDescriptorSet[7]')
    local binfo = ffi.new('VkDescriptorBufferInfo[7]')
    for i = 0, 6 do
        binfo[i].buffer = buffers[i + 1]
        binfo[i].offset = 0
        binfo[i].range = VK.VK_WHOLE_SIZE -- VK_WHOLE_SIZE (rest of the buffer)
        writes[i].sType = VK.STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
        writes[i].dstSet = set
        writes[i].dstBinding = i
        writes[i].dstArrayElement = 0
        writes[i].descriptorCount = 1
        writes[i].descriptorType = VK.DESCRIPTOR_TYPE_STORAGE_BUFFER
        writes[i].pBufferInfo = binfo + i
    end
    fn.updateDescriptorSets(ctx.device, 7, writes, 0, nil)
    return true
end

-- record_dispatch(ctx, cmd, pipe, set, num_cells) — records the compute
-- dispatch for one network (one uniform profile) into `cmd`. Caller must have
-- written config + inputs and must submit with a fence.
function _M.record_dispatch(ctx, cmd, pipe, set, num_cells)
    local fn = ctx.fn.device
    local groups = math.ceil(num_cells / 64)
    fn.cmdBindPipeline(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline)
    local setarr = ffi.new('VkDescriptorSet[1]', set)
    fn.cmdBindDescriptorSets(cmd, VK.PIPELINE_BIND_POINT_COMPUTE, pipe.layout, 0, 1, setarr, 0, nil)
    fn.cmdDispatch(cmd, groups, 1, 1)
end

-- drop(): destroys pipelines, layouts, set layouts and the pool.
function PipelineCache:drop()
    if not self.ctx or not ffi then
        self.by_key = {}
        self.order = {}
        return
    end
    local fn = self.ctx.fn.device
    for _, key in ipairs(self.order) do
        local p = self.by_key[key]
        if p then
            if fn.destroyPipeline then fn.destroyPipeline(self.ctx.device, p.pipeline, nil) end
            if fn.destroyPipelineLayout then fn.destroyPipelineLayout(self.ctx.device, p.layout, nil) end
            if fn.destroyDescriptorSetLayout then fn.destroyDescriptorSetLayout(self.ctx.device, p.set_layout, nil) end
        end
    end
    if self.pool ~= 0 and fn.destroyDescriptorPool then
        fn.destroyDescriptorPool(self.ctx.device, self.pool, nil)
        self.pool = 0
    end
    self.by_key = {}
    self.order = {}
end

return _M
