-- nn/vulkan_worker.lua — GPU worker thread lifecycle (fences, drain, shutdown).
--
-- The worker exists only when backend=gpu. It owns its own Vulkan context
-- (instance/device/queue/command pool are created on the worker), a pipeline
-- cache, an arena set, one command buffer and one fence.
--
-- Threading: love.thread is used IF available (LÖVE); otherwise a same-thread
-- passthrough keeps the module functional in plain LuaJIT. Across the thread
-- boundary we transfer ONLY plain Lua numeric buffers/bytes — NEVER FFI cdata
-- (cdata cannot cross threads: each love.thread has its own Lua state and its
-- own FFI namespace; a cdata pointer is meaningless there). Similarly only
-- plain io paths are used (love.thread has no love.filesystem).
--
-- Tick identity: every batch carries tick_id + model_hash + backend +
-- precision + row mapping; results are tagged with the same identity and stale
-- results (wrong tick/model) are rejected by wait().
--
-- Device lost: vkQueueSubmit/vkWaitForFences returning VK_ERROR_DEVICE_LOST
-- yields a structured DEVICE_LOST error — there is NEVER a silent fallback to
-- CPU. Shutdown drain: stop accepting -> drain queue -> wait fences ->
-- vkDeviceWaitIdle -> release arenas/pipelines -> terminate.

local vulkan = require('nn.vulkan')
local pipelines = require('nn.vulkan_pipelines')
local arenas = require('nn.vulkan_arenas')
local ffi = vulkan.ffi
local mkerr = vulkan.mkerr
local ERR = vulkan.ERRORS
local VK = vulkan.VK

local _M = { _VERSION = '1.0.0', name = 'vulkan_worker' }

local FENCE_TIMEOUT_NS = 10 ^ 9 -- 1 s per dispatch

-- love.thread availability (never required).
local has_love_thread = false
do
    local ok = pcall(require, 'love.thread')
    has_love_thread = ok
end
_M.has_love_thread = has_love_thread

-- ------------------------------------------------------ worker internals ----
local Worker = {}
Worker.__index = Worker

-- create_host_buffer(ctx, bytes, usage) -> {buffer, memory, f32, u32} | nil, err
local function create_host_buffer(ctx, bytes, usage)
    local fn = ctx.fn.device
    local bci = ffi.new('VkBufferCreateInfo')
    bci.sType = VK.STRUCTURE_TYPE_BUFFER_CREATE_INFO
    bci.size = bytes
    bci.usage = usage
    bci.sharingMode = 0
    local buf = ffi.new('VkBuffer[1]')
    local res = fn.createBuffer(ctx.device, bci, nil, buf)
    if res ~= VK.SUCCESS then
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkCreateBuffer failed: ' .. res)
    end
    local reqs = ffi.new('VkMemoryRequirements')
    fn.getBufferMemoryRequirements(ctx.device, buf[0], reqs)
    local mem_type = vulkan.pick_memory_type(ctx.fn.instance.getPhysicalDeviceMemoryProperties,
        ctx.physical, reqs.memoryTypeBits, VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    if not mem_type then
        mem_type = vulkan.pick_memory_type(ctx.fn.instance.getPhysicalDeviceMemoryProperties,
            ctx.physical, reqs.memoryTypeBits, VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    end
    if not mem_type then
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'no suitable memory type for worker buffer')
    end
    local aci = ffi.new('VkMemoryAllocateInfo')
    aci.sType = VK.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
    aci.allocationSize = reqs.size
    aci.memoryTypeIndex = mem_type
    local mem = ffi.new('VkDeviceMemory[1]')
    res = fn.allocateMemory(ctx.device, aci, nil, mem)
    if res ~= VK.SUCCESS then
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkAllocateMemory failed: ' .. res)
    end
    res = fn.bindBufferMemory(ctx.device, buf[0], mem[0], 0)
    if res ~= VK.SUCCESS then
        fn.freeMemory(ctx.device, mem[0], nil)
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkBindBufferMemory failed: ' .. res)
    end
    local ptr = ffi.new('void*[1]')
    res = fn.mapMemory(ctx.device, mem[0], 0, bytes, 0, ptr)
    if res ~= VK.SUCCESS then
        fn.freeMemory(ctx.device, mem[0], nil)
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkMapMemory failed: ' .. res)
    end
    return {
        bytes = bytes,
        buffer = buf[0],
        memory = mem[0],
        f32 = ffi.cast('float*', ptr[0]),
        u32 = ffi.cast('uint32_t*', ptr[0]),
    }, nil
end

-- new(opts) -> worker | nil, err
-- opts: { spv_dir, max_arena_bytes, experimental_per_layer_dispatch }
-- Creates its own device context on the worker (love.thread: in the thread;
-- passthrough: on the calling thread).
function Worker.new(opts)
    opts = opts or {}
    local ctx, cerr = vulkan.init(opts)
    if not ctx then
        return nil, cerr
    end
    local w, werr = Worker._wrap(ctx, opts)
    if not w then
        vulkan.destroy(ctx)
        return nil, werr
    end
    return w
end

-- ensure_buffer(kind, bytes) -> buffer | nil, err — grow on demand.
function Worker:ensure_buffer(kind, bytes)
    local b = self.buffers[kind]
    if b and b.bytes >= bytes then return b, nil end
    if b then
        local fn = self.ctx.fn.device
        fn.unmapMemory(self.ctx.device, b.memory)
        fn.freeMemory(self.ctx.device, b.memory, nil)
        fn.destroyBuffer(self.ctx.device, b.buffer, nil)
        self.buffers[kind] = nil
    end
    local usage = kind == 'config'
        and VK.BUFFER_USAGE_STORAGE_BUFFER_BIT
        or (VK.BUFFER_USAGE_STORAGE_BUFFER_BIT | VK.BUFFER_USAGE_TRANSFER_SRC_BIT)
    local nb, err = create_host_buffer(self.ctx, bytes, usage)
    if not nb then return nil, err end
    self.buffers[kind] = nb
    return nb, nil
end

-- get_descriptor_set(pipe, buffers) -> set | nil, err
-- Returns the ONE cached descriptor set for a pipeline, creating it on first
-- use. Reuse keeps descriptor-pool usage bounded (a fresh set per dispatch
-- exhausts the pool after ~150 dispatches). Bindings are refreshed in place
-- every dispatch so buffer growth/arena reallocations stay coherent; safe
-- because each dispatch waits for the worker's fence before the next one.
function Worker:get_descriptor_set(pipe, buffers)
    local cached = self.descriptor_sets[pipe.key]
    if cached then
        local ok, err = pipelines.update_descriptor_set(self.ctx, cached.set, buffers)
        if not ok then return nil, err end
        return cached.set, nil
    end
    local set, derr = pipelines.create_descriptor_set(self.ctx, pipe, self.cache, buffers)
    if not set then return nil, derr end
    self.descriptor_sets[pipe.key] = { set = set, pipe = pipe }
    return set, nil
end

-- --------------------------------------------------------- batch execution --
-- run_batch(batch) -> result table (never raises). A batch:
--   { tick_id, model_hash, backend='vulkan', precision, profile={...},
--     layer_map={...}?, rows={ {genome_id, profile, item} , ... },
--     config={ numNodes, fanIn }, inputs={...}, num_cells }
-- Result:
--   { tick_id, model_hash, backend, precision, ok=true, rows=..., outputs={...} }
-- or { tick_id, ok=false, err_class, message }.
function Worker:run_batch(batch)
    local result = {
        tick_id = batch.tick_id,
        model_hash = batch.model_hash,
        backend = batch.backend or 'vulkan',
        precision = batch.precision,
        rows = batch.rows,
    }
    local function fail(err)
        result.ok = false
        result.err_class = err.class or 'VULKAN_INITIALIZATION_FAILED'
        result.message = err.message or tostring(err)
        return result
    end

    if not self.alive then
        return fail(mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'worker is shut down'))
    end
    if self.device_lost then
        return fail(mkerr(ERR.DEVICE_LOST, 'worker marked device-lost; no further work accepted'))
    end

    -- 1. Uniform-profile gate (N2): nonuniform per-layer maps are rejected
    --    unless the explicit experimental flag is set.
    if batch.layer_map then
        local ok, uerr = pipelines.validate_uniform_profile(batch.layer_map, self.opts)
        if not ok then return fail(uerr) end
    end

    -- 2. Pipeline: cached per profile, NEVER rebuilt per tick.
    local profile = batch.profile or {}
    local pipe, perr, cached = self.cache:get(profile)
    if not pipe then return fail(perr) end

    -- 3. Stage rows into the arenas (cached by genome identity + profile).
    -- Per-tick batch descriptor per item:
    --   { network_id, profile_id, payload_base, scale_base, offset_base, special_base }
    local staged_rows = {}
    for i, row in ipairs(batch.rows or {}) do
        local row_profile = row.profile or profile
        local bases, serr = self.arenas:stage(row.genome_id, row_profile, row.item)
        if not bases then return fail(serr) end
        staged_rows[i] = {
            genome_id = row.genome_id,
            network_id = row_profile.network_id,
            profile_id = pipelines.profile_key(row_profile),
            payload_base = bases.payload_base,
            scale_base = bases.scale_base,
            offset_base = bases.offset_base,
            special_base = bases.special_base,
            row_start = row.row_start or 0,
            row_count = row.row_count or batch.num_cells,
        }
    end

    -- 4. Input/output/config buffers.
    local num_cells = batch.num_cells or 1
    local num_nodes = batch.config and batch.config.numNodes or (batch.rows and #batch.rows)
    local fan_in = batch.config and batch.config.fanIn or 1
    local inputs = batch.inputs -- plain Lua array, cell-major: cell* fanIn + k
    local input_bytes = math.max(1, num_cells * fan_in * 4)
    local output_bytes = math.max(1, num_cells * num_nodes * 4)
    -- Config SSBO (v2 ABI): 8 header words + 4 words of per-cell bases
    -- (payload/scale/offset/special) per batch row.
    local config_bytes = math.max(1, (8 + 4 * num_cells) * 4)
    local in_buf, ierr = self:ensure_buffer('input', input_bytes)
    if not in_buf then return fail(ierr) end
    local out_buf, oerr = self:ensure_buffer('output', output_bytes)
    if not out_buf then return fail(oerr) end
    local cfg_buf, cerr = self:ensure_buffer('config', config_bytes)
    if not cfg_buf then return fail(cerr) end

    -- Map each cell to the staged row covering it. The runtime stages one row
    -- per cell (distinct per-cell genomes); a single row with row_count=N
    -- (all cells sharing one genome) also maps every cell to that row.
    local cell_rows = {}
    for c = 1, num_cells do
        local idx = c - 1
        local found = nil
        for _, sr in ipairs(staged_rows) do
            local start = sr.row_start or 0
            local count = sr.row_count or batch.num_cells or 1
            if idx >= start and idx < start + count then found = sr break end
        end
        cell_rows[c] = found or staged_rows[1]
    end

    -- Config words (see shader ABI v2, forward_packed.comp):
    --   cfg[0..1] numNodes/fanIn, cfg[2..3] input/output strides,
    --   cfg[4] numCells, cfg[5] flags (bit0: activate the input layer on the
    --   first-layer dispatch), cfg[6..7] input/output base (cells stacked from 0),
    --   cfg[8 + c*4 ..] per-cell payloadBase (BYTE offset into the payload
    --   arena) and scaleBase/offsetBase/specialBase (FLOAT indices into their
    --   arenas — arena bases are byte offsets, hence /4).
    cfg_buf.u32[0] = num_nodes
    cfg_buf.u32[1] = fan_in
    cfg_buf.u32[2] = fan_in -- inputStride (floats per cell)
    cfg_buf.u32[3] = num_nodes -- outputStride (floats per cell)
    cfg_buf.u32[4] = num_cells
    cfg_buf.u32[5] = batch.config and batch.config.flags or 0
    cfg_buf.u32[6] = 0 -- inputBase: cells stacked from 0
    cfg_buf.u32[7] = 0 -- outputBase: cells stacked from 0
    for c = 1, num_cells do
        local b = cell_rows[c]
        local row = 8 + (c - 1) * 4
        cfg_buf.u32[row]     = b and b.payload_base or 0
        cfg_buf.u32[row + 1] = b and math.floor(b.scale_base / 4) or 0
        cfg_buf.u32[row + 2] = b and math.floor(b.offset_base / 4) or 0
        cfg_buf.u32[row + 3] = b and math.floor(b.special_base / 4) or 0
    end

    -- Inputs: cell-major flat array (plain Lua numbers).
    local n_in = num_cells * fan_in
    for i = 1, n_in do
        in_buf.f32[i - 1] = inputs[i] or 0
    end

    -- 5. Descriptor set for (pipe, buffers) — REUSED per pipeline across
    --    dispatches (MUST-2): allocating a fresh set per dispatch exhausts the
    --    bounded pool. Bindings are updated in place when buffers change.
    local buffers = {
        in_buf.buffer,         -- 0 input
        self.arenas.arenas.payload.buffer, -- 1 payload
        self.arenas.arenas.scale.buffer,   -- 2 scale
        self.arenas.arenas.offset.buffer,  -- 3 offset
        self.arenas.arenas.special.buffer, -- 4 special
        out_buf.buffer,        -- 5 output
        cfg_buf.buffer,        -- 6 config
    }
    local set, derr = self:get_descriptor_set(pipe, buffers)
    if not set then return fail(derr) end

    -- 6. Record the dispatch.
    local fn = self.ctx.fn.device
    local cb = self.cmd
    local bbi = ffi.new('VkCommandBufferBeginInfo')
    bbi.sType = VK.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
    bbi.flags = VK.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    local res = fn.beginCommandBuffer(cb, bbi)
    if res ~= VK.SUCCESS then return fail(mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkBeginCommandBuffer failed: ' .. res)) end

    -- Pre-dispatch barrier: host writes (arena payload/scale/offset/special,
    -- input, config) visible to the compute shader.
    local mb = ffi.new('VkMemoryBarrier[1]')
    mb[0].sType = VK.STRUCTURE_TYPE_MEMORY_BARRIER
    mb[0].srcAccessMask = VK.ACCESS_HOST_WRITE_BIT
    mb[0].dstAccessMask = VK.ACCESS_SHADER_READ_BIT | VK.ACCESS_SHADER_WRITE_BIT
    fn.cmdPipelineBarrier(cb, VK.PIPELINE_STAGE_HOST_BIT, VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, mb, 0, nil, 0, nil)

    pipelines.record_dispatch(self.ctx, cb, pipe, set, num_cells)

    -- Post-dispatch barrier: shader output visible to the host for readback.
    mb[0].srcAccessMask = VK.ACCESS_SHADER_WRITE_BIT
    mb[0].dstAccessMask = VK.ACCESS_HOST_READ_BIT
    fn.cmdPipelineBarrier(cb, VK.PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK.PIPELINE_STAGE_HOST_BIT,
        0, 1, mb, 0, nil, 0, nil)

    res = fn.endCommandBuffer(cb)
    if res ~= VK.SUCCESS then return fail(mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkEndCommandBuffer failed: ' .. res)) end

    -- 7. Submit + wait.
    local si = ffi.new('VkSubmitInfo')
    si.sType = VK.STRUCTURE_TYPE_SUBMIT_INFO
    si.commandBufferCount = 1
    local cbs = ffi.new('VkCommandBuffer[1]', cb)
    si.pCommandBuffers = cbs
    res = fn.queueSubmit(self.ctx.queue, 1, si, self.fence)
    if res == VK.ERROR_DEVICE_LOST then
        self.device_lost = true
        return fail(mkerr(ERR.DEVICE_LOST, 'vkQueueSubmit returned VK_ERROR_DEVICE_LOST'))
    end
    if res ~= VK.SUCCESS then return fail(mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkQueueSubmit failed: ' .. res)) end

    res = fn.waitForFences(self.ctx.device, 1, ffi.new('VkFence[1]', self.fence), 1, FENCE_TIMEOUT_NS)
    if res == VK.ERROR_DEVICE_LOST then
        self.device_lost = true
        return fail(mkerr(ERR.DEVICE_LOST, 'vkWaitForFences returned VK_ERROR_DEVICE_LOST'))
    end
    if res ~= VK.SUCCESS then return fail(mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkWaitForFences failed: ' .. res)) end
    fn.resetFences(self.ctx.device, 1, ffi.new('VkFence[1]', self.fence))

    -- 8. Readback: plain Lua numeric buffers (thread-transferable).
    -- Shader layout is 0-based: cell c occupies [(c-1)*numNodes, c*numNodes).
    local outputs = {}
    for c = 1, num_cells do
        local cell_out = {}
        for node = 0, num_nodes - 1 do
            cell_out[node + 1] = out_buf.f32[(c - 1) * num_nodes + node]
        end
        outputs[c] = cell_out
    end
    result.ok = true
    result.outputs = outputs
    result.num_cells = num_cells
    result.num_nodes = num_nodes
    result.cached_pipeline = cached
    result.staged_rows = staged_rows
    self.last_submitted_tick = batch.tick_id
    return result
end

-- ------------------------------------------------------------ passthrough ----
-- Same-thread worker: used when love.thread is unavailable (plain LuaJIT). The
-- public API mirrors the threaded worker (submit/wait/shutdown) so callers are
-- agnostic.
local Passthrough = {}
Passthrough.__index = Passthrough

function Passthrough.new(opts)
    local inner, err = Worker.new(opts)
    if not inner then return nil, err end
    return setmetatable({
        inner = inner,
        results = {},
        alive = true,
    }, Passthrough)
end

function Passthrough:submit(batch)
    if not self.alive then
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'worker shutdown')
    end
    local res = self.inner:run_batch(batch)
    self.results[#self.results + 1] = res
    return true
end

-- wait(expected_tick, model_hash) -> results. Stale results (wrong tick or
-- model) are rejected: kept results must match both, otherwise dropped.
function Passthrough:wait(expected_tick, model_hash)
    local out = {}
    local keep = {}
    for _, r in ipairs(self.results) do
        if (expected_tick == nil or r.tick_id == expected_tick) and
           (model_hash == nil or r.model_hash == model_hash) then
            out[#out + 1] = r
        else
            keep[#keep + 1] = r -- stale: dropped on next wait
        end
    end
    self.results = keep
    return out
end

-- shutdown(): drain (nothing in-flight in passthrough — synchronous), wait for
-- the device, then release arenas/pipelines/command buffer/fence/context.
function Passthrough:shutdown()
    if not self.alive then return true end
    self.alive = false
    local inner = self.inner
    local fn = inner.ctx.fn.device
    if not inner.device_lost then
        local ok = pcall(fn.deviceWaitIdle, inner.ctx.device)
        if not ok then
            -- device lost during drain: proceed to release handles
        end
    end
    inner.cache:drop()
    inner.arenas:drop()
    inner.descriptor_sets = {} -- pool destroyed with the cache; sets are dead
    for _, b in pairs(inner.buffers) do
        if b then
            fn.unmapMemory(inner.ctx.device, b.memory)
            fn.freeMemory(inner.ctx.device, b.memory, nil)
            fn.destroyBuffer(inner.ctx.device, b.buffer, nil)
        end
    end
    inner.buffers = {}
    fn.freeCommandBuffers(inner.ctx.device, inner.ctx.command_pool, 1, ffi.new('VkCommandBuffer[1]', inner.cmd))
    fn.destroyFence(inner.ctx.device, inner.fence, nil)
    vulkan.destroy(inner.ctx)
    return true
end

-- --------------------------------------------------------------- LÖVE ----
-- love.thread worker: the device context is created INSIDE the thread. Only
-- plain Lua data crosses the channels (never FFI cdata — cdata cannot cross
-- threads). Shutdown drains the channel, waits, and terminates the thread.
local THREAD_SOURCE = [[
-- nn/vulkan_worker thread body (love.thread). Plain data only across channels.
local in_ch = love.thread.getChannel(ARGS[1])
local out_ch = love.thread.getChannel(ARGS[2])
local function emit(t) out_ch:push(t) end

local vulkan = require('nn.vulkan')
local worker = require('nn.vulkan_worker')
local ctx, err = vulkan.init()
if not ctx then
    emit({ type = 'ready', ok = false, err_class = err.class, message = err.message })
    return
end
local w = worker._passthrough_on(ctx)
if not w then
    vulkan.destroy(ctx)
    emit({ type = 'ready', ok = false, err_class = 'VULKAN_INITIALIZATION_FAILED', message = 'worker wrap failed' })
    return
end
emit({ type = 'ready', ok = true })

while true do
    local batch = in_ch:demand()
    if batch and batch.type == 'shutdown' then break end
    if batch then
        local res = w:submit(batch)
        emit({ type = 'result', result = res })
    end
end
-- w:shutdown() destroys the context; no second vulkan.destroy here.
w:shutdown()
emit({ type = 'bye' })
]]

local LoveThread = {}
LoveThread.__index = LoveThread

function LoveThread.new(opts)
    if not has_love_thread then
        return nil, mkerr(ERR.VULKAN_UNSUPPORTED_PLATFORM, 'love.thread not available')
    end
    local love_thread = require('love.thread')
    local suffix = tostring(math.random(1, 10 ^ 9))
    local in_ch = love_thread.newChannel('nn_vk_in_' .. suffix)
    local out_ch = love_thread.newChannel('nn_vk_out_' .. suffix)
    local th = love_thread.newThread(THREAD_SOURCE, in_ch:getName(), out_ch:getName())
    th:start()

    local self = setmetatable({
        thread = th,
        in_ch = in_ch,
        out_ch = out_ch,
        alive = true,
    }, LoveThread)

    -- Wait for the worker to be ready (or report init failure).
    local ready = out_ch:demand()
    if not (ready and ready.ok) then
        th:kill()
        self.alive = false
        return nil, mkerr(ready and ready.err_class or ERR.VULKAN_INITIALIZATION_FAILED,
            (ready and ready.message) or 'worker thread init failed')
    end
    return self
end

function LoveThread:submit(batch)
    if not self.alive then
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'worker shutdown')
    end
    self.in_ch:push(batch)
    return true
end

function LoveThread:wait(expected_tick, model_hash)
    local out = {}
    while true do
        local msg = self.out_ch:pop()
        if not msg then break end
        if msg.type == 'result' then
            local r = msg.result
            if (expected_tick == nil or r.tick_id == expected_tick) and
               (model_hash == nil or r.model_hash == model_hash) then
                out[#out + 1] = r
            end
        end
    end
    return out
end

function LoveThread:shutdown()
    if not self.alive then return true end
    self.alive = false
    self.in_ch:push({ type = 'shutdown' })
    self.thread:join()
    return true
end

-- ---------------------------------------------------------------- factory ----
-- new(opts) -> worker | nil, err. Prefers love.thread; falls back to the
-- same-thread passthrough so the module works in plain LuaJIT.
function _M.new(opts)
    opts = opts or {}
    if has_love_thread then
        local w, err = LoveThread.new(opts)
        if w then return w end
        -- fall through to passthrough if the thread path failed
    end
    return Passthrough.new(opts)
end

-- _passthrough_on(ctx): used by the love.thread body to wrap an already
-- initialized context into the same-thread executor.
function _M._passthrough_on(ctx)
    local inner, err = Worker._wrap(ctx, nil)
    if not inner then return nil, err end
    return setmetatable({ inner = inner, results = {}, alive = true }, Passthrough)
end

-- Worker._wrap(ctx): build a Worker around a pre-existing context (thread path).
function Worker._wrap(ctx, opts)
    opts = opts or {}
    local pcache = pipelines.PipelineCache.new(ctx, { spv_dir = opts.spv_dir })
    local aset, aerr = arenas.ArenaSet.new(ctx, opts)
    if not aset then return nil, aerr end
    local cb = ffi.new('VkCommandBuffer[1]')
    local cbai = ffi.new('VkCommandBufferAllocateInfo')
    cbai.sType = VK.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
    cbai.commandPool = ctx.command_pool
    cbai.level = VK.COMMAND_BUFFER_LEVEL_PRIMARY
    cbai.commandBufferCount = 1
    local res = ctx.fn.device.allocateCommandBuffers(ctx.device, cbai, cb)
    if res ~= VK.SUCCESS then
        aset:drop()
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkAllocateCommandBuffers failed: ' .. res)
    end
    local fence = ffi.new('VkFence[1]')
    local fci = ffi.new('VkFenceCreateInfo')
    fci.sType = VK.STRUCTURE_TYPE_FENCE_CREATE_INFO
    res = ctx.fn.device.createFence(ctx.device, fci, nil, fence)
    if res ~= VK.SUCCESS then
        ctx.fn.device.freeCommandBuffers(ctx.device, ctx.command_pool, 1, cb)
        aset:drop()
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkCreateFence failed: ' .. res)
    end
    return setmetatable({
        ctx = ctx,
        opts = opts,
        cache = pcache,
        arenas = aset,
        cmd = cb[0],
        fence = fence[0],
        buffers = {},
        descriptor_sets = {},
        results = {},
        alive = true,
        device_lost = false,
    }, Worker)
end

return _M
