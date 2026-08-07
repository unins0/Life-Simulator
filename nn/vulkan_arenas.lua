-- nn/vulkan_arenas.lua — separate VkBuffer arenas, per-entry alignment, staging.
--
-- Four independent VkBuffers, each bound at OFFSET ZERO:
--   payload_arena (uint8 weights, packed bytes)
--   scale_arena   (float, one per block)
--   offset_arena  (float, one per block; used by fp4/fp2)
--   special_arena (float, {bias, threshold, dead} per node)
-- Separate buffers at offset 0 sidestep minStorageBufferOffsetAlignment
-- (commonly 256 bytes). IF a combined-buffer path is ever added it MUST query
-- limits.minStorageBufferOffsetAlignment and align each entry base to it.
--
-- Per-entry alignment: payload entries are aligned to 4 bytes (a block-8 fp2
-- entry can be 30 bytes, e.g. 120 weights / 4 per byte -> 30 bytes -> rounded
-- to 32); scale/offset/special entries are aligned to their element size (4).
--
-- Staging is per-tick: the caller packs changed genomes on the CPU and uploads
-- changed ranges; a (genome identity, profile) cache reuses already-staged
-- entries instead of re-uploading. Arena growth reallocates to a larger buffer
-- and copies the old contents; VULKAN_ARENA_EXHAUSTED if a configured
-- max_arena_bytes is exceeded.

local vulkan = require('nn.vulkan')
local pipelines = require('nn.vulkan_pipelines')
local ffi = vulkan.ffi
local mkerr = vulkan.mkerr
local ERR = vulkan.ERRORS
local VK = vulkan.VK

local _M = { _VERSION = '1.0.0', name = 'vulkan_arenas' }

local ARENA_NAMES = { 'payload', 'scale', 'offset', 'special' }
_M.ARENA_NAMES = ARENA_NAMES

function _M.align_up(n, alignment)
    alignment = alignment or 4
    if alignment <= 1 then return n end
    local r = n % alignment
    return r == 0 and n or (n + alignment - r)
end

-- pick_memory_type(ctx, type_bits, required_props) -> index | nil
local function pick_memory_type(ctx, type_bits, required)
    return vulkan.pick_memory_type(ctx.fn.instance.getPhysicalDeviceMemoryProperties,
        ctx.physical, type_bits, required)
end

-- create_buffer(ctx, size, usage) -> buffer, memory, mem_props, err
-- mem_props: VkMemoryPropertyFlags of the chosen type (host_visible/coherent).
local function create_buffer(ctx, size, usage)
    local fn = ctx.fn.device
    local bci = ffi.new('VkBufferCreateInfo')
    bci.sType = VK.STRUCTURE_TYPE_BUFFER_CREATE_INFO
    bci.size = size
    bci.usage = usage
    bci.sharingMode = 0 -- VK_SHARING_MODE_EXCLUSIVE
    local buf = ffi.new('VkBuffer[1]')
    local res = fn.createBuffer(ctx.device, bci, nil, buf)
    if res ~= VK.SUCCESS then
        return nil, nil, nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkCreateBuffer failed: ' .. res)
    end

    local reqs = ffi.new('VkMemoryRequirements')
    fn.getBufferMemoryRequirements(ctx.device, buf[0], reqs)

    -- Prefer host-visible memory (CPU uploads weights each tick); fall back to
    -- device-local (slower upload path via staging, not implemented in the
    -- skeleton — host-visible is the expected path on desktop/APU drivers).
    local mem_props
    local mem_type = pick_memory_type(ctx, reqs.memoryTypeBits, VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    if mem_type then
        mem_props = VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT
    else
        mem_type = pick_memory_type(ctx, reqs.memoryTypeBits, VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        mem_props = VK.MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    end
    if not mem_type then
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, nil, nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'no suitable memory type for arena buffer')
    end

    -- Read the exact property flags of the chosen type (coherence matters for
    -- whether we must flush/invalidate mapped ranges).
    local props = ffi.new('VkPhysicalDeviceMemoryProperties')
    ctx.fn.instance.getPhysicalDeviceMemoryProperties(ctx.physical, props)
    mem_props = props.memoryTypes[mem_type].propertyFlags

    local aci = ffi.new('VkMemoryAllocateInfo')
    aci.sType = VK.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
    aci.allocationSize = reqs.size
    aci.memoryTypeIndex = mem_type
    local mem = ffi.new('VkDeviceMemory[1]')
    res = fn.allocateMemory(ctx.device, aci, nil, mem)
    if res ~= VK.SUCCESS then
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, nil, nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkAllocateMemory failed: ' .. res)
    end
    res = fn.bindBufferMemory(ctx.device, buf[0], mem[0], 0)
    if res ~= VK.SUCCESS then
        fn.freeMemory(ctx.device, mem[0], nil)
        fn.destroyBuffer(ctx.device, buf[0], nil)
        return nil, nil, nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkBindBufferMemory failed: ' .. res)
    end
    return buf[0], mem[0], mem_props, nil
end

-- ---------------------------------------------------------------- Arena ----
local Arena = {}
Arena.__index = Arena

function Arena.new(ctx, name, capacity, usage, opts)
    local max_bytes = opts.max_arena_bytes or (256 << 20) -- 256 MiB default cap
    -- The initial allocation must respect max_arena_bytes so the exhaust guard
    -- is consistent from the first alloc onward.
    if capacity > max_bytes then capacity = max_bytes end
    local self = setmetatable({
        ctx = ctx,
        name = name,
        capacity = capacity,
        usage = usage,
        max_bytes = max_bytes,
        cursor = 0,
        buffer = 0,
        memory = 0,
        mapped = nil,
        host_visible = false,
        coherent = false,
    }, Arena)
    local ok, err = self:_create(capacity)
    if not ok then
        return nil, err
    end
    return self
end

function Arena:_create(capacity)
    local fn = self.ctx.fn.device
    local buf, mem, mem_props, err = create_buffer(self.ctx, capacity, self.usage)
    if not buf then return nil, err end
    self.buffer = buf
    self.memory = mem
    self.capacity = capacity
    self.host_visible = (mem_props & VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT) ~= 0
    self.coherent = (mem_props & VK.MEMORY_PROPERTY_HOST_COHERENT_BIT) ~= 0

    if self.host_visible then
        local ptr = ffi.new('void*[1]')
        local res = fn.mapMemory(self.ctx.device, mem, 0, capacity, 0, ptr)
        if res ~= VK.SUCCESS then
            return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkMapMemory failed for arena ' .. self.name)
        end
        self.mapped = ffi.cast('uint8_t*', ptr[0])
    end
    return true
end

-- alloc(bytes, alignment) -> base (byte offset within the arena) | nil, err
function Arena:alloc(bytes, alignment)
    local base = _M.align_up(self.cursor, alignment or 4)
    local need = base + bytes
    if need > self.capacity then
        local ok, err = self:_grow(need)
        if not ok then return nil, err end
    end
    self.cursor = need
    return base
end

-- Grow by reallocating to a larger buffer and copying the old contents.
-- The copy must preserve committed bytes regardless of memory type:
--   host-visible -> host-visible: map both and memcpy;
--   anything else (device-local fallback): vkCmdCopyBuffer via a one-shot
--   command buffer + fence (the old contents are NOT silently dropped).
function Arena:_grow(need)
    if need > self.max_bytes then
        return nil, mkerr(ERR.VULKAN_ARENA_EXHAUSTED,
            ('arena %q needs %d bytes, exceeds max_arena_bytes=%d'):format(self.name, need, self.max_bytes))
    end
    local new_capacity = self.capacity
    while new_capacity < need do
        new_capacity = new_capacity * 2
    end
    if new_capacity > self.max_bytes then new_capacity = self.max_bytes end

    local old_buf, old_mem = self.buffer, self.memory
    local fn = self.ctx.fn.device
    local buf, mem, mem_props, err = create_buffer(self.ctx, new_capacity, self.usage)
    if not buf then return nil, err end
    local new_host_visible = (mem_props & VK.MEMORY_PROPERTY_HOST_VISIBLE_BIT) ~= 0

    -- Copy old contents into the new allocation.
    if self.mapped and new_host_visible then
        -- Fast path: map both and memcpy.
        local new_ptr = ffi.new('void*[1]')
        local res = fn.mapMemory(self.ctx.device, mem, 0, new_capacity, 0, new_ptr)
        if res ~= VK.SUCCESS then
            fn.freeMemory(self.ctx.device, mem, nil)
            fn.destroyBuffer(self.ctx.device, buf, nil)
            return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkMapMemory failed during arena grow')
        end
        ffi.copy(new_ptr[0], self.mapped, self.cursor) -- keep committed bytes
        fn.unmapMemory(self.ctx.device, mem)
    elseif self.cursor > 0 then
        -- Device-local-only path: device-side copy. Flush any unflushed host
        -- writes first so the copy sees the committed bytes.
        if self.host_visible and not self.coherent then
            local mr = ffi.new('VkMappedMemoryRange')
            mr.sType = VK.STRUCTURE_TYPE_MAPPED_MEMORY_RANGE
            mr.memory = old_mem
            mr.offset = 0
            mr.size = VK.VK_WHOLE_SIZE
            fn.flushMappedMemoryRanges(self.ctx.device, 1, mr)
        end
        local ok, cerr = copy_buffer_device(self.ctx, buf, old_buf, self.cursor)
        if not ok then
            fn.freeMemory(self.ctx.device, mem, nil)
            fn.destroyBuffer(self.ctx.device, buf, nil)
            return nil, cerr
        end
    end

    -- Release the old buffer/memory (only unmap what was actually mapped).
    if self.mapped then fn.unmapMemory(self.ctx.device, old_mem) end
    fn.freeMemory(self.ctx.device, old_mem, nil)
    fn.destroyBuffer(self.ctx.device, old_buf, nil)

    self.buffer = buf
    self.memory = mem
    self.capacity = new_capacity
    self.host_visible = new_host_visible
    self.coherent = (mem_props & VK.MEMORY_PROPERTY_HOST_COHERENT_BIT) ~= 0
    self.mapped = nil
    if self.host_visible then
        local ptr = ffi.new('void*[1]')
        local res = fn.mapMemory(self.ctx.device, mem, 0, new_capacity, 0, ptr)
        if res ~= VK.SUCCESS then
            return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED, 'vkMapMemory failed after arena grow')
        end
        self.mapped = ffi.cast('uint8_t*', ptr[0])
    end
    return true
end

-- write(base, data): copy bytes (string or array of byte numbers) into the
-- arena at `base`. No-op when the arena is not host-visible.
function Arena:write(base, data, len)
    if not self.mapped then return end
    local n = len or #data
    if type(data) == 'string' then
        ffi.copy(self.mapped + base, data, n)
    else
        local tmp = ffi.new('uint8_t[?]', n)
        for i = 1, n do tmp[i - 1] = data[i] end
        ffi.copy(self.mapped + base, tmp, n)
    end
    self:_flush(base, n)
end

-- write_f32(base, floats): copy a Lua array of numbers as float32.
function Arena:write_f32(base, floats, len)
    if not self.mapped then return end
    local n = len or #floats
    local tmp = ffi.new('float[?]', n)
    for i = 1, n do tmp[i - 1] = floats[i] end
    ffi.copy(self.mapped + base, tmp, n * 4)
    self:_flush(base, n * 4)
end

-- Flush host writes to the device when the memory is not coherent. We flush
-- the whole allocation (offset 0) to stay clear of nonCoherentAtomSize
-- alignment constraints for partial ranges.
function Arena:_flush(base, bytes)
    if self.coherent or not self.host_visible then return end
    local mr = ffi.new('VkMappedMemoryRange')
    mr.sType = VK.STRUCTURE_TYPE_MAPPED_MEMORY_RANGE
    mr.memory = self.memory
    mr.offset = 0
    mr.size = VK.VK_WHOLE_SIZE -- VK_WHOLE_SIZE
    self.ctx.fn.device.flushMappedMemoryRanges(self.ctx.device, 1, mr)
end

function Arena:drop()
    local fn = self.ctx.fn.device
    if self.mapped then fn.unmapMemory(self.ctx.device, self.memory) end
    if self.memory ~= 0 then fn.freeMemory(self.ctx.device, self.memory, nil) end
    if self.buffer ~= 0 then fn.destroyBuffer(self.ctx.device, self.buffer, nil) end
    self.mapped = nil
    self.memory = 0
    self.buffer = 0
    self.cursor = 0
end

-- ------------------------------------------------------------- ArenaSet ----
-- The four arenas of a worker, plus the (genome identity, profile) staging
-- cache that reuses already-uploaded entries.
local ArenaSet = {}
ArenaSet.__index = ArenaSet

local ARENA_USAGE = {
    payload = VK.BUFFER_USAGE_STORAGE_BUFFER_BIT | VK.BUFFER_USAGE_TRANSFER_SRC_BIT
        | VK.BUFFER_USAGE_TRANSFER_DST_BIT,
    scale = VK.BUFFER_USAGE_STORAGE_BUFFER_BIT | VK.BUFFER_USAGE_TRANSFER_SRC_BIT
        | VK.BUFFER_USAGE_TRANSFER_DST_BIT,
    offset = VK.BUFFER_USAGE_STORAGE_BUFFER_BIT | VK.BUFFER_USAGE_TRANSFER_SRC_BIT
        | VK.BUFFER_USAGE_TRANSFER_DST_BIT,
    special = VK.BUFFER_USAGE_STORAGE_BUFFER_BIT | VK.BUFFER_USAGE_TRANSFER_SRC_BIT
        | VK.BUFFER_USAGE_TRANSFER_DST_BIT,
}

-- Fence timeout for the one-shot grow-copy submission (1 s).
local FENCE_TIMEOUT_NS = 10 ^ 9

-- copy_buffer_device(ctx, dst_buf, src_buf, bytes): copy old arena contents
-- into a new (possibly device-local) buffer with a one-shot command buffer +
-- fence. Used by Arena:_grow when the fast host-visible memcpy path is
-- unavailable. Returns true or nil, err.
local function copy_buffer_device(ctx, dst_buf, src_buf, bytes)
    local fn = ctx.fn.device
    local cb = ffi.new('VkCommandBuffer[1]')
    local cbai = ffi.new('VkCommandBufferAllocateInfo')
    cbai.sType = VK.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
    cbai.commandPool = ctx.command_pool
    cbai.level = VK.COMMAND_BUFFER_LEVEL_PRIMARY
    cbai.commandBufferCount = 1
    local res = fn.allocateCommandBuffers(ctx.device, cbai, cb)
    if res ~= VK.SUCCESS then
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED,
            'vkAllocateCommandBuffers failed during arena grow copy: ' .. res)
    end
    local fence = ffi.new('VkFence[1]')
    local fci = ffi.new('VkFenceCreateInfo')
    fci.sType = VK.STRUCTURE_TYPE_FENCE_CREATE_INFO
    res = fn.createFence(ctx.device, fci, nil, fence)
    if res ~= VK.SUCCESS then
        fn.freeCommandBuffers(ctx.device, ctx.command_pool, 1, cb)
        return nil, mkerr(ERR.VULKAN_INITIALIZATION_FAILED,
            'vkCreateFence failed during arena grow copy: ' .. res)
    end
    local function cleanup()
        fn.destroyFence(ctx.device, fence[0], nil)
        fn.freeCommandBuffers(ctx.device, ctx.command_pool, 1, cb)
    end
    local bbi = ffi.new('VkCommandBufferBeginInfo')
    bbi.sType = VK.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
    bbi.flags = VK.COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    res = fn.beginCommandBuffer(cb[0], bbi)
    if res ~= VK.SUCCESS then cleanup() return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkBeginCommandBuffer failed: ' .. res) end
    -- Make prior writes (host uploads or earlier transfers) visible to the copy.
    local mb = ffi.new('VkMemoryBarrier[1]')
    mb[0].sType = VK.STRUCTURE_TYPE_MEMORY_BARRIER
    mb[0].srcAccessMask = VK.ACCESS_HOST_WRITE_BIT | VK.ACCESS_TRANSFER_WRITE_BIT
    mb[0].dstAccessMask = VK.ACCESS_TRANSFER_READ_BIT
    fn.cmdPipelineBarrier(cb[0], VK.PIPELINE_STAGE_HOST_BIT | VK.PIPELINE_STAGE_TRANSFER_BIT,
        VK.PIPELINE_STAGE_TRANSFER_BIT, 0, 1, mb, 0, nil, 0, nil)
    local region = ffi.new('VkBufferCopy[1]')
    region[0].srcOffset = 0
    region[0].dstOffset = 0
    region[0].size = bytes
    fn.cmdCopyBuffer(cb[0], src_buf, dst_buf, 1, region)
    res = fn.endCommandBuffer(cb[0])
    if res ~= VK.SUCCESS then cleanup() return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkEndCommandBuffer failed: ' .. res) end
    local si = ffi.new('VkSubmitInfo')
    si.sType = VK.STRUCTURE_TYPE_SUBMIT_INFO
    si.commandBufferCount = 1
    local cbs = ffi.new('VkCommandBuffer[1]', cb[0])
    si.pCommandBuffers = cbs
    res = fn.queueSubmit(ctx.queue, 1, si, fence[0])
    if res ~= VK.SUCCESS then cleanup() return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkQueueSubmit failed: ' .. res) end
    res = fn.waitForFences(ctx.device, 1, fence, 1, FENCE_TIMEOUT_NS)
    cleanup()
    if res ~= VK.SUCCESS then return nil, mkerr(ERR.VULKAN_PIPELINE_FAILED, 'vkWaitForFences failed: ' .. res) end
    return true
end

function ArenaSet.new(ctx, opts)
    opts = opts or {}
    local self = setmetatable({
        ctx = ctx,
        opts = opts,
        arenas = {},
        staged = {}, -- key (genome_id|profile_key) -> {payload_base, scale_base, offset_base, special_base}
    }, ArenaSet)
    for _, name in ipairs(ARENA_NAMES) do
        local capacity = opts[name .. '_capacity'] or opts.capacity or (4 << 20)
        local arena, err = Arena.new(ctx, name, capacity, ARENA_USAGE[name], opts)
        if not arena then return nil, err end
        self.arenas[name] = arena
    end
    return self
end

-- stage(genome_id, profile, item) -> bases | nil, err
-- item: { payload = bytes|string, scales = {..}, offsets = {..}, specials = {..} }
-- Uploads changed data (payload bytes; scale/offset/special floats) into the
-- arenas and returns per-entry bases, cached by (genome identity, profile) so
-- unchanged genomes are never re-uploaded per tick.
function ArenaSet:stage(genome_id, profile, item)
    local key = tostring(genome_id) .. '|' .. pipelines.profile_key(profile)
    local cached = self.staged[key]
    if cached then return cached, nil, true end

    local payload = self.arenas.payload
    local scale = self.arenas.scale
    local offset = self.arenas.offset
    local special = self.arenas.special

    -- Payload: byte-length may not be a multiple of 4 (e.g. 120 fp2 weights at
    -- block 8 = 30 bytes); align the entry to 4 so the GPU word reads are safe.
    local payload_len = item.payload_len
    if not payload_len then
        payload_len = type(item.payload) == 'string' and #item.payload or #item.payload
    end
    local payload_base, err = payload:alloc(payload_len, 4)
    if not payload_base then return nil, err end
    if type(item.payload) == 'string' then
        payload:write(payload_base, item.payload, payload_len)
    elseif profile.format == 'fp32' or profile.format == 'fp16' then
        -- fp32/fp16 payloads are float arrays (fp32 weights); write as floats.
        payload:write_f32(payload_base, item.payload, math.floor(payload_len / 4))
    else
        payload:write(payload_base, item.payload, payload_len)
    end

    local scale_base, err = scale:alloc(#item.scales * 4, 4)
    if not scale_base then return nil, err end
    scale:write_f32(scale_base, item.scales, #item.scales)

    local offset_base = 0
    if item.offsets and #item.offsets > 0 then
        offset_base, err = offset:alloc(#item.offsets * 4, 4)
        if not offset_base then return nil, err end
        offset:write_f32(offset_base, item.offsets, #item.offsets)
    end

    local special_base, err = special:alloc(#item.specials * 4, 4)
    if not special_base then return nil, err end
    special:write_f32(special_base, item.specials, #item.specials)

    local bases = {
        payload_base = payload_base,
        scale_base = scale_base,
        offset_base = offset_base,
        special_base = special_base,
        cache_key = key,
    }
    self.staged[key] = bases
    return bases, nil, false
end

-- drop(): release all four arenas (buffers, memory, mappings).
function ArenaSet:drop()
    for _, name in ipairs(ARENA_NAMES) do
        if self.arenas[name] then self.arenas[name]:drop() end
    end
    self.arenas = {}
    self.staged = {}
end

_M.ArenaSet = ArenaSet
_M.Arena = Arena
-- Exported for tests: the one-shot device copy used by Arena:_grow on the
-- device-local-only fallback path.
_M.copy_buffer_device = copy_buffer_device

return _M
