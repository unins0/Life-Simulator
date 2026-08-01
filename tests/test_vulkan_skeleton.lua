-- tests/test_vulkan_skeleton.lua — tests for the nn/vulkan backend skeleton.
--
-- Conventions (see tests/test_runner.lua): the file returns a table of
-- { name = function() ... end }; assertions raise on failure; the runner
-- reports PASS/FAIL and exits non-zero on any failure. Runs under BOTH plain
-- `lua` (no ffi) and `luajit` (ffi). Tests that need ffi/a loader skip
-- (pass) with a printed note when unavailable.
--
-- Coverage (per the vulkan skeleton spec):
--   (1) module import never fails
--   (2) init() returns nil + structured error instead of raising
--   (3) capabilities reflect reality
--   (4) ABI golden scaffolding (sizeof/offsetof vs pinned values)
--   (5) loadByte math: pure-Lua reference for fp8/fp4/fp2 bit ordering
--   (6) shader source: '<=' activation, word-addressed loadByte, 'not <'
--   (7) nonuniform layer map -> runtime rejection error path
--   (8) pipeline cache key stability
--   (+) optional real-GPU end-to-end dispatch when a loader + .spv exist

local vulkan = require('nn.vulkan')
local pipelines = require('nn.vulkan_pipelines')
local arenas = require('nn.vulkan_arenas')
local worker_mod = require('nn.vulkan_worker')

-- ---------------------------------------------------------------- helpers --
local function file_exists(path)
    local f = io.open(path, 'rb')
    if f then f:close() return true end
    return false
end

local function read_file(path)
    local f = io.open(path, 'rb')
    if not f then return nil end
    local data = f:read('*a')
    f:close()
    return data
end

-- Pure-Lua mirrors of the shader decode math (see forward_packed.comp).
local function pack_words_le(bytes)
    local words = {}
    for i = 1, #bytes, 4 do
        local w = 0
        for j = 0, 3 do
            w = w | ((bytes[i + j] or 0) << (j * 8))
        end
        if w < 0 then w = w + 2 ^ 32 end -- LuaJIT bitwise ops are signed int32
        words[#words + 1] = w
    end
    return words
end

-- Word-addressed byte load, identical math to the shader loadByte():
--   byte at addr = (payload_u[addr >> 2] >> ((addr & 3) << 3)) & 0xFF
local function loadByte(payload_u, addr)
    local word = payload_u[math.floor(addr / 4) + 1] -- 1-based Lua array
    return (word >> ((addr % 4) * 8)) & 0xFF
end

local function decode_fp8(payload_bytes, payload_base, logical_index, scale)
    local q = loadByte(pack_words_le(payload_bytes), payload_base + logical_index) - 128
    return scale * q
end

local function decode_fp4(payload_bytes, payload_base, logical_index, offset, scale)
    local byte_index = math.floor(logical_index / 2)
    local shift = (logical_index % 2) * 4
    local q = (loadByte(pack_words_le(payload_bytes), payload_base + byte_index) >> shift) & 0xF
    return offset + scale * q
end

local function decode_fp2(payload_bytes, payload_base, logical_index, offset, scale)
    local byte_index = math.floor(logical_index / 4)
    local shift = (logical_index % 4) * 2
    local q = (loadByte(pack_words_le(payload_bytes), payload_base + byte_index) >> shift) & 0x3
    return offset + scale * q
end

-- Reference forward for one node with packed weights (fp8 model).
local function reference_forward_fp8(payload_bytes, fan_in, inputs, scale, bias, threshold, dead)
    local acc = 0
    for k = 0, fan_in - 1 do
        acc = acc + decode_fp8(payload_bytes, 0, k, scale) * inputs[k + 1]
    end
    local value = acc + bias
    if value <= threshold then value = dead end -- inclusive dead-zone, NOT '<'
    return value
end

-- ------------------------------------------------------------------ tests --
local tests = {}

tests['module import never fails'] = function()
    assert(type(vulkan) == 'table')
    assert(type(pipelines) == 'table')
    assert(type(arenas) == 'table')
    assert(type(worker_mod) == 'table')
    assert(vulkan.name == 'vulkan')
end

tests['init returns structured error or success, never raises'] = function()
    local ok_load = vulkan.can_load()
    if not ok_load then
        -- Loader/ffi absent: init() MUST return nil + structured error.
        local ctx, err = vulkan.init()
        assert(ctx == nil, 'init() must return nil when the loader is missing')
        assert(type(err) == 'table' and err.class, 'init() must return a structured error table')
        assert(err.class == vulkan.ERRORS.VULKAN_UNSUPPORTED_PLATFORM
            or err.class == vulkan.ERRORS.VULKAN_INITIALIZATION_FAILED,
            ('unexpected error class: %s'):format(tostring(err.class)))
        return
    end
    -- Loader present: attempt a real init; success or structured error both pass.
    local ctx, err, already = vulkan.init()
    if ctx then
        print(('NOTE: loader present; init() succeeded on %q (already=%s)'):format(
            tostring(ctx.props and ctx.props.name), tostring(already)))
        vulkan.destroy(ctx)
    else
        assert(type(err) == 'table' and err.class,
            'init() must return a structured error, not raise')
        print(('NOTE: loader present but init() failed: %s %s'):format(
            tostring(err.class), tostring(err.message)))
    end
end

tests['capabilities reflect reality'] = function()
    local caps = vulkan.query_capabilities()
    assert(type(caps) == 'table', 'capabilities must be a table')
    assert(type(caps.vulkan) == 'boolean', 'caps.vulkan must be boolean')
    assert(type(caps.caps) == 'table' and type(caps.extensions) == 'table',
        'caps.caps and caps.extensions must be tables')
    assert(caps.caps.word_addressed_payload == true,
        'word-addressed uint[] payload must be the mandatory primary path')
    assert(caps.backend == 'gpu')
    if not caps.vulkan then
        print('NOTE: capabilities report no Vulkan (loader/ICD absent or not initialized)')
    else
        assert(type(caps.device) == 'table' and caps.device.name, 'device info missing')
        assert(type(caps.extensions.sixteen_bit_storage) == 'boolean')
        assert(caps.caps.f16_arithmetic == caps.extensions.shader_float16_int8,
            'f16 arithmetic must be gated on VK_KHR_shader_float16_int8')
    end
end

tests['ABI golden scaffolding (sizeof/offsetof)'] = function()
    if not vulkan.ffi then
        print('SKIP: no ffi (plain Lua) — ABI assertions are LuaJIT-only')
        return
    end
    local ok, err = vulkan.abi_check()
    assert(ok, ('ABI mismatch: %s %s'):format(
        tostring(err and err.class), tostring(err and err.message)))
    -- Spot checks against pinned values (LP64), independent of abi_check.
    local ffi = vulkan.ffi
    if ffi.abi('64bit') then
        assert(ffi.sizeof('VkApplicationInfo') == 48)
        assert(ffi.offsetof('VkApplicationInfo', 'apiVersion') == 44)
        assert(ffi.sizeof('VkBufferCreateInfo') == 56)
        assert(ffi.offsetof('VkBufferCreateInfo', 'size') == 24)
        assert(ffi.sizeof('VkSubmitInfo') == 72)
        assert(ffi.sizeof('VkDescriptorBufferInfo') == 24)
        assert(ffi.sizeof('VkSpecializationMapEntry') == 16)
    end
    -- The golden table covers the full spec'd struct set.
    local count = 0
    for _ in pairs(vulkan.ABI_GOLDEN) do count = count + 1 end
    assert(count >= 27, ('ABI golden table covers %d structs, expected >= 27'):format(count))
end

tests['loadByte math matches the shader (fp8/fp4/fp2 bit ordering)'] = function()
    -- Word layout is little-endian: byte at addr N is bits (N%4)*8 of word N//4.
    local bytes = { 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 }
    local words = pack_words_le(bytes)
    assert(words[1] == 0x78563412, 'word 0 must be little-endian packed')
    assert(words[2] == 0xF0DEBC9A, 'word 1 must be little-endian packed')
    assert(loadByte(words, 0) == 0x12)
    assert(loadByte(words, 1) == 0x34)
    assert(loadByte(words, 3) == 0x78)
    assert(loadByte(words, 4) == 0x9A)
    assert(loadByte(words, 5) == 0xBC)
    assert(loadByte(words, 7) == 0xF0)

    -- fp8: q = byte - 128, per-block scale.
    assert(decode_fp8({ 132, 130, 131 }, 0, 0, 0.5) == 2.0) -- (132-128)*0.5
    assert(decode_fp8({ 132, 130, 131 }, 0, 1, 0.5) == 1.0)
    assert(decode_fp8({ 132, 130, 131 }, 0, 2, 0.5) == 1.5)
    assert(decode_fp8({ 0, 255, 1 }, 0, 0, 1.0) == -128.0)
    assert(decode_fp8({ 0, 255, 1 }, 0, 1, 1.0) == 127.0)

    -- fp4: two weights per byte; even logicalIndex -> LOW nibble, odd -> HIGH.
    local b = 0xAB -- low nibble 0xB (11), high nibble 0xA (10)
    local fb = { b }
    assert(decode_fp4(fb, 0, 0, 0.0, 1.0) == 11, 'even fp4 index must read the low nibble')
    assert(decode_fp4(fb, 0, 1, 0.0, 1.0) == 10, 'odd fp4 index must read the high nibble')
    assert(decode_fp4(fb, 0, 0, 2.0, 0.5) == 2.0 + 0.5 * 11, 'fp4 offset+scale')

    -- fp2: four weights per byte, low bits first: 0b11_10_01_00.
    local f2 = { 0xE4 } -- 0b11100100
    assert(decode_fp2(f2, 0, 0, 0.0, 1.0) == 0)
    assert(decode_fp2(f2, 0, 1, 0.0, 1.0) == 1)
    assert(decode_fp2(f2, 0, 2, 0.0, 1.0) == 2)
    assert(decode_fp2(f2, 0, 3, 0.0, 1.0) == 3)

    -- Reference forward with dead-zone (inclusive '<=').
    assert(reference_forward_fp8({ 132 }, 1, { 1 }, 1.0, 0.0, 2.0, 7.7) == 4.0,
        '4.0 > 2.0 must NOT trigger the dead zone')
    assert(reference_forward_fp8({ 132 }, 1, { 0.5 }, 1.0, 0.0, 2.0, 7.7) == 7.7,
        '2.0 <= 2.0 must trigger the dead zone (inclusive comparison)')
end

tests['forward_packed.comp source constraints'] = function()
    local src = read_file('nn/shaders/forward_packed.comp')
    assert(src, 'nn/shaders/forward_packed.comp must exist')
    assert(src:find('shader_abi', 1, true), 'shader must carry a shader_abi version comment')
    -- Activation: inclusive '<=' dead-zone, not '<'-only.
    assert(src:find('value <= threshold', 1, true),
        "activation must compare with '<=' (inclusive dead-zone)")
    assert(src:find("not <", 1, true) and not src:find('value < threshold', 1, true),
        "shader must carry a comment asserting the comparison is not '<'-only")
    -- Word-addressed loadByte on a uint[] payload.
    assert(src:find('payload_u', 1, true), 'payload SSBO must be uint[] (word-addressed)')
    assert(src:find('>> 2u', 1, true), 'loadByte must use byteAddr >> 2 for the word index')
    assert(src:find('0xFF', 1, true), 'loadByte must mask with 0xFF')
    assert(src:find('byteAddr & 3u', 1, true), 'loadByte must use (addr & 3) for the lane')
    -- Block decomposition formulas.
    assert(src:find('logicalIndex >> BLOCK_LOG2', 1, true), 'block = logicalIndex >> blockLog2')
    assert(src:find('inBlock', 1, true), 'inBlock must be computed')
    assert(src:find('logicalIndex & 1u) << 2u', 1, true), 'fp4 shift = (logicalIndex & 1) << 2')
    assert(src:find('logicalIndex & 3u) << 1u', 1, true), 'fp2 shift = (logicalIndex & 3) << 1')
end

tests['nonuniform layer map is runtime-rejected'] = function()
    local uniform = {
        { format = 'fp8', block_size = 16 },
        { format = 'fp8', block_size = 16 },
        { format = 'fp8', block_size = 16 },
    }
    local res, uerr = pipelines.validate_uniform_profile(uniform)
    assert(res and res.uniform == true, 'uniform layer map must validate')
    assert(uerr == nil)

    local nonuniform = {
        { format = 'fp8', block_size = 16 },
        { format = 'fp8', block_size = 16 },
        { format = 'fp4', block_size = 16 }, -- different format at layer 3
    }
    local ok, err = pipelines.validate_uniform_profile(nonuniform)
    assert(ok == nil, 'nonuniform map must be rejected at runtime')
    assert(type(err) == 'table' and err.class == vulkan.ERRORS.VULKAN_PIPELINE_FAILED,
        'rejection must be a structured VULKAN_PIPELINE_FAILED error')
    assert((err.message or ''):find('per_layer') or (err.message or ''):find('per%-layer'),
        'rejection message must mention the per-layer dispatch flag')

    -- The explicit experimental flag unlocks per-layer dispatch.
    local allowed = pipelines.validate_uniform_profile(nonuniform,
        { experimental_per_layer_dispatch = true })
    assert(allowed ~= nil, 'experimental_per_layer_dispatch must allow nonuniform maps')
end

tests['pipeline cache key is stable and canonical'] = function()
    local function profile(network_id, format, block_size)
        return {
            network_id = network_id,
            format = format,
            block_size = block_size,
            compute_format = 'fp32',
            scale_type = 'fp32',
            offset_type = 'fp32',
            quant_abi = 1,
        }
    end
    local k1 = pipelines.profile_key(profile('seed', 'fp8', 16))
    local k2 = pipelines.profile_key(profile('seed', 'fp8', 16))
    assert(k1 == k2, 'identical profiles must produce identical keys')
    assert(k1 ~= pipelines.profile_key(profile('seed', 'fp4', 16)), 'format must affect the key')
    assert(k1 ~= pipelines.profile_key(profile('seed', 'fp8', 8)), 'block size must affect the key')
    assert(k1 ~= pipelines.profile_key(profile('spore', 'fp8', 16)), 'network_id must affect the key')

    local p = profile('spore', 'fp4', 32)
    p.quant_abi = 2
    assert(k1 ~= pipelines.profile_key(p), 'quant ABI version must affect the key')

    -- All components must be present in the key.
    local key = pipelines.profile_key(profile('seed', 'fp8', 16))
    for _, part in ipairs({ 'seed', 'fp8', '16', 'fp32', '1' }) do
        assert(key:find(part, 1, true), ('key must contain %q'):format(part))
    end
end

tests['arena staging aligns payload entries and caches by genome+profile'] = function()
    -- Pure-Lua parts of the arena logic: per-entry alignment + staging keys.
    assert(arenas.align_up(0, 4) == 0)
    assert(arenas.align_up(1, 4) == 4)
    assert(arenas.align_up(30, 4) == 32, '120 fp2 weights / block8 = 30 bytes -> align to 32')
    assert(arenas.align_up(32, 4) == 32)
    -- Staging cache key derivation mirrors (genome identity, profile).
    local p = { network_id = 'seed', format = 'fp8', block_size = 16, quant_abi = 1 }
    local key_a = 'g42|' .. pipelines.profile_key(p)
    local key_b = 'g42|' .. pipelines.profile_key(p)
    local key_c = 'g43|' .. pipelines.profile_key(p)
    assert(key_a == key_b, 'same genome+profile must reuse the same staging slot')
    assert(key_a ~= key_c, 'different genome must use a different staging slot')
end

tests['optional real-GPU end-to-end dispatch'] = function()
    if not vulkan.ffi or not vulkan.can_load() then
        print('SKIP: no Vulkan loader/ffi')
        return
    end
    if not file_exists('nn/shaders/spv/forward_packed.spv') then
        print('SKIP: no compiled .spv (build artifacts; see nn/shaders/spv/README.md)')
        return
    end
    local w, werr = worker_mod.new()
    if not w then
        print(('SKIP: worker init failed (%s %s)'):format(
            tostring(werr and werr.class), tostring(werr and werr.message)))
        return
    end
    local ok_shutdown = false
    local ok, serr = w:submit({
        tick_id = 42,
        model_hash = 'e2e',
        backend = 'vulkan',
        precision = 'fp8',
        profile = {
            network_id = 'test', format = 'fp8', block_size = 8,
            compute_format = 'fp32', scale_type = 'fp32',
            offset_type = 'fp32', quant_abi = 1,
        },
        num_cells = 2,
        config = { numNodes = 2, fanIn = 3 },
        inputs = { 1, 2, 3, 4, 5, 6 },
        rows = {
            { genome_id = 'g1', row_start = 0, row_count = 2,
              item = {
                  payload = string.char(132, 130, 131, 140, 138, 137),
                  payload_len = 6,
                  scales = { 0.5 },
                  offsets = {},
                  specials = { 0.1, 0.5, 0.0, 0.2, 0.7, 0.0 },
              } },
        },
    })
    if not ok then
        w:shutdown()
        print(('SKIP: submit failed (%s %s)'):format(tostring(serr), tostring(ok)))
        return
    end
    local results = w:wait(42, 'e2e')
    w:shutdown()
    ok_shutdown = true
    assert(#results == 1, 'expected exactly one result for tick 42')
    local r = results[1]
    assert(r.ok, ('GPU run failed: %s %s'):format(tostring(r.err_class), tostring(r.message)))
    assert(r.tick_id == 42 and r.model_hash == 'e2e', 'result must carry tick identity')
    assert(r.backend == 'vulkan' and r.precision == 'fp8', 'result must carry backend+precision')

    -- Verify against the pure-Lua reference. The 6-byte payload decodes to
    -- q=(4,2,3,12,10,9) with scale 0.5; the matrix is stored INPUT-MAJOR
    -- (input_node * out_c + output_node), so per output node the weights are
    -- {4,3,10} (inputs 1..3 -> node 1) and {2,12,9} (inputs 1..3 -> node 2).
    local function ref(fan_in, inputs, qs, scale, bias, threshold, dead)
        local acc = 0
        for k = 0, fan_in - 1 do
            acc = acc + (scale * qs[k + 1]) * inputs[k + 1]
        end
        local value = acc + bias
        if value <= threshold then value = dead end
        return value
    end
    for c = 1, 2 do
        local inputs = c == 1 and { 1, 2, 3 } or { 4, 5, 6 }
        local exp1 = ref(3, inputs, { 4, 3, 10 }, 0.5, 0.1, 0.5, 0.0)
        local exp2 = ref(3, inputs, { 2, 12, 9 }, 0.5, 0.2, 0.7, 0.0)
        assert(math.abs(r.outputs[c][1] - exp1) < 1e-4,
            ('cell %d node1: GPU %.6f vs reference %.6f'):format(c, r.outputs[c][1], exp1))
        assert(math.abs(r.outputs[c][2] - exp2) < 1e-4,
            ('cell %d node2: GPU %.6f vs reference %.6f'):format(c, r.outputs[c][2], exp2))
    end
    print('NOTE: real-GPU fp8 dispatch verified against the Lua reference')
end

tests['stale results are rejected by tick identity'] = function()
    if not vulkan.ffi or not vulkan.can_load() then
        print('SKIP: no Vulkan loader/ffi')
        return
    end
    local w, werr = worker_mod.new()
    if not w then
        print(('SKIP: worker init failed (%s)'):format(tostring(werr and werr.message)))
        return
    end
    local profile = { network_id = 't', format = 'fp8', block_size = 8,
        compute_format = 'fp32', scale_type = 'fp32', offset_type = 'fp32', quant_abi = 1 }
    local function mk(tick, hash, inputs)
        return {
            tick_id = tick, model_hash = hash, backend = 'vulkan', precision = 'fp8',
            profile = profile, num_cells = 1, config = { numNodes = 1, fanIn = 1 },
            inputs = inputs,
            rows = { { genome_id = ('g' .. tick), row_start = 0, row_count = 1,
                item = { payload = string.char(132), payload_len = 1,
                         scales = { 1.0 }, offsets = {}, specials = { 0.0, -1e9, 0.0 } } } },
        }
    end
    w:submit(mk(1, 'h1', { 1 }))
    w:submit(mk(2, 'h2', { 2 }))
    -- Waiting for tick 2 must reject tick 1 as stale.
    local res = w:wait(2, 'h2')
    assert(#res == 1 and res[1].tick_id == 2, 'stale results must be rejected')
    w:shutdown()
end

-- MUST-2 regression: descriptor sets are REUSED per pipeline, so hundreds of
-- dispatches must not exhaust the bounded descriptor pool (~150 sets before
-- the fix bricked the worker permanently).
tests['descriptor sets are reused across 600 dispatches'] = function()
    if not vulkan.ffi or not vulkan.can_load() then
        print('SKIP: no Vulkan loader/ffi')
        return
    end
    if not file_exists('nn/shaders/spv/forward_packed.spv') then
        print('SKIP: no compiled .spv (build artifacts; see nn/shaders/spv/README.md)')
        return
    end
    local w, werr = worker_mod.new()
    if not w then
        print(('SKIP: worker init failed (%s %s)'):format(
            tostring(werr and werr.class), tostring(werr and werr.message)))
        return
    end
    local profile = { network_id = 't', format = 'fp8', block_size = 8,
        compute_format = 'fp32', scale_type = 'fp32', offset_type = 'fp32', quant_abi = 1 }
    for tick = 1, 600 do
        local ok, serr = w:submit({
            tick_id = tick, model_hash = 'stress', backend = 'vulkan', precision = 'fp8',
            profile = profile, num_cells = 1, config = { numNodes = 1, fanIn = 1 },
            inputs = { 1 },
            rows = { { genome_id = 'gs', row_start = 0, row_count = 1,
                item = { payload = string.char(132), payload_len = 1,
                         scales = { 1.0 }, offsets = {}, specials = { 0.0, -1e9, 0.0 } } } },
        })
        assert(ok, ('dispatch %d must be accepted (%s)'):format(tick, tostring(serr)))
    end
    local results = w:wait(600, 'stress')
    assert(#results == 1, ('expected the final tick result, got %d'):format(#results))
    local r = results[1]
    assert(r and r.ok, ('stress dispatch failed: %s %s'):format(
        tostring(r and r.err_class), tostring(r and r.message)))
    assert(r.outputs and r.outputs[1] and math.abs(r.outputs[1][1] - 4.0) < 1e-4,
        ('stress output must stay correct (got %s)'):format(tostring(r.outputs and r.outputs[1][1])))
    -- Bounded growth: 600 dispatches through ONE pipeline must have allocated
    -- exactly ONE descriptor set (passthrough worker exposes inner.state).
    if w.inner and w.inner.descriptor_sets then
        local n = 0
        for _ in pairs(w.inner.descriptor_sets) do n = n + 1 end
        assert(n == 1, ('descriptor sets must be reused (found %d)'):format(n))
    end
    w:shutdown()
    print('NOTE: 600 real-GPU dispatches completed with a single reused descriptor set')
end

-- SHOULD-6 regression: arena growth must preserve already-staged contents
-- (the old code silently dropped them on device-local-only drivers).
tests['arena grow preserves staged contents'] = function()
    if not vulkan.ffi or not vulkan.can_load() then
        print('SKIP: no Vulkan loader/ffi')
        return
    end
    local ctx, cerr = vulkan.init()
    if not ctx then
        print(('SKIP: init failed (%s %s)'):format(tostring(cerr and cerr.class), tostring(cerr and cerr.message)))
        return
    end
    -- Tiny capacity forces the payload arena to grow within a few stages.
    local aset, aerr = arenas.ArenaSet.new(ctx, { capacity = 256, max_arena_bytes = 1 << 20 })
    if not aset then
        vulkan.destroy(ctx)
        print(('SKIP: ArenaSet init failed (%s)'):format(tostring(aerr and aerr.message)))
        return
    end
    local profile = { network_id = 'g', format = 'fp8', block_size = 8,
        compute_format = 'fp32', scale_type = 'fp32', offset_type = 'fp32', quant_abi = 1 }
    local first
    for i = 1, 4 do
        -- ~200-byte payload entries; the 256-byte arena must grow on stage 2+.
        local payload = {}
        for b = 1, 200 do payload[b] = (i * 37 + b) % 256 end
        local payload_s = string.char(unpack(payload))
        local item = {
            payload = payload_s,
            payload_len = #payload_s,
            scales = { 0.5 },
            offsets = {},
            specials = { 0.1, 0.5, 0.0 },
        }
        local bases, serr = aset:stage(('g%d'):format(i), profile, item)
        assert(bases, ('stage %d failed: %s'):format(i, tostring(serr and serr.message)))
        if i == 1 then first = { bases = bases, payload = payload_s } end
    end
    -- The payload arena must have grown beyond the initial 256 bytes.
    assert(aset.arenas.payload.capacity > 256, 'arena must have grown during staging')
    -- Contents of the FIRST entry must survive the growth unchanged.
    local arena = aset.arenas.payload
    assert(arena.host_visible,
        'this driver must expose a host-visible arena for the readback (skip device-local-only)')
    local mem = vulkan.ffi.cast('const uint8_t*', arena.mapped)
    for i = 1, #first.payload do
        assert(mem[first.bases.payload_base + i - 1] == first.payload:byte(i),
            ('arena grow lost payload byte %d'):format(i))
    end
    aset:drop()
    vulkan.destroy(ctx)
    print('NOTE: arena grow preserved all staged payload bytes')
end

-- SHOULD-6: the device-side grow copy (vkCmdCopyBuffer path) must copy bytes
-- exactly, independent of the driver's memory type (forced directly).
tests['device copy buffer path preserves bytes'] = function()
    if not vulkan.ffi or not vulkan.can_load() then
        print('SKIP: no Vulkan loader/ffi')
        return
    end
    local ctx, cerr = vulkan.init()
    if not ctx then
        print(('SKIP: init failed (%s %s)'):format(tostring(cerr and cerr.class), tostring(cerr and cerr.message)))
        return
    end
    -- ArenaSet buffers carry STORAGE | TRANSFER_SRC | TRANSFER_DST usage.
    local aset, aerr = arenas.ArenaSet.new(ctx, { capacity = 512, max_arena_bytes = 1 << 20 })
    if not aset then
        vulkan.destroy(ctx)
        print(('SKIP: ArenaSet init failed (%s)'):format(tostring(aerr and aerr.message)))
        return
    end
    local bytes = {}
    for i = 1, 64 do bytes[i] = (i * 7 + 3) % 256 end
    local blob = string.char(unpack(bytes))
    local sa = aset.arenas.payload
    local da = aset.arenas.scale
    sa:write(0, blob, #blob)
    local ok, copyerr = arenas.copy_buffer_device(ctx, da.buffer, sa.buffer, #blob)
    assert(ok, ('device copy failed: %s'):format(tostring(copyerr and copyerr.message)))
    if not da.host_visible or not da.mapped then
        aset:drop()
        vulkan.destroy(ctx)
        print('SKIP: destination arena is device-local-only (no host readback on this driver)')
        return
    end
    -- Invalidate the destination host cache, then read back and compare.
    if not da.coherent then
        local mr = vulkan.ffi.new('VkMappedMemoryRange')
        mr.sType = vulkan.VK.STRUCTURE_TYPE_MAPPED_MEMORY_RANGE
        mr.memory = da.memory
        mr.offset = 0
        mr.size = vulkan.VK.VK_WHOLE_SIZE
        ctx.fn.device.invalidateMappedMemoryRanges(ctx.device, 1, mr)
    end
    local mem = vulkan.ffi.cast('const uint8_t*', da.mapped)
    for i = 1, #blob do
        assert(mem[i - 1] == blob:byte(i),
            ('device copy lost byte %d'):format(i))
    end
    aset:drop()
    vulkan.destroy(ctx)
    print('NOTE: vkCmdCopyBuffer arena path verified byte-for-byte')
end

return tests
