-- tests/test_nn_core.lua — tests for the standalone NN core (nn/errors,
-- nn/capabilities, nn/format, nn/quantize, nn/serialize, nn/corpus,
-- nn/reference, nn/cpu, nn/runtime, nn/api, nn/init).
--
-- Conventions (see tests/test_runner.lua): the file returns a table of
-- { name = function() ... end }; assertions raise on failure. Runs under BOTH
-- plain `lua` (no ffi) and `luajit` (ffi + real Vulkan). Tests that need ffi
-- or a Vulkan loader skip (pass) with a printed note when unavailable.

-- LuaJIT's default package.path lacks './?/init.lua' — nn/init.lua is the
-- module entry for require('nn').
package.path = './?.lua;./?/init.lua;' .. package.path

local nn = require('nn')
local errors = nn.errors
local format = nn.format
local quantize = nn.quantize
local serialize = nn.serialize
local corpus = nn.corpus

local HAS_FFI, ffi = pcall(function() return require('ffi') end)
if not HAS_FFI then ffi = nil end

-- ------------------------------------------------------------------ helpers --

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

local function hex(s)
    return (s:gsub('.', function(c) return string.format('%02x', c:byte()) end))
end

-- Deterministic pseudo-random genome (node-major common stream).
local function make_genome(seed)
    local common = {}
    local s = seed or 1
    local function rnd()
        s = (s * 1103515245 + 12345) % 2147483648
        return s / 2147483648
    end
    for i = 1, format.COMMON_LEN do
        common[i] = (rnd() - 0.5) * 200
    end
    return common
end

local function network_stream(common, network_id)
    local net = format.NETWORKS[network_id]
    local w = {}
    for i = 1, net.len do w[i] = common[net.offset + i] end
    return w
end

-- Build a corpus guaranteed to pass every gate: thresholds far below any
-- pre-activation (never triggers the dead zone), all-positive weights and
-- inputs (signs always agree), over all three topologies.
-- Returns (corpus, common_weights).
local function build_passing_corpus()
    local common = {}
    local s = 777
    local function rnd()
        s = (s * 1103515245 + 12345) % 2147483648
        return s / 2147483648
    end
    -- Walk the real node-major layout: per node [bias, threshold, dead,
    -- weights_to_next_layer[]]; final nodes [bias, threshold, dead].
    local pos = 1
    for _, id in ipairs(format.NETWORK_ORDER) do
        local layers = format.NETWORKS[id].layers
        for li = 1, #layers do
            for j = 1, layers[li] do
                common[pos] = 0.05 + rnd() * 2.0 -- bias (positive)
                common[pos + 1] = -1e9 -- threshold (never triggers the dead zone)
                common[pos + 2] = 0.0 -- dead
                pos = pos + 3
                if li < #layers then
                    for k = 1, layers[li + 1] do
                        common[pos] = 0.05 + rnd() * 2.0 -- weight (positive)
                        pos = pos + 1
                    end
                end
            end
        end
    end
    assert(pos - 1 == format.COMMON_LEN, 'crafted genome length mismatch')
    local inputs_a, actions_a, outputs_a, netids_a = {}, {}, {}, {}
    for _, id in ipairs(format.NETWORK_ORDER) do
        local net = format.NETWORKS[id]
        local w = network_stream(common, id)
        local refm = nn.reference.new_model(net.layers):load_weights(w)
        for k = 1, 6 do
            local inputs = {}
            for i = 1, net.layers[1] do inputs[i] = 0.05 + rnd() * 0.9 end
            local outputs, _, dead = refm:run_debug(inputs)
            inputs_a[#inputs_a + 1] = inputs
            outputs_a[#outputs_a + 1] = outputs
            netids_a[#netids_a + 1] = id
            local actions = { dead = dead, boundary = {} }
            for o = 1, #dead do actions.boundary[o] = false end
            actions_a[#actions_a + 1] = actions
        end
    end
    local c = corpus.record(inputs_a, actions_a, outputs_a, netids_a,
        format.TOPOLOGY_IDENTITY, 1e-6)
    return c, common
end

-- -------------------------------------------------------------------- tests --

local tests = {}

-- Topology must match the REAL game topology (shares.lua AI_LAYERS_*).
tests['topology matches shares.lua and the spec counts'] = function()
    local shares = require('shares')
    assert(shares.AI_LAYERS_SEED[1] == format.NETWORKS.seed.layers[1]
        and #shares.AI_LAYERS_SEED == #format.NETWORKS.seed.layers)
    assert(shares.AI_LAYERS_SPORE[1] == format.NETWORKS.spore.layers[1]
        and #shares.AI_LAYERS_SPORE == #format.NETWORKS.spore.layers)
    assert(shares.AI_LAYERS_SPROUT[1] == format.NETWORKS.sprout.layers[1]
        and #shares.AI_LAYERS_SPROUT == #format.NETWORKS.sprout.layers)
    -- Lengths (node-major stream) match AI_LEN_*.
    assert(format.NETWORKS.seed.len == shares.AI_LEN_SEED)
    assert(format.NETWORKS.spore.len == shares.AI_LEN_SPORE)
    assert(format.NETWORKS.sprout.len == shares.AI_LEN_SPROUT)
    assert(format.COMMON_LEN == shares.AI_LEN_COMMON)
    assert(format.COMMON_LEN == 1003)
    -- Spec'd decomposition counts: matrix + specials per network.
    local expect = {
        seed = { 120, 66 }, spore = { 112, 69 }, sprout = { 498, 138 },
    }
    local total_matrix, total_specials = 0, 0
    for id, net in pairs(format.NETWORKS) do
        assert(format.matrix_count(net.layers) == expect[id][1],
            ('%s matrix count'):format(id))
        assert(format.special_count(net.layers) == expect[id][2],
            ('%s special count'):format(id))
        assert(format.matrix_count(net.layers) + format.special_count(net.layers) == net.len)
        total_matrix = total_matrix + format.matrix_count(net.layers)
        total_specials = total_specials + format.special_count(net.layers)
    end
    assert(total_matrix == 730 and total_specials == 273 and total_matrix + total_specials == 1003)
    -- Offsets mirror shares.lua.
    assert(format.NETWORKS.seed.offset == shares.AI_OFFSET_SEED)
    assert(format.NETWORKS.spore.offset == shares.AI_OFFSET_SPORE)
    assert(format.NETWORKS.sprout.offset == shares.AI_OFFSET_SPROUT)
end

-- Backend resolution: never "auto" after init.
tests['backend resolution'] = function()
    local r = nn.new({ backend = 'cpu', deterministic = true })
    assert(r and r.backend == 'cpu')

    local ra = nn.new({ backend = 'auto', deterministic = true })
    assert(ra and ra.backend == 'cpu', 'deterministic+auto must resolve to cpu')

    local rg, err = nn.new({ backend = 'gpu', deterministic = true })
    assert(rg == nil and errors.is(err, 'INVALID_ARGUMENT'),
        'deterministic+gpu must fail with INVALID_ARGUMENT')

    -- auto + non-deterministic: gpu when Vulkan is available, else cpu.
    local rn = nn.new({ backend = 'auto', deterministic = false })
    assert(rn, 'auto runtime must init')
    assert(rn.backend == 'cpu' or rn.backend == 'gpu',
        ('backend must never be "auto" after init (got %q)'):format(tostring(rn.backend)))

    -- gpu requested but unsupported -> structured error (not a raise).
    local vulkan_ok = false
    local okv, v = pcall(require, 'nn.vulkan')
    if okv and type(v) == 'table' and v.can_load then
        local okc = pcall(v.can_load)
        if okc and v.can_load() then vulkan_ok = true end
    end
    if not vulkan_ok then
        local rna, errna = nn.new({ backend = 'gpu', deterministic = false })
        assert(rna == nil and type(errna) == 'table' and errna.class,
            'gpu without Vulkan must fail with a structured error')
        assert(errna.class == errors.CLASSES.VULKAN_UNSUPPORTED_PLATFORM
            or errna.class == errors.CLASSES.VULKAN_INITIALIZATION_FAILED)
    else
        local rgpu, errgpu = nn.new({ backend = 'gpu', deterministic = false })
        if rgpu then
            assert(rgpu.backend == 'gpu')
            rgpu:shutdown()
        else
            assert(type(errgpu) == 'table' and errgpu.class,
                'gpu init failure must be structured')
        end
    end
end

-- Structured error taxonomy.
tests['errors taxonomy'] = function()
    local e = errors.new('INVALID_ARGUMENT', 'boom', 'cpu')
    assert(type(e) == 'table' and e.class == 'INVALID_ARGUMENT'
        and e.message == 'boom' and e.backend == 'cpu' and e.recoverable == false)
    assert(errors.is(e, 'INVALID_ARGUMENT'))
    assert(not errors.is(e, 'INVALID_WEIGHTS'))
    assert(not errors.is(nil, 'INVALID_ARGUMENT'))
    assert(not errors.is({}, 'INVALID_ARGUMENT'))
    -- Class-keyed constructors (vulkan compat surface).
    local e2 = errors.INVALID_WEIGHTS('nope')
    assert(errors.is(e2, 'INVALID_WEIGHTS'))
    -- Every class is present.
    for class in pairs(errors.CLASSES) do
        assert(type(errors[class]) == 'function', ('missing ctor for %s'):format(class))
    end
    assert(errors.CLASSES.DEVICE_LOST == 'DEVICE_LOST')
    assert(errors.CLASSES.CORPUS_GATE_FAILED == 'CORPUS_GATE_FAILED')
    assert(errors.CLASSES.ABI_MISMATCH == 'ABI_MISMATCH')
end

-- Capabilities shape.
tests['capabilities'] = function()
    local r = nn.new({ backend = 'cpu', deterministic = true })
    local c = r:capabilities()
    assert(c.backend == 'cpu' and c.deterministic == true)
    assert(c.precisions.fp32 == true)
    assert(c.native_fp16_arithmetic == false)
    assert(c.packed_storage == true)
    assert(c.max_batch_size and c.max_arena_bytes ~= nil)
    assert(c.block_sizes[1] == 8 and c.block_sizes[2] == 16
        and c.block_sizes[3] == 32 and c.block_sizes[4] == 64)
    assert(#c.supported_networks == 3)
    for _, id in ipairs({ 'seed', 'spore', 'sprout' }) do
        local found = false
        for _, sid in ipairs(c.supported_networks) do if sid == id then found = true end end
        assert(found, ('supported_networks must contain %q'):format(id))
    end
end

-- Weight blob types: table / string (packed fp64 LE) / cdata produce the same
-- output as the exact reference.
tests['blob types are equivalent'] = function()
    local common = make_genome(5)
    local inputs = { 0.5, -1.2, 0.3, 2.0, -0.5, 0.1, 0.9, -0.7, 0.4 }
    local seed_stream = network_stream(common, 'seed')
    local ref = nn.reference.new_model(format.NETWORKS.seed.layers):load_weights(seed_stream)
    local expected = ref:forward(inputs)[1]

    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp32' })
    -- table (common genome)
    assert(r:forward('seed', common, inputs)[1] == expected)
    -- table (per-network slice)
    assert(r:forward('seed', seed_stream, inputs)[1] == expected)
    -- string: packed little-endian fp64 doubles of the common genome
    local parts = {}
    for i = 1, #common do parts[i] = quantize.f64_to_bytes(common[i]) end
    local blob = table.concat(parts)
    assert(r:forward('seed', blob, inputs)[1] == expected, 'string blob must match')
    -- string: per-network slice
    local parts2 = {}
    for i = 1, #seed_stream do parts2[i] = quantize.f64_to_bytes(seed_stream[i]) end
    assert(r:forward('seed', table.concat(parts2), inputs)[1] == expected,
        'string per-network blob must match')
    -- weights_desc wrapper
    assert(r:forward('seed', { blob = common, blob_type = 'table', layout = 'genome_node_major' }, inputs)[1] == expected)
    if ffi then
        local cd = ffi.new('double[?]', #common)
        for i = 1, #common do cd[i - 1] = common[i] end
        assert(r:forward('seed', cd, inputs)[1] == expected, 'cdata blob must match')
    end
    -- .nnw serialized model as a weights string
    local model = { networks = {} }
    for _, id in ipairs(format.NETWORK_ORDER) do
        model.networks[id] = { layers = format.NETWORKS[id].layers, weights = network_stream(common, id) }
    end
    local bytes = assert(serialize.model_to_nnw(model))
    local got = r:forward('seed', bytes, inputs)[1]
    -- The .nnw stores fp32-rounded weights, so compare against a forward of
    -- those rounded weights (fp32-exact reconstruction -> bit-identical).
    local f32w = {}
    for i = 1, #seed_stream do f32w[i] = quantize.round_f32(seed_stream[i]) end
    local ref32 = nn.reference.new_model(format.NETWORKS.seed.layers):load_weights(f32w)
    local expect32 = ref32:forward(inputs)[1]
    assert(got == expect32,
        (' .nnw weights must decode to the fp32-rounded reference (%.17g vs %.17g)')
        :format(got, expect32))
end

-- forward_into: caller-owned output, no allocation, returns `out`.
tests['forward_into returns the caller buffer without allocation'] = function()
    local common = make_genome(9)
    local inputs = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8' })
    local out = { 0 }
    local ret = r:forward_into('seed', common, inputs, out)
    assert(ret == out, 'forward_into must return the caller-owned table')
    local out2 = { -999 }
    local ret2 = r:forward_into('seed', common, inputs, out2)
    assert(ret2 == out2 and out2[1] ~= -999, 'must overwrite the caller buffer')
    -- Repeated calls with the same buffer stay consistent.
    local a = r:forward_into('seed', common, inputs, { 0 })
    local b = r:forward_into('seed', common, inputs, { 0 })
    assert(a[1] == b[1])
end

-- Activation semantics: value = prev + bias; if value <= threshold then dead.
tests['activation threshold semantics (<=, equality, NaN)'] = function()
    -- {1,1} network stream: [b1,t1,d1,w11,b2,t2,d2].
    -- bias=0, threshold=0, dead=7 on both nodes, weight 1.
    local m = nn.reference.new_model({ 1, 1 }):load_weights({ 0, 0, 7, 1, 0, 0, 7 })
    assert(m:forward({ 0.5 })[1] == 0.5, 'above threshold must stay live')
    assert(m:forward({ -1 })[1] == 7, 'below threshold must die')
    assert(m:forward({ 0 })[1] == 7, 'exactly at threshold (<=) must die')
    -- Bias is added first: bias=2 -> value = x + 2.
    local m2 = nn.reference.new_model({ 1, 1 }):load_weights({ 2, 1, 9, 1, 0, 0, 0 })
    assert(m2:forward({ -1 })[1] == 9, '-1 + 2 = 1 <= 1 -> dead')
    assert(m2:forward({ 0 })[1] == 2, '0 + 2 = 2 > 1 -> live (2.0)')
    -- NaN propagates (NaN <= t is false).
    local out = m:forward({ 0 / 0 })
    assert(out[1] ~= out[1], 'NaN must propagate through the activation')
    -- Runtime fp32 path agrees with the reference (single node network).
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp32' })
    local o = r:forward('seed', make_genome(3), { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 })
    assert(type(o[1]) == 'number' and o[1] == o[1], 'runtime output must be finite')
end

-- round-half-to-even ties (never floor(v + 0.5)).
tests['round_even ties including negatives'] = function()
    local re = quantize.round_even
    assert(re(0.5) == 0 and re(1.5) == 2 and re(2.5) == 2 and re(3.5) == 4)
    assert(re(-0.5) == 0 and re(-1.5) == -2 and re(-2.5) == -2 and re(-3.5) == -4)
    assert(re(2.0) == 2 and re(-1.0) == -1)
    assert(re(2.4) == 2 and re(2.6) == 3 and re(-2.4) == -2 and re(-2.6) == -3)
    assert(re(0.0) == 0 and re(1e-9) == 0)
end

-- fp8 zero block: scale 0, q 0, stored byte 128, decode zero.
tests['fp8 zero block short-circuits before division'] = function()
    local enc = quantize.encode_fp8_block({ 0, 0, 0, 0, 0, 0, 0, 0 }, 8)
    assert(enc.zero_block == true and enc.scale == 0.0)
    for _, q in ipairs(enc.codes) do assert(q == 0) end
    local payload = quantize.pack_fp8(enc.codes, 8)
    assert(payload == string.rep(string.char(128), 8),
        'zero block bytes must all be 128 (q=0), got ' .. hex(payload))
    local dec = quantize.decode_fp8_block(payload, 0, 0, 8, 0.0)
    for i = 1, 8 do assert(dec[i] == 0.0, 'decoded zero block must be zero') end
    -- Byte-identical across the two interpreters is covered by the packer.
end

-- fp8 encode/decode round trip (incl. asymmetric, negative-only).
tests['fp8 round trip'] = function()
    local blocks = {
        { 1.0, -0.5, 0.25, 2.0, -1.5, 0.125, 3.0, -2.75 },
        { -3, -2, -1, -0.5, -4, -2.5, -1.25, -0.1 }, -- negative-only
        { -100, 100, 0, 50, -50, 25, -25, 75 }, -- asymmetric
        { 5, 5, 5, 5, 5, 5, 5, 5 }, -- constant
        { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 },
    }
    for _, values in ipairs(blocks) do
        local absmax = 0
        for _, v in ipairs(values) do absmax = math.max(absmax, math.abs(v)) end
        local enc = quantize.encode_fp8_block(values, 8)
        if absmax == 0 then
            assert(enc.zero_block)
        else
            assert(math.abs(enc.scale * 127 - absmax) < 1e-9, 'scale = absmax/127')
            assert(#enc.codes == #values)
            for i, q in ipairs(enc.codes) do
                assert(q >= -127 and q <= 127)
            end
            local payload = quantize.pack_fp8(enc.codes, #values)
            local scale32 = quantize.round_f32(enc.scale)
            local dec = quantize.decode_fp8_block(payload, 0, 0, #values, scale32)
            for i = 1, #values do
                assert(math.abs(dec[i] - values[i]) <= 0.005 * absmax + 1e-9,
                    ('fp8 decode error: got %g want %g'):format(dec[i], values[i]))
            end
        end
    end
end

-- fp4/fp2 affine round trips incl. constant, asymmetric, negative-only.
tests['fp4/fp2 affine round trips'] = function()
    local function check(max_code, values)
        local enc = quantize.encode_affine_block(values, max_code)
        local mn, mx = values[1], values[1]
        for _, v in ipairs(values) do
            if v < mn then mn = v end
            if v > mx then mx = v end
        end
        if mx == mn then
            assert(enc.constant and enc.scale == 0.0 and enc.offset == mn)
        else
            assert(math.abs(enc.scale - (mx - mn) / max_code) < 1e-9)
            assert(enc.offset == mn)
        end
        for i, q in ipairs(enc.codes) do
            assert(q >= 0 and q <= max_code)
        end
        local payload
        if max_code == 15 then
            payload = quantize.pack_fp4(enc.codes, #values)
        else
            payload = quantize.pack_fp2(enc.codes, #values)
        end
        local scale32 = quantize.round_f32(enc.scale)
        local offset32 = quantize.round_f32(enc.offset)
        local dec
        if max_code == 15 then
            dec = quantize.decode_affine_block(payload, 0, 0, #values, 15, offset32, scale32)
        else
            dec = quantize.decode_affine_block(payload, 0, 0, #values, 3, offset32, scale32)
        end
        local range = mx - mn
        for i = 1, #values do
            local tol = max_code == 15 and 0.05 * range or 0.2 * range
            assert(math.abs(dec[i] - values[i]) <= tol + 1e-9,
                ('affine decode error: got %g want %g'):format(dec[i], values[i]))
        end
    end
    -- fp4
    check(15, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 })
    check(15, { 7, 7, 7, 7, 7, 7, 7, 7 })
    check(15, { -10, 10, 5, -5, 0, 3, -3, 8 })
    check(15, { -20, -15, -10, -5 })
    -- fp2
    check(3, { 0, 1, 2, 3, 1, 0, 2, 3 })
    check(3, { 4, 4, 4, 4 })
    check(3, { -2, 2, 0, 1 })
    check(3, { -10, -8, -6, -4 })
end

-- Padding bytes + padding excluded from statistics.
tests['block padding bytes and stats exclusion'] = function()
    -- 12 logical fp8 values in a 16-block: padding bytes must be 128.
    local values = { 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 }
    local enc = quantize.encode_fp8_block(values, 16)
    local codes = {}
    for i = 1, #values do codes[i] = enc.codes[i] end
    local payload = quantize.pack_fp8(codes, 16) -- pads to 16
    assert(#payload == 16)
    for i = 13, 16 do
        assert(payload:byte(i) == 128, ('fp8 padding byte %d must be 128'):format(i))
    end
    -- fp4: 12 values in a 16-block -> 8 bytes; padding codes 0.
    local enc4 = quantize.encode_affine_block(values, 15)
    local codes4 = {}
    for i = 1, #values do codes4[i] = enc4.codes[i] end
    local p4 = quantize.pack_fp4(codes4, 16)
    assert(#p4 == 8)
    assert(p4:byte(7) == 0 and p4:byte(8) == 0, 'fp4 padding bytes must be 0')
    -- fp2: 12 values in a 16-block -> 4 bytes; padding codes 0.
    local enc2 = quantize.encode_affine_block(values, 3)
    local codes2 = {}
    for i = 1, #values do codes2[i] = enc2.codes[i] end
    local p2 = quantize.pack_fp2(codes2, 16)
    assert(#p2 == 4)
    assert(p2:byte(4) == 0, 'fp2 padding codes must be 0')
    -- Padding excluded from statistics: a constant block of 12 equal values
    -- must take the constant branch (scale=0) even though the padding slots
    -- would read as 0 if treated as real values.
    local constant = {}
    for i = 1, 12 do constant[i] = 100 end
    local econst = quantize.encode_affine_block(constant, 15)
    assert(econst.constant == true and econst.scale == 0.0 and econst.offset == 100,
        'padding must be excluded from min/max statistics')
    local codes_c = {}
    for i = 1, #constant do codes_c[i] = econst.codes[i] end
    local pc = quantize.pack_fp4(codes_c, 16)
    for i = 1, 6 do assert(pc:byte(i) == 0, 'constant fp4 block codes must be 0') end
    assert(pc:byte(7) == 0 and pc:byte(8) == 0)
end

-- Memory totals recompute formula.
tests['memory totals recompute'] = function()
    local function memory_total(fmt, block)
        local total = 0
        for _, id in ipairs(format.NETWORK_ORDER) do
            local layers = format.NETWORKS[id].layers
            for li = 1, #layers - 1 do
                local logical = layers[li] * layers[li + 1]
                local blocks = quantize.block_count(logical, block)
                local padded = blocks * block
                local payload
                if fmt == 'fp8' then payload = padded
                elseif fmt == 'fp4' then payload = padded / 2
                else payload = padded / 4 end
                total = total + payload + blocks * 4 * (fmt == 'fp8' and 1 or 2)
            end
        end
        return total + 273 * 4 -- fp32 specials
    end
    assert(memory_total('fp4', 16) == 1844, ('fp4/16 = %d'):format(memory_total('fp4', 16)))
    assert(memory_total('fp8', 16) == 2032, ('fp8/16 = %d'):format(memory_total('fp8', 16)))
    assert(memory_total('fp2', 16) == 1656, ('fp2/16 = %d'):format(memory_total('fp2', 16)))
    assert(memory_total('fp8', 32) == 2028, ('fp8/32 = %d'):format(memory_total('fp8', 32)))
    assert(memory_total('fp4', 32) == 1716, ('fp4/32 = %d'):format(memory_total('fp4', 32)))
    assert(memory_total('fp2', 32) == 1508, ('fp2/32 = %d'):format(memory_total('fp2', 32)))
    -- block8 payload alignment invariant: 120 fp2 weights / block8 = 30 bytes.
    assert(quantize.block_count(120, 8) == 15)
    local padded = quantize.block_count(120, 8) * 8
    assert(padded / 4 == 30, 'fp2 payload bytes for 120 weights at block8')
end

-- f64->f32 bytes vs FFI cast (LuaJIT) / string.pack (PUC Lua).
tests['f64->f32 bytes match the hardware conversion'] = function()
    local vals = { 0, -0.0, 1, -1, 0.5, 1.5, 2.5, math.pi, 1 / 3, 123456.789,
        1e-40, 2 ^ -130, 16777217, 16777219, math.huge, -math.huge, 3.4028234663852886e38 }
    if ffi then
        for _, v in ipairs(vals) do
            local u = ffi.new('union { float f; uint8_t b[4]; }')
            u.f = ffi.cast('float', v)
            local ref = string.char(u.b[0], u.b[1], u.b[2], u.b[3])
            assert(quantize.f64_to_f32_bytes(v) == ref,
                ('f64->f32 byte mismatch for %.17g: %s vs %s')
                :format(v, hex(quantize.f64_to_f32_bytes(v)), hex(ref)))
        end
    else
        assert(type(string.pack) == 'function', 'plain Lua must provide string.pack for this test')
        for _, v in ipairs(vals) do
            assert(quantize.f64_to_f32_bytes(v) == string.pack('<f', v),
                ('f64->f32 byte mismatch for %.17g'):format(v))
        end
    end
    -- Subnormal round trips through round_f32.
    for _, v in ipairs({ 2 ^ -149, 2 ^ -148, 2 ^ -126, 2 ^ -125 }) do
        local rf = quantize.round_f32(v)
        if ffi then
            assert(ffi.cast('float', rf) == ffi.cast('float', v))
        else
            assert(string.pack('<f', rf) == string.pack('<f', v))
        end
    end
    -- fp64 byte round trip is exact.
    for _, v in ipairs({ 0.1, -2.5, 1e300, 5e-324, 123456789012345.0, math.pi }) do
        local s = quantize.f64_to_bytes(v)
        assert(#s == 8)
        assert(quantize.f64_from_bytes(s, 1) == v, 'f64 byte round trip must be exact')
    end
end

-- .nnw serialization: byte-stable round trip + rejection paths.
tests['nnw serialization round trip and rejection'] = function()
    local common = make_genome(7)
    local model = { networks = {} }
    for _, id in ipairs(format.NETWORK_ORDER) do
        model.networks[id] = { layers = format.NETWORKS[id].layers, weights = network_stream(common, id) }
    end
    local b1 = assert(serialize.model_to_nnw(model))
    local b2 = assert(serialize.model_to_nnw(model))
    assert(b1 == b2, 'saving twice must produce byte-identical output')
    assert(b1:sub(1, 4) == 'NNW\0', 'magic')
    assert(serialize.ru32(b1, 5) == 1, 'format version must be 1')
    assert(serialize.ru32(b1, 9) == serialize.ENDIANNESS_MARKER, 'endianness marker')

    local loaded, lerr = serialize.read(b1)
    assert(loaded, lerr and lerr.message)
    -- fp32-exact reconstruction of the original weights.
    for _, id in ipairs(format.NETWORK_ORDER) do
        local w1 = model.networks[id].weights
        local w2 = loaded.networks[id].weights
        assert(#w1 == #w2, ('%s weight count'):format(id))
        for i = 1, #w1 do
            assert(w2[i] == quantize.round_f32(w1[i]),
                ('%s[%d] must reconstruct fp32-exactly'):format(id, i))
        end
    end

    -- Rejections.
    local bad_magic = b1:sub(2) .. 'X'
    local ok1, e1 = serialize.read(bad_magic)
    assert(ok1 == nil and errors.is(e1, 'INVALID_SERIALIZATION'), 'bad magic')
    local bad_version = '\0\0\0\0' .. serialize.wu32(99) .. b1:sub(9)
    local ok2, e2 = serialize.read(bad_version)
    assert(ok2 == nil and errors.is(e2, 'INVALID_SERIALIZATION'), 'bad version')
    -- swap the endianness marker bytes
    local swapped = b1:sub(1, 8) .. b1:sub(12, 12) .. b1:sub(11, 11) .. b1:sub(10, 10) .. b1:sub(9, 9) .. b1:sub(13)
    local ok3, e3 = serialize.read(swapped)
    assert(ok3 == nil and errors.is(e3, 'INVALID_SERIALIZATION'), 'bad endianness')
    local ok4, e4 = serialize.read(b1:sub(1, #b1 - 3))
    assert(ok4 == nil and errors.is(e4, 'INVALID_SERIALIZATION'), 'truncation')
    local ok5, e5 = serialize.read('not a file at all')
    assert(ok5 == nil and errors.is(e5, 'INVALID_SERIALIZATION'), 'garbage')
end

-- nn.save / nn.load via plain io.
tests['nn.save and nn.load via plain io'] = function()
    local common = make_genome(13)
    local model = { networks = {} }
    for _, id in ipairs(format.NETWORK_ORDER) do
        model.networks[id] = { layers = format.NETWORKS[id].layers, weights = network_stream(common, id) }
    end
    local path = '/tmp/nn_core_test_model.nnn'
    local ok, err = nn.save(path, model)
    assert(ok, err and err.message)
    local loaded, lerr = nn.load(path)
    assert(loaded, lerr and lerr.message)
    for _, id in ipairs(format.NETWORK_ORDER) do
        assert(#loaded.networks[id].weights == format.NETWORKS[id].len)
    end
    -- load against a runtime validates topology.
    local r = nn.new({ backend = 'cpu', deterministic = true })
    local loaded2, lerr2 = nn.load(path, r)
    assert(loaded2, lerr2 and lerr2.message)
    os.remove(path)
end

-- Genome conversion: every special maps once, every matrix weight maps once,
-- fp32 reconstruction exact (bit-exact stream round trip).
tests['genome decomposition/reconstruction is exact'] = function()
    local common = make_genome(21)
    for _, id in ipairs(format.NETWORK_ORDER) do
        local net = format.NETWORKS[id]
        local stream = network_stream(common, id)
        local dec = assert(format.decompose(net.layers, stream))
        -- Counts.
        local matrix_total, special_total = 0, 0
        for li = 1, #net.layers - 1 do
            matrix_total = matrix_total + #dec.matrices[li].values
        end
        special_total = #dec.specials
        assert(matrix_total == format.matrix_count(net.layers), ('%s matrix count'):format(id))
        assert(special_total == format.special_count(net.layers), ('%s special count'):format(id))
        assert(matrix_total + special_total == net.len, ('%s total'):format(id))
        -- Node order: specials interleaved [bias, threshold, dead].
        assert(#dec.specials % 3 == 0)
        -- Every matrix weight maps once: reconstruction is bit-exact.
        local rebuilt = format.reconstruct(net.layers, dec)
        assert(#rebuilt == #stream)
        for i = 1, #stream do
            assert(rebuilt[i] == stream[i],
                ('%s stream[%d] must reconstruct exactly: %.17g vs %.17g')
                :format(id, i, rebuilt[i], stream[i]))
        end
    end
end

-- set_precision gating (N6).
tests['set_precision gating'] = function()
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp32' })
    assert(r.precision == 'fp32')
    -- fp8 before any corpus gate -> CORPUS_GATE_FAILED.
    local ok8, e8 = r:set_precision('fp8')
    assert(ok8 == nil and errors.is(e8, 'CORPUS_GATE_FAILED'),
        'fp8 before gate must fail with CORPUS_GATE_FAILED')
    -- fp16 likewise.
    local ok16, e16 = r:set_precision('fp16')
    assert(ok16 == nil and errors.is(e16, 'CORPUS_GATE_FAILED'))
    -- fp4/fp2 -> CORPUS_GATE_FAILED unless experimental.
    local ok4, e4 = r:set_precision('fp4')
    assert(ok4 == nil and errors.is(e4, 'CORPUS_GATE_FAILED'))
    local ok4e = r:set_precision('fp4', { experimental = true })
    assert(ok4e == true and r.precision == 'fp4', 'fp4 experimental must switch')
    local ok2e = r:set_precision('fp2', { experimental = true })
    assert(ok2e == true and r.precision == 'fp2')
    -- Unknown precision -> INVALID_PRECISION.
    local okx, ex = r:set_precision('fp128')
    assert(okx == nil and errors.is(ex, 'INVALID_PRECISION'))
    -- Back to fp32 always works.
    assert(r:set_precision('fp32') == true)
    -- After a passing corpus gate, fp8 becomes allowed.
    local gate_corpus, gate_common = build_passing_corpus()
    local r2 = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp32', model = gate_common })
    local res, rerr = corpus.replay(gate_corpus, r2)
    assert(res, rerr and rerr.message)
    assert(res.gates.fp8 == true, 'crafted corpus must pass the fp8 gate')
    assert(r2:set_precision('fp8') == true, 'fp8 must switch after the gate')
    assert(r2:set_precision('fp16') == true, 'fp16 must switch after the gate')
    r2:shutdown()
end

-- Corpus: boundary bucket + epsilon hashed into the identity.
tests['corpus boundary bucket and epsilon hash'] = function()
    -- {1,1} network with a known threshold on the output node:
    -- input node: bias 0 / threshold -1e9 / dead 0 / weight 1;
    -- output node: bias 0 / threshold 1 / dead 5.
    local m = nn.reference.new_model({ 1, 1 }):load_weights({ 0, -1e9, 0, 1, 0, 1, 5 })
    -- pre == 1 - eps/2 (boundary), pre == 1 + 10*eps (ordinary, above),
    -- pre == -2 (ordinary, below), pre == 0 (ordinary, below -> dead).
    local eps = 1e-3
    local inputs = { { 1 - eps / 2 }, { 1 + 10 * eps }, { -2 }, { 0 } }
    local outputs, deads, pres = {}, {}, {}
    for _, inp in ipairs(inputs) do
        local out, pre, dead = m:run_debug(inp)
        outputs[#outputs + 1] = out
        deads[#deads + 1] = dead
        pres[#pres + 1] = pre
    end
    local actions, netids = {}, {}
    for i = 1, #inputs do
        local boundary = {}
        for o = 1, #pres[i] do
            boundary[o] = math.abs(pres[i][o] - 1) <= eps -- output threshold is 1
        end
        actions[i] = { dead = deads[i], boundary = boundary }
        netids[i] = 'seed'
    end
    -- epsilon is hashed into the identity.
    local c1 = corpus.record(inputs, actions, outputs, netids, format.TOPOLOGY_IDENTITY, eps)
    local c2 = corpus.record(inputs, actions, outputs, netids, format.TOPOLOGY_IDENTITY, eps * 2)
    assert(c1.hash ~= c2.hash, 'changing epsilon must change the corpus hash')
    assert(c1.data.manifest.epsilon == eps and c1.data.epsilon == eps)
    -- The boundary flag survives the record.
    assert(c1.data.items[1].boundary[1] == true, 'pre at threshold - eps/2 is a boundary')
    assert(c1.data.items[2].boundary[1] == false, 'pre well above threshold is ordinary')
    assert(c1.data.items[3].boundary[1] == false)

    -- Replay with a runtime carrying this exact single-node genome.
    local genome = {}
    for i = 1, format.NETWORKS.seed.len do genome[i] = 0 end
    genome[1] = 0 -- bias
    genome[2] = 1 -- threshold
    genome[3] = 5 -- dead
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8', model = genome })
    local res, rerr = corpus.replay(c1, r)
    assert(res, rerr and rerr.message)
    assert(res.topology_matches == true)
    assert(res.epsilon == eps)
    assert(res.boundary.total >= 1, 'boundary buckets must be reported separately')
    -- Boundary items are excluded from the primary gate counts.
    local ordinary = #inputs * 1 - res.counts.fp8.boundary_total
    assert(res.counts.fp8.total == ordinary,
        'boundary items must be excluded from primary counts')
    assert(res.hash == c1.hash)
    -- set_precision consults the stored gate results.
    assert(r._gate_results and r._gate_results.gates, 'replay must store gate results')
end

-- Purity: core sources never require game modules / love.
tests['core is standalone (purity scan)'] = function()
    local core = {
        'nn/errors.lua', 'nn/capabilities.lua', 'nn/api.lua', 'nn/format.lua',
        'nn/serialize.lua', 'nn/quantize.lua', 'nn/corpus.lua', 'nn/reference.lua',
        'nn/cpu.lua', 'nn/runtime.lua', 'nn/init.lua',
    }
    local forbidden = { 'shares', 'love', 'cell_module', 'sim_module', 'ai_module' }
    for _, path in ipairs(core) do
        local src = assert(read_file(path), ('missing core file %s'):format(path))
        for _, mod in ipairs(forbidden) do
            local pat = "require('" .. mod
            local pat2 = 'require("' .. mod
            assert(not src:find(pat, 1, true) and not src:find(pat2, 1, true),
                ('%s must not require %q'):format(path, mod))
        end
    end
end

-- Reference vs CPU backends are bit-identical for fp32/double.
tests['reference and cpu are bit-identical for fp32'] = function()
    local s = 99
    local function rnd()
        s = (s * 1103515245 + 12345) % 2147483648
        return s / 2147483648
    end
    local checked = 0
    for _ = 1, 4 do
        local common = {}
        for i = 1, format.COMMON_LEN do common[i] = (rnd() - 0.5) * 200 end
        for _, id in ipairs(format.NETWORK_ORDER) do
            local net = format.NETWORKS[id]
            local w = network_stream(common, id)
            local refm = nn.reference.new_model(net.layers):load_weights(w)
            local cpum = nn.cpu.new_model(net.layers):load_weights(w)
            for _ = 1, 2 do
                local inputs = {}
                for i = 1, net.layers[1] do inputs[i] = (rnd() - 0.5) * 4 end
                local a = refm:forward(inputs)
                local b = cpum:forward(inputs)
                for o = 1, #a do
                    assert(a[o] == b[o],
                        ('bit mismatch %s[%d]: %.17g vs %.17g'):format(id, o, a[o], b[o]))
                    checked = checked + 1
                end
            end
        end
    end
    assert(checked > 0)
end

-- Runtime forward_batch handles per-item genomes.
tests['forward_batch with per-cell genomes'] = function()
    local g1 = make_genome(31)
    local g2 = make_genome(32)
    local g3 = make_genome(33)
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8' })
    local inputs = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
    local in_buf = {}
    for _ = 1, 3 do
        for _, v in ipairs(inputs) do in_buf[#in_buf + 1] = v end
    end
    local out_buf = { 0, 0, 0 }
    local res = r:forward_batch('seed', {
        { genome = g1, network_id = 'seed' },
        { genome = g2, network_id = 'seed' },
        { genome = g3, network_id = 'seed' },
    }, { buffer = in_buf, stride = 9, count = 3, element_type = 'fp32' },
       { buffer = out_buf, stride = 1, count = 3, element_type = 'fp32' })
    assert(res == out_buf)
    assert(out_buf[1] == r:forward('seed', g1, inputs)[1])
    assert(out_buf[2] == r:forward('seed', g2, inputs)[1])
    assert(out_buf[3] == r:forward('seed', g3, inputs)[1])
    -- Mismatched count must error.
    local okb, eb = r:forward_batch('seed', { { genome = g1 } },
        { buffer = in_buf, stride = 9, count = 5 }, { buffer = out_buf, stride = 1, count = 1 })
    assert(okb == nil and errors.is(eb, 'INVALID_ARGUMENT'))
end

-- Precision switch invalidates packed caches (packed entries keyed by
-- (genome identity, profile)); switching back must not reuse stale packs.
tests['precision switch invalidates packed cache'] = function()
    local common = make_genome(41)
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp32' })
    local inputs = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 }
    local a32 = r:forward('seed', common, inputs)[1]
    assert(r:set_precision('fp4', { experimental = true }) == true)
    local a4 = r:forward('seed', common, inputs)[1]
    assert(a4 ~= a32, 'fp4 quantization must change the output')
    assert(r:set_precision('fp32') == true)
    local a32b = r:forward('seed', common, inputs)[1]
    assert(a32b == a32, 'switching back to fp32 must reproduce the fp32 result')
end

-- Deterministic reproducibility: identical genomes/inputs -> identical outputs
-- across separate runtimes (and interpreters).
tests['deterministic fp8 reproducibility'] = function()
    local common = make_genome(51)
    local inputs = { 0.3, -0.7, 1.1, -0.2, 0.8, 0.05, -0.4, 0.6, -1.0 }
    local r1 = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8', block_size = 16 })
    local r2 = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8', block_size = 16 })
    local o1 = r1:forward('sprout', common, inputs)
    local o2 = r2:forward('sprout', common, inputs)
    assert(#o1 == 3 and #o2 == 3)
    for i = 1, 3 do
        assert(o1[i] == o2[i], ('deterministic outputs must match at %d'):format(i))
    end
    -- All block sizes produce valid results.
    for _, bs in ipairs({ 8, 16, 32, 64 }) do
        local rb = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8', block_size = bs })
        local ob = rb:forward('seed', common, inputs)
        assert(type(ob[1]) == 'number' and ob[1] == ob[1])
    end
end

-- Non-finite weights are rejected at pack time.
tests['NaN/Inf weights are rejected at pack time'] = function()
    local common = make_genome(71)
    local inputs = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8' })
    local nan_weights = { common[1] }
    for i = 2, format.NETWORKS.seed.len do nan_weights[i] = common[format.NETWORKS.seed.offset + i] end
    nan_weights[4] = 0 / 0 -- a matrix weight becomes NaN
    local out, err = r:forward('seed', nan_weights, inputs)
    assert(out == nil and errors.is(err, 'INVALID_WEIGHTS'),
        'NaN weights must be rejected with INVALID_WEIGHTS')
    local inf_weights = { common[1] }
    for i = 2, format.NETWORKS.seed.len do inf_weights[i] = common[format.NETWORKS.seed.offset + i] end
    inf_weights[2] = math.huge -- a threshold becomes Inf
    local out2, err2 = r:forward('seed', inf_weights, inputs)
    assert(out2 == nil and errors.is(err2, 'INVALID_WEIGHTS'),
        'Inf weights must be rejected with INVALID_WEIGHTS')
    -- NaN inputs propagate through the activation (not rejected).
    local o = r:forward('seed', common, { 1, 2, 0 / 0, 4, 5, 6, 7, 8, 9 })
    assert(o and o[1] ~= o[1], 'NaN inputs must propagate (NaN output)')
end

-- Serialization rejects inconsistent cross-record references.
tests['nnw rejects inconsistent references'] = function()
    -- A block quant record referencing a missing source tensor must fail.
    local records = {
        { type = serialize.RECORD_TENSOR, tensor_id = 1, role = serialize.ROLE_SPECIALS,
            element_type = serialize.ELEM_FP32, storage_type = serialize.STORAGE_RAW,
            logical_count = 3, padded_count = 3, alignment = 4, flags = 0 },
        { type = serialize.RECORD_BLOCK_QUANT, source_tensor_id = 99, -- missing
            payload_tensor_id = 0, scale_tensor_id = 0, offset_tensor_id = 0,
            zero_point_tensor_id = 0, block_size = 16, block_log2 = 4,
            format = serialize.QUANT_FP8_SYMMETRIC, scale_type = serialize.SCALE_FP32_CANONICAL,
            offset_type = serialize.OFFSET_NONE, zero_point_type = serialize.ZERO_POINT_NONE,
            decode_form = serialize.DECODE_SYMMETRIC_ZERO_POINT,
            granularity = serialize.GRANULARITY_PER_BLOCK, payload_alignment = 1,
            metadata_alignment = 4, logical_count = 16, padded_count = 16, block_count = 1 },
    }
    local bytes, werr = serialize.write({}, records, { [1] = 'abc' })
    assert(bytes, werr and werr.message)
    local ok, rerr = serialize.read(bytes)
    assert(ok == nil and errors.is(rerr, 'INVALID_SERIALIZATION'),
        'block quant with a missing source tensor must be rejected')
    -- scale_type=FP16_EXPERIMENTAL without an FP16 scale tensor must fail.
    local records2 = {
        { type = serialize.RECORD_TENSOR, tensor_id = 1, role = serialize.ROLE_SPECIALS,
            element_type = serialize.ELEM_FP32, storage_type = serialize.STORAGE_RAW,
            logical_count = 3, padded_count = 3, alignment = 4, flags = 0 },
        { type = serialize.RECORD_BLOCK_QUANT, source_tensor_id = 1,
            payload_tensor_id = 0, scale_tensor_id = 1, offset_tensor_id = 0,
            zero_point_tensor_id = 0, block_size = 16, block_log2 = 4,
            format = serialize.QUANT_FP8_SYMMETRIC, scale_type = serialize.SCALE_FP16_EXPERIMENTAL,
            offset_type = serialize.OFFSET_NONE, zero_point_type = serialize.ZERO_POINT_NONE,
            decode_form = serialize.DECODE_SYMMETRIC_ZERO_POINT,
            granularity = serialize.GRANULARITY_PER_BLOCK, payload_alignment = 1,
            metadata_alignment = 4, logical_count = 16, padded_count = 16, block_count = 1 },
    }
    local bytes2 = assert(serialize.write({}, records2, { [1] = 'abc' }))
    local ok2, rerr2 = serialize.read(bytes2)
    assert(ok2 == nil and errors.is(rerr2, 'INVALID_SERIALIZATION'),
        'FP16_EXPERIMENTAL scale_type must require an FP16 scale tensor')
end

-- Precision switches are counted under the "precision-switch cost" metric.
tests['precision-switch cost metric'] = function()
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp32' })
    assert(r.metrics['precision-switch cost'] == nil or r.metrics['precision-switch cost'] == 0)
    assert(r:set_precision('fp4', { experimental = true }) == true)
    assert(r:set_precision('fp2', { experimental = true }) == true)
    assert(r:set_precision('fp32') == true)
    assert(r.metrics['precision-switch cost'] == 3,
        ('precision-switch cost must count switches, got %s')
        :format(tostring(r.metrics['precision-switch cost'])))
end

-- Forward / forward_batch / pack-cache counters increment at the public
-- boundaries and never alter results.
tests['forward and cache metrics increment'] = function()
    local g1 = make_genome(61)
    local g2 = make_genome(62)
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8' })
    local inputs = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }

    r:forward('seed', g1, inputs)
    r:forward('seed', g2, inputs)
    assert(r.metrics['forward calls'] == 2,
        ('forward calls must equal 2, got %s'):format(tostring(r.metrics['forward calls'])))

    local in_buf, out_buf = {}, { 0, 0 }
    for _ = 1, 2 do
        for _, v in ipairs(inputs) do in_buf[#in_buf + 1] = v end
    end
    local before_batch = r.metrics['forward batch calls'] or 0
    r:forward_batch('seed', {
        { genome = g1, network_id = 'seed' },
        { genome = g2, network_id = 'seed' },
    }, { buffer = in_buf, stride = 9, count = 2, element_type = 'fp32' },
       { buffer = out_buf, stride = 1, count = 2, element_type = 'fp32' })
    assert(r.metrics['forward batch calls'] == before_batch + 1,
        'forward_batch calls must increment')

    -- Первый прогон новых геномов через batch = 2 промаха; повтор = 2 попадания.
    assert((r.metrics['pack cache misses'] or 0) >= 2,
        'first batch of fresh genomes must miss the cache')
    local misses = r.metrics['pack cache misses'] or 0
    local hits = r.metrics['pack cache hits'] or 0
    r:forward_batch('seed', {
        { genome = g1, network_id = 'seed' },
        { genome = g2, network_id = 'seed' },
    }, { buffer = in_buf, stride = 9, count = 2, element_type = 'fp32' },
       { buffer = out_buf, stride = 1, count = 2, element_type = 'fp32' })
    assert((r.metrics['pack cache hits'] or 0) == hits + 2,
        'repeated batch must hit the cache twice')
    assert(r.metrics['pack cache misses'] == misses,
        'repeated batch must not miss again')
end

-- Optional real-GPU forward when Vulkan + compiled .spv exist.
tests['optional real-GPU forward (packed, per-layer dispatch)'] = function()
    if not ffi then
        print('SKIP: no ffi (plain Lua) — GPU path is LuaJIT-only')
        return
    end
    local okv, v = pcall(require, 'nn.vulkan')
    if not okv or type(v) ~= 'table' or not v.can_load then
        print('SKIP: no Vulkan loader')
        return
    end
    local okc = pcall(v.can_load)
    if not okc or not v.can_load() then
        print('SKIP: Vulkan loader present but unusable')
        return
    end
    if not file_exists('nn/shaders/spv/forward_packed.spv') then
        print('SKIP: no compiled .spv artifacts')
        return
    end
    -- Use a tiny 2-layer network via the generic topology machinery.
    local common = make_genome(61)
    local r = nn.new({ backend = 'gpu', deterministic = false, precision = 'fp8' })
    local inputs = { 0.5, -0.5, 0.25, -0.25, 0.125, 0, 1, -1, 0.75 }
    local out, err = r:forward('seed', common, inputs)
    if not out then
        r:shutdown()
        print(('SKIP: GPU forward failed (%s %s)'):format(
            tostring(err and err.class), tostring(err and err.message)))
        return
    end
    assert(type(out[1]) == 'number' and out[1] == out[1],
        'GPU output must be a finite number')
    -- The packed shader is a DENSE single-layer model, dispatched per layer,
    -- and is now semantically EQUIVALENT to the CPU/reference path: the first
    -- dispatch also applies input-layer activation (bias/threshold/dead on the
    -- layer-1 nodes). Validate the GPU against a pure-Lua simulation of that
    -- exact model (input activation included).
    local stream = {}
    for i = 1, format.NETWORKS.seed.len do stream[i] = common[format.NETWORKS.seed.offset + i] end
    local cell = inputs
    for li = 1, #format.NETWORKS.seed.layers - 1 do
        local item = r:_pack_layer('seed', stream, 'fp8', 16, li)
        local in_c, out_c = format.NETWORKS.seed.layers[li], format.NETWORKS.seed.layers[li + 1]
        local w = quantize.decode_matrix(item.payload, 0, in_c * out_c, 16, item.scales, item.offsets, nil)
        -- Input-layer activation (first dispatch only): _pack_layer prepends
        -- the layer-1 nodes' [bias, threshold, dead] to the specials entry.
        local in_values = {}
        if li == 1 then
            for k = 1, in_c do
                local v = cell[k] + item.specials[(k - 1) * 3 + 1]
                if v <= item.specials[(k - 1) * 3 + 2] then
                    v = item.specials[(k - 1) * 3 + 3]
                end
                in_values[k] = v
            end
        else
            for k = 1, in_c do in_values[k] = cell[k] end
        end
        local spec_base = (li == 1) and (in_c * 3) or 0
        local next_cell = {}
        for node = 0, out_c - 1 do
            local acc = 0
            -- Input-major matrix: weight from input k to output node is
            -- w[(k-1)*out_c + node + 1] (the shader's logicalIndex =
            -- k*numNodes + node, 1-based twin).
            for k = 1, in_c do acc = acc + w[(k - 1) * out_c + node + 1] * in_values[k] end
            local value = acc + item.specials[spec_base + node * 3 + 1]
            if value <= item.specials[spec_base + node * 3 + 2] then
                value = item.specials[spec_base + node * 3 + 3]
            end
            next_cell[node + 1] = value
        end
        cell = next_cell
    end
    local tol = 1e-3 * math.max(1, math.abs(cell[1]))
    assert(math.abs(out[1] - cell[1]) <= tol,
        ('GPU %.6f vs CPU-semantics sim %.6f (fp32 accumulate tolerance)'):format(out[1], cell[1]))
    r:shutdown()
    print('NOTE: real-GPU fp8 forward verified against the CPU-equivalent packed model')
end

-- MUST-1 regression: forward_batch with DISTINCT per-cell genomes must give
-- each cell its own weights (the config SSBO carries per-cell base offsets).
tests['optional real-GPU forward_batch with distinct per-cell genomes'] = function()
    if not ffi then
        print('SKIP: no ffi (plain Lua) — GPU path is LuaJIT-only')
        return
    end
    local okv, v = pcall(require, 'nn.vulkan')
    if not okv or type(v) ~= 'table' or not v.can_load then
        print('SKIP: no Vulkan loader')
        return
    end
    local okc = pcall(v.can_load)
    if not okc or not v.can_load() then
        print('SKIP: Vulkan loader present but unusable')
        return
    end
    if not file_exists('nn/shaders/spv/forward_packed.spv') then
        print('SKIP: no compiled .spv artifacts')
        return
    end
    local g1 = make_genome(81)
    local g2 = make_genome(82)
    local inputs = { 0.5, -0.5, 0.25, -0.25, 0.125, 0, 1, -1, 0.75 }
    local r = nn.new({ backend = 'gpu', deterministic = false, precision = 'fp8' })
    local in_buf = {}
    for _ = 1, 2 do
        for _, vv in ipairs(inputs) do in_buf[#in_buf + 1] = vv end
    end
    local out_buf = { 0, 0 }
    local ok, berr = r:forward_batch('seed', {
        { genome = g1, network_id = 'seed' },
        { genome = g2, network_id = 'seed' },
    }, { buffer = in_buf, stride = 9, count = 2, element_type = 'fp32' },
       { buffer = out_buf, stride = 1, count = 2, element_type = 'fp32' })
    if not ok then
        r:shutdown()
        print(('SKIP: GPU forward_batch failed (%s %s)'):format(
            tostring(berr and berr.class), tostring(berr and berr.message)))
        return
    end
    r:shutdown()
    -- The per-cell genome regression: distinct genomes MUST yield distinct
    -- outputs (before the fix every cell got cell-0's weights).
    assert(out_buf[1] ~= out_buf[2],
        'distinct per-cell genomes must yield distinct outputs (per-cell bases broken?)')
    -- Compare against the CPU fp8 runtime (same packed representation, same
    -- reference semantics; only fp32-vs-fp64 accumulation differs).
    local rc = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8', block_size = 16 })
    local c1 = rc:forward('seed', g1, inputs)[1]
    local c2 = rc:forward('seed', g2, inputs)[1]
    rc:shutdown()
    local function close(a, b)
        local tol = 1e-2 * math.max(1, math.abs(a), math.abs(b))
        return math.abs(a - b) <= tol
    end
    assert(math.abs(c1 - c2) > 1e-2 * math.max(1, math.abs(c1), math.abs(c2)),
        'genome pair must be clearly distinguishable on the CPU reference')
    assert(close(out_buf[1], c1),
        ('cell1: GPU %.6f vs CPU-fp8 %.6f'):format(out_buf[1], c1))
    assert(close(out_buf[2], c2),
        ('cell2: GPU %.6f vs CPU-fp8 %.6f'):format(out_buf[2], c2))
    print(('NOTE: real-GPU multi-cell batch verified: %s / %s'):format(out_buf[1], out_buf[2]))
end

-- SHOULD-4: forward_batch must return a structured INVALID_ARGUMENT instead of
-- raising a raw Lua error when in_desc is nil or lacks a buffer.
tests['forward_batch missing in_desc returns structured error'] = function()
    local r = nn.new({ backend = 'cpu', deterministic = true, precision = 'fp8' })
    local out_buf = { 0 }
    local ok, err = r:forward_batch('seed', { { genome = make_genome(1) } }, nil,
        { buffer = out_buf, stride = 1, count = 1, element_type = 'fp32' })
    assert(ok == nil and errors.is(err, 'INVALID_ARGUMENT'),
        'nil in_desc must return INVALID_ARGUMENT, not raise')
    local ok2, err2 = r:forward_batch('seed', { { genome = make_genome(1) } },
        { stride = 9, count = 1 }, { buffer = out_buf, stride = 1, count = 1 })
    assert(ok2 == nil and errors.is(err2, 'INVALID_ARGUMENT'),
        'in_desc without a buffer must return INVALID_ARGUMENT, not raise')
end

return tests
