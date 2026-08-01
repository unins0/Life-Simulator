-- tests/test_compat_ai.lua — compatibility adapter tests.
--
-- nn/compat_ai_module.lua must be behaviorally equivalent to ai_module
-- (genWeights / mutateWeights / run / acquire / release) while running on
-- the standalone nn/ package, and must stay pure (no love / game-module
-- requires). Runs under BOTH `lua` (table genomes) and `luajit` (ffi cdata
-- genomes); equivalence is checked against the REAL ai_module on real
-- genomes produced by ai_module.genWeights.

package.path = './?.lua;./?/init.lua;' .. package.path

local shares    = require('shares')
local ai_module = require('ai_module')
local compat    = require('nn.compat_ai_module')
local H         = require('tests.helpers')

local tests = {}

local HAS_FFI
do
    local ok
    ok, _ = pcall(function() return require('ffi') end)
    HAS_FFI = ok
end

-- Adapter wired to the REAL game topology/config. The adapter itself must not
-- require shares; all configuration arrives through the constructor.
local adapter = compat.new({
    topology = {
        { id = 'seed',   layers = shares.AI_LAYERS_SEED },
        { id = 'spore',  layers = shares.AI_LAYERS_SPORE },
        { id = 'sprout', layers = shares.AI_LAYERS_SPROUT },
    },
    genome_len = shares.AI_LEN_COMMON,
    default_network = 'seed',
})

local NETWORKS = {
    { id = 'seed',   layers = shares.AI_LAYERS_SEED,   offset = shares.AI_OFFSET_SEED },
    { id = 'spore',  layers = shares.AI_LAYERS_SPORE,  offset = shares.AI_OFFSET_SPORE },
    { id = 'sprout', layers = shares.AI_LAYERS_SPROUT, offset = shares.AI_OFFSET_SPROUT },
}

local INPUT_KINDS = { 'zeros', 'ones', 'alt', 'neg', 'extreme', 'rand' }

local function make_inputs(in_size, kind)
    local t = {}
    for i = 1, in_size do
        if kind == 'zeros' then
            t[i] = 0
        elseif kind == 'ones' then
            t[i] = 1
        elseif kind == 'alt' then
            t[i] = (i % 2 == 1) and 0.5 or -0.5
        elseif kind == 'neg' then
            t[i] = -i * 0.1
        elseif kind == 'extreme' then
            t[i] = (i % 2 == 1) and 1e4 or -1e4
        else
            t[i] = math.random() * 2 - 1
        end
    end
    return t
end

-- Copy a genome (wrapper or raw table/cdata) into a plain 1-based table.
local function copy_genome(entry)
    local w = entry.data or entry
    local is_cdata = type(w) == 'cdata'
    local out = {}
    for i = 1, #w do out[i] = is_cdata and w[i - 1] or w[i] end
    return out
end

-- Place a genome at a high slot index (addGenome never scans that far, so
-- tests cannot clobber the real slot store).
local function place_at(store, idx, genome)
    store[idx] = { data = genome }
end

-- ---------------------------------------------------------------------------
-- 1. Configuration mirrors the game.
-- ---------------------------------------------------------------------------
tests['adapter topology mirrors the game config'] = function()
    assert(adapter.genome_len == shares.AI_LEN_COMMON,
        ('genome_len %d, expected AI_LEN_COMMON %d')
        :format(adapter.genome_len, shares.AI_LEN_COMMON))
    assert(adapter.networks_by_id.seed.offset == shares.AI_OFFSET_SEED)
    assert(adapter.networks_by_id.spore.offset == shares.AI_OFFSET_SPORE)
    assert(adapter.networks_by_id.sprout.offset == shares.AI_OFFSET_SPROUT)
    assert(adapter.networks_by_id.seed.len == shares.AI_LEN_SEED)
    assert(adapter.networks_by_id.spore.len == shares.AI_LEN_SPORE)
    assert(adapter.networks_by_id.sprout.len == shares.AI_LEN_SPROUT)
    assert(adapter.networks_by_layers['9,12,1'] == 'seed')
    assert(adapter.networks_by_layers['6,16,1'] == 'spore')
    assert(adapter.networks_by_layers['9,18,16,3'] == 'sprout')
    assert(adapter.GENOME_INIT_MULT == ai_module.GENOME_INIT_MULT)
    assert(adapter.GENOME_MUTATION_STRENGHT == ai_module.GENOME_MUTATION_STRENGHT)
    -- A default-constructed instance must also mirror the game.
    local dflt = compat.new()
    assert(dflt.genome_len == shares.AI_LEN_COMMON)
    assert(dflt.networks_by_id.sprout.offset == shares.AI_OFFSET_SPROUT)
    assert(dflt.default_network == 'seed')
end

-- ---------------------------------------------------------------------------
-- 2. run() equivalence vs ai_module on all topologies (real genomes, several
-- hand-crafted + random inputs; tolerance 1e-9, fp64 both sides).
-- ---------------------------------------------------------------------------
tests['run matches ai_module on all topologies'] = function()
    math.randomseed(42)
    local slot = ai_module.genWeights()
    local entry = shares.CELL_GENOMES[slot] -- wrapper (cdata on luajit)

    for _, net in ipairs(NETWORKS) do
        for _, kind in ipairs(INPUT_KINDS) do
            for trial = 1, 3 do
                local inputs = make_inputs(net.layers[1], kind)
                local expected = ai_module.run(entry, net.layers, net.offset, inputs)
                local actual = adapter.run(entry, net.layers, net.offset, inputs)
                H.assertClose(actual, expected, 1e-9,
                    ('compat vs ai_module (%s, %s, trial %d)'):format(net.id, kind, trial))
            end
        end
    end
end

-- ---------------------------------------------------------------------------
-- 3. run() equivalence on raw table genomes (legacy path both sides).
-- ---------------------------------------------------------------------------
tests['run matches ai_module on raw table genomes'] = function()
    math.randomseed(7)
    local slot = ai_module.genWeights()
    local genome = H.dataCopy(slot) -- pure 1-based table
    for _, net in ipairs(NETWORKS) do
        local inputs = make_inputs(net.layers[1], 'rand')
        local expected = ai_module.run(genome, net.layers, net.offset, inputs)
        local actual = adapter.run(genome, net.layers, net.offset, inputs)
        H.assertClose(actual, expected, 1e-9, ('raw table (%s)'):format(net.id))
    end
end

-- ---------------------------------------------------------------------------
-- 4. FFI vs table paths: on luajit ai_module stores cdata genomes; the
-- adapter must accept cdata and produce the same outputs as the table copy.
-- ---------------------------------------------------------------------------
tests['run matches on ffi cdata genomes (luajit only)'] = function()
    if not HAS_FFI then
        print('SKIP: no ffi (plain lua) — cdata genomes do not exist')
        return
    end
    math.randomseed(11)
    local slot = ai_module.genWeights()
    local entry = shares.CELL_GENOMES[slot]
    assert(type(entry.data) == 'cdata', 'luajit genomes must be stored as cdata')
    local tbl = H.dataCopy(slot)
    for _, net in ipairs(NETWORKS) do
        local inputs = make_inputs(net.layers[1], 'alt')
        local via_cdata = adapter.run(entry, net.layers, net.offset, inputs)
        local via_table = adapter.run(tbl, net.layers, net.offset, inputs)
        H.assertClose(via_cdata, via_table, 1e-9, ('cdata vs table (%s)'):format(net.id))
        local via_ai = ai_module.run(entry, net.layers, net.offset, inputs)
        H.assertClose(via_cdata, via_ai, 1e-9, ('cdata vs ai_module (%s)'):format(net.id))
    end
end

-- ---------------------------------------------------------------------------
-- 5. The adapter routes through the nn cpu runtime (fp32-exact) when asked
--    for the default 'auto' backend.
-- ---------------------------------------------------------------------------
tests['adapter runs on the nn package'] = function()
    math.randomseed(3)
    local slot = adapter.genWeights()
    local out = adapter.run(adapter.genomes[slot],
        shares.AI_LAYERS_SEED, shares.AI_OFFSET_SEED, { 1, 1, 1, 1, 1, 1, 1, 1, 1 })
    assert(out and #out == 1 and out[1] == out[1], 'seed run must return one finite output')
    local rt = adapter._runtime
    assert(rt and rt.backend == 'cpu', 'adapter must route through the nn cpu runtime')
    assert(rt.precision == 'fp32', 'adapter must use fp32 (exact) precision')
    assert(rt:capabilities().supported_networks[1] == 'seed')
end

-- ---------------------------------------------------------------------------
-- 6. cpu (runtime) and reference backends both agree with ai_module.
-- ---------------------------------------------------------------------------
tests['cpu and reference backends agree with ai_module'] = function()
    local ref_adapter = compat.new({
        topology = {
            { id = 'seed',   layers = shares.AI_LAYERS_SEED },
            { id = 'spore',  layers = shares.AI_LAYERS_SPORE },
            { id = 'sprout', layers = shares.AI_LAYERS_SPROUT },
        },
        genome_len = shares.AI_LEN_COMMON,
        backend = 'reference',
    })
    math.randomseed(31)
    local slot = ai_module.genWeights()
    local entry = shares.CELL_GENOMES[slot]
    for _, net in ipairs(NETWORKS) do
        local inputs = make_inputs(net.layers[1], 'alt')
        local expected = ai_module.run(entry, net.layers, net.offset, inputs)
        local via_ref = ref_adapter.run(entry, net.layers, net.offset, inputs)
        H.assertClose(via_ref, expected, 1e-9, ('reference backend (%s)'):format(net.id))
        local via_cpu = adapter.run(entry, net.layers, net.offset, inputs)
        H.assertClose(via_cpu, expected, 1e-9, ('cpu backend (%s)'):format(net.id))
    end
end

-- ---------------------------------------------------------------------------
-- 7. Builtin fallback (nn package unavailable) is still bit-equivalent.
-- ---------------------------------------------------------------------------
tests['builtin fallback forward agrees with ai_module'] = function()
    local builtin = compat.new({
        topology = {
            { id = 'seed',   layers = shares.AI_LAYERS_SEED },
            { id = 'spore',  layers = shares.AI_LAYERS_SPORE },
            { id = 'sprout', layers = shares.AI_LAYERS_SPROUT },
        },
        genome_len = shares.AI_LEN_COMMON,
        backend = 'builtin',
    })
    math.randomseed(17)
    local slot = ai_module.genWeights()
    local entry = shares.CELL_GENOMES[slot]
    for _, net in ipairs(NETWORKS) do
        local inputs = make_inputs(net.layers[1], 'rand')
        local expected = ai_module.run(entry, net.layers, net.offset, inputs)
        local got = builtin.run(entry, net.layers, net.offset, inputs)
        H.assertClose(got, expected, 1e-9, ('builtin fallback (%s)'):format(net.id))
    end
end

-- ---------------------------------------------------------------------------
-- 8. genWeights: valid length, fresh counter, range, run round-trip.
-- ---------------------------------------------------------------------------
tests['genWeights produces a valid genome that round-trips'] = function()
    math.randomseed(99)
    local slot = adapter.genWeights()
    local place = adapter.genomes[slot]
    assert(place and place.data, 'genWeights must create a slot entry')
    assert(#place.data == adapter.genome_len,
        ('genome length %d, expected %d'):format(#place.data, adapter.genome_len))
    assert(place.counter == 0, 'fresh genome must start with counter 0')
    for i = 1, #place.data do
        assert(math.abs(place.data[i]) <= adapter.GENOME_INIT_MULT + 1e-9,
            ('weight[%d] out of range: %g'):format(i, place.data[i]))
    end

    -- Own run round-trips: same genome + inputs -> identical outputs, finite.
    for _, net in ipairs(NETWORKS) do
        local inputs = make_inputs(net.layers[1], 'rand')
        local o1 = adapter.run(place, net.layers, net.offset, inputs)
        local o2 = adapter.run(place, net.layers, net.offset, inputs)
        H.assertEqual(o1, o2, ('round-trip (%s)'):format(net.id))
        for o = 1, #o1 do
            assert(o1[o] == o1[o], ('output %d must be finite (%s)'):format(o, net.id))
        end
    end

    -- run(genome, inputs) convenience form uses the configured default network.
    local via2 = adapter.run(place, make_inputs(9, 'zeros'))
    local via4 = adapter.run(place, shares.AI_LAYERS_SEED, shares.AI_OFFSET_SEED,
        make_inputs(9, 'zeros'))
    H.assertEqual(via2, via4, '2-arg convenience form must match the 4-arg seed call')
end

-- ---------------------------------------------------------------------------
-- 9. genWeights determinism: same seed -> bit-identical to ai_module.
-- ---------------------------------------------------------------------------
tests['genWeights matches ai_module under the same seed'] = function()
    math.randomseed(4242)
    local s1 = ai_module.genWeights()
    local d1 = H.dataCopy(s1)
    math.randomseed(4242)
    local s2 = adapter.genWeights()
    local d2 = adapter.genomes[s2].data
    assert(#d2 == #d1, ('length %d vs %d'):format(#d2, #d1))
    for i = 1, #d1 do
        assert(d2[i] == d1[i],
            ('genWeights diverged at %d: %.17g vs %.17g'):format(i, d2[i], d1[i]))
    end
end

-- ---------------------------------------------------------------------------
-- 10. mutateWeights: length kept, parent untouched, differs, deviation
--     bounded by strength, and the mutated genome still runs.
-- ---------------------------------------------------------------------------
tests['mutateWeights changes weights and still runs'] = function()
    math.randomseed(123)
    local parent = adapter.genWeights()
    local parentEntry = adapter.genomes[parent] -- object reference (slot may be reused)
    local snapshot = copy_genome(parentEntry)

    local child = adapter.mutateWeights(parent)
    local childEntry = adapter.genomes[child]
    assert(childEntry ~= parentEntry, 'mutate must return a new genome object')
    assert(#childEntry.data == adapter.genome_len,
        ('mutated genome has %d weights, expected %d')
        :format(#childEntry.data, adapter.genome_len))

    -- Parent object untouched (copy-on-write).
    for i = 1, #snapshot do
        assert(parentEntry.data[i] == snapshot[i],
            ('parent data changed at %d'):format(i))
    end

    -- Child differs in at least one position.
    local changed = false
    for i = 1, #snapshot do
        if childEntry.data[i] ~= snapshot[i] then changed = true break end
    end
    assert(changed, 'mutateWeights returned a genome identical to the parent')

    -- Single-step distinct-position mutation: |dev| < strength per position.
    local strength = adapter.GENOME_MUTATION_STRENGHT
    for i = 1, #snapshot do
        local dev = math.abs(childEntry.data[i] - snapshot[i])
        assert(dev <= strength + 1e-9,
            ('deviation %.10g at %d exceeds strength %g'):format(dev, i, strength))
    end

    -- Mutated genome still runs on every topology.
    for _, net in ipairs(NETWORKS) do
        local inputs = make_inputs(net.layers[1], 'rand')
        local out = adapter.run(childEntry, net.layers, net.offset, inputs)
        assert(#out == net.layers[#net.layers],
            ('mutated run (%s) must produce %d outputs')
            :format(net.id, net.layers[#net.layers]))
        for o = 1, #out do assert(out[o] == out[o], 'output must be finite') end
    end
end

-- ---------------------------------------------------------------------------
-- 11. mutateWeights determinism: same seed + same parent -> bit-identical to
--     ai_module (identical RNG draw order).
-- ---------------------------------------------------------------------------
tests['mutateWeights matches ai_module under the same seed'] = function()
    math.randomseed(2024)
    local gA = ai_module.genWeights()
    local P = H.dataCopy(gA) -- shared parent genome (plain table)

    -- Same parent placed in both stores at high, never-scanned slots.
    local SA, SB = 800011, 800012
    place_at(shares.CELL_GENOMES, SA, P)
    place_at(adapter.genomes, SB, P)

    math.randomseed(77)
    local childA = ai_module.mutateWeights(SA)
    local dA = H.dataCopy(childA)

    math.randomseed(77)
    local childB = adapter.mutateWeights(SB)
    local dB = adapter.genomes[childB].data

    assert(#dB == #dA, ('length %d vs %d'):format(#dB, #dA))
    for i = 1, #dA do
        assert(dB[i] == dA[i],
            ('mutate diverged at %d: %.17g vs %.17g'):format(i, dB[i], dA[i]))
    end

    -- Cleanup: drop the raw high-slot parents.
    shares.CELL_GENOMES[SA] = nil
    adapter.genomes[SB] = nil
end

-- ---------------------------------------------------------------------------
-- 12. acquire/release lifecycle mirrors ai_module (counter tracking, double
--     release errors, missing-slot errors/no-op, bounded slot reuse).
-- ---------------------------------------------------------------------------
tests['acquire/release lifecycle mirrors ai_module'] = function()
    math.randomseed(5)
    local slot = adapter.genWeights()
    local place = adapter.genomes[slot]
    assert(place.counter == 0, 'fresh genome must start with counter 0')

    for i = 1, 5 do
        adapter.acquire(slot)
        assert(place.counter == 1, ('after acquire #%d counter = %d, expected 1')
            :format(i, place.counter))
        adapter.release(slot)
        assert(place.counter == 0, ('after release #%d counter = %d, expected 0')
            :format(i, place.counter))
    end
    assert(place.counter == 0, 'counter must return to 0 after balanced cycles')

    -- Double release of a live slot errors with a message.
    adapter.acquire(slot)
    adapter.release(slot)
    local ok, err = pcall(adapter.release, slot)
    assert(not ok, 'second release must error (counter <= 0)')
    assert(type(err) == 'string' and err ~= '', 'release error must carry a message')

    -- Acquire of a missing slot errors.
    local ok2, err2 = pcall(adapter.acquire, 999999)
    assert(not ok2, 'acquire of missing slot must error')
    assert(type(err2) == 'string' and err2 ~= '', 'acquire error must carry a message')

    -- Release of a missing slot is an idempotent no-op.
    local before = #adapter.genomes
    assert(adapter.release(999999) == nil, 'release of missing slot returns nothing')
    assert(#adapter.genomes == before, 'no-op release must not change the store')

    -- Freed slots are reused: the max index must not grow.
    local max1 = 0
    for i = 1, 20 do
        local s = adapter.genWeights()
        if s > max1 then max1 = s end
        adapter.acquire(s)
        adapter.release(s)
    end
    local max2 = 0
    for i = 1, 20 do
        local s = adapter.genWeights()
        if s > max2 then max2 = s end
    end
    assert(max2 <= max1,
        ('slots not reused: max index %d after release, was %d'):format(max2, max1))
end

-- ---------------------------------------------------------------------------
-- 13. Module-level surface (drop-in) and slot-number run.
-- ---------------------------------------------------------------------------
tests['module-level surface and slot-number run'] = function()
    math.randomseed(8)
    local s = compat.genWeights() -- module-level default instance
    local def = compat.default()
    assert(def.genomes[s], 'module surface must use a shared default instance')

    local out = compat.run(s, shares.AI_LAYERS_SPORE, shares.AI_OFFSET_SPORE,
        { 1, 1, 1, 1, 1, 1 })
    assert(#out == 1 and out[1] == out[1], 'slot-number run must work')

    compat.acquire(s)
    assert(def.genomes[s].counter == 1, 'module acquire must track the counter')
    compat.release(s)
    assert(def.genomes[s].counter == 0, 'module release must track the counter')
end

-- ---------------------------------------------------------------------------
-- 14. Purity: the adapter source must not require love / game modules.
-- ---------------------------------------------------------------------------
tests['compat adapter source is pure'] = function()
    local f = io.open('nn/compat_ai_module.lua', 'rb')
    assert(f, 'nn/compat_ai_module.lua must exist')
    local src = f:read('*a')
    f:close()
    for _, mod in ipairs({ 'love', 'shares', 'sim_module', 'cell_module', 'ai_module' }) do
        assert(not src:find("require('" .. mod, 1, true)
            and not src:find('require("' .. mod, 1, true),
            ('nn/compat_ai_module.lua must not require %q'):format(mod))
    end
end

-- ---------------------------------------------------------------------------
-- 15. Purity: the core nn/ modules must stay free of game requires (mirrors
--     the test_nn_core scan; vulkan_worker's love.thread is LÖVE-only and
--     outside the pure core list, as in the existing test).
-- ---------------------------------------------------------------------------
tests['core nn modules stay pure (no game requires)'] = function()
    local core = {
        'nn/errors.lua', 'nn/capabilities.lua', 'nn/api.lua', 'nn/format.lua',
        'nn/serialize.lua', 'nn/quantize.lua', 'nn/corpus.lua', 'nn/reference.lua',
        'nn/cpu.lua', 'nn/runtime.lua', 'nn/init.lua',
    }
    local forbidden = { 'love', 'shares', 'sim_module', 'cell_module', 'ai_module' }
    for _, path in ipairs(core) do
        local f = io.open(path, 'rb')
        assert(f, ('missing core file %s'):format(path))
        local src = f:read('*a')
        f:close()
        for _, mod in ipairs(forbidden) do
            assert(not src:find("require('" .. mod, 1, true)
                and not src:find('require("' .. mod, 1, true),
                ('%s must not require %q'):format(path, mod))
        end
    end
end

return tests
