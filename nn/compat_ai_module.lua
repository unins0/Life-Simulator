-- nn/compat_ai_module.lua — game <-> NN compatibility adapter.
--
-- Exposes ai_module's public surface (genWeights, mutateWeights, run,
-- acquire, release) implemented on top of the standalone nn/ package (cpu /
-- reference backends, never Vulkan). This is the ONLY module that translates
-- between the game's node-major AI_LEN_COMMON-double genome and the NN
-- package's (topology + per-network weights blob) representation.
--
-- Purity: this file requires only the nn package (pcall-gated, with a clean
-- built-in fallback) and must stay free of love and game-module requires.
-- Topology/config arrive via compat.new(opts); the genome-slot store is
-- injectable through opts.genomes (defaults to an internal table). The
-- module-level surface delegates to a lazily-created default instance, so
-- require('nn.compat_ai_module') works as a drop-in for ai_module.

local M = {}

local rand = math.random

-- Real game topology mirror (data only, no requires). Offsets and lengths
-- are computed from the order below; tests pin them against shares.lua.
local DEFAULT_NETWORKS = {
    { id = 'seed',   layers = { 9, 12, 1 } },
    { id = 'spore',  layers = { 6, 16, 1 } },
    { id = 'sprout', layers = { 9, 18, 16, 3 } },
}

-- Node-major stream length for a layer list (shares.countWeights formula):
--   n += layers[i] * (3 + layers[i+1]); n += layers[last] * 3
local function count_weights(layers)
    local n = 0
    for i = 1, #layers - 1 do
        n = n + layers[i] * (3 + layers[i + 1])
    end
    return n + layers[#layers] * 3
end

local function layers_key(layers)
    return table.concat(layers, ',')
end

-- --------------------------------------------------------------------------
-- nn package loader (pcall-gated; never loads Vulkan).
-- --------------------------------------------------------------------------
local NN

local function get_nn()
    if NN ~= nil then return NN end
    local ok, nn = pcall(require, 'nn')
    -- LuaJIT's default package.path lacks './?/init.lua'; retry once after
    -- appending it (same fix the repo's tests use for require('nn')).
    if not ok and not package.path:find('?/init%.lua', 1, true) then
        package.path = './?/init.lua;' .. package.path
        ok, nn = pcall(require, 'nn')
    end
    NN = ok and nn or false
    return NN
end

-- --------------------------------------------------------------------------
-- Clean fallback: exact ai_module.run-table semantics when nn is unavailable.
-- --------------------------------------------------------------------------
local scratch = {}

local function forward_builtin(w, layers, offset_idx, inputs)
    local len = #layers
    local idx = 1 + offset_idx
    local offset = 0
    local data = scratch
    local total = 0
    for i = 1, len do total = total + layers[i] end
    for i = 1, total do data[i] = nil end
    for i = 1, layers[1] do data[i] = inputs[i] or 0 end

    for i = 2, len do
        local layer = layers[i]
        local prev = layers[i - 1]
        local next_offset = offset + prev
        for j = 1, prev do
            local value = (data[j + offset] or 0.0) + (w[idx] or 0)
            if value <= (w[idx + 1] or 0) then
                value = w[idx + 2] or 0
            end
            for k = 1, layer do
                local ofs = next_offset + k
                data[ofs] = (data[ofs] or 0.0) + value * (w[idx + k + 2] or 0)
            end
            idx = idx + layer + 3
        end
        offset = next_offset
    end

    local result = {}
    for i = 1, layers[len] do
        local value = data[offset + i] + (w[idx] or 0)
        if value <= (w[idx + 1] or 0) then
            value = w[idx + 2] or 0
        end
        result[i] = value
        idx = idx + 3
    end
    return result
end

-- --------------------------------------------------------------------------
-- Slot lifecycle (mirrors ai_module.addGenome: the first slot whose entry is
-- nil or has counter == 0 is reused, so the store stays dense and bounded).
-- --------------------------------------------------------------------------
local function add_genome(self, genome)
    if genome.data == nil then
        genome = { data = genome, is_ffi = false }
    end
    local store = self.genomes
    local idx = 0
    while true do
        idx = idx + 1
        local place = store[idx]
        if place == nil or place.counter == 0 then
            store[idx] = genome
            genome.counter = 0
            return idx
        end
    end
end

-- Distinct random positions to mutate: Fisher-Yates shuffle, take the first
-- k = rand(1, n) — a position is never mutated twice, so each deviation is a
-- single step (rand()-0.5)*2*strength in (-strength, strength).
local function sample_positions(n)
    local perm = {}
    for i = 1, n do perm[i] = i end
    for i = n, 2, -1 do
        local j = rand(1, i)
        perm[i], perm[j] = perm[j], perm[i]
    end
    return perm, rand(1, n)
end

-- Read a genome's i-th 1-based weight across wrapper / raw / cdata forms.
local function genome_at(genome, i)
    local w = genome.data or genome
    if type(w) == 'cdata' then return w[i - 1] or 0 end
    return w[i] or 0
end

-- Translate a genome into a per-network fp64 stream (the nn weights blob).
local function slice_stream(genome, offset, len)
    local stream = {}
    for i = 1, len do stream[i] = genome_at(genome, offset + i) end
    return stream
end

-- --------------------------------------------------------------------------
-- Factory: compat.new(opts) -> adapter instance.
--   opts.topology        ordered array { {id=.., layers=..}, ... }; defaults
--                        to the real game topology seed/spore/sprout
--   opts.genome_len      common genome length (default computed from topology)
--   opts.default_network id used by the run(genome, inputs) convenience form
--   opts.backend         'auto'  -> cpu runtime for game topologies, else the
--                                   reference backend, else the builtin when
--                                   the nn package is unavailable
--                        'cpu' | 'reference' | 'builtin'
--   opts.precision       only 'fp32' is exact vs ai_module (default 'fp32')
--   opts.genomes         slot-store injection point (default internal table)
-- --------------------------------------------------------------------------
function M.new(opts)
    opts = opts or {}
    local order = opts.topology or DEFAULT_NETWORKS
    local networks, by_id, by_layers = {}, {}, {}
    local offset = 0
    for i, n in ipairs(order) do
        local id = n.id or n[1]
        local layers = n.layers or n[2]
        local net = { id = id, layers = layers, offset = offset, len = count_weights(layers) }
        networks[i] = net
        by_id[id] = net
        by_layers[layers_key(layers)] = id
        offset = offset + net.len
    end
    local first_id = order[1] and (order[1].id or order[1][1])

    local self = {
        networks = networks,
        networks_by_id = by_id,
        networks_by_layers = by_layers,
        genome_len = opts.genome_len or offset,
        default_network = opts.default_network or first_id,
        backend = opts.backend or 'auto',
        precision = opts.precision or 'fp32',
        genomes = opts.genomes or {},
        _runtime = false,
    }

    -- ai_module-compatible config constants.
    self.GENOME_INIT_MULT = 100.0
    self.GENOME_MUTATION_STRENGHT = 0.1
    self.GENOME_MUTATION_CHANCE = 0.1

    local function runtime()
        if self._runtime ~= false then return self._runtime end
        local nn_mod = get_nn()
        if not nn_mod then
            self._runtime = nil
            return nil
        end
        self._runtime = nn_mod.new({
            backend = 'cpu',
            deterministic = true,
            precision = self.precision,
        })
        return self._runtime
    end

    -- genWeights(mult?) -> slot: fresh node-major genome in the slot store.
    function self.genWeights(mult)
        local scale = (mult or self.GENOME_INIT_MULT) * 2
        local data = {}
        for i = 1, self.genome_len do
            data[i] = (rand() - 0.5) * scale
        end
        return add_genome(self, { data = data })
    end

    -- mutateWeights(slot, strenght?) -> new slot; the parent is untouched
    -- (copy-on-write), exactly like ai_module.
    function self.mutateWeights(slot, strenght)
        local place = self.genomes[slot]
        if place == nil then
            error(('genome mutate: slot %s does not exist'):format(tostring(slot)), 2)
        end
        strenght = strenght or self.GENOME_MUTATION_STRENGHT
        local scale = strenght * 2
        local new_data = {}
        for i = 1, self.genome_len do new_data[i] = genome_at(place, i) end
        local perm, k = sample_positions(self.genome_len)
        for p = 1, k do
            local i = perm[p]
            new_data[i] = new_data[i] + (rand() - 0.5) * scale
        end
        return add_genome(self, { data = new_data })
    end

    -- run(genome, layers, idx_offset, inputs) — ai_module's exact signature.
    -- Also accepts run(genome, inputs) using the configured default network
    -- and run(slot, ...) where slot indexes this adapter's store.
    function self.run(genome, layers, offset_idx, inputs)
        if inputs == nil then
            local def = by_id[self.default_network]
            if def == nil then
                error('compat run: 2-arg form needs a default_network (see compat.new)', 2)
            end
            inputs = layers -- 2nd arg is the inputs table in the 2-arg form
            offset_idx = def.offset
            layers = def.layers
        end
        if type(genome) == 'number' then
            local place = self.genomes[genome]
            if place == nil then
                error(('genome run: slot %s does not exist'):format(tostring(genome)), 2)
            end
            genome = place
        end
        local stream = slice_stream(genome, offset_idx, count_weights(layers))
        local nn_mod = get_nn()
        local want = self.backend
        if nn_mod and want ~= 'builtin' then
            local id = by_layers[layers_key(layers)]
            if id and (want == 'auto' or want == 'cpu') then
                local rt = runtime()
                if rt then
                    local out = rt:forward(id, stream, inputs)
                    if out then return out end
                end
            end
            -- Reference backend (and the fallback for unknown topologies).
            local ref = nn_mod.reference
            return ref.new_model(layers):load_weights(stream):forward(inputs)
        end
        return forward_builtin(stream, layers, 0, inputs)
    end

    -- Slot ownership: acquire/release mirror ai_module's counter contract.
    function self.acquire(slot)
        local place = self.genomes[slot]
        if place == nil then
            error(('genome acquire: slot %s does not exist'):format(tostring(slot)), 2)
        end
        place.counter = (place.counter or 0) + 1
        return slot
    end

    -- Idempotent for missing slots; releasing a live slot whose counter is
    -- already <= 0 is an invariant violation and errors (double release).
    function self.release(slot)
        local place = self.genomes[slot]
        if place == nil then return end
        if place.counter == nil or place.counter <= 0 then
            error(('genome release: slot %s has counter %s (double release?)'):format(
                tostring(slot), tostring(place.counter)), 2)
        end
        place.counter = place.counter - 1
    end

    function self.shutdown()
        if self._runtime then self._runtime:shutdown() end
        self._runtime = false
    end

    return self
end

-- --------------------------------------------------------------------------
-- Module-level surface backed by a lazily-created default instance.
-- --------------------------------------------------------------------------
local function default()
    if M._default == nil then M._default = M.new() end
    return M._default
end
M.default = default
M.genWeights = function(...) return default().genWeights(...) end
M.mutateWeights = function(...) return default().mutateWeights(...) end
M.run = function(...) return default().run(...) end
M.acquire = function(...) return default().acquire(...) end
M.release = function(...) return default().release(...) end
M.count_weights = count_weights

return M
