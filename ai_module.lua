local M = {}

-- Configuration
M.GENOME_INIT_MULT         = 100.0
M.GENOME_MUTATION_STRENGHT = 0.1
M.GENOME_MUTATION_CHANCE   = 0.1

local shares = require('shares')
local rand = math.random

local AI_LEN_COMMON = shares.AI_LEN_COMMON

-- FFI backend (LuaJIT inside LÖVE; plain lua falls back to tables).
-- Arrays are 'double' (not 'float'): float32 rounding breaks the strict mutation
-- bound (|new-old| measured up to 0.1000003815 vs strength 0.1) and the
-- ffi/table run equivalence at 1e-9 (measured |ffi-run - table-run| up to ~143
-- with float scratch, 0 with double).
local HAS_FFI, ffi = pcall(function() return require('ffi') end)

-- Max nodes across the three networks (SEED 9+12+1=22, SPORE 6+16+1=23,
-- SPROUT 9+18+16+3=46) — size of the shared data scratch buffer.
local max_nodes = 0
do
    local networks = {shares.AI_LAYERS_SEED, shares.AI_LAYERS_SPORE, shares.AI_LAYERS_SPROUT}
    for _, layers in ipairs(networks) do
        local n = 0
        for i = 1, #layers do n = n + layers[i] end
        if n > max_nodes then max_nodes = n end
    end
end

local data_scratch, w_scratch
if HAS_FFI then
    data_scratch = ffi.new('double[?]', max_nodes)
    w_scratch    = ffi.new('double[?]', AI_LEN_COMMON)
end

-- Genome slot: {data = <weights>, counter = <refs>, is_ffi = <bool>}
-- Weights layout: (bias, threshold, dead-constant, *weights to next layer) per node
local function addGenome(genome)
    local CELL_GENOMES = shares.CELL_GENOMES
    if genome.data == nil then
        genome = {data = genome, is_ffi = false} -- wrap raw weights
    end
    local idx = 0
    while true do
        idx = idx + 1
        local place = CELL_GENOMES[idx]
        if place == nil or place.counter == 0 then
            CELL_GENOMES[idx] = genome
            genome.counter = 0
            return idx
        end
    end
end

-- Ownership of genome references (BUG-1): every live AI cell (typ >= 4) owns
-- exactly one reference in cell[11]. acquire/release keep the slot counter in
-- sync with the number of live references so slots with counter == 0 can be
-- reused by addGenome and CELL_GENOMES stays dense and bounded.
function M.acquire(slot)
    local place = shares.CELL_GENOMES[slot]
    if place == nil then
        error(('genome acquire: slot %s does not exist'):format(tostring(slot)), 2)
    end
    place.counter = place.counter + 1
    return slot
end

-- Idempotent for already-removed slots (CELL_GENOMES[slot] == nil); releasing
-- a live slot whose counter is already <= 0 is an invariant violation -> error.
function M.release(slot)
    local place = shares.CELL_GENOMES[slot]
    if place == nil then return end -- slot already removed/never existed
    if place.counter == nil or place.counter <= 0 then
        error(('genome release: slot %s has counter %s (double release?)'):format(
            tostring(slot), tostring(place.counter)), 2)
    end
    place.counter = place.counter - 1
end

-- Distinct random positions to mutate: Fisher-Yates shuffle of {1..w_len},
-- take k = rand(1, w_len) — no position is ever mutated twice, so a position
-- deviates by at most (rand()-0.5)*2*strength ∈ (-strength, strength).
local function samplePositions(w_len)
    local perm = {}
    for i = 1, w_len do perm[i] = i end
    for i = w_len, 2, -1 do
        local j = rand(1, i)
        perm[i], perm[j] = perm[j], perm[i]
    end
    return perm, rand(1, w_len)
end

-- ---------------------------------------------------------------------------
-- Table backend (plain lua fallback)
-- ---------------------------------------------------------------------------
local function genWeights_table(mult)
    local scale = (mult or M.GENOME_INIT_MULT) * 2
    local data, idx = {}, 1
    for i = 1, AI_LEN_COMMON do
        data[i] = (rand() - 0.5) * scale
    end
    return addGenome({data = data, is_ffi = false})
end

local function mutateWeights_table(genome_idx, strenght)
    local slot = shares.CELL_GENOMES[genome_idx]
    local weights = slot.data or slot -- raw legacy table compat
    strenght    = strenght or M.GENOME_MUTATION_STRENGHT
    local scale = strenght * 2
    local new_weights = {}
    for i = 1, AI_LEN_COMMON do new_weights[i] = weights[i] end
    local perm, k = samplePositions(AI_LEN_COMMON)
    for p = 1, k do
        local idx = perm[p]
        new_weights[idx] = new_weights[idx] + (rand() - 0.5) * scale
    end
    return addGenome({data = new_weights, is_ffi = false})
end

-- Module-level scratch for the table backend: reused across calls (no per-call
-- allocation). Slots are nilled before each run, so `(data[i] or 0.0)` reads
-- stay exact. result is still a fresh table — never the shared scratch.
local data_scratch_t = {}

local function run_table(weights, layers, idx_offset, inputs)
    local len    = #layers
    local idx    = 1 + idx_offset
    local offset = 0
    local w      = weights.data or weights -- wrapper or raw legacy table
    local data   = data_scratch_t
    local total_nodes = 0
    for i = 1, len do total_nodes = total_nodes + layers[i] end
    for i = 1, total_nodes do data[i] = nil end -- zero only the used slots
    for i = 1, layers[1] do data[i] = inputs[i] end

    for i = 2, len do
        local layer       = layers[i]
        local prev_layer  = layers[i - 1]
        local next_offset = offset + prev_layer
        for j = 1, prev_layer do
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

-- ---------------------------------------------------------------------------
-- FFI backend: hot loop works on cdata; all weights 0-based (cursor idx is
-- 1-based, so a table access w[i] becomes w[i-1]).
-- ---------------------------------------------------------------------------
local function genWeights_ffi(mult)
    local scale = (mult or M.GENOME_INIT_MULT) * 2
    local data = ffi.new('double[?]', AI_LEN_COMMON)
    for i = 1, AI_LEN_COMMON do
        data[i - 1] = (rand() - 0.5) * scale
    end
    return addGenome({data = data, is_ffi = true})
end

local function mutateWeights_ffi(genome_idx, strenght)
    local slot = shares.CELL_GENOMES[genome_idx]
    local weights = slot.data or slot -- raw legacy table compat
    strenght    = strenght or M.GENOME_MUTATION_STRENGHT
    local scale = strenght * 2
    local new_data = ffi.new('double[?]', AI_LEN_COMMON)
    if type(weights) == 'cdata' then
        ffi.copy(new_data, weights, AI_LEN_COMMON * ffi.sizeof('double'))
    else
        for i = 1, AI_LEN_COMMON do new_data[i - 1] = weights[i] or 0 end
    end
    local perm, k = samplePositions(AI_LEN_COMMON)
    for p = 1, k do
        local idx = perm[p]
        new_data[idx - 1] = new_data[idx - 1] + (rand() - 0.5) * scale
    end
    return addGenome({data = new_data, is_ffi = true})
end

local function run_ffi(weights, layers, idx_offset, inputs)
    local len    = #layers
    local idx    = 1 + idx_offset
    local offset = 0
    -- wrapper (cdata) — use directly; table wrapper / raw legacy table — copy
    -- into the scratch once (~1003 doubles) and compute on cdata.
    local w0 = weights.data or weights
    local w
    if type(w0) == 'cdata' then
        w = w0
    else
        for i = 1, AI_LEN_COMMON do w_scratch[i - 1] = w0[i] or 0 end
        w = w_scratch
    end

    local total = 0
    for i = 1, len do total = total + layers[i] end
    ffi.fill(data_scratch, total * ffi.sizeof('double'))
    local data = data_scratch
    for i = 1, layers[1] do data[i - 1] = inputs[i] or 0 end

    for i = 2, len do
        local layer       = layers[i]
        local prev_layer  = layers[i - 1]
        local next_offset = offset + prev_layer
        for j = 1, prev_layer do
            local value = data[j + offset - 1] + w[idx - 1]
            if value <= w[idx] then
                value = w[idx + 1]
            end

            for k = 1, layer do
                local ofs = next_offset + k - 1
                data[ofs] = data[ofs] + value * w[idx + k + 1]
            end
            idx = idx + layer + 3
        end
        offset = next_offset
    end

    -- Fresh Lua table — never the shared scratch (callers keep the reference).
    local result = {}
    for i = 1, layers[len] do
        local value = data[offset + i - 1] + w[idx - 1]
        if value <= w[idx] then
            value = w[idx + 1]
        end
        result[i] = value
        idx = idx + 3
    end
    return result
end

-- Backend selected once at module load.
M.genWeights    = HAS_FFI and genWeights_ffi or genWeights_table
M.mutateWeights = HAS_FFI and mutateWeights_ffi or mutateWeights_table
M.run           = HAS_FFI and run_ffi or run_table

return M
