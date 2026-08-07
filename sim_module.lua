-- sim_module.lua — pure-Lua simulation core (no love.*).
-- Mechanically extracted from main.lua: tick(), addCell(), removeCell(),
-- calcSunFactor() and the data part of regenMap(). All constants and state are
-- read from the `state` argument (the shares table); all rendering goes through
-- the `view` callbacks (setCell / clearCell / updateMinerals) provided by
-- main.lua. Requires ai_module/cell_module (both are pure Lua too).
--
-- Behavior is 1:1 with the old main.lua except ONE structural change:
-- the apply-phase spore movement now uses moveCell() (the old code called
-- addCell() on the still-occupied source slot, so spores never moved).
-- Plus conservation fixes: death/spawn/transfer refunds (incl. the CELL_COSTS
-- reserve and the MAP_ENERGY ledger) make energy and minerals lossless.
--
-- Compute/apply contract: compute (the CELL_QUEUE loop) writes only to the
-- local BUFFER_* tables; apply is the sequential mutator (death/transfer/
-- spawn/update/movement). BUFFER_SPAWN carries only serializable number arrays.

local ai_module   = require('ai_module')
local cell_module = require('cell_module')

local M = {}

local rand  = math.random
local floor = math.floor
local pi2   = math.pi * 2

local x_offsets = {1, 0, -1, 0}
local y_offsets = {0, 1, 0, -1}

-- Sun factor formula from main.lua (DAY_DURATION, SUN_MIN/MAX, dsun, pi2 are
-- read from state or recomputed locally — the module has no shares upvalues).
function M.calcSunFactor(state, step)
    local DAY_DURATION = state.DAY_DURATION
    local dsun         = state.SUN_MAX - state.SUN_MIN
    local phase = (step % DAY_DURATION) / DAY_DURATION - 0.5
    return state.SUN_MIN + dsun * (0.5 - 0.5 * math.cos(pi2 * phase))
end

-- Same semantics as the old main.lua addCell: occupancy check, map write,
-- CELL_COUNTER + 1 and CELL_QUEUE[CELL_COUNTER] = idx. No genome acquire here
-- (the cell owns its reference from cell_module.initCell).
function M.addCell(state, cell, view)
    local MAP_CELLS = state.MAP_CELLS
    if not cell or MAP_CELLS[cell[1]] then return false end
    MAP_CELLS[cell[1]]        = cell
    state.MAP_TYPES[cell[1]]  = cell[2]

    state.CELL_COUNTER = state.CELL_COUNTER + 1
    state.CELL_QUEUE[state.CELL_COUNTER] = cell[1]
    view.setCell(cell[1], cell)

    -- MAP_ENERGY is a ledger of energy on empty tiles; when a cell appears on a
    -- tile, the stored energy moves into the cell (conservation: no sink, no loss).
    local e = state.MAP_ENERGY[cell[1]]
    if e and e ~= 0 then
        cell[4] = cell[4] + e
        state.MAP_ENERGY[cell[1]] = 0
    end
    return true
end

-- Same semantics as the old main.lua removeCell.
function M.removeCell(state, idx, view)
    local MAP_CELLS = state.MAP_CELLS
    local cell = MAP_CELLS[idx]
    if not cell then return false end
    MAP_CELLS[idx] = nil
    state.MAP_TYPES[idx] = nil

    -- Genome ownership: AI cells (typ >= 4) release their ref here so the
    -- slot counter can reach 0 and be reused (acquired in initCell).
    if cell[2] >= 4 then
        ai_module.release(cell[11])
    end
    view.clearCell(idx)
    return true
end

-- Apply-phase spore movement: moves the cell at `from` to `to`, keeping
-- MAP_CELLS/MAP_TYPES consistent. Does NOT touch CELL_COUNTER/CELL_QUEUE.
-- This is the (only) structural fix of this refactor: the old main.lua called
-- addCell(cell) while the source slot was still occupied, so spores never moved.
function M.moveCell(state, from, to, view)
    local MAP_CELLS = state.MAP_CELLS
    if from == to or not MAP_CELLS[from] or MAP_CELLS[to] then return false end
    local cell = MAP_CELLS[from]
    MAP_CELLS[from] = nil
    state.MAP_TYPES[from] = nil
    cell[1] = to
    MAP_CELLS[to] = cell
    state.MAP_TYPES[to] = cell[2]
    view.clearCell(from)
    view.setCell(to, cell)
    -- CELL_QUEUE stores cell indices; keep it in sync with the new position.
    local queue = state.CELL_QUEUE
    for qi = 1, state.CELL_COUNTER do
        if queue[qi] == from then queue[qi] = to; break end
    end
    return true
end

-- Data part of the old regenMap() (no graphics): fresh tables, counter/step/
-- sun_factor reset, minerals re-randomized. MAP_TYPES is 0-filled exactly like
-- the original regenMap so the Leaf/Root `if MAP_TYPES[parent] then` checks
-- behave identically (0 is truthy, nil is not).
function M.reset(state)
    local MAP_CELLS    = {}
    local MAP_TYPES    = {}
    local MAP_MINERALS = {}
    state.MAP_CELLS    = MAP_CELLS
    state.MAP_TYPES    = MAP_TYPES
    state.MAP_MINERALS = MAP_MINERALS
    state.MAP_ENERGY   = {}
    state.CELL_GENOMES = {}
    state.CELL_QUEUE   = {}
    state.CELL_COUNTER = 0
    state.step         = 0
    state.sun_factor   = 1.0

    local MINERALS_MIN, MINERALS_MAX = state.MINERALS_MIN, state.MINERALS_MAX
    for i = 1, state.MAP_SIZE do
        MAP_CELLS[i]    = nil
        MAP_TYPES[i]    = 0
        MAP_MINERALS[i] = rand(MINERALS_MIN, MINERALS_MAX)
    end
end

-- The old main.lua tick(), verbatim, with:
--   shares.X          -> state.X
--   step/sun_factor   -> state.step / state.sun_factor
--   updateMinerals()  -> view.updateMinerals()
--   cell_batch set*   -> view.setCell(idx, cell)
--   addCell(cell)     -> M.addCell(state, cell, view)
--   removeCell(idx)   -> M.removeCell(state, idx, view)
--   apply-phase move  -> M.moveCell(state, idx_from, idx_to, view)  (fixes spores)
--   regenMap() tail   -> return state.CELL_COUNTER <= 0
-- Returns true when no cells are left (main.lua then calls regenMap()).
function M.tick(state, view)
    local CELL_ENERGY_CONS  = state.CELL_ENERGY_CONS
    local CELL_AGES         = state.CELL_AGES
    local CELL_COSTS        = state.CELL_COSTS
    local CELL_GENOMES      = state.CELL_GENOMES
    local AI_LAYERS_SEED    = state.AI_LAYERS_SEED
    local AI_LAYERS_SPORE   = state.AI_LAYERS_SPORE
    local AI_LAYERS_SPROUT  = state.AI_LAYERS_SPROUT
    local AI_OFFSET_SEED    = state.AI_OFFSET_SEED
    local AI_OFFSET_SPORE   = state.AI_OFFSET_SPORE
    local AI_OFFSET_SPROUT  = state.AI_OFFSET_SPROUT
    local LEAF_ENERGY_GEN   = state.LEAF_ENERGY_GEN
    local ROOT_MINERAL_EXTR = state.ROOT_MINERAL_EXTR

    local MAP_CELLS    = state.MAP_CELLS
    local MAP_TYPES    = state.MAP_TYPES
    local MAP_MINERALS = state.MAP_MINERALS
    local MAP_ENERGY   = state.MAP_ENERGY
    local CELL_QUEUE   = state.CELL_QUEUE
    local idx2pos      = state.idx2pos
    local pos2idx      = state.pos2idx
    local initCell     = cell_module.initCell

    local BUFFER_ENERGY  = {}
    local BUFFER_MINERAL = {}
    local BUFFER_EXTR    = {} -- {from, to, ...}
    local BUFFER_SPAWN   = {}
    local BUFFER_DEATH   = {}
    local BUFFER_UPDATE  = {}
    local BUFFER_MOVING  = {} -- {from, to, ...}
    local BUFFER_RELEASE = {} -- genome slots to release (deferred from compute)
    local extr_idx, spawn_idx   = 0, 0
    local death_idx, update_idx = 0, 0
    local move_idx, release_idx = 0, 0
    local ai_calls              = 0

    -- Indices written into BUFFER_ENERGY/BUFFER_MINERAL this tick (dedup),
    -- so the transfer phase skips the full-map scan.
    local touched = {}
    local touched_set = {}
    local function track(i)
        if not touched_set[i] then
            touched_set[i] = true
            touched[#touched + 1] = i
        end
    end
    -- One reusable input table for all ai_module.run calls this tick.
    local data = {}
    -- Reusable result table: every run_into result is consumed within the
    -- same tick before the next run_into overwrites it.
    local ai_out = {}

    state.step = state.step + 1
    state.sun_factor = M.calcSunFactor(state, state.step)

    for i = 1, #CELL_QUEUE do
        local idx  = CELL_QUEUE[i]
        local cell = MAP_CELLS[idx]
        local typ  = MAP_TYPES[idx]

        if cell == nil then
            goto continue
        end

        cell[4] = cell[4] - CELL_ENERGY_CONS[typ]
        cell[6] = cell[6] + 1
        if cell[4] > 0 and cell[6] < CELL_AGES[typ] then

            if     typ == 1 then -- Leaf
                local parent_idx = cell[7]
                if MAP_TYPES[parent_idx] then
                    BUFFER_ENERGY[parent_idx] = (BUFFER_ENERGY[parent_idx] or 0.0) + LEAF_ENERGY_GEN * state.sun_factor
                    track(parent_idx)
                else
                    death_idx = death_idx + 1
                    BUFFER_DEATH[death_idx] = idx
                end

            elseif typ == 2 then -- Root
                local parent_idx = cell[7]
                if MAP_TYPES[parent_idx] then
                    local x, y = idx2pos(idx)
                    extr_idx = extr_idx + 2
                    -- Cardinal-direction extraction, never the Root's own tile.
                    local dir = rand(1, 4)
                    BUFFER_EXTR[extr_idx - 1] = pos2idx(x + x_offsets[dir], y + y_offsets[dir])
                    BUFFER_EXTR[extr_idx]     = cell[7]
                else
                    death_idx = death_idx + 1
                    BUFFER_DEATH[death_idx] = idx
                end

            elseif typ == 3 then -- Stem
                local n = 1
                local targets = {}
                for j = 1, 3 do
                    local t_idx = cell[7 + j]
                    local target = MAP_TYPES[t_idx]
                    if target and target > 2 then
                        targets[n] = t_idx
                        n = n + 1
                    end
                end
                if n == 1 then
                    death_idx = death_idx + 1
                    BUFFER_DEATH[death_idx] = idx
                else
                    cell[4] = cell[4] / n
                    cell[5] = cell[5] / n
                    for j = 1, n - 1 do
                        local t_idx = targets[j]
                        BUFFER_ENERGY[t_idx]  = (BUFFER_ENERGY[t_idx]  or 0.0) + cell[4]
                        BUFFER_MINERAL[t_idx] = (BUFFER_MINERAL[t_idx] or 0.0) + cell[5]
                        track(t_idx)
                    end
                end

            elseif typ == 4 then -- Seed
                data[1] = cell[3]
                data[2] = cell[4]
                data[3] = cell[5]
                data[4] = cell[6]
                data[5] = state.sun_factor
                for j = 1, 4 do
                    data[5 + j] = (MAP_TYPES[cell[6 + j]] or 0)
                end
                ai_calls = ai_calls + 1
                local action = ai_module.run_into(
                    CELL_GENOMES[cell[11]],
                    AI_LAYERS_SEED,
                    AI_OFFSET_SEED,
                    data,
                    ai_out
                )[1]
                if action > 0.0 then
                    cell[2] = 6
                    cell[5] = cell[5] + CELL_COSTS[4] - CELL_COSTS[6]
                    update_idx = update_idx + 1
                    BUFFER_UPDATE[update_idx] = idx
                end

            elseif typ == 5 then -- Spore
                local x, y = idx2pos(idx)
                local dir = cell[3] + 1
                local target_idx = pos2idx(
                    (x + x_offsets[dir]),
                    (y + y_offsets[dir])
                )
                cell[7] = nil
                local target_type = MAP_TYPES[target_idx]
                data[1] = cell[3]
                data[2] = cell[4]
                data[3] = cell[5]
                data[4] = cell[6]
                data[5] = state.sun_factor
                data[6] = target_type or 0
                ai_calls = ai_calls + 1
                local action = floor(ai_module.run_into(
                    CELL_GENOMES[cell[11]],
                    AI_LAYERS_SPORE,
                    AI_OFFSET_SPORE,
                    data,
                    ai_out
                )[1]) % 5
                if     action == 1 then
                    cell[3] = (cell[3] - 1) % 4
                    update_idx = update_idx + 1
                    BUFFER_UPDATE[update_idx] = idx
                elseif action == 2 then
                    cell[3] = (cell[3] + 1) % 4
                    update_idx = update_idx + 1
                    BUFFER_UPDATE[update_idx] = idx
                elseif action == 3 then
                    move_idx = move_idx + 2
                    BUFFER_MOVING[move_idx - 1] = idx
                    BUFFER_MOVING[move_idx]     = target_idx
                elseif action == 4 then
                    cell[2] = 4
                    cell[5] = cell[5] + CELL_COSTS[5] - CELL_COSTS[4]
                    for j = 1, 4 do
                        local dir = (cell[3] + j + 1) % 4 + 1
                        cell[6 + j] = pos2idx(x + x_offsets[dir], y + y_offsets[dir])
                    end
                    update_idx = update_idx + 1
                    BUFFER_UPDATE[update_idx] = idx
                end

            elseif typ == 6 then -- Sprout
                data[1] = cell[3]
                data[2] = cell[4]
                data[3] = cell[5]
                data[4] = cell[6]
                data[5] = state.sun_factor
                for j = 1, 4 do
                    data[5 + j] = (MAP_TYPES[cell[6 + j]] or 0)
                end
                ai_calls = ai_calls + 1
                local res = ai_module.run_into(
                    CELL_GENOMES[cell[11]],
                    AI_LAYERS_SPROUT,
                    AI_OFFSET_SPROUT,
                    data,
                    ai_out
                )
                local n = 0
                local shared_energy = cell[4] / 4
                cell[4] = shared_energy
                for j = 1, 3 do
                    local typ = floor(res[j]) % 7
                    local cost = CELL_COSTS[typ]
                    if typ > 0 and cell[5] > cost then
                        cell[5] = cell[5] - cost
                        n = n + 1
                        spawn_idx = spawn_idx + 1
                        -- Serializable spawn request (plain numbers, no cell
                        -- tables / no initCell/acquire in compute):
                        -- {typ, dir, target_idx, energy_share, parent_idx, genome_slot}
                        -- genome_slot is captured BEFORE the release below (children inherit it).
                        BUFFER_SPAWN[spawn_idx] = {
                            typ,
                            (cell[3] + j - 2) % 4,
                            cell[7 + j],
                            shared_energy,
                            idx,
                            cell[11],
                        }
                    else cell[4] = cell[4] + shared_energy
                    end
                end
                if n > 0 then
                    cell[2] = 3
                    -- Morph 6 -> 3: keep the CELL_COSTS reserve invariant like the
                    -- Seed->Sprout and Spore->Seed morphs (reserve must not leak).
                    cell[5] = cell[5] + CELL_COSTS[6] - CELL_COSTS[3]
                    update_idx = update_idx + 1
                    BUFFER_UPDATE[update_idx] = idx
                    -- BUG-1: Sprout -> Stem drops the genome ref (Stem is not an
                    -- AI cell); release is DEFERRED to apply (BUFFER_RELEASE —
                    -- compute must not touch genome counters). Children got their
                    -- own slot copy above, before this release.
                    release_idx = release_idx + 1
                    BUFFER_RELEASE[release_idx] = cell[11]
                    cell[11] = nil
                end
            end
        else
            death_idx = death_idx + 1
            BUFFER_DEATH[death_idx] = idx
        end

        ::continue::
    end

    -- Death phase: process the dying set in two passes so the result is
    -- independent of the order BUFFER_DEATH was filled. A cell's resources go
    -- to a parent that is alive AND survives this tick; otherwise (no parent,
    -- dying parent, or Spore) they drop to the tile ledgers.
    local dying = {}
    for i = 1, death_idx do
        dying[BUFFER_DEATH[i]] = true
    end

    for i = 1, death_idx do -- Killing cells
        local idx  = BUFFER_DEATH[i]
        local cell = MAP_CELLS[idx]
        local drop = (MAP_TYPES[idx] == 5) -- Spores always drop to the tile
        if not drop then
            local parent = MAP_CELLS[cell[7]]
            if parent and not dying[cell[7]] then
                -- Refund to a surviving parent: energy + minerals + the
                -- CELL_COSTS reserve the parent paid at spawn (no cost leak).
                parent[4] = parent[4] + cell[4]
                parent[5] = parent[5] + cell[5] + CELL_COSTS[cell[2]]
            else
                drop = true
            end
        end
        if drop then
            -- Spore / no surviving parent: energy to the MAP_ENERGY ledger,
            -- minerals + cost reserve back to MAP_MINERALS (no loss either).
            MAP_ENERGY[idx] = (MAP_ENERGY[idx] or 0) + cell[4]
            MAP_MINERALS[idx] = MAP_MINERALS[idx] + cell[5] + CELL_COSTS[cell[2]]
            view.updateMinerals(idx)
        end
        M.removeCell(state, idx, view)
    end

    for i = 1, release_idx do -- Genome releases (deferred from compute)
        ai_module.release(BUFFER_RELEASE[i])
    end

    local write_idx = 0
    for i = 1, state.CELL_COUNTER do -- Removing dead cell from queue
        local idx = CELL_QUEUE[i]
        if MAP_CELLS[idx] then
            write_idx = write_idx + 1
            CELL_QUEUE[write_idx] = idx
        end
    end
    for i = write_idx + 1, state.CELL_COUNTER do
        CELL_QUEUE[i] = nil
    end
    state.CELL_COUNTER = write_idx

    for ti = 1, #touched do -- Resource transfering
        local i = touched[ti]
        local energy   = BUFFER_ENERGY[i]  or 0.0
        local minerals = BUFFER_MINERAL[i] or 0.0
        if energy ~= 0.0 or minerals ~= 0.0 then
            local cell = MAP_CELLS[i]
            if cell then
                cell[4] = cell[4] + energy
                cell[5] = cell[5] + minerals
            else
                MAP_MINERALS[i] = MAP_MINERALS[i] + minerals
                -- energy to MAP_ENERGY ledger instead of dropping it
                MAP_ENERGY[i] = (MAP_ENERGY[i] or 0) + energy
            end
        end
    end

    for i = 1, extr_idx, 2 do -- Mineral extraction
        local extr_to = BUFFER_EXTR[i + 1]
        local cell    = MAP_CELLS[extr_to]
        if cell then
            local extr_from = BUFFER_EXTR[i]
            local minerals  = math.min(MAP_MINERALS[extr_from], ROOT_MINERAL_EXTR)
            MAP_MINERALS[extr_from] = MAP_MINERALS[extr_from] - minerals
            cell[5] = cell[5] + minerals
            view.updateMinerals(extr_from)
        end
    end

    for i = 1, spawn_idx do -- Cell spawning
        local req        = BUFFER_SPAWN[i]
        local typ        = req[1]
        local dir        = req[2]
        local target_idx = req[3]
        local energy     = req[4]
        local parent_idx = req[5]
        local genome     = req[6]
        local x, y       = idx2pos(target_idx)
        local child      = initCell(typ, x, y, dir,
            {energy = energy, minerals = 0, parent = parent_idx, genome = genome})
        if not M.addCell(state, child, view) then
            -- BUG-1: the child acquired its genome reference in initCell; since
            -- it did not appear on the map, return the reference so the slot can
            -- reach counter 0 and be reused. release is nil-safe for non-AI
            -- children (cell[11] == nil) — exactly one release per failed spawn,
            -- and none on a successful addCell.
            ai_module.release(child[11])
            local parent = MAP_CELLS[parent_idx]
            if parent then
                parent[4] = parent[4] + child[4]
                parent[5] = parent[5] + child[5] + CELL_COSTS[child[2]]
            else
                -- No parent: refund energy to the MAP_ENERGY ledger and
                -- minerals + cost reserve to MAP_MINERALS (no loss).
                MAP_ENERGY[target_idx] = (MAP_ENERGY[target_idx] or 0) + child[4]
                MAP_MINERALS[target_idx] = MAP_MINERALS[target_idx] + child[5] + CELL_COSTS[child[2]]
                view.updateMinerals(target_idx)
            end
        end
    end

    for i = 1, update_idx do -- Updating cells
        local idx  = BUFFER_UPDATE[i]
        local cell = MAP_CELLS[idx]
        MAP_TYPES[idx] = cell[2]
        view.setCell(idx, cell)
    end

    for i = 1, move_idx, 2 do -- Moving cells
        local idx_from, idx_to = BUFFER_MOVING[i], BUFFER_MOVING[i + 1]
        M.moveCell(state, idx_from, idx_to, view)
    end

    -- moves/extracts are pair-buffered, so their recorded counts are halved.
    local stats = {
        births   = spawn_idx,
        deaths   = death_idx,
        moves    = move_idx / 2,
        updates  = update_idx,
        extracts = extr_idx / 2,
        ai_calls = ai_calls,
        cells    = state.CELL_COUNTER,
    }
    return state.CELL_COUNTER <= 0, stats
end

return M
