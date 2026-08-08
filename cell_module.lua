local shares    = require('shares')
local ai_module = require('ai_module')

local M = {}

local rand = math.random

local x_offsets = {1, 0, -1, 0}
local y_offsets = {0, 1, 0, -1}

function M.initCell(typ, x, y, direction, args)
    -- Common attributes
    args           = args or {}
    local energy   = args.energy   or shares.CELL_INIT_ENERGY
    local minerals = args.minerals or shares.CELL_INIT_MINERALS
    local parent   = args.parent

    x, y = (x - 1) % shares.MAP_WIDTH + 1, (y - 1) % shares.MAP_HEIGHT + 1
    if not parent then
        local back = (direction + 1) % 4 + 1
        parent = shares.pos2idx(x + x_offsets[back], y + y_offsets[back])
    end 
    direction      = direction % 4
    local idx      = shares.pos2idx(x, y)

    local cell = {
        idx,
        typ,
        direction,
        energy,
        minerals,
        0, -- age
        parent,
    }

    if typ >= 3 and typ ~= 5 then
        local pos2idx = shares.pos2idx
        for i = 0, 2 do
            local dir = (direction + i + 2) % 4 + 1
            cell[8 + i] = pos2idx(x + x_offsets[dir], y + y_offsets[dir])
        end
    end
    if typ >= 4 then
        local genome = args.genome
        if genome == nil then genome = ai_module.genWeights() 
        elseif rand() < ai_module.GENOME_MUTATION_CHANCE then 
            genome = ai_module.mutateWeights(genome)
        end
        cell[11] = genome
        -- BUG-1 (genome aliasing): the cell owns its reference from the moment of
        -- creation. acquire() bumps the slot counter right here, before any
        -- addCell, so a second mutating AI child spawned by the same Sprout
        -- cannot have addGenome reuse this slot (counter == 0) and overwrite it.
        ai_module.acquire(genome)
    end

    return cell
end

return M