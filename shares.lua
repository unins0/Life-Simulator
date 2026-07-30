local M = {}

-- Map configuration
M.MAP_WIDTH  = 100
M.MAP_HEIGHT = 100
M.MAP_SIZE   = M.MAP_WIDTH * M.MAP_HEIGHT

M.DAY_DURATION = 1024
M.SUN_MIN      = 0.0
M.SUN_MAX      = 1.0
M.MINERALS_MIN = 0
M.MINERALS_MAX = 256

-- Cell configuration
M.CELL_INIT_ENERGY   = 256.0
M.CELL_INIT_MINERALS = 256.0

M.LEAF_ENERGY_GEN   = 20
M.ROOT_MINERAL_EXTR = 20

M.CELL_AGES = {
    100, -- Leaf
    100, -- Root
    100, -- Stem
    500, -- Seed
    300, -- Spore
    100, -- Sprout
}

M.CELL_ENERGY_CONS = {
    0.0, -- Leaf
    0.0, -- Root
    0.0, -- Stem
    0.0, -- Seed
    0.0, -- Spore
    0.0, -- Sprout
}

M.CELL_COSTS = {
    1.0,  -- Leaf
    1.0,  -- Root
    1.0,  -- Stem
    3.0, -- Seed
    2.0, -- Spore
    2.0,  -- Sprout
}

M.CELL_COLORS = {
    {0.0, 1.0, 0.0}, -- Leaf
    {1.0, 0.0, 0.0}, -- Root
    {0.5, 0.5, 0.5}, -- Stem
    {1.0, 1.0, 0.0}, -- Seed
    {0.5, 0.0, 1.0}, -- Spore
    {1.0, 0.5, 0.0}, -- Sprout
}

M.AI_LAYERS_SEED   = {9, 12, 1}
M.AI_LAYERS_SPORE  = {6, 16, 1}
M.AI_LAYERS_SPROUT = {9, 18, 16, 3}

-- Cached data
local floor = math.floor

local function countWeights(layers)
    local len = #layers
    local n = 0
    for i = 1, len - 1 do
        local layer, next_layer = layers[i], layers[i + 1]
        n = n + layer * (3 + next_layer)
    end
    return n + layers[len] * 3
end

M.AI_LEN_SEED   = countWeights(M.AI_LAYERS_SEED)
M.AI_LEN_SPORE  = countWeights(M.AI_LAYERS_SPORE)
M.AI_LEN_SPROUT = countWeights(M.AI_LAYERS_SPROUT)
M.AI_LEN_COMMON = M.AI_LEN_SEED + M.AI_LEN_SPORE + M.AI_LEN_SPROUT

M.AI_OFFSET_SEED   = 0
M.AI_OFFSET_SPORE  = M.AI_LEN_SEED
M.AI_OFFSET_SPROUT = M.AI_LEN_SEED + M.AI_LEN_SPORE

M.CELL_NAMES = {'Leaf', 'Root', 'Stem', 'Seed', 'Spore', 'Sprout'}

M.MAP_CELLS      = {}
M.MAP_TYPES      = {}
M.MAP_MINERALS   = {}
M.CELL_GENOMES   = {}
M.CELL_QUEUE     = {}
M.CELL_COUNTER   = 0

-- Some useful functions
function M.clamp(value, lo, hi)
    if value < lo then return lo end
    if value > hi then return hi end
    return value
end

function M.pos2idx(x, y)
    local MAP_WIDTH = M.MAP_WIDTH
    return ((x - 1) % MAP_WIDTH + 1) + ((y - 1) % M.MAP_HEIGHT * MAP_WIDTH)
end

function M.idx2pos(idx)
    local MAP_WIDTH = M.MAP_WIDTH
    return (idx - 1) % MAP_WIDTH + 1, floor((idx - 1) / MAP_WIDTH) + 1
end

return M