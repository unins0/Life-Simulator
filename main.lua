-- Simulation configuration
local TPS = 3

local VIEW_MODES = {'Normal', 'Energy', 'Cell Minerals', 'Map Minerals'}

local shares      = require('shares')
local cell_module = require('cell_module')
local ai_module   = require('ai_module')

local LG     = love.graphics

-- Clocks-n-Timers
local sun_factor = 1.0
local step          = 0
local tps_threshold = 1.0 / TPS
local tps_timer     = 0.0
local pause         = true

-- Camera variables
local screen_width, screen_height = LG.getDimensions()
local view_mode   = 0 -- 0: normal, 1: energy, 2: cell minerals, 3: map minerals
local target_cell = {idx = 0, x = 0, y = 0, cell = nil}
local camera_x    = (screen_width - shares.MAP_WIDTH) / 2
local camera_y    = (screen_height - shares.MAP_HEIGHT) / 2
local camera_zoom = 1.0
local world_x     = 0.0
local world_y     = 0.0
local highlight_x = 0
local highlight_y = 0
local draw_interface   = true
local is_mouse_pressed = false

local cell_sprites
local cell_batch
local mineral_batch

-- Boring cached data
local rand   = math.random
local floor  = math.floor
local remove = table.remove
local pi2    = math.pi * 2
local dsun   = shares.SUN_MAX - shares.SUN_MIN
local x_offsets = {1, 0, -1, 0}
local y_offsets = {0, 1, 0, -1}

local function calcSunFactor(step)
    local DAY_DURATION = shares.DAY_DURATION
    local phase = (step % DAY_DURATION) / DAY_DURATION - 0.5
    return shares.SUN_MIN + dsun * (0.5 - 0.5 * math.cos(pi2 * phase))
end

function regenMap()
    local MAP_CELLS    = shares.MAP_CELLS
    local MAP_TYPES    = shares.MAP_TYPES
    local MAP_MINERALS = shares.MAP_MINERALS
    MAP_CELLS    = {}
    MAP_TYPES    = {}
    MAP_MINERALS = {}
    shares.CELL_GENOMES = {}
    shares.CELL_QUEUE   = {}
    shares.CELL_COUNTER = 0
    step = 0

    local MINERALS_MIN, MINERALS_MAX = shares.MINERALS_MIN, shares.MINERALS_MAX
    for i = 1, shares.MAP_SIZE do
        MAP_CELLS[i]    = nil
        MAP_TYPES[i]    = 0
        MAP_MINERALS[i] = rand(MINERALS_MIN, MINERALS_MAX)
    end

    shares.MAP_CELLS    = MAP_CELLS
    shares.MAP_TYPES    = MAP_TYPES
    shares.MAP_MINERALS = MAP_MINERALS

    initCellBatch()
    initMineralBatch()

    addCell(cell_module.initCell(
        6, -- Sprout
        rand(1, shares.MAP_WIDTH),
        rand(1, shares.MAP_HEIGHT),
        rand(0, 3)
    ))
end

function initCellBatch()
    local MAP_WIDTH = shares.MAP_WIDTH
    cell_batch:clear()
    cell_batch:setColor(0.0, 0.5, 1.0, 0.1)
    for y = 1, shares.MAP_HEIGHT do 
        for x = 1, MAP_WIDTH do    
            cell_batch:add(cell_sprites[0], x, y, 0, 0.125, 0.125, 4, 4)
        end
    end
end

function initMineralBatch()
    local MAP_WIDTH    = shares.MAP_WIDTH
    local MAP_MINERALS = shares.MAP_MINERALS
    local MINERALS_MAX = shares.MINERALS_MAX
    local pos2idx      = shares.pos2idx
    mineral_batch:clear()
    for y = 1, shares.MAP_HEIGHT do
        for x = 1, MAP_WIDTH do
            local a = MAP_MINERALS[pos2idx(x, y)] / MINERALS_MAX
            mineral_batch:setColor(0.0, 0.0, a, 0.5)
            mineral_batch:add(x, y, 0, 1, 1, 0.5, 0.5)
        end
    end
end

function updateMinerals(idx)
    local x, y = shares.idx2pos(idx)
    local a = shares.MAP_MINERALS[idx] / shares.MINERALS_MAX
    mineral_batch:setColor(0.0, 0.0, a, 0.5)
    mineral_batch:set(idx, x, y, 0, 1, 1, 0.5, 0.5)
end

function addCell(cell)
    local MAP_CELLS = shares.MAP_CELLS
    if not cell or MAP_CELLS[cell[1]] then return false end
    MAP_CELLS[cell[1]]        = cell
    shares.MAP_TYPES[cell[1]] = cell[2]

    shares.CELL_COUNTER = shares.CELL_COUNTER + 1
    shares.CELL_QUEUE[shares.CELL_COUNTER] = cell[1]
    if cell[2] >= 4 then 
        local genome = shares.CELL_GENOMES[cell[11]]
        genome.counter = genome.counter + 1
    end
    local x, y = shares.idx2pos(cell[1])
    local r, g, b = shares.CELL_COLORS[cell[2]]
    cell_batch:setColor(r, g, b)
    cell_batch:set(
        cell[1],
        cell_sprites[cell[2]],
        x,
        y,
        cell[3] / 2 * math.pi,
        0.125,
        0.125,
        4, 4
    )
    return true
end

function removeCell(idx)
    local MAP_CELLS = shares.MAP_CELLS
    local cell = MAP_CELLS[idx]
    if not cell then return false end
    MAP_CELLS[idx] = nil
    shares.MAP_TYPES[idx] = nil

    if cell[2] >= 4 then 
        local genome = shares.CELL_GENOMES[cell[11]]
        genome.counter = genome.counter - 1
    end
    local x, y = shares.idx2pos(idx)
    cell_batch:setColor(0.0, 0.5, 1.0, 0.1)
    cell_batch:set(idx, cell_sprites[0], x, y, 0, 0.125, 0.125, 4, 4)
    return true
end

function tick()
    -- Cache
    local MAP_WIDTH  = shares.MAP_WIDTH
    local MAP_HEIHGT = shares.MAP_HEIGHT

    local CELL_ENERGY_CONS  = shares.CELL_ENERGY_CONS
    local CELL_AGES         = shares.CELL_AGES
    local CELL_COSTS        = shares.CELL_COSTS
    local CELL_COLORS       = shares.CELL_COLORS
    local CELL_GENOMES      = shares.CELL_GENOMES
    local AI_LAYERS_SEED    = shares.AI_LAYERS_SEED
    local AI_LAYERS_SPORE   = shares.AI_LAYERS_SPORE
    local AI_LAYERS_SPROUT  = shares.AI_LAYERS_SPROUT
    local AI_OFFSET_SEED    = shares.AI_OFFSET_SEED
    local AI_OFFSET_SPORE   = shares.AI_OFFSET_SPORE
    local AI_OFFSET_SPROUT  = shares.AI_OFFSET_SPROUT
    local LEAF_ENERGY_GEN   = shares.LEAF_ENERGY_GEN
    local ROOT_MINERAL_EXTR = shares.ROOT_MINERAL_EXTR

    local MAP_CELLS    = shares.MAP_CELLS
    local MAP_TYPES    = shares.MAP_TYPES
    local MAP_MINERALS = shares.MAP_MINERALS
    local CELL_QUEUE   = shares.CELL_QUEUE
    local idx2pos      = shares.idx2pos
    local pos2idx      = shares.pos2idx
    local initCell     = cell_module.initCell

    local BUFFER_ENERGY  = {}
    local BUFFER_MINERAL = {}
    local BUFFER_EXTR    = {} -- {from, to, ...}
    local BUFFER_SPAWN   = {}
    local BUFFER_DEATH   = {}
    local BUFFER_UPDATE  = {}
    local BUFFER_MOVING  = {} -- {from, to, ...}
    local extr_idx, spawn_idx   = 0, 0
    local death_idx, update_idx = 0, 0
    local move_idx = 0

    step = step + 1
    sun_factor = calcSunFactor(step)

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
                    BUFFER_ENERGY[parent_idx] = (BUFFER_ENERGY[parent_idx] or 0.0) + LEAF_ENERGY_GEN * sun_factor
                else
                    death_idx = death_idx + 1 
                    BUFFER_DEATH[death_idx] = idx
                end

            elseif typ == 2 then -- Root
                local parent_idx = cell[7]
                if MAP_TYPES[parent_idx] then
                    local x, y = idx2pos(idx)
                    extr_idx = extr_idx + 2
                    BUFFER_EXTR[extr_idx - 1] = pos2idx(x + rand(-1, 1), y + rand(-1, 1))
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
                    end
                end

            elseif typ == 4 then -- Seed
                local x, y = idx2pos(idx)
                local data = {
                    cell[3], 
                    cell[4],
                    cell[5],
                    cell[6],
                    sun_factor,
                }
                for j = 1, 4 do
                    data[5 + j] = (MAP_TYPES[cell[6 + j]] or 0)
                end
                local action = ai_module.run(
                    CELL_GENOMES[cell[11]],
                    AI_LAYERS_SEED,
                    AI_OFFSET_SEED,
                    data
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
                local data = {
                    cell[3],
                    cell[4],
                    cell[5],
                    cell[6],
                    sun_factor,
                    target_type or 0,
                }
                local action = floor(ai_module.run(
                    CELL_GENOMES[cell[11]],
                    AI_LAYERS_SPORE,
                    AI_OFFSET_SPORE,
                    data
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
                    BUFFER_MOVING[move_idx ]    = target_idx
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
                local x, y = idx2pos(idx)
                local data = {
                    cell[3], 
                    cell[4],
                    cell[5],
                    cell[6],
                    sun_factor,
                }
                for j = 1, 4 do
                    data[5 + j] = (MAP_TYPES[cell[6 + j]] or 0)
                end
                local res = ai_module.run(
                    CELL_GENOMES[cell[11]],
                    AI_LAYERS_SPROUT,
                    AI_OFFSET_SPROUT,
                    data
                )
                local childs = {}
                local n = 0
                local shared_energy = cell[4] / 4
                cell[4] = shared_energy
                for j = 1, 3 do
                    local typ = floor(res[j]) % 7
                    local cost = CELL_COSTS[typ]
                    if typ > 0 and cell[5] > cost then
                        cell[5] = cell[5] - cost
                        local x, y = idx2pos(cell[7 + j])
                        local child = initCell(
                            typ,
                            x, y,
                            (cell[3] + j - 2) % 4,
                            {
                                energy   = shared_energy,
                                minerals = 0,
                                parent   = idx,
                                genome   = cell[11],
                            }
                        )
                        n = n + 1
                        spawn_idx = spawn_idx + 1
                        BUFFER_SPAWN[spawn_idx] = child
                    else cell[4] = cell[4] + shared_energy
                    end
                end
                if n > 0 then
                    cell[2] = 3
                    update_idx = update_idx + 1
                    BUFFER_UPDATE[update_idx] = idx
                end
            end
        else
            death_idx = death_idx + 1
            BUFFER_DEATH[death_idx] = idx
        end

        ::continue::
    end

    for i = 1, death_idx do -- Killing cells
        local idx = BUFFER_DEATH[i]
        local cell = MAP_CELLS[idx]
        local drop_minerals = false
        if MAP_TYPES[idx] ~= 5 then    
            local parent = MAP_CELLS[cell[7]]
            if parent then
                parent[4] = parent[4] + cell[4]
                parent[5] = parent[5] + cell[5]
            else drop_minerals = true
            end
        else drop_minerals = true
        end
        if drop_minerals then
            MAP_MINERALS[idx] = MAP_MINERALS[idx] + cell[5] + CELL_COSTS[cell[2]]
            updateMinerals(idx)
        end
        removeCell(idx)
    end

    local write_idx = 0
    for i = 1, shares.CELL_COUNTER do -- Removing dead cell from queue
        local idx = CELL_QUEUE[i]
        if MAP_CELLS[idx] then
            write_idx = write_idx + 1
            CELL_QUEUE[write_idx] = idx
        end
    end
    for i = write_idx + 1, shares.CELL_COUNTER do
        CELL_QUEUE[i] = nil
    end
    shares.CELL_COUNTER = write_idx

    for i = 1, shares.MAP_SIZE do -- Resource transfering
        local energy   = (BUFFER_ENERGY[i]  or 0.0)
        local minerals = (BUFFER_MINERAL[i] or 0.0)
        if energy ~= 0.0 or minerals ~= 0.0 then
            local cell = MAP_CELLS[i]
            if cell then
                cell[4] = cell[4] + energy
                cell[5] = cell[5] + minerals
            else
                MAP_MINERALS[i] = MAP_MINERALS[i] + minerals
                --[[print(
                    'Resource miss\n' ..
                    i, energy, minerals
                )]]
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
            updateMinerals(extr_from)
        end
    end

    for i = 1, spawn_idx do -- Cell spawning
        local cell = BUFFER_SPAWN[i]
        if not addCell(cell) then
            local parent = MAP_CELLS[cell[7]]
            if parent then
                parent[4] = parent[4] + cell[4]
                parent[5] = parent[5] + cell[5] + CELL_COSTS[cell[2]]
            else
                local idx = cell[1]
                MAP_MINERALS[idx] = MAP_MINERALS[idx] + cell[5] + CELL_COSTS[cell[2]]
                updateMinerals(idx)
            end
        end
    end

    for i = 1, update_idx do -- Updating cells
        local idx  = BUFFER_UPDATE[i]
        local cell = MAP_CELLS[idx]
        MAP_TYPES[idx] = cell[2]
        local x, y = idx2pos(idx)
        local r, g, b = CELL_COLORS[cell[2]]
        cell_batch:setColor(r, g, b)
        cell_batch:set(
            idx,
            cell_sprites[cell[2]],
            x,
            y,
            cell[3] / 2 * math.pi,
            0.125,
            0.125,
            4, 4
        )
    end

    for i = 1, move_idx, 2 do -- Moving cells
        local idx_from, idx_to = BUFFER_MOVING[i], BUFFER_MOVING[i + 1]
        local cell = MAP_CELLS[idx_from]
        if addCell(cell) then
            cell[1] = idx_to
            MAP_CELLS[idx_from] = nil
        end
    end

    if shares.CELL_COUNTER <= 0 then regenMap() end
end

function love.load()
    --[[function update_minerals()
        local sx, sy = math.max(math.floor(-camera_x / camera_zoom), 1), math.max(math.floor(-camera_y / camera_zoom), 1)
        local w, h = math.min(math.ceil(screen_width / camera_zoom) + sx, MAP_WIDTH), math.min(math.ceil(screen_height / camera_zoom) + sy, MAP_HEIGHT)
        for y = sy, h do
            for x = sx, w do
                local c
                local idx = pos2idx(x, y)
                if Map.minerals[idx] then c = Map.minerals[idx] / MINERALS_MAX else c = 0.0 end
                mineral_batch:setColor(0.0, 0.0, 1.0, c)
                mineral_batch:set(idx, x + 3.5, y + 3.5, 0, 1, 1, 4, 4)
            end
        end
    end]]

    local cell_atlas = LG.newImage('cell_sprites.png')
    cell_atlas:setFilter('nearest')
    cell_sprites = {}
    for i = 0, 6 do cell_sprites[i] = LG.newQuad(8 * i, 0, 8, 8, cell_atlas) end
    cell_batch = LG.newSpriteBatch(cell_atlas, MAP_SIZE)

    local img_data = love.image.newImageData(1, 1)
    img_data:setPixel(0, 0, 1, 1, 1, 1)
    local rectimg = LG.newImage(img_data)
    rectimg:setFilter('nearest')
    mineral_batch = LG.newSpriteBatch(rectimg, MAP_SIZE)

    regenMap()
end

function love.update(dt)
    if not pause then tps_timer = tps_timer + dt end
    while tps_timer >= tps_threshold do
        tps_timer = tps_timer - tps_threshold
        tick()
    end
end

function love.draw()
    screen_width, screen_height = LG.getDimensions()

    LG.translate(camera_x, camera_y)
    LG.scale(camera_zoom, camera_zoom)

    local sf = sun_factor
    LG.setColor(sf, sf, sf)
    LG.rectangle('fill', 0.5, 0.5, shares.MAP_WIDTH, shares.MAP_HEIGHT)

    LG.setColor(1.0, 1.0, 1.0)
    LG.draw(cell_batch)

    if view_mode == 3 then LG.draw(mineral_batch) end

    LG.setColor(0.0, 0.5, 1.0, 0.5)
    LG.rectangle('fill', highlight_x - 0.5, highlight_y - 0.5, 1.0, 1.0)
    LG.setColor(1.0, 0.8, 0.0, 0.5)
    LG.rectangle('fill', target_cell.x - 0.6, target_cell.y - 0.6, 1.2, 1.2)

    -- User Interface
    if draw_interface then
        LG.setColor(1.0, 0.0, 0.5)
        LG.scale(1 / camera_zoom, 1 / camera_zoom)
        LG.translate(-camera_x, -camera_y)

        LG.print(
            'FPS: '      .. love.timer.getFPS() ..
            '\nTPS: '    .. math.floor(1 / tps_threshold) ..
            '\nStep: '   .. step ..
            '\nSun: '    .. string.format('%.2f', sun_factor) ..
            '\nCells: '  .. shares.CELL_COUNTER ..
            '\nGenomes: '.. #shares.CELL_GENOMES ..
            '\nX, Y: '   .. highlight_x .. ' ' .. highlight_y ..
            '\nTarget: ' .. target_cell.x .. ' ' .. target_cell.y .. ' ' .. target_cell.idx,
            10, 10
        )

        LG.print(
            'View Mode: ' .. VIEW_MODES[view_mode + 1] ..
            '\nZoom: '    .. string.format('%.2f', camera_zoom),
            10, screen_height - 30
        )

        local cell = target_cell.cell
        if cell then 
            LG.print(
                'Idx: '        .. cell[1] ..
                '\nType: '     .. string.format('real %s, on map %s', shares.CELL_NAMES[cell[2]], shares.CELL_NAMES[shares.MAP_TYPES[cell[1]]]) ..
                '\nDir: '      .. cell[3] ..
                '\nEnergy: '   .. cell[4] ..
                '\nMinerals: ' .. cell[5] ..
                '\nAge: '      .. cell[6] ..
                '\nSurrs: '    .. tostring(cell[7]) ..
                ' '            .. tostring(cell[8]) ..
                ' '            .. tostring(cell[9]) ..
                ' '            .. tostring(cell[10]),
                10, 130
            )
        end

        if pause then
            LG.setColor(1.0, 0.0, 0.0)
            LG.print('Pause', screen_width / 2, 30, 0, 2, 2, 18)
        end
    end
end

function love.mousepressed(x, y, button, istouch)
    if button == 1 then is_mouse_pressed = true
    elseif button == 2 then
        target_cell.idx  = shares.pos2idx(highlight_x, highlight_y)
        target_cell.x    = highlight_x
        target_cell.y    = highlight_y
        target_cell.cell = shares.MAP_CELLS[shares.pos2idx(highlight_x, highlight_y)]
    end
end

function love.mousereleased(x, y, button, istouch)
    if button == 1 then is_mouse_pressed = false end
end

function love.mousemoved(x, y, dx, dy, isTouch)
    if is_mouse_pressed then camera_x, camera_y = camera_x + dx, camera_y + dy end
    world_x, world_y = (x - camera_x) / camera_zoom, (y - camera_y) / camera_zoom
    highlight_x, highlight_y = math.floor(world_x + 0.5), math.floor(world_y + 0.5)
end

function love.wheelmoved(x, y)
    local mx, my = love.mouse.getPosition()
    camera_zoom = camera_zoom * 1.1 ^ y
    camera_x = mx - world_x * camera_zoom
    camera_y = my - world_y * camera_zoom
end

function love.keypressed(key, scancode, isrepeat)
    if     key == 'space' then pause = not(pause) 
    elseif key == 'up'    then tps_threshold = shares.clamp(tps_threshold / 1.1, 0.002, 1.0)
    elseif key == 'down'  then tps_threshold = shares.clamp(tps_threshold * 1.1, 0.002, 1.0)
    elseif key == 'u'     then draw_interface = not(draw_interface)
    elseif key == 'e'     then view_mode = (view_mode + 1) % 4
    elseif key == 'r'     then regenMap()
    end
end

function love.quit()
    -- Just for case
end