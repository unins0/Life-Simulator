-- Simulation configuration
local TPS = 3

local VIEW_MODES = {'Normal', 'Energy', 'Cell Minerals', 'Map Minerals'}

local shares      = require('shares')
local cell_module = require('cell_module')
local sim_module  = require('sim_module')

local LG     = love.graphics

-- Clocks-n-Timers
local tps_threshold = 1.0 / TPS
local tps_timer     = 0.0
local pause         = true

-- Console metrics (1s cadence; os.clock is read-only / determinism-safe).
local report_timer    = 0.0
local report_clock    = 0.0
local report_ticks    = 0
local report_acc      = {births=0, deaths=0, moves=0, updates=0, extracts=0, ai_calls=0}
local extinct_steps   = 0
local extinct_peak    = 0

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
local rand = math.random

-- Graphics callbacks for sim_module (the sim core is pure Lua and renders
-- through these). Each callback is identical to the sprite code that used to
-- live in addCell/removeCell/updateMinerals. Defined before regenMap so it can
-- pass `view` down to sim_module.addCell.
local view = {
    setCell = function(idx, cell)
        local x, y = shares.idx2pos(idx)
        local r, g, b = shares.CELL_COLORS[cell[2]]
        cell_batch:setColor(r, g, b)
        cell_batch:set(idx, cell_sprites[cell[2]], x, y, cell[3] / 2 * math.pi, 0.125, 0.125, 4, 4)
    end,
    clearCell = function(idx)
        local x, y = shares.idx2pos(idx)
        cell_batch:setColor(0.0, 0.5, 1.0, 0.1)
        cell_batch:set(idx, cell_sprites[0], x, y, 0, 0.125, 0.125, 4, 4)
    end,
    updateMinerals = function(idx)
        local x, y = shares.idx2pos(idx)
        local a = shares.MAP_MINERALS[idx] / shares.MINERALS_MAX
        mineral_batch:setColor(0.0, 0.0, a, 0.5)
        mineral_batch:set(idx, x, y, 0, 1, 1, 0.5, 0.5)
    end,
}

function regenMap()
    sim_module.reset(shares)
    initCellBatch()
    initMineralBatch()
    sim_module.addCell(shares, cell_module.initCell(
        6, -- Sprout
        rand(1, shares.MAP_WIDTH),
        rand(1, shares.MAP_HEIGHT),
        rand(0, 3)
    ), view)
    extinct_steps = 0
    extinct_peak  = shares.CELL_COUNTER
    print(('[init] map=%dx%d tps=%g cells=%d'):format(
        shares.MAP_WIDTH, shares.MAP_HEIGHT, 1 / tps_threshold, shares.CELL_COUNTER))
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

function love.load()
    local cell_atlas = LG.newImage('cell_sprites.png')
    cell_atlas:setFilter('nearest')
    cell_sprites = {}
    for i = 0, 6 do cell_sprites[i] = LG.newQuad(8 * i, 0, 8, 8, cell_atlas) end
    cell_batch = LG.newSpriteBatch(cell_atlas, shares.MAP_SIZE)

    local img_data = love.image.newImageData(1, 1)
    img_data:setPixel(0, 0, 1, 1, 1, 1)
    local rectimg = LG.newImage(img_data)
    rectimg:setFilter('nearest')
    mineral_batch = LG.newSpriteBatch(rectimg, shares.MAP_SIZE)

    regenMap()
end

function love.update(dt)
    if not pause then tps_timer = tps_timer + dt end
    while tps_timer >= tps_threshold do
        tps_timer = tps_timer - tps_threshold
        local t0 = os.clock()
        local extinct, stats = sim_module.tick(shares, view)
        local dt_tick = os.clock() - t0
        report_clock = report_clock + dt_tick
        report_ticks = report_ticks + 1
        extinct_steps = extinct_steps + 1
        if shares.CELL_COUNTER > extinct_peak then extinct_peak = shares.CELL_COUNTER end
        if stats then
            report_acc.births   = report_acc.births   + stats.births
            report_acc.deaths   = report_acc.deaths   + stats.deaths
            report_acc.moves    = report_acc.moves    + stats.moves
            report_acc.updates  = report_acc.updates  + stats.updates
            report_acc.extracts = report_acc.extracts + stats.extracts
            report_acc.ai_calls = report_acc.ai_calls + stats.ai_calls
        end
        if extinct then
            print(('[extinct] survived=%d peak=%d'):format(extinct_steps, extinct_peak))
            regenMap()
        end
        report_timer = report_timer + tps_threshold
        if report_timer >= 1.0 then
            local avg_ms = report_clock / report_ticks * 1000
            print(('[1s] tick=%.3fms cells=%d b=%d d=%d m=%d u=%d x=%d ai=%d'):format(
                avg_ms,
                shares.CELL_COUNTER,
                report_acc.births, report_acc.deaths, report_acc.moves,
                report_acc.updates, report_acc.extracts, report_acc.ai_calls))
            report_timer = report_timer - 1.0
            report_clock = 0.0
            report_ticks = 0
            report_acc.births, report_acc.deaths = 0, 0
            report_acc.moves, report_acc.updates = 0, 0
            report_acc.extracts, report_acc.ai_calls = 0, 0
        end
    end
end

function love.draw()
    screen_width, screen_height = LG.getDimensions()

    LG.translate(camera_x, camera_y)
    LG.scale(camera_zoom, camera_zoom)

    local sf = shares.sun_factor
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
            '\nStep: '   .. shares.step ..
            '\nSun: '    .. string.format('%.2f', shares.sun_factor) ..
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
        print(('[pause] state=%s'):format(pause and 'paused' or 'running'))
    elseif key == 'up'    then tps_threshold = shares.clamp(tps_threshold / 1.1, 0.002, 1.0)
    elseif key == 'down'  then tps_threshold = shares.clamp(tps_threshold * 1.1, 0.002, 1.0)
    elseif key == 'u'     then draw_interface = not(draw_interface)
    elseif key == 'e'     then view_mode = (view_mode + 1) % 4
    elseif key == 'r'     then regenMap()
    end
end

function love.quit()
end