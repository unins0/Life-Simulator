-- test_sim.lua — тесты симуляции: тик, фотосинтез-трансфер, смерти,
-- moveCell, инварианты очереди клеток и таблиц карты.
-- Запуск (из корня проекта): lua tests/test_runner.lua tests/test_sim.lua
-- Продакшн-код (sim_module) в этих тестах не модифицируется.

local shares      = require('shares')
local sim_module  = require('sim_module')
local cell_module = require('cell_module')

-- no-op view: рендеринг в тестах не нужен.
local view = {
    setCell        = function() end,
    clearCell      = function() end,
    updateMinerals = function() end,
}

local tests = {}

local abs = math.abs

-- ---------------------------------------------------------------------------
-- Служебное: ledgers консервации. Энергия сохраняется между cell[4] и
-- MAP_ENERGY; минералы — между cell[5], MAP_MINERALS и резервом CELL_COSTS
-- (каждая живая клетка «несёт» резерв, который родитель заплатил за спавн).
-- ---------------------------------------------------------------------------
local function energyLedger()
    local e = 0
    for _, c in pairs(shares.MAP_CELLS) do e = e + c[4] end
    for _, v in pairs(shares.MAP_ENERGY) do e = e + v end
    return e
end

local function mineralLedger()
    local m = 0
    for _, c in pairs(shares.MAP_CELLS) do m = m + c[5] + shares.CELL_COSTS[c[2]] end
    for _, v in pairs(shares.MAP_MINERALS) do m = m + v end
    return m
end

local function assertClose(a, b, tol, what)
    assert(abs(a - b) <= tol,
        ('%s: %.12g != %.12g (|delta| = %.3g > %g)'):format(what, a, b, abs(a - b), tol))
end

-- MAP_ENERGY «пуст» — все записи nil или 0.
local function mapEnergyEmpty()
    for _, v in pairs(shares.MAP_ENERGY) do
        if v and v ~= 0 then return false end
    end
    return true
end

-- ---------------------------------------------------------------------------
-- 1. Фотосинтез: трансфер в существующую клетку.
-- Stem выживает в тик только при наличии ребёнка typ > 2 (Leaf/Root
-- «детьми» стема не считаются — это поведение sim_module), поэтому рядом
-- ставится Sprout в один из детских слотов стема. Дифференциальный запуск
-- (с листом и без) с одинаковым seed изолирует ровно вклад листа: стем
-- получает строго LEAF_ENERGY_GEN * sun_factor, MAP_ENERGY не задевается.
-- ---------------------------------------------------------------------------
tests['leaf transfers energy to a surviving stem'] = function()
    local stem_idx = shares.pos2idx(10, 10)
    -- reset() обнуляет step; первый тик инкрементит его до 1, поэтому
    -- солнце в обоих прогонах считается для step == 1 (шаг из предыдущих
    -- тестов того же процесса на результат не влияет).
    local sun = sim_module.calcSunFactor(shares, 1)

    local function run(with_leaf)
        math.randomseed(7)
        sim_module.reset(shares)
        local stem = cell_module.initCell(3, 10, 10, 0, {energy = 100, minerals = 50})
        sim_module.addCell(shares, stem, view)
        -- Sprout в детском слоте стема (9,10): иначе стем умер бы (нет детей typ>2).
        local sprout = cell_module.initCell(6, 9, 10, 0, {energy = 10, minerals = 10})
        sim_module.addCell(shares, sprout, view)
        if with_leaf then
            local leaf = cell_module.initCell(1, 11, 10, 0,
                {energy = 40, minerals = 10, parent = stem_idx})
            sim_module.addCell(shares, leaf, view)
        end
        sim_module.tick(shares, view)
        return stem
    end

    local stem_a = run(false)
    local stem_b = run(true)

    assert(shares.MAP_CELLS[stem_idx] ~= nil, 'stem died during tick')
    local expected = shares.LEAF_ENERGY_GEN * sun
    assertClose(stem_b[4] - stem_a[4], expected, 1e-6,
        'stem energy must grow by exactly LEAF_ENERGY_GEN * sun_factor')
    assert(mapEnergyEmpty(), 'MAP_ENERGY must stay empty when the parent cell is alive')
end

-- ---------------------------------------------------------------------------
-- 2. Фотосинтез: трансфер в пустую цель (ledger MAP_ENERGY).
-- Пустой тайл после reset имеет MAP_TYPES == 0 (truthy!), поэтому лист жив и
-- производит энергию; цель не клетка -> энергия оседает в MAP_ENERGY[parent].
-- ---------------------------------------------------------------------------
tests['leaf transfers energy to an empty tile ledger'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local parent_idx = shares.pos2idx(20, 20) -- пустой тайл, MAP_TYPES == 0
    local leaf = cell_module.initCell(1, 11, 10, 0,
        {energy = 40, minerals = 10, parent = parent_idx})
    assert(sim_module.addCell(shares, leaf, view))

    local min_before = shares.MAP_MINERALS[parent_idx]
    sim_module.tick(shares, view)

    local sun = sim_module.calcSunFactor(shares, 1)
    assertClose(shares.MAP_ENERGY[parent_idx], shares.LEAF_ENERGY_GEN * sun, 1e-6,
        'MAP_ENERGY[parent] must equal LEAF_ENERGY_GEN * sun_factor')
    assert(shares.MAP_MINERALS[parent_idx] == min_before,
        'MAP_MINERALS[parent] must not change (leaf produces no minerals)')
end

-- ---------------------------------------------------------------------------
-- 3. Смерть без родителя: ресурсы падают на тайл, ledgers сохраняются.
-- cell[6] = CELL_AGES[3] -> после инкремента возраста клетка умирает.
-- Резерв CELL_COSTS[3] возвращается в MAP_MINERALS вместе с минералами.
-- ---------------------------------------------------------------------------
tests['cell death without a parent conserves ledgers'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local idx = shares.pos2idx(10, 10)
    local stem = cell_module.initCell(3, 10, 10, 0,
        {energy = 50, minerals = 30, parent = shares.pos2idx(20, 20)})
    stem[6] = shares.CELL_AGES[3] -- умрёт в первый же тик
    assert(sim_module.addCell(shares, stem, view))

    local min_at_tile = shares.MAP_MINERALS[idx]
    local e0, m0 = energyLedger(), mineralLedger()
    sim_module.tick(shares, view)

    assert(shares.MAP_CELLS[idx] == nil, 'stem must be removed after death')
    assertClose(shares.MAP_ENERGY[idx], 50, 1e-6,
        'stem energy must drop to the MAP_ENERGY ledger')
    assertClose(shares.MAP_MINERALS[idx], min_at_tile + 30 + shares.CELL_COSTS[3], 1e-6,
        'MAP_MINERALS must gain minerals + CELL_COSTS reserve')
    assertClose(energyLedger(), e0, 1e-6, 'energy ledger must be conserved')
    assertClose(mineralLedger(), m0, 1e-6, 'mineral ledger must be conserved')
end

-- ---------------------------------------------------------------------------
-- 4. Родитель и ребёнок умирают в один тик: энергия ребёнка НЕ уходит
-- умирающему родителю, а падает на тайл ребёнка; лист не фотосинтезирует
-- (умер до ветки живой логики), поэтому суммарная энергия постоянна.
-- ---------------------------------------------------------------------------
tests['parent and child dying in the same tick conserve ledgers'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local stem_idx = shares.pos2idx(10, 10)
    local leaf_idx = shares.pos2idx(11, 10)
    local stem = cell_module.initCell(3, 10, 10, 0, {energy = 50, minerals = 30})
    local leaf = cell_module.initCell(1, 11, 10, 0,
        {energy = 30, minerals = 20, parent = stem_idx})
    stem[6] = shares.CELL_AGES[3] -- оба умрут в один тик
    leaf[6] = shares.CELL_AGES[1]
    sim_module.addCell(shares, stem, view)
    sim_module.addCell(shares, leaf, view)

    local e0, m0 = energyLedger(), mineralLedger()
    sim_module.tick(shares, view)

    assert(shares.MAP_CELLS[stem_idx] == nil and shares.MAP_CELLS[leaf_idx] == nil,
        'both cells must be removed')
    assertClose(shares.MAP_ENERGY[leaf_idx], 30, 1e-6,
        'leaf energy must go to its tile, NOT to the dying parent')
    assertClose(shares.MAP_ENERGY[stem_idx], 50, 1e-6,
        'stem energy must go to its tile')
    assertClose(energyLedger(), e0, 1e-6,
        'energy ledger must be conserved (leaf produced no energy while dying)')
    assertClose(mineralLedger(), m0, 1e-6, 'mineral ledger must be conserved')
end

-- ---------------------------------------------------------------------------
-- 5. moveCell: перемещение в свободную цель.
-- Прямой вызов (без тика): MAP_CELLS/MAP_TYPES/cell[1] согласованы,
-- CELL_COUNTER не меняется, запись очереди обновляется (from -> to).
-- ---------------------------------------------------------------------------
tests['moveCell moves a spore to a free tile'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local from = shares.pos2idx(10, 10)
    local to   = shares.pos2idx(11, 10) -- свободно (направление dir=0 -> +x)
    local spore = cell_module.initCell(5, 10, 10, 0, {energy = 10, minerals = 5})
    sim_module.addCell(shares, spore, view)

    local counter_before = shares.CELL_COUNTER
    local ok = sim_module.moveCell(shares, from, to, view)
    assert(ok == true, 'moveCell must succeed on a free tile')

    assert(shares.MAP_CELLS[from] == nil, 'source slot must be cleared')
    local moved = shares.MAP_CELLS[to]
    assert(moved ~= nil, 'target slot must hold the cell')
    assert(moved[1] == to, 'cell[1] must follow the new index')
    assert(shares.MAP_TYPES[from] == nil and shares.MAP_TYPES[to] == 5,
        'MAP_TYPES must stay consistent with the new position')
    assert(shares.CELL_COUNTER == counter_before,
        'moveCell must not touch CELL_COUNTER')

    local n_from, n_to = 0, 0
    for i = 1, shares.CELL_COUNTER do
        if shares.CELL_QUEUE[i] == from then n_from = n_from + 1 end
        if shares.CELL_QUEUE[i] == to   then n_to   = n_to + 1 end
    end
    assert(n_from == 0 and n_to == 1,
        ('CELL_QUEUE must track the move: from absent (%d), to present exactly once (%d)')
        :format(n_from, n_to))
end

-- ---------------------------------------------------------------------------
-- 6. moveCell: занятая цель -> no-op (false), ничего не меняется.
-- ---------------------------------------------------------------------------
tests['moveCell to an occupied tile is a no-op'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local from = shares.pos2idx(10, 10)
    local to   = shares.pos2idx(11, 10)
    local spore = cell_module.initCell(5, 10, 10, 0, {energy = 10, minerals = 5})
    local stem  = cell_module.initCell(3, 11, 10, 0, {energy = 20, minerals = 7})
    sim_module.addCell(shares, spore, view)
    sim_module.addCell(shares, stem, view)

    local counter_before = shares.CELL_COUNTER
    local queue_before   = {}
    for i = 1, counter_before do queue_before[i] = shares.CELL_QUEUE[i] end

    local ok = sim_module.moveCell(shares, from, to, view)
    assert(ok == false, 'moveCell must fail on an occupied tile')

    assert(shares.MAP_CELLS[from] == spore, 'spore must stay at its tile')
    assert(shares.MAP_CELLS[to] == stem, 'occupying cell must stay in place')
    assert(shares.CELL_COUNTER == counter_before, 'CELL_COUNTER must not change')
    for i = 1, counter_before do
        assert(shares.CELL_QUEUE[i] == queue_before[i], 'CELL_QUEUE must not change')
    end
    assert(spore[4] == 10 and spore[5] == 5, 'spore resources must not be touched')
    assert(stem[4] == 20 and stem[5] == 7, 'occupant resources must not be touched')
end

-- ---------------------------------------------------------------------------
-- 7. Очередь без дублей после тиков: 50 тиков споры-«ростка», seed = 7.
-- После КАЖДОГО тика: #CELL_QUEUE == CELL_COUNTER, значения 1..CELL_COUNTER
-- уникальны и ссылаются на живые клетки, MAP_TYPES согласован с MAP_CELLS.
-- ---------------------------------------------------------------------------
tests['CELL_QUEUE stays dense, unique and consistent after ticks'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local x = math.random(1, shares.MAP_WIDTH)
    local y = math.random(1, shares.MAP_HEIGHT)
    local sprout = cell_module.initCell(6, x, y, math.random(0, 3),
        {energy = 256, minerals = 256})
    assert(sim_module.addCell(shares, sprout, view), 'initial sprout must be added')

    for t = 1, 50 do
        sim_module.tick(shares, view)
        local queue, counter = shares.CELL_QUEUE, shares.CELL_COUNTER

        assert(#queue == counter,
            ('tick %d: #CELL_QUEUE=%d != CELL_COUNTER=%d'):format(t, #queue, counter))

        local seen = {}
        for i = 1, counter do
            local idx = queue[i]
            assert(idx ~= nil, ('tick %d: CELL_QUEUE[%d] is nil'):format(t, i))
            assert(not seen[idx],
                ('tick %d: duplicate index %d in CELL_QUEUE'):format(t, idx))
            seen[idx] = true
            assert(shares.MAP_CELLS[idx] ~= nil,
                ('tick %d: queued index %d is not a live cell'):format(t, idx))
        end

        for idx, c in pairs(shares.MAP_CELLS) do
            assert(shares.MAP_TYPES[idx] == c[2],
                ('tick %d: MAP_TYPES[%d]=%s != MAP_CELLS[%d][2]=%s')
                :format(t, idx, tostring(shares.MAP_TYPES[idx]), idx, tostring(c[2])))
        end
    end
end

-- ---------------------------------------------------------------------------
-- 8. MAP_CELLS/MAP_TYPES согласованы после reset и addCell/removeCell.
-- После reset: MAP_TYPES[i] == 0 для всех тайлов, MAP_CELLS пуст.
-- После addCell -> removeCell: обе записи по idx == nil.
-- ---------------------------------------------------------------------------
tests['MAP_CELLS and MAP_TYPES stay consistent on add/remove'] = function()
    sim_module.reset(shares)

    for i = 1, shares.MAP_SIZE do
        assert(shares.MAP_TYPES[i] == 0,
            ('MAP_TYPES[%d] = %s after reset, expected 0'):format(i, tostring(shares.MAP_TYPES[i])))
    end
    assert(next(shares.MAP_CELLS) == nil, 'MAP_CELLS must be empty after reset')

    local idx = shares.pos2idx(5, 5)
    local stem = cell_module.initCell(3, 5, 5, 0)
    assert(sim_module.addCell(shares, stem, view), 'addCell must succeed')
    assert(shares.MAP_CELLS[idx] == stem and shares.MAP_TYPES[idx] == 3,
        'addCell must write MAP_CELLS and MAP_TYPES')

    assert(sim_module.removeCell(shares, idx, view), 'removeCell must succeed')
    assert(shares.MAP_CELLS[idx] == nil, 'MAP_CELLS[idx] must be nil after removeCell')
    assert(shares.MAP_TYPES[idx] == nil, 'MAP_TYPES[idx] must be nil after removeCell')
end

-- ---------------------------------------------------------------------------
-- 9. Root извлекает из кардинального соседа, но НИКОГДА из собственного тайла:
-- dir = rand(1,4) выбирает один из четырёх соседей (x_offsets/y_offsets),
-- поэтому MAP_MINERALS собственного тайла не может быть затронут. Родитель-стем
-- на (9,10) выживает: его детский слот (8,10) занят Seed'ом (typ==4 > 2), а
-- Root на (10,10) имеет typ==2 и в расчёт n не идёт -> n == 2 (себе + Seed).
-- Экстракция проходит ровно один раз: один сосед теряет
-- min(значение, ROOT_MINERAL_EXTR), стем получает ту же сумму, ledgers
-- сохраняются (переток MAP_MINERALS -> cell[5] нейтрален).
-- ---------------------------------------------------------------------------
tests['root never extracts from its own tile'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local stem_idx = shares.pos2idx(9, 10)
    local root_idx = shares.pos2idx(10, 10)
    local stem = cell_module.initCell(3, 9, 10, 0, {energy = 100, minerals = 50})
    sim_module.addCell(shares, stem, view)
    -- Seed в детском слоте стема (8,10): стем с направлением 0 имеет
    -- cell[8..10] = (8,10), (9,9), (10,10). Иначе он умер бы (нет детей typ>2).
    local seed = cell_module.initCell(4, 8, 10, 0, {energy = 10, minerals = 5})
    sim_module.addCell(shares, seed, view)
    local root = cell_module.initCell(2, 10, 10, 0,
        {energy = 100, minerals = 40, parent = stem_idx})
    sim_module.addCell(shares, root, view)

    -- MAP_MINERALS вручную: свой тайл 100, соседи 90/80/70/60 (все > EXTR=20).
    local neighbors = {
        shares.pos2idx(9, 10),  -- 90, -x
        shares.pos2idx(11, 10), -- 80, +x
        shares.pos2idx(10, 9),  -- 70, -y
        shares.pos2idx(10, 11), -- 60, +y
    }
    local initial = {90, 80, 70, 60}
    shares.MAP_MINERALS[root_idx] = 100
    for i = 1, 4 do shares.MAP_MINERALS[neighbors[i]] = initial[i] end

    local e0, m0 = energyLedger(), mineralLedger()
    sim_module.tick(shares, view)

    assert(shares.MAP_CELLS[stem_idx] ~= nil, 'stem must survive the tick')
    assert(shares.MAP_CELLS[root_idx] ~= nil, 'root must survive the tick')
    assert(shares.MAP_CELLS[seed[1]] ~= nil, 'seed child must survive the tick')
    assert(shares.MAP_MINERALS[root_idx] == 100,
        'root tile minerals must be untouched (extraction is never from the own tile)')

    -- Ровно один из четырёх соседей потерял min(значение, ROOT_MINERAL_EXTR);
    -- все значения > EXTR, поэтому дельта == EXTR == 20.
    local decreased = 0
    for i = 1, 4 do
        local delta = initial[i] - shares.MAP_MINERALS[neighbors[i]]
        if delta == 0 then -- не тронут
        elseif delta == shares.ROOT_MINERAL_EXTR then
            decreased = decreased + 1
        else
            assert(false, ('neighbor %d minerals changed by %g (expected 0 or %g)')
                :format(i, delta, shares.ROOT_MINERAL_EXTR))
        end
    end
    assert(decreased == 1,
        ('exactly one cardinal neighbor must lose minerals, got %d'):format(decreased))

    -- Стём поделился пополам (n == 2) и получил экстракцию в apply-фазе.
    assertClose(stem[4], 50, 1e-6, 'stem energy after 1:2 split')
    assertClose(stem[5], 25 + shares.ROOT_MINERAL_EXTR, 1e-6,
        'stem minerals must grow by exactly the extracted amount')
    assertClose(energyLedger(), e0, 1e-6, 'energy ledger must be conserved')
    assertClose(mineralLedger(), m0, 1e-6, 'mineral ledger must be conserved')
end

-- ---------------------------------------------------------------------------
-- 10. Стём делит энергию/минералы поровну между собой и живыми детьми typ>2.
-- Stem (10,10) dir 0: cell[8..10] = (9,10), (10,9), (11,10). Два Seed на
-- (9,10) и (11,10) -> n == 3: стем оставляет 1/3 (40/20), каждому семени
-- уходит по 1/3 (40/20), которые СУММИРУЮТСЯ с их собственными 10/5.
-- До тика: 120/60 + 10/5 + 10/5 = 140/70. После: 40/20 + 50/25 + 50/25 = 140/70.
-- ВНИМАНИЕ: Seed — AI-клетка и может морфиться 4->6 (cell[5] += COSTS[4]-COSTS[6]
-- == +1). Морф нейтрален для резерва CELL_COSTS (3 -> 2), но сдвигает cell[5]
-- на 1, и решение AI зависит от бэкенда (lua/luajit). Поэтому по минералам
-- проверяем резерв-инвариант cell[5] + CELL_COSTS[typ] (всегда == 28 на семя,
-- == 21 на стем), а не голую сумму cell[5].
-- ---------------------------------------------------------------------------
tests['stem splits energy and minerals conservatively'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local stem = cell_module.initCell(3, 10, 10, 0, {energy = 120, minerals = 60})
    sim_module.addCell(shares, stem, view)
    local seed1 = cell_module.initCell(4, 9, 10, 0, {energy = 10, minerals = 5})
    local seed2 = cell_module.initCell(4, 11, 10, 0, {energy = 10, minerals = 5})
    sim_module.addCell(shares, seed1, view)
    sim_module.addCell(shares, seed2, view)

    local seed1_idx, seed2_idx = seed1[1], seed2[1]
    local e0, m0 = energyLedger(), mineralLedger()
    sim_module.tick(shares, view)

    assert(shares.MAP_CELLS[stem[1]] ~= nil, 'stem must survive the tick (n == 3)')
    assert(shares.MAP_CELLS[seed1_idx] ~= nil, 'seed 1 must survive the tick')
    assert(shares.MAP_CELLS[seed2_idx] ~= nil, 'seed 2 must survive the tick')

    -- 1/3 остаётся стему, 2/3 уходят детям.
    assertClose(stem[4], 40, 1e-6, 'stem keeps 1/3 of its energy')
    assertClose(stem[5], 20, 1e-6, 'stem keeps 1/3 of its minerals')
    assertClose(seed1[4], 50, 1e-6, 'seed 1 receives +40 energy')
    assertClose(seed2[4], 50, 1e-6, 'seed 2 receives +40 energy')

    -- Резерв-инвариант: 5 (свои) + 20 (от стема) + 3 (резерв Seed) == 28.
    local function reserve(c) return c[5] + shares.CELL_COSTS[c[2]] end
    assertClose(reserve(seed1), 28, 1e-6, 'seed 1 minerals + reserve conserved')
    assertClose(reserve(seed2), 28, 1e-6, 'seed 2 minerals + reserve conserved')

    -- Суммы: энергия 120 + 2*10 = 140; минералы+резервы 61 + 2*8 = 77.
    assertClose(stem[4] + seed1[4] + seed2[4], 140, 1e-6, 'total energy conserved')
    assertClose(reserve(stem) + reserve(seed1) + reserve(seed2), 77, 1e-6,
        'total minerals (incl. CELL_COSTS reserve) conserved')
    assertClose(energyLedger(), e0, 1e-6, 'energy ledger must be conserved')
    assertClose(mineralLedger(), m0, 1e-6, 'mineral ledger must be conserved')
end

-- ---------------------------------------------------------------------------
-- 11. Sprout: неудачный спавн -> полный рефанд энергии и минералов.
-- Sprout (10,10) dir 0: слоты детей cell[8..10] = (9,10), (10,9), (11,10).
-- Все три заняты Seed'ами (typ==4): они не умирают за тик и не передают
-- ресурсов спруту, поэтому все три addCell в spawn-фазе проваливаются.
-- ВАЖНО: n в ветке Sprout считает ПОПЫТКИ спавна (typ > 0), а не успешные
-- размещения — поэтому при всех неудачах спрут всё равно морфится 6->3
-- (cell[5] += COSTS[6]-COSTS[3] == +1), и эта поправка входит в ожидания.
-- Стем-оккупанты тут не подошли бы: стем без детей typ>2 умирает (n == 1) и
-- освобождает тайл — тогда спавн бы удался (и морф-поправка была бы другой).
-- Рефанд: энергия shared (50) возвращается через parent[4] += cell[4] на
-- каждой неудаче (50 + 3*50 == 200); минералы -cost + (0 + COSTS[typ]) == 0
-- на каждой неудаче (50 + морф-поправка 1 == 51).
-- ---------------------------------------------------------------------------
tests['sprout failed spawn refunds everything'] = function()
    math.randomseed(7)
    sim_module.reset(shares)

    local sprout = cell_module.initCell(6, 10, 10, 0, {energy = 200, minerals = 50})
    sim_module.addCell(shares, sprout, view)

    local occupiers = {}
    for j = 1, 3 do
        local x, y = shares.idx2pos(sprout[7 + j])
        local seed = cell_module.initCell(4, x, y, 0, {energy = 10, minerals = 5})
        occupiers[j] = seed
        sim_module.addCell(shares, seed, view)
    end

    local counter_before = shares.CELL_COUNTER
    local e0, m0 = energyLedger(), mineralLedger()
    sim_module.tick(shares, view)

    assert(shares.MAP_CELLS[sprout[1]] ~= nil, 'sprout must survive the tick')
    assert(sprout[2] == 3,
        'sprout must morph 6->3 (n counts spawn attempts; every placement failed)')

    -- shared == 200/4 == 50; каждая из трёх неудач возвращает свои 50.
    assertClose(sprout[4], 200, 1e-6, 'all failed spawns must refund their energy share')
    -- Минералы вернулись полностью: -cost + (0 + COSTS[typ]) == 0, плюс морф 6->3.
    assertClose(sprout[5], 50 + shares.CELL_COSTS[6] - shares.CELL_COSTS[3], 1e-6,
        'all failed spawns must refund their minerals (plus morph reserve adjustment)')

    assert(shares.CELL_COUNTER == counter_before,
        'no new cell may appear (every spawn placement failed)')
    for j = 1, 3 do
        assert(shares.MAP_CELLS[occupiers[j][1]] ~= nil,
            ('occupying seed %d must stay in place'):format(j))
    end
    assertClose(energyLedger(), e0, 1e-6, 'energy ledger must be conserved')
    assertClose(mineralLedger(), m0, 1e-6, 'mineral ledger must be conserved')
end

-- ---------------------------------------------------------------------------
-- 12. Глобальная консервация за 200 тиков (seed = 11, стартовый Sprout).
-- Расход энергии в конфиге равен нулю, поэтому энергетический леджер
-- (cell[4] живых + MAP_ENERGY) меняется РОВНО на фотосинтез выживающих
-- листьев: лист (typ == 1) с истинным MAP_TYPES[cell[7]] (0 на пустом тайле
-- тоже истинно, как в симуляции), cell[4] > 0 и cell[6] < CELL_AGES[1] ДО
-- тика даёт LEAF_ENERGY_GEN * sun. Солнце считается как
-- calcSunFactor(shares, shares.step + 1) — step берётся ДО вызова tick
-- (tick сам инкрементирует step). Минеральный леджер (cell[5] +
-- CELL_COSTS[typ] + MAP_MINERALS) строго постоянен. Цикл идёт все 200
-- итераций даже при вымирании (tick вернёт true — ledgers и очередь
-- остаются консистентными).
-- ---------------------------------------------------------------------------
tests['global conservation over 200 ticks'] = function()
    math.randomseed(11)
    sim_module.reset(shares)

    local x = math.random(1, shares.MAP_WIDTH)
    local y = math.random(1, shares.MAP_HEIGHT)
    local sprout = cell_module.initCell(6, x, y, math.random(0, 3),
        {energy = 256, minerals = 256})
    assert(sim_module.addCell(shares, sprout, view), 'initial sprout must be added')

    for t = 1, 200 do
        -- Ожидаемый прирост энергии: только выживающие листья фотосинтезируют.
        local sun = sim_module.calcSunFactor(shares, shares.step + 1)
        local expected = 0
        for _, c in pairs(shares.MAP_CELLS) do
            if c[2] == 1 and shares.MAP_TYPES[c[7]]
                and c[4] > 0 and c[6] < shares.CELL_AGES[1] then
                expected = expected + shares.LEAF_ENERGY_GEN * sun
            end
        end

        local e_before, m_before = energyLedger(), mineralLedger()
        sim_module.tick(shares, view)

        assertClose(energyLedger(), e_before + expected, 1e-6,
            ('tick %d: energy ledger grew by photosynthesis only'):format(t))
        assertClose(mineralLedger(), m_before, 1e-6,
            ('tick %d: mineral ledger must stay constant'):format(t))

        -- Очередь: плотная (# == CELL_COUNTER), без дублей, только живые.
        local queue, counter = shares.CELL_QUEUE, shares.CELL_COUNTER
        assert(#queue == counter,
            ('tick %d: #CELL_QUEUE=%d != CELL_COUNTER=%d'):format(t, #queue, counter))
        local seen = {}
        for i = 1, counter do
            local idx = queue[i]
            assert(idx ~= nil, ('tick %d: CELL_QUEUE[%d] is nil'):format(t, i))
            assert(not seen[idx],
                ('tick %d: duplicate index %d in CELL_QUEUE'):format(t, idx))
            seen[idx] = true
            assert(shares.MAP_CELLS[idx] ~= nil,
                ('tick %d: queued index %d is not a live cell'):format(t, idx))
        end

        -- MAP_TYPES согласован с MAP_CELLS для всех живых клеток.
        for idx, c in pairs(shares.MAP_CELLS) do
            assert(shares.MAP_TYPES[idx] == c[2],
                ('tick %d: MAP_TYPES[%d]=%s != MAP_CELLS[%d][2]=%s')
                :format(t, idx, tostring(shares.MAP_TYPES[idx]), idx, tostring(c[2])))
        end
    end
end

-- ---------------------------------------------------------------------------
-- 13. Счётчики genome-слотов против живых AI-клеток за 200 тиков.
-- Каждая живая клетка typ >= 4 владеет ровно одной ссылкой на слот
-- (acquire в initCell, release при removeCell и при морфе Sprout->Stem),
-- поэтому после КАЖДОГО тика сумма counter по всем слотам CELL_GENOMES
-- (итерация через pairs) обязана равняться числу живых клеток typ >= 4
-- по MAP_CELLS; ни один counter не отрицателен.
-- ---------------------------------------------------------------------------
tests['genome counters match live AI cells over 200 ticks'] = function()
    math.randomseed(11)
    sim_module.reset(shares)

    local x = math.random(1, shares.MAP_WIDTH)
    local y = math.random(1, shares.MAP_HEIGHT)
    local sprout = cell_module.initCell(6, x, y, math.random(0, 3),
        {energy = 256, minerals = 256})
    assert(sim_module.addCell(shares, sprout, view), 'initial sprout must be added')

    for t = 1, 200 do
        sim_module.tick(shares, view)

        local counter_sum = 0
        for _, place in pairs(shares.CELL_GENOMES) do
            assert(place.counter ~= nil and place.counter >= 0,
                ('tick %d: genome counter %s is negative'):format(t, tostring(place.counter)))
            counter_sum = counter_sum + place.counter
        end

        local ai_count = 0
        for _, c in pairs(shares.MAP_CELLS) do
            if c[2] >= 4 then ai_count = ai_count + 1 end
        end

        assert(counter_sum == ai_count,
            ('tick %d: genome counter sum %d != live AI cells %d')
            :format(t, counter_sum, ai_count))
    end
end

-- ---------------------------------------------------------------------------
-- 14. tick возвращает true при вымирании.
-- Пустая карта: CELL_COUNTER == 0 -> tick сразу возвращает true.
-- Один Stem с энергией 0: расход CONS == 0, поэтому cell[4] остаётся 0,
-- ветка `cell[4] > 0` не выполняется, клетка умирает в первый же тик,
-- MAP_CELLS пуст, tick возвращает true.
-- ---------------------------------------------------------------------------
tests['tick returns true on extinction'] = function()
    -- Первая проверка: пустая карта (reset, ни одной клетки).
    math.randomseed(11)
    sim_module.reset(shares)
    assert(sim_module.tick(shares, view) == true,
        'tick must return true when the map is empty')

    -- Вторая проверка: клетка с нулевой энергией умирает в первый тик.
    sim_module.reset(shares)
    local stem = cell_module.initCell(3, 10, 10, 0, {energy = 0, minerals = 10})
    assert(sim_module.addCell(shares, stem, view), 'stem must be added')
    assert(sim_module.tick(shares, view) == true,
        'tick must return true when the last cell dies')
    assert(next(shares.MAP_CELLS) == nil, 'MAP_CELLS must be empty after extinction')
end

return tests
