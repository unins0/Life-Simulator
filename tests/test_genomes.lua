-- test_genomes.lua — тесты владения genome-слотами (BUG-1).
-- Проверяют единый механизм acquire/release из ai_module, переиспользование
-- слотов с counter == 0, ограниченность CELL_GENOMES, запрет двойного
-- release/acquire отсутствующего слота и инварианты морфов между AI-типами.
-- Запуск (из корня проекта): lua tests/test_runner.lua tests/test_genomes.lua
-- (работает и с lua 5.5 — табличный бэкенд, и с luajit — FFI-бэкенд).

package.path = './tests/?.lua;' .. package.path

local shares       = require('shares')
local ai_module    = require('ai_module')
local cell_module  = require('cell_module')
local H            = require('tests.helpers')

local tests = {}

-- Гарантированно отсутствующий слот (addGenome никогда не дойдёт до таких
-- индексов, а тесты не оставляют плотных записей выше ~нескольких сотен).
local MISSING_SLOT = 999999

-- ---------------------------------------------------------------------------
-- 1. acquire/release баланс: counter строго следует за числом вызовов.
-- ---------------------------------------------------------------------------
tests['acquire/release balance'] = function()
    math.randomseed(42)
    local slot       = ai_module.genWeights()
    local place      = shares.CELL_GENOMES[slot]
    assert(place.counter == 0,
        ('fresh genome must start with counter 0, got %d'):format(place.counter))

    for i = 1, 5 do
        ai_module.acquire(slot)
        assert(place.counter == 1,
            ('after acquire #%d counter = %d, expected 1'):format(i, place.counter))
        ai_module.release(slot)
        assert(place.counter == 0,
            ('after release #%d counter = %d, expected 0'):format(i, place.counter))
    end
    assert(place.counter == 0, 'counter must return to 0 after balanced cycles')
end

-- ---------------------------------------------------------------------------
-- 2. Переиспользование слотов: освобождённые слоты (counter == 0) занимаются
-- addGenome снова; максимальный индекс не растёт.
-- ---------------------------------------------------------------------------
tests['freed slots are reused'] = function()
    math.randomseed(42)
    local K = 20

    local max1 = 0
    local slots = {}
    for i = 1, K do
        local s = ai_module.genWeights()
        slots[i] = s
        if s > max1 then max1 = s end
    end
    for i = 1, K do
        ai_module.acquire(slots[i])
        ai_module.release(slots[i]) -- counter обратно в 0
    end

    local max2 = 0
    for i = 1, K do
        local s = ai_module.genWeights()
        if s > max2 then max2 = s end
    end

    assert(max2 <= max1,
        ('slots not reused: max index %d after release, was %d'):format(max2, max1))
end

-- ---------------------------------------------------------------------------
-- 3. Ограниченность таблицы: цикл «создать 50 → освободить все» × 10;
-- #CELL_GENOMES не растёт неограниченно.
-- ---------------------------------------------------------------------------
tests['CELL_GENOMES stays bounded under churn'] = function()
    math.randomseed(42)
    for iter = 1, 10 do
        local slots = {}
        for i = 1, 50 do
            slots[i] = ai_module.genWeights()
        end
        for i = 1, 50 do
            ai_module.acquire(slots[i])
            ai_module.release(slots[i])
        end
        assert(#shares.CELL_GENOMES <= 100,
            ('#CELL_GENOMES = %d after iteration %d, expected <= 100')
            :format(#shares.CELL_GENOMES, iter))
    end
end

-- ---------------------------------------------------------------------------
-- 4. Запрет двойного release живого слота: второй вызов обязан упасть.
-- ---------------------------------------------------------------------------
tests['double release of a live slot errors'] = function()
    math.randomseed(42)
    local slot = ai_module.genWeights()
    ai_module.acquire(slot)
    ai_module.release(slot) -- counter 0
    local ok, err = pcall(ai_module.release, slot)
    assert(not ok, 'second release must error (counter <= 0)')
    assert(type(err) == 'string' and err ~= '',
        'release error must carry a message')
end

-- ---------------------------------------------------------------------------
-- 5. acquire отсутствующего слота → error.
-- ---------------------------------------------------------------------------
tests['acquire of missing slot errors'] = function()
    math.randomseed(42)
    assert(shares.CELL_GENOMES[MISSING_SLOT] == nil, 'precondition: slot must be missing')
    local ok, err = pcall(ai_module.acquire, MISSING_SLOT)
    assert(not ok, 'acquire of missing slot must error')
    assert(type(err) == 'string' and err ~= '',
        'acquire error must carry a message')
end

-- ---------------------------------------------------------------------------
-- 5b. release отсутствующего (уже удалённого) слота — идемпотентный no-op.
-- ---------------------------------------------------------------------------
tests['release of missing slot is a no-op'] = function()
    math.randomseed(42)
    assert(shares.CELL_GENOMES[MISSING_SLOT] == nil, 'precondition: slot must be missing')
    local before = #shares.CELL_GENOMES
    assert(ai_module.release(MISSING_SLOT) == nil, 'release must return nothing')
    assert(#shares.CELL_GENOMES == before, 'no-op release must not change the table')
end

-- ---------------------------------------------------------------------------
-- 6. Не-AI клетки (Leaf/Root/Stem) не владеют геномом: cell[11] == nil.
-- ---------------------------------------------------------------------------
tests['Leaf/Root/Stem cells do not own a genome'] = function()
    math.randomseed(42)
    for typ = 1, 3 do
        local cell = cell_module.initCell(typ, 1, 1, 0)
        assert(cell[11] == nil,
            ('cell type %d must not own a genome, cell[11] = %s')
            :format(typ, tostring(cell[11])))
    end
end

-- ---------------------------------------------------------------------------
-- 7. AI-клетки (Seed/Spore/Sprout) владеют геномом; initCell уже acquire'ит
-- (клетка владеет ссылкой с момента создания, counter 1), addCell больше не
-- acquire'ит, removeCell (release) возвращает counter в 0.
-- ---------------------------------------------------------------------------
tests['AI cells own a genome and initCell acquires'] = function()
    math.randomseed(42)
    for typ = 4, 6 do
        local cell = cell_module.initCell(typ, 1, 1, 0)
        assert(cell[11] ~= nil, ('cell type %d must own a genome'):format(typ))
        local place = shares.CELL_GENOMES[cell[11]]
        assert(place ~= nil, ('slot %s must exist'):format(tostring(cell[11])))
        assert(place.counter == 1,
            ('initCell must acquire: counter = %d, expected 1'):format(place.counter))
        ai_module.release(cell[11]) -- уборка (как removeCell)
        assert(place.counter == 0, 'release must bring counter back to 0')
    end
end

-- ---------------------------------------------------------------------------
-- 8. Морфы между AI-типами (Seed->Sprout, Spore->Seed) не трогают counter и
-- сохраняют cell[11].
-- ---------------------------------------------------------------------------
tests['Seed->Sprout and Spore->Seed keep the genome reference'] = function()
    math.randomseed(42)

    -- Seed (4) -> Sprout (6)
    local seed = cell_module.initCell(4, 1, 1, 0)
    local seedSlot = seed[11]
    local counterBefore = shares.CELL_GENOMES[seedSlot].counter -- 1 (initCell)
    seed[2] = 6 -- морф, как в tick()
    assert(seed[11] == seedSlot, 'Seed->Sprout must keep cell[11]')
    assert(shares.CELL_GENOMES[seedSlot].counter == counterBefore,
        'Seed->Sprout must not change the genome counter')
    ai_module.release(seedSlot)

    -- Spore (5) -> Seed (4)
    local spore = cell_module.initCell(5, 1, 1, 0)
    local sporeSlot = spore[11]
    counterBefore = shares.CELL_GENOMES[sporeSlot].counter -- 1 (initCell)
    spore[2] = 4 -- морф, как в tick()
    assert(spore[11] == sporeSlot, 'Spore->Seed must keep cell[11]')
    assert(shares.CELL_GENOMES[sporeSlot].counter == counterBefore,
        'Spore->Seed must not change the genome counter')
    ai_module.release(sporeSlot)
end

-- ---------------------------------------------------------------------------
-- 9. Инвариант «counter == числу живых ссылок»: случайная последовательность
-- create/acquire/release; после каждого шага сверяем сумму counter'ов с
-- собственным счётчиком живых acquire.
-- ---------------------------------------------------------------------------
tests['counter equals live references under random churn'] = function()
    math.randomseed(12345)
    local live  = 0   -- сколько ссылок мы сейчас считаем живыми
    local owned = {}  -- slot -> число наших живых ссылок на него

    local function track(slot)
        owned[slot] = (owned[slot] or 0) + 1
        live = live + 1
    end
    local function untrack(slot)
        owned[slot] = owned[slot] - 1
        live = live - 1
        if owned[slot] == 0 then owned[slot] = nil end
    end
    local function ownedSlots()
        local out = {}
        for s in pairs(owned) do out[#out + 1] = s end
        return out
    end

    for step = 1, 200 do
        local r = math.random(10)
        if r <= 4 then
            -- create: genWeights возвращает слот с counter 0, acquire делает 1
            local slot = ai_module.genWeights()
            ai_module.acquire(slot)
            track(slot)
        elseif r <= 7 then
            -- acquire ещё одну ссылку на существующий геном
            local slots = ownedSlots()
            if #slots > 0 then
                local slot = slots[math.random(#slots)]
                ai_module.acquire(slot)
                track(slot)
            end
        else
            -- release одной из наших ссылок
            local slots = ownedSlots()
            if #slots > 0 then
                local slot = slots[math.random(#slots)]
                ai_module.release(slot)
                untrack(slot)
            end
        end

        -- Сверка: сумма counter'ов всех плотных слотов == числу живых ссылок.
        local sum = 0
        local n = #shares.CELL_GENOMES
        for i = 1, n do
            sum = sum + shares.CELL_GENOMES[i].counter
        end
        assert(sum == live,
            ('step %d: counter sum = %d, live references = %d'):format(step, sum, live))
    end
end

-- ---------------------------------------------------------------------------
-- 10. Регрессия aliasing (BUG-1): Sprout порождает двух МУТИРУЮЩИХ AI-детей
-- подряд — до фикса addGenome переиспользовал слот первого ребёнка (counter 0
-- до addCell) и затирал его мутацию. Теперь initCell acquire'ит сразу, поэтому
-- второй ребёнок обязан получить ДРУГОЙ слот.
-- ---------------------------------------------------------------------------
tests['two mutating AI children get distinct genome slots'] = function()
    local savedChance = ai_module.GENOME_MUTATION_CHANCE

    -- Фаза 1 (детерминированная): chance = 1.0 -> оба ребёнка гарантированно
    -- мутируют; окно aliasing воспроизводится на каждом прогоне.
    ai_module.GENOME_MUTATION_CHANCE = 1.0
    math.randomseed(424242)
    local pSlot = ai_module.genWeights()
    ai_module.acquire(pSlot) -- родитель жив на карте, counter 1
    local parentCounter = shares.CELL_GENOMES[pSlot].counter
    assert(parentCounter == 1, ('parent counter = %d, expected 1'):format(parentCounter))

    local childA = cell_module.initCell(4, 5, 5, 0, {genome = pSlot})
    local childB = cell_module.initCell(4, 7, 7, 0, {genome = pSlot})
    local slotA, slotB = childA[11], childB[11]

    assert(slotA ~= pSlot, 'child A must have mutated (forced chance)')
    assert(slotB ~= pSlot, 'child B must have mutated (forced chance)')
    assert(slotA ~= slotB,
        ('aliasing: both children share slot %d (mutation of A was overwritten)')
        :format(slotA))

    -- Мутировавшие геномы различны хотя бы в одной позиции.
    local dataA = H.dataCopy(slotA)
    local dataB = H.dataCopy(slotB)
    local differ = false
    for i = 1, #dataA do
        if dataA[i] ~= dataB[i] then differ = true break end
    end
    assert(differ, 'mutated children produced identical genomes')

    -- Оба слота acquired в initCell (counter 1); слот родителя не тронут.
    assert(shares.CELL_GENOMES[slotA].counter == 1,
        ('slot A counter = %d, expected 1'):format(shares.CELL_GENOMES[slotA].counter))
    assert(shares.CELL_GENOMES[slotB].counter == 1,
        ('slot B counter = %d, expected 1'):format(shares.CELL_GENOMES[slotB].counter))
    assert(shares.CELL_GENOMES[pSlot].counter == parentCounter,
        'parent slot counter must not change when children mutate')

    -- Уборка: 1 (родитель) + 1 (A) + 1 (B).
    ai_module.release(childB[11])
    ai_module.release(childA[11])
    ai_module.release(pSlot)
    ai_module.GENOME_MUTATION_CHANCE = savedChance

    -- Фаза 2: метание окна по 50 разным seed'ам. Chance поднят до 0.5, чтобы
    -- оба ребёнка мутировали в ~25% прогонов и инвариант проверялся надёжно;
    -- каждый прогон детерминирован своим seed'ом.
    ai_module.GENOME_MUTATION_CHANCE = 0.5
    local windowHits = 0
    for i = 1, 50 do
        math.randomseed(1000 + i * 97)
        local p = ai_module.genWeights()
        ai_module.acquire(p)
        local a = cell_module.initCell(4, 5, 5, 0, {genome = p})
        local b = cell_module.initCell(4, 7, 7, 0, {genome = p})
        local sa, sb = a[11], b[11]
        if sa ~= p and sb ~= p then
            windowHits = windowHits + 1
            assert(sa ~= sb,
                ('seed %d: aliasing — both children mutated into slot %d')
                :format(1000 + i * 97, sa))
            assert(shares.CELL_GENOMES[sa].counter == 1
                and shares.CELL_GENOMES[sb].counter == 1,
                'mutated children slots must each have counter 1')
            local dA = H.dataCopy(sa)
            local dB = H.dataCopy(sb)
            local diff = false
            for j = 1, #dA do
                if dA[j] ~= dB[j] then diff = true break end
            end
            assert(diff, ('seed %d: mutated children produced identical genomes')
                :format(1000 + i * 97))
        end
        -- Уборка: 3 acquire (p + a + b) = 3 release.
        ai_module.release(b[11])
        ai_module.release(a[11])
        ai_module.release(p)
    end
    ai_module.GENOME_MUTATION_CHANCE = savedChance
    io.write(('  (aliasing window observed %d/50 seed runs)\n'):format(windowHits))
end

-- ---------------------------------------------------------------------------
-- 11. Баланс создания->смерти: initCell acquire'ит (counter 1), addCell не
-- acquire'ит (имитация записи на карту), removeCell release'ит (counter 0).
-- ---------------------------------------------------------------------------
tests['create->death balance (initCell/acquire, addCell, removeCell/release)'] = function()
    math.randomseed(7)
    local cell = cell_module.initCell(4, 3, 3, 0)
    local slot = cell[11]
    local place = shares.CELL_GENOMES[slot]
    assert(place ~= nil and place.counter == 1,
        ('initCell must acquire: counter = %d, expected 1')
        :format(place and place.counter or -1))

    -- Имитация addCell: запись на карту без acquire.
    local idx = cell[1]
    shares.MAP_CELLS[idx] = cell
    shares.MAP_TYPES[idx] = cell[2]
    assert(place.counter == 1,
        ('addCell must not acquire: counter = %d, expected 1'):format(place.counter))

    -- Имитация removeCell: release.
    shares.MAP_CELLS[idx] = nil
    shares.MAP_TYPES[idx] = nil
    ai_module.release(slot)
    assert(place.counter == 0,
        ('removeCell must release: counter = %d, expected 0'):format(place.counter))
end

-- ---------------------------------------------------------------------------
-- 12. Неудачный спавн (рефанд при занятой цели): initCell наследует слот
-- родителя и acquire'ит (counter = parent+1); release(cell[11]) возвращает его
-- (counter = parent).
-- ---------------------------------------------------------------------------
tests['failed spawn releases the acquired reference'] = function()
    math.randomseed(11)
    local savedChance = ai_module.GENOME_MUTATION_CHANCE
    ai_module.GENOME_MUTATION_CHANCE = 0.0 -- гарантированное наследование слота

    local pSlot = ai_module.genWeights()
    ai_module.acquire(pSlot) -- родитель
    local parentCounter = shares.CELL_GENOMES[pSlot].counter -- 1

    local child = cell_module.initCell(4, 9, 9, 0, {genome = pSlot})
    assert(child[11] == pSlot, 'child must inherit the parent slot (chance forced to 0)')
    assert(shares.CELL_GENOMES[pSlot].counter == parentCounter + 1,
        ('initCell must acquire the parent slot: counter = %d, expected %d')
        :format(shares.CELL_GENOMES[pSlot].counter, parentCounter + 1))

    -- Имитация рефанда при занятой цели: release(cell[11]) — ровно один раз.
    ai_module.release(child[11])
    assert(shares.CELL_GENOMES[pSlot].counter == parentCounter,
        ('failed spawn must release: counter = %d, expected %d')
        :format(shares.CELL_GENOMES[pSlot].counter, parentCounter))

    ai_module.release(pSlot) -- уборка
    ai_module.GENOME_MUTATION_CHANCE = savedChance
end

return tests
