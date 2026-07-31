-- test_ai.lua — тесты ai_module: длина генома, генерация, мутация, прогон сети.
-- Запуск (из корня проекта): lua tests/test_runner.lua tests/test_ai.lua
-- Доступ к весам — через tests/helpers.lua (адаптер, переживёт рефакторинг
-- представления генома на обёртку {data=..., counter=..., is_ffi=...}).

package.path = './tests/?.lua;' .. package.path

local shares    = require('shares')
local ai_module = require('ai_module')
local H         = require('tests.helpers')

local tests = {}

-- ---------------------------------------------------------------------------
-- Эталонная (медленная, очевидно-корректная) реализация прямой сети.
-- Раскладка весов как в ai_module.run: на узел (bias, threshold, dead,
-- *веса в следующий слой); финальный слой: (bias, threshold, dead).
-- Неинициализированные данные считаются 0.
-- ---------------------------------------------------------------------------
local function referenceRun(weights, layers, offset, inputs)
    local nlayers  = #layers
    local out_size = layers[nlayers]
    local w = 1 + offset
    local prev, nxt = {}, {}
    for i = 1, layers[1] do prev[i] = inputs[i] or 0.0 end

    for l = 2, nlayers do
        local next_size = layers[l]
        for j = 1, layers[l - 1] do
            local bias, thr, dead = weights[w], weights[w + 1], weights[w + 2]
            local value = (prev[j] or 0.0) + bias
            if value <= thr then value = dead end
            for k = 1, next_size do
                nxt[k] = (nxt[k] or 0.0) + value * weights[w + 2 + k]
            end
            w = w + 3 + next_size
        end
        prev, nxt = nxt, {}
    end

    local result = {}
    for i = 1, out_size do
        local bias, thr, dead = weights[w], weights[w + 1], weights[w + 2]
        local value = (prev[i] or 0.0) + bias
        if value <= thr then value = dead end
        result[i] = value
        w = w + 3
    end
    return result
end

-- ---------------------------------------------------------------------------
-- 1. Длина генома по конфигам.
-- ---------------------------------------------------------------------------
tests['genome length matches configs'] = function()
    math.randomseed(42)

    local slot = ai_module.genWeights()
    assert(H.dataLen(slot) == shares.AI_LEN_COMMON,
        ('genWeights() produced %d weights, expected AI_LEN_COMMON=%d')
        :format(H.dataLen(slot), shares.AI_LEN_COMMON))

    -- Длина сети по формуле: n += layer*(3+next_layer); n += layers[len]*3
    local function countWeights(layers)
        local len = #layers
        local n = 0
        for i = 1, len - 1 do
            local layer, next_layer = layers[i], layers[i + 1]
            n = n + layer * (3 + next_layer)
        end
        return n + layers[len] * 3
    end

    assert(countWeights({9, 12, 1}) == shares.AI_LEN_SEED,
        ('formula({9,12,1}) = %d, expected AI_LEN_SEED=%d')
        :format(countWeights({9, 12, 1}), shares.AI_LEN_SEED))
    assert(countWeights({6, 16, 1}) == shares.AI_LEN_SPORE,
        ('formula({6,16,1}) = %d, expected AI_LEN_SPORE=%d')
        :format(countWeights({6, 16, 1}), shares.AI_LEN_SPORE))
    assert(countWeights({9, 18, 16, 3}) == shares.AI_LEN_SPROUT,
        ('formula({9,18,16,3}) = %d, expected AI_LEN_SPROUT=%d')
        :format(countWeights({9, 18, 16, 3}), shares.AI_LEN_SPROUT))

    assert(shares.AI_LEN_COMMON == shares.AI_LEN_SEED + shares.AI_LEN_SPORE + shares.AI_LEN_SPROUT,
        'AI_LEN_COMMON != AI_LEN_SEED + AI_LEN_SPORE + AI_LEN_SPROUT')
end

-- ---------------------------------------------------------------------------
-- 2. Диапазон генерации.
-- ---------------------------------------------------------------------------
tests['genWeights range within GENOME_INIT_MULT'] = function()
    math.randomseed(42)

    -- mult по умолчанию = GENOME_INIT_MULT = 100: (rand()-0.5)*scale, scale=2*mult
    local slot = ai_module.genWeights()
    for i = 1, H.dataLen(slot) do
        local w = H.weightAt(slot, i)
        assert(math.abs(w) <= ai_module.GENOME_INIT_MULT + 1e-9,
            ('weight[%d] = %.10g, |w| > GENOME_INIT_MULT'):format(i, w))
    end

    -- вариант mult = 1.0 -> |w| <= 1.0
    local slot1 = ai_module.genWeights(1.0)
    for i = 1, H.dataLen(slot1) do
        local w = H.weightAt(slot1, i)
        assert(math.abs(w) <= 1.0 + 1e-9,
            ('weight[%d] = %.10g, |w| > 1.0'):format(i, w))
    end
end

-- ---------------------------------------------------------------------------
-- 3. Мутация — длина сохраняется.
-- ---------------------------------------------------------------------------
tests['mutateWeights keeps length'] = function()
    math.randomseed(42)
    local parent = ai_module.genWeights()
    local child  = ai_module.mutateWeights(parent)
    assert(H.dataLen(child) == shares.AI_LEN_COMMON,
        ('mutated genome has %d weights, expected %d')
        :format(H.dataLen(child), shares.AI_LEN_COMMON))
end

-- ---------------------------------------------------------------------------
-- 4. Мутация — копия, не мутация родителя.
-- ---------------------------------------------------------------------------
tests['mutateWeights does not alter parent'] = function()
    math.randomseed(42)
    local parentSlot          = ai_module.genWeights()
    local parentData, parentFfi = H.genomeData(parentSlot) -- объект данных родителя
    local snapshot            = H.dataCopy(parentSlot)     -- копия ДО вызова

    local childSlot = ai_module.mutateWeights(parentSlot)

    -- Родительский объект не изменился (мутация работает с копией).
    -- addGenome может переиспользовать слот родителя, поэтому читаем данные
    -- по захваченной ссылке через helpers (ffi-массивы читаются 1-based).
    for i = 1, #snapshot do
        assert(H.rawAt(parentData, parentFfi, i) == snapshot[i],
            ('parent data changed at %d: %.17g vs %.17g')
            :format(i, H.rawAt(parentData, parentFfi, i), snapshot[i]))
    end

    -- Ребёнок отличается от родителя хотя бы в одной позиции.
    local changed = false
    for i = 1, #snapshot do
        if H.weightAt(childSlot, i) ~= snapshot[i] then changed = true break end
    end
    assert(changed, 'mutateWeights returned a genome identical to the parent')
end

-- ---------------------------------------------------------------------------
-- 5. Мутация — строгая граница отклонений (>= 200 итераций, фиксированный seed).
-- Позиции выбираются БЕЗ возврата (k = rand(1, w_len) различных позиций),
-- поэтому каждая позиция отклоняется ровно на один шаг
-- (rand()-0.5)*2*strength ∈ (-strength, strength): |new[i]-parent[i]| < strength.
-- ---------------------------------------------------------------------------
tests['mutateWeights deviation bounded'] = function()
    math.randomseed(42)
    local parentSlot  = ai_module.genWeights()
    local parentEntry = shares.CELL_GENOMES[parentSlot] -- запись слота целиком (обёртка или raw)
    local snapshot    = H.dataCopy(parentSlot)
    local strength    = ai_module.GENOME_MUTATION_STRENGHT

    local iterations = 200
    local n = #snapshot
    local maxDev = 0
    local sumDev, changedPos = 0, 0

    for iter = 1, iterations do
        local childSlot = ai_module.mutateWeights(parentSlot)
        local anyDiff = false
        for i = 1, n do
            local diff = math.abs(H.weightAt(childSlot, i) - snapshot[i])
            if diff > maxDev then maxDev = diff end
            if diff > 0 then
                anyDiff = true
                changedPos = changedPos + 1
                sumDev = sumDev + diff
            end
        end
        -- k = rand(1, w_len) >= 1 всегда: каждая итерация обязана изменить
        -- хотя бы одну позицию.
        assert(anyDiff, ('iteration %d produced no change'):format(iter))

        -- addGenome переиспользует слоты с counter == 0: возвращаем родителя в слот,
        -- чтобы каждая итерация мутировала именно исходный геном.
        shares.CELL_GENOMES[parentSlot] = parentEntry
    end

    -- Выбор позиций без возврата делает strength честной границей:
    -- ни одна позиция не накапливает более одного шага.
    assert(maxDev <= strength + 1e-9,
        ('max |dev| %.10g > strength + 1e-9 = %g'):format(maxDev, strength + 1e-9))

    -- Средний модуль отклонения по изменённым позициям ≈ E|шаг| = strength/2 = 0.05
    -- (для uniform(-strength, strength)); ловит регрессии масштаба шага (scale=2*strength).
    local meanDev = sumDev / changedPos
    assert(meanDev >= 0.04 and meanDev <= 0.075,
        ('mean |dev| %.10g вне ожидаемого диапазона [0.04, 0.075]'):format(meanDev))
end

-- ---------------------------------------------------------------------------
-- 6. run() против эталонной реализации (3 сети, 100 прогонов каждая).
-- ---------------------------------------------------------------------------
tests['run matches reference implementation'] = function()
    math.randomseed(42)

    local slot   = ai_module.genWeights() -- случайный геном (все три сегмента)
    -- Эталонная реализация читает 1-based таблицу: dataCopy работает для обоих
    -- бэкендов (ffi-массив переводится в таблицу с сохранением значений).
    local genome = H.dataCopy(slot)

    local networks = {
        {shares.AI_LAYERS_SEED,   shares.AI_OFFSET_SEED},
        {shares.AI_LAYERS_SPORE,  shares.AI_OFFSET_SPORE},
        {shares.AI_LAYERS_SPROUT, shares.AI_OFFSET_SPROUT},
    }

    for _, net in ipairs(networks) do
        local layers, offset = net[1], net[2]
        local in_size = layers[1]
        for trial = 1, 100 do
            local inputs = {}
            for i = 1, in_size do inputs[i] = math.random() * 2 - 1 end

            local expected = referenceRun(genome, layers, offset, inputs)
            local actual   = ai_module.run(shares.CELL_GENOMES[slot], layers, offset, inputs)

            H.assertClose(actual, expected, 1e-9,
                ('run() vs reference, layers {%s}, trial %d'):format(table.concat(layers, ','), trial))
        end
    end
end

-- ---------------------------------------------------------------------------
-- 7. Synthetic genome — точные значения.
-- Раскладка весов: на каждый НЕ-выходной узел (bias, threshold, dead, *веса в
-- следующий слой), на выходной узел (bias, threshold, dead).
-- Для layers={1,1} это ровно 7 весов и формула из ТЗ:
--   value = input + w1; if value <= w2 then value = w3
--   out   = value*w4 + w5; if out <= w6 then out = w7
-- Для layers={1,1,1} countWeights даёт 11 весов (4+4+3) — средний узел добавляет
-- (bias, threshold, dead, вес), поэтому формула превращается в цепочку из двух
-- активаций; это проверяется отдельно.
-- ---------------------------------------------------------------------------
tests['synthetic genome exact values'] = function()
    math.randomseed(42)

    -- Прямая запись в shares.CELL_GENOMES (addGenome в ai_module локальная;
    -- слоты 800001+ недостижимы для неё, т.к. она сканирует слоты с начала).
    local SLA, SLB = 800001, 800002

    -- Часть A: layers={1,1}, 7 весов:
    --   w1=bias, w2=threshold, w3=dead, w4=out-weight,
    --   w5=out-bias, w6=out-threshold, w7=out-dead.
    local gA = {}
    for i = 1, shares.AI_LEN_COMMON do gA[i] = 0 end
    gA[1] =  0.5    -- bias
    gA[2] =  1.0    -- threshold
    gA[3] = -0.25   -- dead
    gA[4] = -2.0    -- out-weight
    gA[5] =  0.0    -- out-bias
    gA[6] =  0.0    -- out-threshold
    gA[7] =  0.75   -- out-dead
    shares.CELL_GENOMES[SLA] = gA

    local function expectRun(slot, layers, input, want, what)
        local got = ai_module.run(shares.CELL_GENOMES[slot], layers, 0, {input})
        assert(#got == 1 and math.abs(got[1] - want) <= 1e-9,
            ('%s: input=%g: got %g, want %g'):format(what, input, got[1], want))
    end

    -- (а) value проходит (2+0.5=2.5 > 1), out умирает (2.5*(-2)=-5 <= 0) -> out-dead
    expectRun(SLA, {1, 1}, 2.0, 0.75, 'layers={1,1}')
    -- (б) value умирает (-2+0.5=-1.5 <= 1) -> dead=-0.25; out=0.5 > 0 -> проходит
    expectRun(SLA, {1, 1}, -2.0, 0.5, 'layers={1,1}')
    -- граница: value == threshold (1.0 <= 1.0) -> dead-constant
    expectRun(SLA, {1, 1}, 0.5, 0.5, 'layers={1,1}')

    -- Часть B: layers={1,1,1}, 11 весов (цепочка из двух активаций):
    --   v1 = input + w1;  if v1 <= w2 then v1 = w3
    --   v2 = v1*w4 + w5;  if v2 <= w6 then v2 = w7
    --   out = v2*w8 + w9; if out <= w10 then out = w11
    local gB = {}
    for i = 1, shares.AI_LEN_COMMON do gB[i] = 0 end
    gB[1] = 0.5;  gB[2] = 1.0;  gB[3] = -0.25; gB[4] = 2.0
    gB[5] = -1.0; gB[6] = 0.5;  gB[7] = -0.5;  gB[8] = 1.0
    gB[9] = 0.25; gB[10] = 2.0; gB[11] = 0.0
    shares.CELL_GENOMES[SLB] = gB

    -- вход 1: v1=1.5>1 (проходит), v2=1.5*2-1=2.0>0.5 (проходит),
    --         out=2.0+0.25=2.25>2 (проходит) -> 2.25
    expectRun(SLB, {1, 1, 1}, 1.0, 2.25, 'layers={1,1,1}')
    -- вход 0: v1=0.5<=1 (dead=-0.25), v2=-0.25*2-1=-1.5<=0.5 (dead=-0.5),
    --         out=-0.5+0.25=-0.25<=2 (dead=0) -> 0
    expectRun(SLB, {1, 1, 1}, 0.0, 0.0, 'layers={1,1,1}')
end

-- ---------------------------------------------------------------------------
-- 8. AI_OFFSET_*: сегменты не перекрываются; run с SEED-оффсетом даёт ожидаемый
-- результат, а SPORE/SPROUT не зависят от seed-сегмента (эквивалент нулевого входа).
-- ---------------------------------------------------------------------------
tests['AI_OFFSET_* segment isolation'] = function()
    math.randomseed(42)

    -- Неперекрытие сегментов.
    assert(shares.AI_OFFSET_SPORE == shares.AI_LEN_SEED,
        'AI_OFFSET_SPORE != AI_LEN_SEED')
    assert(shares.AI_OFFSET_SPROUT == shares.AI_LEN_SEED + shares.AI_LEN_SPORE,
        'AI_OFFSET_SPROUT != AI_LEN_SEED + AI_LEN_SPORE')
    assert(shares.AI_OFFSET_SPROUT + shares.AI_LEN_SPROUT == shares.AI_LEN_COMMON,
        'segments do not tile the whole genome')

    -- G: заполнен ТОЛЬКО seed-сегмент (известные ненулевые значения), остальное — нули.
    local G = {}
    for i = 1, shares.AI_LEN_COMMON do
        if i <= shares.AI_LEN_SEED then
            G[i] = ((i % 3) - 1.5) * 0.5 -- {-0.75, -0.25, 0.25}, всегда ненулевые
        else
            G[i] = 0
        end
    end
    local Gz = {}
    for i = 1, shares.AI_LEN_COMMON do Gz[i] = 0 end

    local SG, SGZ = 900002, 900003 -- прямые слоты (addGenome до них не дойдёт)
    shares.CELL_GENOMES[SG]  = G
    shares.CELL_GENOMES[SGZ] = Gz

    -- SEED-run даёт ожидаемый результат (сверка с эталоном).
    local in9 = {}
    for i = 1, 9 do in9[i] = math.random() * 2 - 1 end
    local expectedSeed = referenceRun(G, shares.AI_LAYERS_SEED, shares.AI_OFFSET_SEED, in9)
    local actualSeed   = ai_module.run(shares.CELL_GENOMES[SG], shares.AI_LAYERS_SEED, shares.AI_OFFSET_SEED, in9)
    H.assertClose(actualSeed, expectedSeed, 1e-9, 'seed run')

    -- SPORE/SPROUT: не зависят от seed-сегмента и эквивалентны run с нулевым входом.
    local cases = {
        {shares.AI_LAYERS_SPORE,  shares.AI_OFFSET_SPORE,  6},
        {shares.AI_LAYERS_SPROUT, shares.AI_OFFSET_SPROUT, 9},
    }
    for _, c in ipairs(cases) do
        local layers, offset, in_size = c[1], c[2], c[3]
        local inputs, zeros = {}, {}
        for i = 1, in_size do
            inputs[i] = math.random() * 2 - 1
            zeros[i]  = 0
        end

        local withSeed  = ai_module.run(shares.CELL_GENOMES[SG],  layers, offset, inputs)
        local zeroSeg   = ai_module.run(shares.CELL_GENOMES[SGZ], layers, offset, inputs)
        local zeroInput = ai_module.run(shares.CELL_GENOMES[SG],  layers, offset, zeros)

        H.assertClose(withSeed, zeroSeg, 1e-9, 'independence from seed segment')
        H.assertClose(withSeed, zeroInput, 1e-9, 'equivalent to zero input')

        -- обнулённый сегмент даёт ровно нулевые выходы
        for i = 1, #withSeed do
            assert(withSeed[i] == 0, ('zeroed segment output[%d] = %g, expected 0'):format(i, withSeed[i]))
        end
    end
end

-- ---------------------------------------------------------------------------
-- 9. Вспомогательное: детерминизм при фиксированном seed.
-- ---------------------------------------------------------------------------
tests['determinism with fixed randomseed'] = function()
    math.randomseed(42)
    local s1 = ai_module.genWeights()
    local d1 = H.dataCopy(s1)

    math.randomseed(42)
    local s2 = ai_module.genWeights()
    for i = 1, #d1 do
        assert(H.weightAt(s2, i) == d1[i],
            ('weight[%d] differs after re-seed: %g vs %g'):format(i, H.weightAt(s2, i), d1[i]))
    end
end

return tests
