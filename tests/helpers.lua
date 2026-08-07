-- helpers.lua — адаптер доступа к данным геномов для тестов.
-- Изолирует внутреннее представление весов: сейчас это сырая таблица,
-- после рефакторинга ai_module — обёртка {data=..., counter=..., is_ffi=...}.
-- Тесты должны читать веса ТОЛЬКО через этот адаптер.

local shares = require('shares')

local H = {}

-- Данные весов слота: обёртка {data} или сырая таблица.
-- Возвращает (данные, is_ffi).
function H.genomeData(slot)
    local g = shares.CELL_GENOMES[slot]
    return (g and g.data) or g, (g and g.is_ffi) or false
end

-- Вес №i (1-based логически; ffi-массивы 0-based).
function H.weightAt(slot, i)
    local d, ffi = H.genomeData(slot)
    if ffi then return d[i - 1] end
    return d[i]
end

-- Чтение элемента по прямой ссылке на данные (данные, is_ffi), индекс 1-based.
-- Нужен, когда слот уже переиспользован addGenome и данные доступны только
-- через ранее захваченную ссылку.
function H.rawAt(data, isFfi, i)
    if isFfi then return data[i - 1] end
    return data[i]
end

-- Длина данных слота.
function H.dataLen(slot)
    local d = H.genomeData(slot)
    if type(d) == 'cdata' then return shares.AI_LEN_COMMON end
    return #d
end

-- Независимая копия данных слота в обычную таблицу (для сравнения "до/после").
function H.dataCopy(slot)
    local d, ffi = H.genomeData(slot)
    local out = {}
    for i = 1, H.dataLen(slot) do
        if ffi then out[i] = d[i - 1] else out[i] = d[i] end
    end
    return out
end

-- Поэлементное сравнение двух массивов с абсолютным допуском eps.
function H.assertClose(a, b, eps, what)
    eps = eps or 1e-9
    what = what or 'values'
    local n = math.max(#a, #b)
    for i = 1, n do
        local av, bv = a[i] or 0, b[i] or 0
        assert(math.abs(av - bv) <= eps,
            ('%s differ at index %d: %.17g vs %.17g'):format(what, i, av, bv))
    end
end

-- Точное поэлементное сравнение.
function H.assertEqual(a, b, what)
    what = what or 'values'
    local n = math.max(#a, #b)
    for i = 1, n do
        local av, bv = a[i] or 0, b[i] or 0
        assert(av == bv, ('%s differ at index %d: %.17g vs %.17g'):format(what, i, av, bv))
    end
end

return H
