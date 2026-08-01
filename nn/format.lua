-- nn/format.lua — topology definitions + node-major genome decomposition.
--
-- The core is standalone (no shares.lua / ai_module.lua requires), so the REAL
-- game topology is mirrored here as data. The values are pinned by
-- tests/test_nn_core.lua against shares.lua directly (AI_LAYERS_*,
-- AI_LEN_*), so a drift in either place fails loudly.
--
-- Genome layout (node-major stream), identical to ai_module's run() reader:
--   for each non-final node:  [bias, threshold, dead_value, weights_to_next_layer[]]
--   for each final node:      [bias, threshold, dead_value]
-- Decomposition splits one network's stream into:
--   (a) one layer-major weight matrix per layer; matrix_index =
--       input_node * output_count + output_node (input-major rows);
--   (b) one interleaved per-node special tensor in node order:
--       [bias1, threshold1, dead1, bias2, ...] (GPU specials layout).
-- Specials stay production fp32; only weight matrices are block-quantized.

local M = {}

-- REAL topology (mirror of shares.lua AI_LAYERS_*).
M.NETWORKS = {
    seed   = { layers = { 9, 12, 1 } },
    spore  = { layers = { 6, 16, 1 } },
    sprout = { layers = { 9, 18, 16, 3 } },
}

-- Concatenation order of the common genome (matches shares.lua offsets).
M.NETWORK_ORDER = { 'seed', 'spore', 'sprout' }

-- Number of node-major stream entries for a layer list.
local function count_weights(layers)
    local n = 0
    for i = 1, #layers - 1 do
        n = n + layers[i] * (3 + layers[i + 1])
    end
    return n + layers[#layers] * 3
end
M.count_weights = count_weights

-- Per-network lengths + offsets inside the common genome.
local common_len, offset = 0, 0
for _, id in ipairs(M.NETWORK_ORDER) do
    local net = M.NETWORKS[id]
    net.len = count_weights(net.layers)
    net.offset = offset
    common_len = common_len + net.len
    offset = offset + net.len
end
M.COMMON_LEN = common_len

-- Topology identity string; hashed into the corpus identity. Changing any
-- layer shape changes this string.
local id_parts = {}
for _, id in ipairs(M.NETWORK_ORDER) do
    id_parts[#id_parts + 1] = id .. '=' .. table.concat(M.NETWORKS[id].layers, '-')
end
M.TOPOLOGY_IDENTITY = 'nn-v1:' .. table.concat(id_parts, ':')

-- Number of matrix weights (sum of input_count*output_count over layers).
function M.matrix_count(layers)
    local n = 0
    for i = 1, #layers - 1 do
        n = n + layers[i] * layers[i + 1]
    end
    return n
end

-- Number of specials (3 per node: bias, threshold, dead_value).
function M.special_count(layers)
    local n = 0
    for i = 1, #layers do
        n = n + layers[i]
    end
    return n * 3
end

-- Total nodes across a layer list.
function M.node_count(layers)
    local n = 0
    for i = 1, #layers do
        n = n + layers[i]
    end
    return n
end

-- Extract one network's node-major slice from the common genome stream.
-- `stream` is a 1-based Lua array of numbers (AI_LEN_COMMON entries).
function M.extract(stream, network_id)
    local net = M.NETWORKS[network_id]
    if not net then
        return nil, ('unknown network %q'):format(tostring(network_id))
    end
    local w = {}
    for i = 1, net.len do
        w[i] = stream[net.offset + i]
    end
    return w
end

-- decompose: node-major stream -> { matrices, specials }.
-- matrices[li] = { input_count, output_count, values } (input-major flat).
-- specials = { bias1, threshold1, dead1, ... } in node order (interleaved).
function M.decompose(layers, stream)
    local matrices, specials, idx = {}, {}, 1
    for li = 1, #layers - 1 do
        local in_c, out_c = layers[li], layers[li + 1]
        local values = {}
        for j = 1, in_c do
            specials[#specials + 1] = stream[idx]
            specials[#specials + 1] = stream[idx + 1]
            specials[#specials + 1] = stream[idx + 2]
            idx = idx + 3
            for k = 1, out_c do
                values[#values + 1] = stream[idx]
                idx = idx + 1
            end
        end
        matrices[li] = { input_count = in_c, output_count = out_c, values = values }
    end
    for j = 1, layers[#layers] do
        specials[#specials + 1] = stream[idx]
        specials[#specials + 1] = stream[idx + 1]
        specials[#specials + 1] = stream[idx + 2]
        idx = idx + 3
    end
    if idx - 1 ~= #stream then
        return nil, ('stream length %d does not match topology (%d)'):format(
            #stream, idx - 1)
    end
    return { matrices = matrices, specials = specials }
end

-- reconstruct: { matrices, specials } -> node-major stream (exact inverse of
-- decompose; every special maps once, every matrix weight maps once).
function M.reconstruct(layers, decomposed)
    local matrices, specials = decomposed.matrices, decomposed.specials
    local stream, idx, si = {}, 1, 0
    for li = 1, #layers - 1 do
        local m = matrices[li]
        for j = 1, m.input_count do
            si = si + 1; stream[idx] = specials[si]; idx = idx + 1
            si = si + 1; stream[idx] = specials[si]; idx = idx + 1
            si = si + 1; stream[idx] = specials[si]; idx = idx + 1
            local base = (j - 1) * m.output_count
            for k = 1, m.output_count do
                stream[idx] = m.values[base + k]
                idx = idx + 1
            end
        end
    end
    for j = 1, layers[#layers] do
        si = si + 1; stream[idx] = specials[si]; idx = idx + 1
        si = si + 1; stream[idx] = specials[si]; idx = idx + 1
        si = si + 1; stream[idx] = specials[si]; idx = idx + 1
    end
    return stream
end

return M
