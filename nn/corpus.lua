-- nn/corpus.lua — recorded evaluation corpus + precision gates.
--
-- A corpus is recorded ONCE (with a reference fp32 model) and replayed
-- byte-for-byte later; it is never regenerated from a per-runtime seeded PRNG.
-- record() freezes inputs/reference-outputs/actions/boundary flags and hashes
-- the manifest (epsilon + topology identity are part of the identity).
--
-- Epsilon boundary bucket (N5): a node is "boundary" iff |pre - threshold| <=
-- eps, with eps precision-specific (fp32: 2^-20*max(1,|t|), fp16:
-- 2^-10*max(1,|t|)). Boundary nodes are reported separately and EXCLUDED from
-- the primary gate; they cannot hide catastrophic non-boundary errors.
--
-- Gates (action agreement on ordinary buckets): fp16 >= 99%, fp8 >= 97%,
-- fp4 >= 90%, fp2 >= 80%, plus bounded-error limits.

local errors = require('nn.errors')
local format = require('nn.format')

local corpus = {}

corpus.GATE_THRESHOLDS = { fp16 = 0.99, fp8 = 0.97, fp4 = 0.90, fp2 = 0.80 }
corpus.GATE_ORDER = { 'fp16', 'fp8', 'fp4', 'fp2' }

-- Epsilon for a precision given the threshold value t.
function corpus.epsilon_for(precision, t)
    t = t or 0
    if precision == 'fp16' then
        return 2 ^ -10 * math.max(1, math.abs(t))
    end
    return 2 ^ -20 * math.max(1, math.abs(t)) -- fp32 and coarser
end

-- Deterministic string hash (arithmetic-only; identical under lua + luajit).
local function hash_string(s)
    local h = 5381
    for i = 1, #s do
        h = (h * 33 + s:byte(i)) % 4294967296
    end
    return h
end

local function num_key(x)
    if x ~= x then return 'nan' end
    if x == math.huge then return 'inf' end
    if x == -math.huge then return '-inf' end
    return ('%.17g'):format(x)
end

-- Canonical byte form of the corpus data (hash input; deterministic).
function corpus.serialize_data(data)
    local parts = { 'NN-CORPUS-1', data.epsilon ~= nil and num_key(data.epsilon) or 'nil',
        data.topology_identity or '' }
    for _, item in ipairs(data.items) do
        parts[#parts + 1] = item.network_id or '?'
        for _, v in ipairs(item.inputs or {}) do parts[#parts + 1] = num_key(v) end
        parts[#parts + 1] = '|'
        for _, v in ipairs(item.outputs or {}) do parts[#parts + 1] = num_key(v) end
        parts[#parts + 1] = '|'
        for _, d in ipairs(item.dead or {}) do parts[#parts + 1] = d and '1' or '0' end
        parts[#parts + 1] = '|'
        for _, b in ipairs(item.boundary or {}) do parts[#parts + 1] = b and '1' or '0' end
        parts[#parts + 1] = ';'
    end
    return table.concat(parts)
end

function corpus.hash(data)
    return string.format('%08x', hash_string(corpus.serialize_data(data)))
end

-- record(inputs, reference_actions, outputs, network_ids, topology_identity,
--        epsilon) -> frozen { data, hash }
--   inputs[i]            = input vector
--   reference_actions[i] = { dead = {bool per output}, boundary = {bool per output} }
--   outputs[i]           = reference output vector
--   network_ids[i]       = network id
--   topology_identity    = identity string (hashed into the manifest)
--   epsilon              = boundary epsilon (stored + hashed)
function corpus.record(inputs, reference_actions, outputs, network_ids, topology_identity, epsilon)
    local items = {}
    for i = 1, #inputs do
        items[i] = {
            network_id = network_ids[i],
            inputs = inputs[i],
            outputs = outputs[i],
            dead = reference_actions[i].dead,
            boundary = reference_actions[i].boundary,
        }
    end
    local data = {
        version = 1,
        topology_identity = topology_identity,
        epsilon = epsilon,
        manifest = {
            version = 1,
            topology_identity = topology_identity,
            epsilon = epsilon,
            item_count = #items,
        },
        items = items,
    }
    return { data = data, hash = corpus.hash(data) }
end

local function sign_of(x)
    if x ~= x then return 0 end
    if x > 0 then return 1 end
    if x < 0 then return -1 end
    return 0
end

-- Evaluate one candidate precision against the recorded reference.
-- Returns a per-precision result table (see replay for the shape).
local function evaluate_precision(c, runtime, p)
    local data = c.data
    local res = {
        precision = p,
        threshold = corpus.GATE_THRESHOLDS[p],
        total = 0,
        agree = 0,
        dead_mismatch = 0,
        boundary_total = 0,
        boundary_mismatch = 0,
        boundary_max_error = 0,
        max_error = 0,
        mean_error = 0,
        err_sum = 0,
        err_count = 0,
        catastrophic = false,
        per_network = {},
        per_output = {},
    }
    local threshold = res.threshold
    local block_size = runtime.block_size
    local function per_net(net_id)
        local pn = res.per_network[net_id]
        if not pn then
            pn = { total = 0, agree = 0, max_error = 0, err_sum = 0, err_count = 0 }
            res.per_network[net_id] = pn
        end
        return pn
    end
    local function per_out(net_id, o)
        local po = res.per_output[net_id]
        if not po then po = {}; res.per_output[net_id] = po end
        if not po[o] then
            po[o] = { total = 0, agree = 0, max_error = 0 }
        end
        return po[o]
    end

    for _, item in ipairs(data.items) do
        local weights = runtime:model_weights(item.network_id)
        if not weights then
            return nil, errors.new('INVALID_ARGUMENT',
                'runtime has no model weights for corpus network ' .. tostring(item.network_id),
                runtime.backend)
        end
        local packed, err = runtime:_pack(item.network_id, weights, p, block_size)
        if not packed then return nil, err end
        local out = {}
        for o = 1, #item.outputs do out[o] = 0 end
        local _, _, dead_flags = runtime:_run_packed(item.network_id, packed, item.inputs, out, true)
        local item_ok = true
        for o, ref_out in ipairs(item.outputs) do
            local ref_dead = item.dead[o]
            local boundary = item.boundary[o]
            local rt_out = out[o]
            local rt_dead = dead_flags[o]
            local e = math.abs(rt_out - ref_out)
            if boundary then
                res.boundary_total = res.boundary_total + 1
                if rt_dead ~= ref_dead then res.boundary_mismatch = res.boundary_mismatch + 1 end
                if e > res.boundary_max_error then res.boundary_max_error = e end
            else
                res.total = res.total + 1
                res.err_sum = res.err_sum + e
                res.err_count = res.err_count + 1
                if e > res.max_error then res.max_error = e end
                local pn = per_net(item.network_id)
                pn.total = pn.total + 1
                pn.err_sum = pn.err_sum + e
                pn.err_count = pn.err_count + 1
                if e > pn.max_error then pn.max_error = e end
                local po = per_out(item.network_id, o)
                po.total = po.total + 1
                if e > po.max_error then po.max_error = e end
                -- Action agreement: dead flags match, and when both live the
                -- signs must match (morph/direction decisions).
                local agree = rt_dead == ref_dead
                if agree and not rt_dead then
                    agree = sign_of(rt_out) == sign_of(ref_out)
                end
                if agree then
                    res.agree = res.agree + 1
                    pn.agree = pn.agree + 1
                    po.agree = po.agree + 1
                else
                    item_ok = false
                    if rt_dead ~= ref_dead then res.dead_mismatch = res.dead_mismatch + 1 end
                end
                -- Bounded-error limit: catastrophic = > 10x the reference
                -- magnitude (catches catastrophic non-boundary errors).
                if e > 10 * math.max(1, math.abs(ref_out)) then
                    res.catastrophic = true
                end
            end
        end
        if not item_ok then res.item_failures = (res.item_failures or 0) + 1 end
    end

    if res.err_count > 0 then res.mean_error = res.err_sum / res.err_count end
    local ratio = res.total > 0 and (res.agree / res.total) or 0
    res.agreement = ratio
    res.pass = ratio >= threshold and not res.catastrophic
    return res
end

-- replay(corpus_obj, runtime) -> result | nil, err
-- Evaluates all gate precisions (fp16/fp8/fp4/fp2) against the recorded
-- reference, without mutating the runtime's active precision. Stores the
-- result on runtime._gate_results (consulted by set_precision).
function corpus.replay(c, runtime)
    if not c or not c.data then
        return nil, errors.new('INVALID_ARGUMENT', 'corpus is not a recorded corpus', runtime.backend)
    end
    if c.data.topology_identity ~= runtime.topology_identity then
        return nil, errors.new('INVALID_TOPOLOGY',
            'corpus topology identity does not match the runtime', runtime.backend)
    end
    local result = {
        topology_matches = true,
        epsilon = c.data.epsilon,
        gates = {},
        counts = {},
        per_network = {},
        per_output = {},
        boundary = { total = 0, mismatched = 0, max_error = 0 },
        item_count = #c.data.items,
        hash = c.hash,
    }
    for _, p in ipairs(corpus.GATE_ORDER) do
        local r, err = evaluate_precision(c, runtime, p)
        if not r then return nil, err end
        result.gates[p] = r.pass
        result.counts[p] = r
        result.per_network[p] = r.per_network
        result.per_output[p] = r.per_output
        result.boundary.total = result.boundary.total + r.boundary_total
        result.boundary.mismatched = result.boundary.mismatched + r.boundary_mismatch
        if r.boundary_max_error > result.boundary.max_error then
            result.boundary.max_error = r.boundary_max_error
        end
    end
    result.pass = result.gates.fp16 and result.gates.fp8 and result.gates.fp4 and result.gates.fp2
    runtime._gate_results = result
    return result
end

return corpus
