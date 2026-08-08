-- nn/runtime.lua — runtime wiring: backend resolution, model handling,
-- forward/forward_into/forward_batch dispatch, set_precision gating,
-- capabilities and shutdown.
--
-- Backend resolution happens ONCE at init (never "auto" afterwards):
--   deterministic=true  + backend="gpu"  -> init error
--   deterministic=true  + backend="auto" -> "cpu"
--   deterministic=false + backend="auto" -> "gpu" if Vulkan init succeeds
--                                          (pcall nn.vulkan) else "cpu"
--   backend="gpu" + Vulkan unsupported   -> error
-- Post-init DEVICE_LOST is a hard structured error; there is NEVER a silent
-- fallback to CPU after init. "reference" is a validation-only mode, never
-- chosen by auto.
--
-- Precision handling (N6): fp16/fp8 switches require the corpus gate to have
-- passed for that precision; fp4/fp2 switches require options.experimental ==
-- true (a passing corpus gate alone is NOT sufficient — selection is
-- experimental-only). fp32 is always allowed. The default profile is
-- all-fp8/block16/fp32-scales/fp32-specials.

local errors = require('nn.errors')
local format = require('nn.format')
local quantize = require('nn.quantize')
local capabilities = require('nn.capabilities')
local reference = require('nn.reference')
local cpu = require('nn.cpu')
local corpus_mod = require('nn.corpus')
local serialize = require('nn.serialize')

local Runtime = {}
Runtime.__index = Runtime

local PRECISIONS = { fp32 = true, fp16 = true, fp8 = true, fp4 = true, fp2 = true }
local PACKED_FORMATS = { fp8 = true, fp4 = true, fp2 = true }
local GATE_THRESHOLDS = corpus_mod.GATE_THRESHOLDS

-- ------------------------------------------------------------- FNV-ish hash --
-- Deterministic string hash (arithmetic-only: LuaJIT bitwise ops are signed
-- int32, so plain arithmetic keeps both interpreters byte-identical).
-- Shared with nn.corpus.
Runtime.hash_string = corpus_mod.hash_string

-- ------------------------------------------------------------ weight blobs --

-- Decode any weights blob into a per-network node-major 1-based Lua table.
--   table  -> numeric array (common genome OR per-network length)
--   cdata  -> LuaJIT double array (0-based, common genome)
--   string -> .nnw serialization (NNW\0 magic) OR packed LE fp64 doubles
-- Returns weights or nil, err.
function Runtime:_decode_weights(weights, blob_type, network_id)
    local net = self.networks[network_id]
    local len_common = format.COMMON_LEN
    local len_net = net.len
    -- weights_desc wrapper: { blob = weights, blob_type = ..., layout = ... }.
    if type(weights) == 'table' and weights.blob ~= nil
        and (weights.blob_type ~= nil or weights.layout ~= nil) then
        blob_type = blob_type or weights.blob_type
        weights = weights.blob
    end
    local t = blob_type or (type(weights) == 'cdata' and 'cdata'
        or type(weights) == 'string' and 'string' or 'table')

    if t == 'table' then
        if #weights == len_common then
            local w = {}
            for i = 1, len_net do w[i] = weights[net.offset + i] end
            return w
        elseif #weights == len_net then
            local w = {}
            for i = 1, len_net do w[i] = weights[i] end
            return w
        end
        return nil, errors.new('INVALID_WEIGHTS',
            ('weights table length %d does not match network %q (%d) or common (%d)')
            :format(#weights, network_id, len_net, len_common), self.backend)
    elseif t == 'cdata' then
        local w = {}
        for i = 1, len_net do
            w[i] = weights[net.offset + i - 1]
        end
        return w
    elseif t == 'string' then
        if weights:sub(1, 4) == serialize.MAGIC then
            local model, err = serialize.read(weights)
            if not model then
                return nil, errors.new('INVALID_SERIALIZATION',
                    ('weights string is not a valid .nnw: %s'):format(err.message),
                    self.backend)
            end
            local w = model.networks[network_id] and model.networks[network_id].weights
            if not w then
                return nil, errors.new('INVALID_SERIALIZATION',
                    ('.nnw has no weights for network %q'):format(network_id), self.backend)
            end
            local out = {}
            for i = 1, #w do out[i] = w[i] end
            return out
        end
        local n = #weights
        if n % 8 ~= 0 then
            return nil, errors.new('INVALID_WEIGHTS',
                'string weights length must be a multiple of 8 (fp64 LE)', self.backend)
        end
        local count = n / 8
        local w = {}
        if count == len_common then
            for i = 1, len_net do
                w[i] = quantize.f64_from_bytes(weights, (net.offset + i - 1) * 8 + 1)
            end
        elseif count == len_net then
            for i = 1, len_net do
                w[i] = quantize.f64_from_bytes(weights, (i - 1) * 8 + 1)
            end
        else
            return nil, errors.new('INVALID_WEIGHTS',
                ('packed string length %d does not match network %q (%d) or common (%d)')
                :format(count, network_id, len_net, len_common), self.backend)
        end
        return w
    end
    return nil, errors.new('INVALID_WEIGHTS', ('unknown blob type %q'):format(t), self.backend)
end

-- Genome identity for the pack cache: object identity (tables/strings/cdata
-- are identity-keyed in Lua).
local function genome_key(weights, profile_key)
    return tostring(weights) .. '|' .. profile_key
end

-- ---------------------------------------------------------------- packing --

-- Validate finiteness of a decoded weight stream (reject NaN/Inf).
local function check_finite(w, backend)
    for i = 1, #w do
        local v = w[i]
        if v ~= v or v == math.huge or v == -math.huge then
            return nil, errors.new('INVALID_WEIGHTS',
                ('non-finite weight at index %d'):format(i), backend)
        end
    end
    return true
end

-- Quantize one layer matrix into { payload, scales, offsets, logical,
-- padded, blocks, max_code }. max_code: nil (fp8) / 15 (fp4) / 3 (fp2).
local function max_code_for(precision)
    if precision == 'fp8' then return nil end
    if precision == 'fp4' then return 15 end
    return 3
end

local function pack_matrix(values, block_size, max_code)
    local logical = #values
    local blocks = quantize.block_count(logical, block_size)
    local padded = blocks * block_size
    local payload_parts, scales, offsets = {}, {}, {}
    local codes
    for bi = 0, blocks - 1 do
        local block = {}
        for j = 1, block_size do
            local v = values[bi * block_size + j]
            block[j] = v or 0 -- padding excluded from statistics below
        end
        -- Only count logical values for statistics.
        local first = bi * block_size
        local stats_n = math.min(block_size, logical - first)
        local stats = {}
        for j = 1, stats_n do stats[j] = values[first + j] end
        local enc
        if max_code == nil then
            enc = quantize.encode_fp8_block(stats, block_size)
            codes = enc.codes
            scales[bi + 1] = quantize.round_f32(enc.scale)
            payload_parts[bi + 1] = quantize.pack_fp8(codes, block_size)
        else
            enc = quantize.encode_affine_block(stats, max_code)
            codes = enc.codes
            scales[bi + 1] = quantize.round_f32(enc.scale)
            offsets[bi + 1] = quantize.round_f32(enc.offset)
            if max_code == 15 then
                payload_parts[bi + 1] = quantize.pack_fp4(codes, block_size)
            else
                payload_parts[bi + 1] = quantize.pack_fp2(codes, block_size)
            end
        end
    end
    return {
        payload = table.concat(payload_parts),
        scales = scales,
        offsets = offsets,
        logical = logical,
        padded = padded,
        blocks = blocks,
        max_code = max_code,
        block_size = block_size,
    }
end

-- Build the packed representation of a network from its node-major stream.
-- precision fp32 -> exact stream; fp16 -> fp16-rounded matrices (no blocks);
-- fp8/fp4/fp2 -> block-packed payload + fp32 scale/offset tensors.
function Runtime:_pack(network_id, stream, precision, block_size)
    local net = self.networks[network_id]
    local ok, err = check_finite(stream, self.backend)
    if not ok then return nil, err end
    local dec = format.decompose(net.layers, stream)
    if not dec then
        return nil, errors.new('INVALID_WEIGHTS', 'stream does not match topology', self.backend)
    end
    local packed = {
        network_id = network_id,
        precision = precision,
        block_size = block_size,
        specials = dec.specials, -- interleaved [bias, threshold, dead] fp32
        layers = net.layers,
    }
    if precision == 'fp32' then
        packed.exact = stream
        return packed
    end
    if precision == 'fp16' then
        local m16 = {}
        for li = 1, #dec.matrices do
            local m = dec.matrices[li]
            local values = {}
            for i = 1, #m.values do
                values[i] = quantize.round_fp16(m.values[i])
            end
            m16[li] = { input_count = m.input_count, output_count = m.output_count, values = values }
        end
        packed.fp16_matrices = m16
        return packed
    end
    local max_code = max_code_for(precision)
    local per_layer = {}
    local all_payload = {}
    local all_scales, all_offsets = {}, {}
    for li = 1, #dec.matrices do
        local m = dec.matrices[li]
        local pm = pack_matrix(m.values, block_size, max_code)
        per_layer[li] = pm
        all_payload[#all_payload + 1] = pm.payload
        for _, s in ipairs(pm.scales) do all_scales[#all_scales + 1] = s end
        for _, o in ipairs(pm.offsets) do all_offsets[#all_offsets + 1] = o end
    end
    packed.payload = table.concat(all_payload)
    packed.scales = all_scales
    packed.offsets = all_offsets
    packed.per_layer = per_layer
    return packed
end

-- Decode a packed network's layer matrices back to fp64 values.
local function unpack_matrices(packed)
    local layers = packed.layers
    local matrices = {}
    if packed.exact then
        local dec = format.decompose(layers, packed.exact)
        return dec.matrices, dec.specials
    elseif packed.fp16_matrices then
        return packed.fp16_matrices, packed.specials
    end
    local offset_byte = 0
    for li = 1, #layers - 1 do
        local pm = packed.per_layer[li]
        local values = quantize.decode_matrix(packed.payload, offset_byte,
            pm.logical, pm.block_size, pm.scales, pm.offsets, pm.max_code)
        matrices[li] = {
            input_count = layers[li],
            output_count = layers[li + 1],
            values = values,
        }
        offset_byte = offset_byte + #pm.payload
    end
    return matrices, packed.specials
end

-- --------------------------------------------------------------- forward --

-- Exact forward over decomposed matrices + interleaved specials. Mirrors the
-- reference activation: value = prev + bias; if value <= threshold then dead.
local function forward_decomposed(layers, matrices, specials, inputs, out, debug)
    local len = #layers
    local data = reference._data
    local total = 0
    for i = 1, len do total = total + layers[i] end
    for i = 1, total do data[i] = nil end
    for i = 1, layers[1] do data[i] = inputs[i] or 0 end

    local offset = 0
    local si = 0 -- specials cursor (interleaved)
    for li = 1, len - 1 do
        local m = matrices[li]
        local layer = layers[li + 1]
        local next_offset = offset + layers[li]
        local w = m.values
        for j = 1, layers[li] do
            si = si + 1; local bias = specials[si]
            si = si + 1; local threshold = specials[si]
            si = si + 1; local dead = specials[si]
            local value = (data[j + offset] or 0) + bias
            if value <= threshold then value = dead end
            local base = (j - 1) * layer
            for k = 1, layer do
                data[next_offset + k] = (data[next_offset + k] or 0) + value * w[base + k]
            end
        end
        offset = next_offset
    end

    local out_count = layers[len]
    local pre, dead_flags
    if debug then pre, dead_flags = {}, {} end
    for o = 1, out_count do
        si = si + 1; local bias = specials[si]
        si = si + 1; local threshold = specials[si]
        si = si + 1; local dead = specials[si]
        local value = (data[offset + o] or 0) + bias
        if debug then pre[o] = value end
        if value <= threshold then
            if debug then dead_flags[o] = true end
            out[o] = dead
        else
            if debug then dead_flags[o] = false end
            out[o] = value
        end
    end
    if debug then return out, pre, dead_flags end
    return out
end

-- Run a packed network on inputs. Returns outputs (table). With debug=true
-- also returns pre-activation values and dead flags per output node.
function Runtime:_run_packed(network_id, packed, inputs, out, debug)
    local layers = self.networks[network_id].layers
    local matrices, specials = unpack_matrices(packed)
    return forward_decomposed(layers, matrices, specials, inputs, out, debug)
end

-- Profile key for the pack cache (mirrors the vulkan pipelines naming).
function Runtime:_profile_key(network_id, precision, block_size)
    return ('%s|%s|%d'):format(network_id, precision, block_size)
end

-- CPU forward core: decode + (cached) pack + run into `out`. No allocation of
-- a new output table; returns `out` (caller-owned).
function Runtime:_forward_cpu(network_id, weights, inputs, out, precision, use_cache)
    local stream, err = self:_decode_weights(weights, nil, network_id)
    if not stream then return nil, err end
    precision = precision or self.precision
    local block_size = self.block_size
    local key, packed
    if use_cache and not self._pack_cache then self._pack_cache = {} end
    if use_cache then
        key = genome_key(weights, self:_profile_key(network_id, precision, block_size))
        packed = self._pack_cache[key]
        if packed then
            self.metrics['pack cache hits'] = (self.metrics['pack cache hits'] or 0) + 1
        end
    end
    if not packed then
        packed, err = self:_pack(network_id, stream, precision, block_size)
        if not packed then return nil, err end
        if use_cache then
            self.metrics['pack cache misses'] = (self.metrics['pack cache misses'] or 0) + 1
            self._pack_cache[key] = packed
        end
    end
    local out_count = self.networks[network_id].layers[#self.networks[network_id].layers]
    for i = 1, out_count do out[i] = 0 end
    local ok = pcall(self._run_packed, self, network_id, packed, inputs, out, false)
    if not ok then
        return nil, errors.new('INVALID_ARGUMENT', 'forward execution failed', self.backend)
    end
    return out
end

-- Runtime:forward — convenience/debug path. On GPU this is a per-call
-- pack+dispatch with no reuse; on CPU it may reuse the pack cache.
function Runtime:forward(network_id, weights, inputs)
    if self._shutdown then
        return nil, errors.new('WORKER_SHUTDOWN', 'runtime is shut down', self.backend)
    end
    if not self.networks[network_id] then
        return nil, errors.new('INVALID_ARGUMENT',
            ('unknown network %q'):format(tostring(network_id)), self.backend)
    end
    self.metrics['forward calls'] = (self.metrics['forward calls'] or 0) + 1
    if self.backend == 'gpu' then
        local ok, err = self:_forward_gpu(network_id, weights, inputs, nil, self.precision)
        if not ok then return nil, err end
        return ok
    end
    local out = {}
    return self:_forward_cpu(network_id, weights, inputs, out, nil, false)
end

-- Runtime:forward_into — caller-owned output buffer, no allocation, no
-- retention. Returns `out`.
function Runtime:forward_into(network_id, weights, inputs, out)
    if self._shutdown then
        return nil, errors.new('WORKER_SHUTDOWN', 'runtime is shut down', self.backend)
    end
    if not self.networks[network_id] then
        return nil, errors.new('INVALID_ARGUMENT',
            ('unknown network %q'):format(tostring(network_id)), self.backend)
    end
    if type(out) ~= 'table' then
        return nil, errors.new('INVALID_ARGUMENT', 'out must be a Lua table', self.backend)
    end
    self.metrics['forward calls'] = (self.metrics['forward calls'] or 0) + 1
    if self.backend == 'gpu' then
        local res, err = self:_forward_gpu(network_id, weights, inputs, out, self.precision)
        if not res then return nil, err end
        return out
    end
    return self:_forward_cpu(network_id, weights, inputs, out, nil, false)
end

-- Runtime:forward_batch — production path. batch_items = array of
-- { genome = weights_blob_or_desc, network_id = id }. in_desc/out_desc:
-- { buffer, stride, count, element_type }. Writes outputs into out_desc.buffer
-- (caller-owned; never allocated here) and returns it.
function Runtime:forward_batch(network_id, batch_items, in_desc, out_desc)
    if self._shutdown then
        return nil, errors.new('WORKER_SHUTDOWN', 'runtime is shut down', self.backend)
    end
    if type(batch_items) ~= 'table' or #batch_items == 0 then
        return nil, errors.new('INVALID_ARGUMENT', 'batch_items must be a non-empty array', self.backend)
    end
    local net = self.networks[network_id]
    if not net then
        return nil, errors.new('INVALID_ARGUMENT',
            ('unknown network %q'):format(tostring(network_id)), self.backend)
    end
    local in_stride = (in_desc and in_desc.stride) or net.layers[1]
    local out_stride = (out_desc and out_desc.stride) or net.layers[#net.layers]
    local in_count = in_desc and in_desc.count or #batch_items
    if in_count ~= #batch_items then
        return nil, errors.new('INVALID_ARGUMENT',
            'in_desc.count must equal the number of batch items', self.backend)
    end
    if not out_desc or not out_desc.buffer then
        return nil, errors.new('INVALID_ARGUMENT', 'out_desc.buffer is required', self.backend)
    end
    if not in_desc or not in_desc.buffer then
        return nil, errors.new('INVALID_ARGUMENT', 'in_desc.buffer is required', self.backend)
    end
    local in_buf = in_desc.buffer
    local out_buf = out_desc.buffer
    self.metrics['forward batch calls'] = (self.metrics['forward batch calls'] or 0) + 1

    if self.backend == 'gpu' then
        local ok, err = self:_forward_batch_gpu(network_id, batch_items, in_desc, out_desc)
        if not ok then return nil, err end
        return out_buf
    end

    -- CPU: decode each item's own genome and run. Packed entries are cached
    -- by (genome identity, profile) — the production caching path.
    if not self._pack_cache then self._pack_cache = {} end
    local out_count = net.layers[#net.layers]
    local precision, block_size = self.precision, self.block_size
    for c, item in ipairs(batch_items) do
        local item_net = item.network_id or network_id
        local item_weights = item.genome or item.weights
        if type(item_weights) == 'table' and item_weights.blob ~= nil then
            item_weights = item_weights.blob -- weights_desc
        end
        local stream, err = self:_decode_weights(item_weights, nil, item_net)
        if not stream then return nil, err end
        local key = genome_key(item_weights, self:_profile_key(item_net, precision, block_size))
        local packed = self._pack_cache[key]
        if packed then
            self.metrics['pack cache hits'] = (self.metrics['pack cache hits'] or 0) + 1
        else
            self.metrics['pack cache misses'] = (self.metrics['pack cache misses'] or 0) + 1
            packed, err = self:_pack(item_net, stream, precision, block_size)
            if not packed then return nil, err end
            self._pack_cache[key] = packed
        end
        local inputs = {}
        local base = (c - 1) * in_stride
        for k = 1, in_stride do
            inputs[k] = in_buf[base + k] or 0
        end
        local out = {}
        for o = 1, out_count do out[o] = 0 end
        self:_run_packed(item_net, packed, inputs, out, false)
        local obase = (c - 1) * out_stride
        for o = 1, out_count do
            out_buf[obase + o] = out[o]
        end
    end
    return out_buf
end

-- ------------------------------------------------------------- precision --

-- Switch the active precision. Gates (N6): fp16/fp8 need the capability AND
-- a passed corpus gate; fp4/fp2 need options.experimental == true (a passed
-- gate alone is NOT sufficient); fp32 is always allowed.
function Runtime:set_precision(p, options)
    if self._shutdown then
        return nil, errors.new('WORKER_SHUTDOWN', 'runtime is shut down', self.backend)
    end
    options = options or {}
    if not PRECISIONS[p] then
        return nil, errors.new('INVALID_PRECISION',
            ('unknown precision %q'):format(tostring(p)), self.backend)
    end
    if p == self.precision then return true end
    if p == 'fp32' then
        if not self.caps.precisions.fp32 then
            return nil, errors.new('UNSUPPORTED_PRECISION', 'fp32 unsupported on this backend', self.backend)
        end
        self.precision = p
        self:_invalidate_cache()
        self.metrics['precision-switch cost'] = (self.metrics['precision-switch cost'] or 0) + 1
        return true
    end
    if not self.caps.precisions[p] then
        return nil, errors.new('UNSUPPORTED_PRECISION',
            ('%s unsupported on backend %q'):format(p, self.backend), self.backend)
    end
    local gated = self._gate_results and self._gate_results.gates
    if p == 'fp16' or p == 'fp8' then
        if not (gated and gated[p]) then
            return nil, errors.new('CORPUS_GATE_FAILED',
                ('precision %s requires a passing corpus gate (action agreement >= %g%%)')
                :format(p, (GATE_THRESHOLDS[p] or 1) * 100), self.backend)
        end
    else -- fp4 / fp2
        if not options.experimental then
            return nil, errors.new('CORPUS_GATE_FAILED',
                ('precision %s requires options.experimental=true (no corpus gate passed yet)')
                :format(p), self.backend)
        end
    end
    self.precision = p
    self:_invalidate_cache()
    self.metrics['precision-switch cost'] = (self.metrics['precision-switch cost'] or 0) + 1
    return true
end

function Runtime:_invalidate_cache()
    self._pack_cache = {}
end

-- ------------------------------------------------------------- capabilities --

function Runtime:capabilities()
    return {
        backend = self.backend,
        deterministic = self.deterministic,
        precisions = {
            fp32 = true,
            fp16 = self.caps.precisions.fp16,
            fp8 = self.caps.precisions.fp8,
            fp4 = self.caps.precisions.fp4,
            fp2 = self.caps.precisions.fp2,
        },
        native_fp16_arithmetic = self.caps.native_fp16_arithmetic or false,
        packed_storage = self.caps.packed_storage or true,
        max_batch_size = self.caps.max_batch_size,
        max_arena_bytes = self.caps.max_arena_bytes,
        block_sizes = { 8, 16, 32, 64 },
        supported_networks = { format.NETWORK_ORDER[1], format.NETWORK_ORDER[2], format.NETWORK_ORDER[3] },
    }
end

-- ---------------------------------------------------------------- shutdown --

function Runtime:shutdown()
    if self._shutdown then return end
    if self.backend == 'gpu' and self._worker then
        -- Worker teardown may be partially complete; best-effort.
        pcall(self._worker.shutdown, self._worker)
        self._worker = nil
    end
    self._shutdown = true
end

-- ---------------------------------------------------------------- gpu path --

local function get_vulkan()
    local ok, v = pcall(require, 'nn.vulkan')
    if not ok or type(v) ~= 'table' then return nil end
    return v
end

-- Probe: is a working GPU backend available?
local function vulkan_available()
    local v = get_vulkan()
    if not v or not v.can_load then return false end
    local ok = pcall(v.can_load)
    if not ok or not v.can_load() then return false end
    local ctx, err = v.init()
    if ctx then
        pcall(v.destroy, ctx)
        return true
    end
    return false
end

function Runtime:_ensure_worker()
    if self._worker then return self._worker end
    local worker_mod = require('nn.vulkan_worker')
    local w, err = worker_mod.new({})
    if not w then
        return nil, err
    end
    self._worker = w
    return w
end

-- Pack ONE layer's matrix for a GPU dispatch. The packed shader is a dense
-- single-layer model whose payload is stored INPUT-MAJOR: flat index
-- k*out_c + node == input_node*out_c + output_node (the format.decompose
-- matrix layout), so the shader's per-node accumulation matches the CPU path
-- weight-for-weight.
function Runtime:_pack_layer(network_id, stream, precision, block_size, li)
    local net = self.networks[network_id]
    local dec = format.decompose(net.layers, stream)
    if not dec then
        return nil, errors.new('INVALID_WEIGHTS', 'stream does not match topology', self.backend)
    end
    local m = dec.matrices[li]
    local out_c = net.layers[li + 1]
    -- This dispatch computes the layer (li+1) nodes; their specials start
    -- after all nodes of layers 1..li.
    local node_start = 0
    for i = 1, li do node_start = node_start + net.layers[i] end
    -- Specials of the computed (output) nodes, interleaved [bias, threshold,
    -- dead] per node — the shader reads special[spBase + 3*node + {0,1,2}].
    local specials = {}
    if li == 1 then
        -- Input-layer activation (GPU == CPU semantics): the reference
        -- activates layer-1 nodes before the fanout, so the FIRST dispatch's
        -- specials entry carries the layer-1 node specials FIRST. The shader
        -- reads them at spBase + 3*k for input node k when the
        -- input-activation flag (config word 5, bit 0) is set, and the
        -- computed nodes' specials at spBase + 3*fanIn + 3*node.
        for j = 1, net.layers[1] do
            local base = (j - 1) * 3
            specials[#specials + 1] = dec.specials[base + 1]
            specials[#specials + 1] = dec.specials[base + 2]
            specials[#specials + 1] = dec.specials[base + 3]
        end
    end
    for j = 1, out_c do
        local base = (node_start + j - 1) * 3
        specials[#specials + 1] = dec.specials[base + 1]
        specials[#specials + 1] = dec.specials[base + 2]
        specials[#specials + 1] = dec.specials[base + 3]
    end
    local max_code = max_code_for(precision)
    local pm = pack_matrix(m.values, block_size, max_code)
    local item = {
        payload = pm.payload,
        scales = pm.scales,
        offsets = pm.offsets,
        specials = specials,
    }
    return item, nil
end

-- GPU forward/forward_batch core: per-layer sequential dispatch (the packed
-- shader is a dense single-layer model; intermediate activations round-trip
-- through the worker readback). Each layer is ONE dispatch — the plan's "one
-- forward pass" is implemented as #layers-1 dispatch ticks over intermediate
-- buffers, which is the experimental per-layer dispatch model. Each cell
-- carries its own genome: rows stage per-cell payload/scales/offsets/specials
-- and the config SSBO carries per-cell base offsets (see forward_packed.comp).
-- The first dispatch sets the input-activation flag so layer-1 nodes are
-- activated exactly like the CPU/reference path. Returns per-cell output
-- vectors or nil, err.
function Runtime:_gpu_forward(network_id, items, precision)
    if not PACKED_FORMATS[precision] then
        return nil, errors.new('UNSUPPORTED_PRECISION',
            ('GPU packed dispatch requires fp8/fp4/fp2 (got %q)'):format(tostring(precision)), 'gpu')
    end
    local w, werr = self:_ensure_worker()
    if not w then
        return nil, errors.new('VULKAN_INITIALIZATION_FAILED',
            ('GPU worker unavailable: %s'):format(tostring(werr and werr.message)), 'gpu')
    end
    local net = self.networks[network_id]
    local layers = net.layers
    local num_cells = #items
    local cell_inputs = {}
    for c = 1, num_cells do cell_inputs[c] = items[c].inputs end

    -- Decode + pack each cell's genome once (per-cell genomes).
    local packed_cells = {}
    for c = 1, num_cells do
        local item_net = items[c].network_id or network_id
        local item_weights = items[c].genome
        if type(item_weights) == 'table' and item_weights.blob ~= nil then
            item_weights = item_weights.blob
        end
        local stream, err = self:_decode_weights(item_weights, nil, item_net)
        if not stream then return nil, err end
        local ok, ferr = check_finite(stream, self.backend)
        if not ok then return nil, ferr end
        packed_cells[c] = { stream = stream }
    end

    for li = 1, #layers - 1 do
        local in_c, out_c = layers[li], layers[li + 1]
        local profile = {
            network_id = network_id,
            format = precision,
            block_size = self.block_size,
            compute_format = 'fp32',
            scale_type = 'fp32',
            offset_type = 'fp32',
            quant_abi = 1,
        }
        local rows, inputs_flat = {}, {}
        for c = 1, num_cells do
            local item, perr = self:_pack_layer(network_id, packed_cells[c].stream,
                precision, self.block_size, li)
            if not item then return nil, perr end
            rows[c] = {
                genome_id = ('%s|%s|%d'):format(tostring(packed_cells[c].stream), network_id, li),
                row_start = c - 1,
                row_count = 1,
                item = item,
            }
            for k = 1, in_c do
                inputs_flat[#inputs_flat + 1] = cell_inputs[c][k] or 0
            end
        end
        local tick = (self._next_tick or 1)
        self._next_tick = tick + 1
        local batch = {
            tick_id = tick,
            model_hash = tostring(self),
            backend = 'vulkan',
            precision = precision,
            profile = profile,
            -- flags bit 0: activate the input layer (first dispatch only) —
            -- the shader then applies bias/threshold/dead to the layer-1
            -- inputs exactly like the CPU/reference path (SHOULD-3).
            config = { numNodes = out_c, fanIn = in_c, flags = (li == 1) and 1 or 0 },
            inputs = inputs_flat,
            num_cells = num_cells,
            rows = rows,
        }
        local sres = w:submit(batch)
        -- submit returns true on accept (result arrives via wait) or a
        -- failure result table on synchronous rejection.
        if type(sres) == 'table' and sres.ok == false then
            local class = sres.err_class or 'VULKAN_PIPELINE_FAILED'
            return nil, errors.new(class, sres.message or 'GPU dispatch failed', 'gpu')
        end
        local t0 = os.clock()
        local waited = w:wait(batch.tick_id, batch.model_hash)
        local res
        for _, r in ipairs(waited or {}) do
            if r.tick_id == batch.tick_id then res = r break end
        end
        local dt = os.clock() - t0
        self.metrics['gpu dispatch count'] = (self.metrics['gpu dispatch count'] or 0) + 1
        self.metrics['gpu dispatch time'] = (self.metrics['gpu dispatch time'] or 0) + dt
        if not res then
            return nil, errors.new('VULKAN_PIPELINE_FAILED',
                'no worker result for dispatched tick', 'gpu')
        end
        if not res.ok then
            local class = res.err_class or 'VULKAN_PIPELINE_FAILED'
            return nil, errors.new(class, res.message or 'GPU dispatch failed', 'gpu')
        end
        for c = 1, num_cells do
            local next_in = {}
            for o = 1, out_c do next_in[o] = res.outputs[c][o] end
            cell_inputs[c] = next_in
        end
    end
    return cell_inputs
end

-- GPU forward into a caller-owned `out` (or a fresh table when out == nil).
function Runtime:_forward_gpu(network_id, weights, inputs, out, precision)
    local items = { { genome = weights, network_id = network_id, inputs = inputs } }
    local results, err = self:_gpu_forward(network_id, items, precision)
    if not results then return nil, err end
    if not out then out = {} end
    local out_count = self.networks[network_id].layers[#self.networks[network_id].layers]
    for o = 1, out_count do out[o] = results[1][o] end
    return out
end

-- GPU forward_batch: dispatch once for all cells, copy readback into the
-- caller-owned out_desc.buffer. Output is never allocated by the runtime.
function Runtime:_forward_batch_gpu(network_id, batch_items, in_desc, out_desc)
    local net = self.networks[network_id]
    local fan_in = net.layers[1]
    local items = {}
    for c, item in ipairs(batch_items) do
        local inputs = {}
        local base = (c - 1) * (in_desc.stride or fan_in)
        for k = 1, fan_in do
            inputs[k] = in_desc.buffer[base + k] or 0
        end
        items[c] = { genome = item.genome or item.weights, network_id = item.network_id or network_id, inputs = inputs }
    end
    local results, err = self:_gpu_forward(network_id, items, self.precision)
    if not results then return nil, err end
    local out_count = net.layers[#net.layers]
    local out_stride = out_desc.stride or out_count
    for c = 1, #batch_items do
        local obase = (c - 1) * out_stride
        for o = 1, out_count do
            out_desc.buffer[obase + o] = results[c][o]
        end
    end
    return true
end

-- ------------------------------------------------------------- construction --

-- Backend resolution (once at init).
local function resolve_backend(requested, deterministic)
    if requested == 'cpu' then
        return 'cpu', nil
    end
    if requested == 'gpu' then
        if deterministic then
            return nil, errors.new('INVALID_ARGUMENT',
                'deterministic=true is incompatible with backend="gpu"', 'gpu')
        end
        if not vulkan_available() then
            return nil, errors.new('VULKAN_UNSUPPORTED_PLATFORM',
                'backend="gpu" requested but Vulkan is unavailable', 'gpu')
        end
        return 'gpu', nil
    end
    if requested == 'auto' then
        if deterministic then return 'cpu', nil end
        if vulkan_available() then return 'gpu', nil end
        return 'cpu', nil
    end
    return nil, errors.new('INVALID_ARGUMENT',
        ('unknown backend %q'):format(tostring(requested)), 'cpu')
end

-- Validate an optional topology override against the embedded one.
local function validate_topology(topology)
    if topology == nil then return true end
    local expected = format.NETWORKS
    for id, net in pairs(expected) do
        local given = topology[id]
        if given == nil then return false end
        local layers = given.layers or given
        if type(layers) ~= 'table' or #layers ~= #net.layers then return false end
        for i = 1, #layers do
            if layers[i] ~= net.layers[i] then return false end
        end
    end
    for id in pairs(topology) do
        if not expected[id] then return false end
    end
    return true
end

-- runtime.new(opts) -> runtime | nil, err.
function Runtime.new(opts)
    opts = opts or {}
    local backend, berr = resolve_backend(opts.backend or 'auto', opts.deterministic == true)
    if not backend then return nil, berr end

    local precision = opts.precision or 'fp8'
    if not PRECISIONS[precision] then
        return nil, errors.new('INVALID_PRECISION',
            ('unknown precision %q'):format(tostring(precision)), backend)
    end
    local block_size = opts.block_size or quantize.DEFAULT_BLOCK_SIZE
    if not quantize.is_valid_block_size(block_size) then
        return nil, errors.new('INVALID_ARGUMENT',
            ('invalid block size %d (must be 8/16/32/64)'):format(block_size), backend)
    end
    if not validate_topology(opts.topology) then
        return nil, errors.new('INVALID_TOPOLOGY',
            'topology must match the embedded game topology (seed/spore/sprout)', backend)
    end

    local self = setmetatable({
        backend = backend,
        deterministic = opts.deterministic == true,
        precision = precision,
        block_size = block_size,
        networks = format.NETWORKS,
        topology_identity = format.TOPOLOGY_IDENTITY,
        caps = capabilities.for_backend(backend, { deterministic = opts.deterministic == true }),
        _pack_cache = {},
        _shutdown = false,
        _gate_results = nil,
        _worker = nil,
        metrics = {},
    }, Runtime)

    -- Model loading: either a serialize model table {networks=...} or a raw
    -- weights blob (common genome); per-network slices are lazy.
    local model = opts.model
    if model ~= nil then
        self.model = model
    end
    return self
end

-- Per-network weights from the loaded model (used by corpus replay).
function Runtime:model_weights(network_id)
    local m = self.model
    if not m then return nil end
    if type(m) == 'table' and m.networks then
        local w = m.networks[network_id] and m.networks[network_id].weights
        if not w then return nil end
        local out = {}
        for i = 1, #w do out[i] = w[i] end
        return out
    end
    local stream, err = self:_decode_weights(m, nil, network_id)
    if stream then return stream end
    return nil
end

-- Corpus gate API: replay a recorded corpus against this runtime and store
-- gate results for set_precision.
function Runtime:load_corpus(c)
    local result, err = corpus_mod.replay(c, self)
    if not result then return nil, err end
    self._gate_results = result
    return result
end

return Runtime
