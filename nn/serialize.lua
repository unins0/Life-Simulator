-- nn/serialize.lua — versioned binary .nnw serialization (pure Lua, no
-- string.pack — LuaJIT lacks it). Byte-stable round trip: save(model) twice
-- produces identical bytes; load() validates magic/version/endianness,
-- truncation and cross-record references.
--
-- File layout:
--   [0]  header (24 bytes): magic "NNW\0", format_version u32, endianness
--        marker u32 (0x01020304), header_size u32, record_count u32, flags u32
--   [24] record directory: record_count fixed-size records (see record_size)
--   [header_size] payload region: tensor byte blobs, each at byte_offset with
--        explicit zero-fill of alignment gaps (relative to the region start)
--
-- Records (each prefixed by a u32 type):
--   TENSOR       (type 1, 52 B): tensor_id, role, element_type, storage_type,
--                logical_count, padded_count (u32), byte_offset, byte_length
--                (u64), alignment, flags (u32)
--   BLOCK_QUANT  (type 2, 76 B): source/payload/scale/offset/zero_point
--                tensor ids, block_size, block_log2, format, scale_type,
--                offset_type, zero_point_type, decode_form, granularity,
--                payload_alignment, metadata_alignment, logical_count,
--                padded_count, block_count (all u32)
--   NETWORK      (type 3, 44 B): network_id (len u32 + 16 zero-padded bytes),
--                layer_count, special_tensor_id, default_format,
--                default_block_size, resolved_profile_id (u32)
--   LAYER        (type 4, 36 B): input_count, output_count,
--                logical_weight_count, padded_weight_count, weight_tensor_id,
--                block_quant_record_id, format_override, block_size_override

local M = {}

local errors = require('nn.errors')
local format = require('nn.format')
local quantize = require('nn.quantize')

M.FORMAT_VERSION = 1
M.MAGIC = 'NNW\0'
M.ENDIANNESS_MARKER = 0x01020304

-- Record types
M.RECORD_TENSOR = 1
M.RECORD_BLOCK_QUANT = 2
M.RECORD_NETWORK = 3
M.RECORD_LAYER = 4

-- Tensor roles
M.ROLE_WEIGHT_MATRIX = 1
M.ROLE_SPECIALS = 2
M.ROLE_SCALE = 3
M.ROLE_OFFSET = 4
M.ROLE_ZERO_POINT = 5

-- Element types
M.ELEM_FP32 = 1
M.ELEM_FP16 = 2
M.ELEM_INT8_CODE = 3
M.ELEM_UINT8_CODE = 4
M.ELEM_PACKED_UINT8 = 5

-- Storage types
M.STORAGE_RAW = 1
M.STORAGE_PACKED_UINT8 = 2
M.STORAGE_WORD_PACKED = 3

-- Block quant enums
M.QUANT_FP8_SYMMETRIC = 1
M.QUANT_FP4_AFFINE = 2
M.QUANT_FP2_AFFINE = 3
M.SCALE_FP32_CANONICAL = 1
M.SCALE_FP16_EXPERIMENTAL = 2
M.OFFSET_NONE = 1
M.OFFSET_FP32_CANONICAL = 2
M.OFFSET_FP16_EXPERIMENTAL = 3
M.ZERO_POINT_NONE = 1
M.ZERO_POINT_UINT8 = 2
M.DECODE_SYMMETRIC_ZERO_POINT = 1
M.DECODE_AFFINE_OFFSET = 2
M.GRANULARITY_PER_BLOCK = 1
M.FORMAT_INHERIT = 1
M.FORMAT_FP8 = 2
M.FORMAT_FP4 = 3
M.FORMAT_FP2 = 4

M.NETWORK_FORMAT = { fp32 = 1, fp16 = 2, fp8 = 3, fp4 = 4, fp2 = 5 }

-- --------------------------------------------------------------- byte I/O --

local function wu32(v)
    v = v % 4294967296
    local b0 = v % 256
    local b1 = math.floor(v / 256) % 256
    local b2 = math.floor(v / 65536) % 256
    local b3 = math.floor(v / 16777216) % 256
    return string.char(b0, b1, b2, b3)
end

local function wu64(v)
    local lo = v % 4294967296
    local hi = math.floor(v / 4294967296)
    return wu32(lo) .. wu32(hi)
end

local function ru32(s, off)
    local b0 = s:byte(off)
    local b1 = s:byte(off + 1)
    local b2 = s:byte(off + 2)
    local b3 = s:byte(off + 3)
    return b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
end

local function ru64(s, off)
    local lo = ru32(s, off)
    local hi = ru32(s, off + 4)
    return lo + hi * 4294967296
end

M.wu32 = wu32
M.ru32 = ru32

-- Record sizes (type u32 + body).
M.RECORD_SIZES = {
    [M.RECORD_TENSOR] = 52,
    [M.RECORD_BLOCK_QUANT] = 76,
    [M.RECORD_NETWORK] = 44,
    [M.RECORD_LAYER] = 36,
}

function M.record_size(rec)
    return M.RECORD_SIZES[rec.type] or 0
end

-- --------------------------------------------------------- record encode --

local function encode_tensor(rec)
    local out = wu32(M.RECORD_TENSOR)
    out = out .. wu32(rec.tensor_id)
    out = out .. wu32(rec.role)
    out = out .. wu32(rec.element_type)
    out = out .. wu32(rec.storage_type)
    out = out .. wu32(rec.logical_count)
    out = out .. wu32(rec.padded_count)
    out = out .. wu64(rec.byte_offset or 0)
    out = out .. wu64(rec.byte_length or 0)
    out = out .. wu32(rec.alignment or 4)
    out = out .. wu32(rec.flags or 0)
    return out
end

local function encode_block_quant(rec)
    local out = wu32(M.RECORD_BLOCK_QUANT)
    out = out .. wu32(rec.source_tensor_id)
    out = out .. wu32(rec.payload_tensor_id)
    out = out .. wu32(rec.scale_tensor_id)
    out = out .. wu32(rec.offset_tensor_id)
    out = out .. wu32(rec.zero_point_tensor_id)
    out = out .. wu32(rec.block_size)
    out = out .. wu32(rec.block_log2)
    out = out .. wu32(rec.format)
    out = out .. wu32(rec.scale_type)
    out = out .. wu32(rec.offset_type)
    out = out .. wu32(rec.zero_point_type)
    out = out .. wu32(rec.decode_form)
    out = out .. wu32(rec.granularity)
    out = out .. wu32(rec.payload_alignment)
    out = out .. wu32(rec.metadata_alignment)
    out = out .. wu32(rec.logical_count)
    out = out .. wu32(rec.padded_count)
    out = out .. wu32(rec.block_count)
    return out
end

local function encode_network(rec)
    local id = rec.network_id or ''
    assert(#id <= 16, 'network_id longer than 16 bytes')
    local pad = id .. string.rep('\0', 16 - #id)
    local out = wu32(M.RECORD_NETWORK)
    out = out .. wu32(#id)
    out = out .. pad
    out = out .. wu32(rec.layer_count)
    out = out .. wu32(rec.special_tensor_id)
    out = out .. wu32(rec.default_format)
    out = out .. wu32(rec.default_block_size)
    out = out .. wu32(rec.resolved_profile_id)
    return out
end

local function encode_layer(rec)
    local out = wu32(M.RECORD_LAYER)
    out = out .. wu32(rec.input_count)
    out = out .. wu32(rec.output_count)
    out = out .. wu32(rec.logical_weight_count)
    out = out .. wu32(rec.padded_weight_count)
    out = out .. wu32(rec.weight_tensor_id)
    out = out .. wu32(rec.block_quant_record_id)
    out = out .. wu32(rec.format_override)
    out = out .. wu32(rec.block_size_override)
    return out
end

local ENCODERS = {
    [M.RECORD_TENSOR] = encode_tensor,
    [M.RECORD_BLOCK_QUANT] = encode_block_quant,
    [M.RECORD_NETWORK] = encode_network,
    [M.RECORD_LAYER] = encode_layer,
}

-- ---------------------------------------------------------- record decode --

local function decode_tensor(s, off, header_size)
    local rec = { type = M.RECORD_TENSOR }
    rec.tensor_id = ru32(s, off + 4)
    rec.role = ru32(s, off + 8)
    rec.element_type = ru32(s, off + 12)
    rec.storage_type = ru32(s, off + 16)
    rec.logical_count = ru32(s, off + 20)
    rec.padded_count = ru32(s, off + 24)
    rec.byte_offset = ru64(s, off + 28)
    rec.byte_length = ru64(s, off + 36)
    rec.alignment = ru32(s, off + 44)
    rec.flags = ru32(s, off + 48)
    return rec
end

local function decode_block_quant(s, off)
    local rec = { type = M.RECORD_BLOCK_QUANT }
    rec.source_tensor_id = ru32(s, off + 4)
    rec.payload_tensor_id = ru32(s, off + 8)
    rec.scale_tensor_id = ru32(s, off + 12)
    rec.offset_tensor_id = ru32(s, off + 16)
    rec.zero_point_tensor_id = ru32(s, off + 20)
    rec.block_size = ru32(s, off + 24)
    rec.block_log2 = ru32(s, off + 28)
    rec.format = ru32(s, off + 32)
    rec.scale_type = ru32(s, off + 36)
    rec.offset_type = ru32(s, off + 40)
    rec.zero_point_type = ru32(s, off + 44)
    rec.decode_form = ru32(s, off + 48)
    rec.granularity = ru32(s, off + 52)
    rec.payload_alignment = ru32(s, off + 56)
    rec.metadata_alignment = ru32(s, off + 60)
    rec.logical_count = ru32(s, off + 64)
    rec.padded_count = ru32(s, off + 68)
    rec.block_count = ru32(s, off + 72)
    return rec
end

local function decode_network(s, off)
    local rec = { type = M.RECORD_NETWORK }
    local len = ru32(s, off + 4)
    if len > 16 then return nil, 'network_id too long' end
    rec.network_id = s:sub(off + 8, off + 7 + len)
    rec.layer_count = ru32(s, off + 24)
    rec.special_tensor_id = ru32(s, off + 28)
    rec.default_format = ru32(s, off + 32)
    rec.default_block_size = ru32(s, off + 36)
    rec.resolved_profile_id = ru32(s, off + 40)
    return rec
end

local function decode_layer(s, off)
    local rec = { type = M.RECORD_LAYER }
    rec.input_count = ru32(s, off + 4)
    rec.output_count = ru32(s, off + 8)
    rec.logical_weight_count = ru32(s, off + 12)
    rec.padded_weight_count = ru32(s, off + 16)
    rec.weight_tensor_id = ru32(s, off + 20)
    rec.block_quant_record_id = ru32(s, off + 24)
    rec.format_override = ru32(s, off + 28)
    rec.block_size_override = ru32(s, off + 32)
    return rec
end

local DECODERS = {
    [M.RECORD_TENSOR] = decode_tensor,
    [M.RECORD_BLOCK_QUANT] = decode_block_quant,
    [M.RECORD_NETWORK] = decode_network,
    [M.RECORD_LAYER] = decode_layer,
}

-- ---------------------------------------------------------------- writing --

-- Build a canonical .nnw byte string from a model:
--   model = { topology_identity = string, block_size = n,
--             networks = { <id> = { layers = {...}, weights = <node-major> } } }
-- Weights/specials are stored RAW as fp32 (production fp32 specials; matrix
-- weights fp32). Quant records describe the default fp8/block_size profile.
function M.write(header, records, payloads)
    -- 1. Compute directory size + assign payload offsets (in record order).
    local directory = {}
    local dir_size = 0
    local tensor_recs = {}
    for i, rec in ipairs(records) do
        local size = M.record_size(rec)
        if size == 0 then
            return nil, errors.new('INVALID_SERIALIZATION',
                ('unknown record type %s at record %d'):format(tostring(rec.type), i))
        end
        dir_size = dir_size + size
        directory[i] = rec
        if rec.type == M.RECORD_TENSOR then
            tensor_recs[#tensor_recs + 1] = rec
        end
    end

    local header_size = 24 + dir_size
    local cursor = 0
    local payload_end = 0
    for _, rec in ipairs(tensor_recs) do
        local blob = payloads[rec.tensor_id] or ''
        local align = rec.alignment or 4
        if align < 1 then align = 1 end
        -- Align the entry start; gaps are zero-filled explicitly below.
        local start = cursor
        local rem = cursor % align
        if rem > 0 then start = cursor + (align - rem) end
        rec.byte_offset = start
        rec.byte_length = #blob
        cursor = start + #blob
        if cursor > payload_end then payload_end = cursor end
    end

    -- 2. Header.
    local out = M.MAGIC
    out = out .. wu32(header.format_version or M.FORMAT_VERSION)
    out = out .. wu32(M.ENDIANNESS_MARKER)
    out = out .. wu32(header_size)
    out = out .. wu32(#records)
    out = out .. wu32(header.flags or 0)
    assert(#out == 24, 'header must be 24 bytes')

    -- 3. Record directory.
    for i, rec in ipairs(directory) do
        out = out .. ENCODERS[rec.type](rec)
    end
    assert(#out == header_size, 'directory size mismatch')

    -- 4. Payload region: emit with explicit zero-fill of alignment gaps.
    local pos = 0
    for _, rec in ipairs(tensor_recs) do
        local blob = payloads[rec.tensor_id] or ''
        local gap = rec.byte_offset - pos
        if gap > 0 then out = out .. string.rep('\0', gap) end
        out = out .. blob
        pos = rec.byte_offset + #blob
    end
    -- Any trailing alignment slack beyond the last payload is explicit zeros.
    -- (pos == payload_end by construction.)
    return out
end

-- Build records+payloads for a model (canonical record order).
function M.model_to_nnw(model, opts)
    opts = opts or {}
    local block_size = opts.block_size or model.block_size or quantize.DEFAULT_BLOCK_SIZE
    local networks = {}
    for _, id in ipairs(format.NETWORK_ORDER) do
        if model.networks[id] then networks[#networks + 1] = id end
    end

    local records, payloads = {}, {}
    local next_tensor = 1
    local tensor_id = {}
    local matrix_tensor = {}

    -- Assign tensor ids first (deterministic order).
    local specials_tensor = {}
    for _, id in ipairs(networks) do
        local net = model.networks[id]
        specials_tensor[id] = next_tensor; next_tensor = next_tensor + 1
        matrix_tensor[id] = {}
        for li = 1, #net.layers - 1 do
            matrix_tensor[id][li] = next_tensor; next_tensor = next_tensor + 1
        end
    end

    -- Network descriptors.
    for _, id in ipairs(networks) do
        local net = model.networks[id]
        records[#records + 1] = {
            type = M.RECORD_NETWORK,
            network_id = id,
            layer_count = #net.layers,
            special_tensor_id = specials_tensor[id],
            default_format = M.NETWORK_FORMAT[opts.format or 'fp8'] or M.FORMAT_FP8,
            default_block_size = block_size,
            resolved_profile_id = opts.profile_id or 0,
        }
    end

    -- Layer descriptors.
    local block_quant_id = {}
    for _, id in ipairs(networks) do
        local net = model.networks[id]
        block_quant_id[id] = {}
        local dec = format.decompose(net.layers, net.weights)
        for li = 1, #net.layers - 1 do
            local m = dec.matrices[li]
            local logical = m.input_count * m.output_count
            local blocks = quantize.block_count(logical, block_size)
            local padded = blocks * block_size
            records[#records + 1] = {
                type = M.RECORD_LAYER,
                input_count = m.input_count,
                output_count = m.output_count,
                logical_weight_count = logical,
                padded_weight_count = padded,
                weight_tensor_id = matrix_tensor[id][li],
                block_quant_record_id = 0, -- patched after ids assigned
                format_override = M.FORMAT_INHERIT,
                block_size_override = 0,
            }
        end
    end

    -- Block quant records.
    local bq_seq = 0
    for _, id in ipairs(networks) do
        local net = model.networks[id]
        for li = 1, #net.layers - 1 do
            local logical = net.layers[li] * net.layers[li + 1]
            local blocks = quantize.block_count(logical, block_size)
            bq_seq = bq_seq + 1
            local bq_id = bq_seq
            block_quant_id[id][li] = bq_id
            records[#records + 1] = {
                type = M.RECORD_BLOCK_QUANT,
                source_tensor_id = matrix_tensor[id][li],
                payload_tensor_id = 0,
                scale_tensor_id = 0,
                offset_tensor_id = 0,
                zero_point_tensor_id = 0,
                block_size = block_size,
                block_log2 = math.floor(math.log(block_size) / math.log(2) + 0.5),
                format = M.QUANT_FP8_SYMMETRIC,
                scale_type = M.SCALE_FP32_CANONICAL,
                offset_type = M.OFFSET_NONE,
                zero_point_type = M.ZERO_POINT_NONE,
                decode_form = M.DECODE_SYMMETRIC_ZERO_POINT,
                granularity = M.GRANULARITY_PER_BLOCK,
                payload_alignment = 1,
                metadata_alignment = 4,
                logical_count = logical,
                padded_count = blocks * block_size,
                block_count = blocks,
            }
        end
    end

    -- Tensor records + payloads (specials, then matrices per layer).
    for _, id in ipairs(networks) do
        local net = model.networks[id]
        local dec = format.decompose(net.layers, net.weights)
        -- Specials tensor: interleaved fp32.
        local spec_blob = {}
        for i = 1, #dec.specials do
            spec_blob[i] = quantize.f64_to_f32_bytes(dec.specials[i])
        end
        local tid = specials_tensor[id]
        payloads[tid] = table.concat(spec_blob)
        records[#records + 1] = {
            type = M.RECORD_TENSOR,
            tensor_id = tid,
            role = M.ROLE_SPECIALS,
            element_type = M.ELEM_FP32,
            storage_type = M.STORAGE_RAW,
            logical_count = #dec.specials,
            padded_count = #dec.specials,
            alignment = 4,
            flags = 0,
        }
        for li = 1, #net.layers - 1 do
            local m = dec.matrices[li]
            local logical = m.input_count * m.output_count
            local blocks = quantize.block_count(logical, block_size)
            local padded = blocks * block_size
            local blob = {}
            for i = 1, #m.values do
                blob[i] = quantize.f64_to_f32_bytes(m.values[i])
            end
            local mtid = matrix_tensor[id][li]
            payloads[mtid] = table.concat(blob)
            records[#records + 1] = {
                type = M.RECORD_TENSOR,
                tensor_id = mtid,
                role = M.ROLE_WEIGHT_MATRIX,
                element_type = M.ELEM_FP32,
                storage_type = M.STORAGE_RAW,
                logical_count = logical,
                padded_count = padded,
                alignment = 4,
                flags = 0,
            }
        end
    end

    -- Patch layer records with block_quant_record_id (scan in directory order).
    local layer_index = 0
    for _, id in ipairs(networks) do
        for li = 1, #model.networks[id].layers - 1 do
            layer_index = layer_index + 1
            local n = 0
            for i = 1, #records do
                if records[i].type == M.RECORD_LAYER then
                    n = n + 1
                    if n == layer_index then
                        records[i].block_quant_record_id = block_quant_id[id][li]
                        break
                    end
                end
            end
        end
    end

    local bytes, err = M.write(
        { format_version = M.FORMAT_VERSION, flags = opts.flags or 0 },
        records, payloads)
    if not bytes then return nil, err end
    return bytes, records, payloads
end

-- ---------------------------------------------------------------- reading --

-- read(bytes) -> model | nil, err
function M.read(bytes)
    if type(bytes) ~= 'string' then
        return nil, errors.new('INVALID_SERIALIZATION', 'expected a byte string')
    end
    if #bytes < 24 then
        return nil, errors.new('INVALID_SERIALIZATION', 'truncated header')
    end
    if bytes:sub(1, 4) ~= M.MAGIC then
        return nil, errors.new('INVALID_SERIALIZATION', 'bad magic')
    end
    local version = ru32(bytes, 5)
    if version ~= M.FORMAT_VERSION then
        return nil, errors.new('INVALID_SERIALIZATION',
            ('unsupported format version %d'):format(version))
    end
    local marker = ru32(bytes, 9)
    if marker ~= M.ENDIANNESS_MARKER then
        return nil, errors.new('INVALID_SERIALIZATION',
            'endianness marker mismatch (not little-endian)')
    end
    local header_size = ru32(bytes, 13)
    local record_count = ru32(bytes, 17)
    local flags = ru32(bytes, 21)
    if header_size < 24 then
        return nil, errors.new('INVALID_SERIALIZATION', 'header_size smaller than header')
    end

    -- Decode the record directory.
    local records, off = {}, 25
    local expected_dir = 0
    for i = 1, record_count do
        if off + 4 > #bytes then
            return nil, errors.new('INVALID_SERIALIZATION', 'truncated record directory')
        end
        local rtype = ru32(bytes, off)
        local size = M.RECORD_SIZES[rtype]
        if not size then
            return nil, errors.new('INVALID_SERIALIZATION',
                ('unknown record type %d'):format(rtype))
        end
        if off + size - 1 > #bytes then
            return nil, errors.new('INVALID_SERIALIZATION', 'truncated record directory')
        end
        local rec, derr = DECODERS[rtype](bytes, off)
        if not rec then
            return nil, errors.new('INVALID_SERIALIZATION', derr)
        end
        records[i] = rec
        off = off + size
        expected_dir = expected_dir + size
    end
    if header_size ~= 24 + expected_dir then
        return nil, errors.new('INVALID_SERIALIZATION',
            'header_size does not match the record directory')
    end

    -- Payload region length.
    local payload_end = 0
    local tensors = {}
    for _, rec in ipairs(records) do
        if rec.type == M.RECORD_TENSOR then
            tensors[rec.tensor_id] = rec
            local endp = rec.byte_offset + rec.byte_length
            if endp > payload_end then payload_end = endp end
        end
    end
    if #bytes ~= header_size + payload_end then
        return nil, errors.new('INVALID_SERIALIZATION',
            ('truncated payload (got %d bytes, expected %d)')
            :format(#bytes, header_size + payload_end))
    end

    -- Cross-record reference validation.
    for _, rec in ipairs(records) do
        if rec.type == M.RECORD_BLOCK_QUANT then
            if not tensors[rec.source_tensor_id] then
                return nil, errors.new('INVALID_SERIALIZATION',
                    'block quant references a missing source tensor')
            end
            for _, field in ipairs({ 'payload_tensor_id', 'scale_tensor_id',
                'offset_tensor_id', 'zero_point_tensor_id' }) do
                local v = rec[field]
                if v and v ~= 0 and not tensors[v] then
                    return nil, errors.new('INVALID_SERIALIZATION',
                        ('block quant references a missing %s tensor'):format(field))
                end
            end
            -- scale_type=FP16_EXPERIMENTAL must mirror in the SCALE tensor.
            if rec.scale_type == M.SCALE_FP16_EXPERIMENTAL then
                local st = tensors[rec.scale_tensor_id]
                if not st or st.element_type ~= M.ELEM_FP16 then
                    return nil, errors.new('INVALID_SERIALIZATION',
                        'FP16_EXPERIMENTAL scale_type requires an FP16 scale tensor')
                end
            end
        end
    end

    -- Slice payloads.
    local payloads = {}
    for _, rec in ipairs(records) do
        if rec.type == M.RECORD_TENSOR then
            local start = header_size + 1 + rec.byte_offset
            payloads[rec.tensor_id] = bytes:sub(start, start + rec.byte_length - 1)
        end
    end

    -- Map layer records back to their network. Tensor ids were assigned in
    -- network order: network k owns the range [s_k, s_k + 1 + (L_k - 1))
    -- where s_k is its specials tensor id and L_k its layer count. Layer
    -- records reference weight tensor ids, so the owning network is the one
    -- whose id range contains the id.
    local net_meta = {}
    for _, rec in ipairs(records) do
        if rec.type == M.RECORD_NETWORK then
            net_meta[#net_meta + 1] = {
                network_id = rec.network_id,
                layer_count = rec.layer_count,
                special_tensor_id = rec.special_tensor_id,
                default_format = rec.default_format,
                default_block_size = rec.default_block_size,
                resolved_profile_id = rec.resolved_profile_id,
            }
        end
    end
    local range_lo, range_hi = {}, {}
    for i, nm in ipairs(net_meta) do
        local next_id = i < #net_meta and net_meta[i + 1].special_tensor_id
            or (10 ^ 9)
        range_lo[i] = nm.special_tensor_id
        range_hi[i] = next_id - 1
    end
    local function owner_of(tid)
        for i, nm in ipairs(net_meta) do
            if tid >= range_lo[i] and tid <= range_hi[i] then
                return nm.network_id
            end
        end
        return nil
    end

    -- Rebuild the model (networks -> layers -> node-major weights via fp32).
    local model = {
        format_version = version,
        flags = flags,
        header_size = header_size,
        networks = {},
        records = records,
        payloads = payloads,
    }
    local id_parts = {}
    for _, nm in ipairs(net_meta) do
        model.networks[nm.network_id] = { layers = {} }
    end
    -- Collect layers per network: first input_count + every output_count.
    for _, rec in ipairs(records) do
        if rec.type == M.RECORD_LAYER then
            local net_id = owner_of(rec.weight_tensor_id)
            if not net_id then
                return nil, errors.new('INVALID_SERIALIZATION',
                    'layer weight tensor id has no owning network')
            end
            local net = model.networks[net_id]
            if not net.layers_inited then
                net.layers_inited = true
                net.layers = { rec.input_count }
            end
            net.layers[#net.layers + 1] = rec.output_count
        end
    end
    -- Reconstruct node-major weights from specials + matrix tensors.
    for net_id, net in pairs(model.networks) do
        local nm
        for _, m in ipairs(net_meta) do
            if m.network_id == net_id then nm = m end
        end
        local spec_blob = payloads[nm.special_tensor_id]
        if not spec_blob then
            return nil, errors.new('INVALID_SERIALIZATION',
                ('missing specials tensor for network %q'):format(net_id))
        end
        local specials = {}
        for i = 1, #spec_blob, 4 do
            specials[#specials + 1] = quantize.f32_from_bytes(
                spec_blob:byte(i), spec_blob:byte(i + 1),
                spec_blob:byte(i + 2), spec_blob:byte(i + 3))
        end
        -- Reconstruct matrices from layer records (in directory order).
        local matrices = {}
        for _, rec in ipairs(records) do
            if rec.type == M.RECORD_LAYER and owner_of(rec.weight_tensor_id) == net_id then
                local blob = payloads[rec.weight_tensor_id]
                local values = {}
                for i = 1, rec.logical_weight_count do
                    local b = (i - 1) * 4 + 1
                    values[i] = quantize.f32_from_bytes(blob:byte(b), blob:byte(b + 1),
                        blob:byte(b + 2), blob:byte(b + 3))
                end
                matrices[#matrices + 1] = {
                    input_count = rec.input_count,
                    output_count = rec.output_count,
                    values = values,
                }
            end
        end
        local stream = format.reconstruct(net.layers, { matrices = matrices, specials = specials })
        net.weights = stream
        net.specials = specials
        net.matrices = matrices
        net.layers_inited = nil
    end
    model.topology_identity = table.concat(id_parts, ':')
    for net_id, net in pairs(model.networks) do
        id_parts[#id_parts + 1] = net_id .. '=' .. table.concat(net.layers, '-')
    end
    model.topology_identity = table.concat(id_parts, ':')
    -- Default block size from the first block quant record / network.
    for _, rec in ipairs(records) do
        if rec.type == M.RECORD_BLOCK_QUANT then
            model.block_size = rec.block_size
            break
        end
    end
    if not model.block_size then
        model.block_size = net_meta[1] and net_meta[1].default_block_size
            or quantize.DEFAULT_BLOCK_SIZE
    end
    return model, nil
end

return M
