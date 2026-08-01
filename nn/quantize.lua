-- nn/quantize.lua — block quantization + IEEE float rounding (pure Lua).
--
-- The quantizer is CPU-authoritative: the pure-Lua implementation and the
-- LuaJIT FFI path MUST produce byte-identical codes/scales/offsets. All
-- rounding happens on fp64 values (no premature fp32 rounding of scales
-- during code selection); scales/offsets are then serialized as fp32
-- (round-to-nearest-even f64->f32) so the stored tensor matches what a GPU
-- path would consume.
--
-- Block packing rules:
--   * block sizes 8/16/32/64 (default 16); per LAYER MATRIX, padding is
--     block_count = ceil(logical/block), padded = block_count*block.
--   * padding is excluded from statistics (absmax / min-max).
--   * fp8 padding byte = 128 (q == 0); fp4/fp2 padding code = 0.
--   * zero blocks short-circuit BEFORE any division (no divide-by-zero).

local M = {}

local floor = math.floor

-- ------------------------------------------------------------------ rounding --

-- round-half-to-even (banker's): 0.5->0, 1.5->2, 2.5->2, -0.5->0,
-- -1.5->-2, -2.5->-2, -3.5->-4. NEVER plain floor(v + 0.5).
function M.round_even(v)
    local f = floor(v)
    local d = v - f
    if d > 0.5 then
        return f + 1
    elseif d == 0.5 then
        if f % 2 ~= 0 then return f + 1 end
        return f
    end
    return f
end

-- Extract IEEE-754 binary64 fields. Returns sign (0/1), exp (11 bits),
-- mantissa (52 bits) as exact Lua numbers. NaN/Inf handled by caller;
-- zero/-0.0 must be detected by the caller first (mant == 0 and exp == 0).
local function extract_f64(x)
    local s = 0
    if x < 0 then s = 1; x = -x end
    local m, e = math.frexp(x)
    local mant = m * 2 ^ 53 - 2 ^ 52
    return s, e + 1022, mant
end

-- Round a 53-bit significand (S, integer) to sig_bits (incl. implicit bit),
-- round-half-to-even on the discarded low bits. shift = 53 - sig_bits.
-- All arithmetic is exact in fp64 (S < 2^53, power-of-two divisors).
local function round_significand(S, shift)
    local q = floor(S / 2 ^ shift)
    local rem = S - q * 2 ^ shift
    local half = 2 ^ (shift - 1)
    if rem > half or (rem == half and q % 2 == 1) then
        q = q + 1
    end
    return q
end

-- Convert an fp64 to an fp32/fp16 value expressed as a 32-bit container
-- (bits stored in the low 24/16 bits as per IEEE). sig_bits=24 -> fp32,
-- sig_bits=11 -> fp16 (half). Matches hardware round-to-nearest-even.
function M.f64_to_float_bits(x, sig_bits, exp_bits)
    local max_exp = 2 ^ exp_bits - 1 -- 255 (fp32) / 31 (fp16)
    local bias = 2 ^ (exp_bits - 1) - 1 -- 127 / 15
    local shift = 53 - sig_bits
    if x ~= x then -- NaN -> canonical quiet NaN (0x7FC00000 fp32 / 0x7E00 fp16)
        return (max_exp << (sig_bits - 1)) | (1 << (sig_bits - 2))
    end
    if x == math.huge or x == -math.huge then
        return (x < 0 and 0x80000000 or 0) | (max_exp << (sig_bits - 1))
    end
    if x == 0 then
        local s = 1 / x == -math.huge and 0x80000000 or 0
        return s
    end
    local sign, e64, mant = extract_f64(x)
    local e = e64 - 1023 + bias -- target exponent field (normal)
    local sign_bit = sign * 0x80000000
    local function done(bits)
        -- LuaJIT bitwise ops are signed int32; normalize to unsigned.
        if bits < 0 then bits = bits + 2 ^ 32 end
        return bits
    end
    if e >= max_exp then -- overflow -> +/-inf (incl. rounding carry)
        return done(sign_bit | (max_exp << (sig_bits - 1)))
    end
    local S = mant + 2 ^ 52
    if e >= 1 then
        local q = round_significand(S, shift)
        if q == 2 ^ sig_bits then -- carry into the exponent
            q = 2 ^ (sig_bits - 1)
            e = e + 1
            if e >= max_exp then
                return done(sign_bit | (max_exp << (sig_bits - 1)))
            end
        end
        return done(sign_bit | (e << (sig_bits - 1)) | (q - 2 ^ (sig_bits - 1)))
    end
    -- Subnormal or zero: value in units of 2^(1-bias-(sig_bits-1)).
    -- Q = round(S / 2^(shift + 1 - e)); Q == 0 -> (+/-)0.
    local q = round_significand(S, shift + 1 - e)
    if q == 0 then
        return sign_bit
    end
    if q >= 2 ^ (sig_bits - 1) then -- rounding carried into normal range
        if q == 2 ^ sig_bits then
            return sign_bit | (2 << (sig_bits - 1)) -- e == 2, mantissa 0
        end
        return sign_bit | (1 << (sig_bits - 1)) | (q - 2 ^ (sig_bits - 1))
    end
    return sign_bit | q -- subnormal: mantissa == value
end

-- fp64 -> fp32 rounded value as an exact fp64 (for decode arithmetic).
function M.round_f32(x)
    local bits = M.f64_to_float_bits(x, 24, 8)
    return M.f32_from_bits(bits)
end

-- 32-bit fp32 container -> exact fp64 value.
function M.f32_from_bits(bits)
    local sign = bits >= 0x80000000 and -1 or 1
    local e = floor(bits / 0x800000) % 256
    local m = bits % 0x800000
    if e == 0 then
        if m == 0 then return sign == -1 and -0.0 or 0.0 end
        return sign * m * 2 ^ -149
    elseif e == 255 then
        if m == 0 then return sign * math.huge end
        return 0 / 0 -- NaN
    end
    return sign * (1 + m * 2 ^ -23) * 2 ^ (e - 127)
end

-- Little-endian bytes of an fp64->fp32 conversion (round-to-nearest-even).
function M.f64_to_f32_bytes(x)
    local bits = M.f64_to_float_bits(x, 24, 8)
    local b0 = bits % 256
    local b1 = floor(bits / 256) % 256
    local b2 = floor(bits / 65536) % 256
    local b3 = floor(bits / 16777216) % 256
    return string.char(b0, b1, b2, b3)
end

-- fp32 value from 4 little-endian bytes (exact fp64 reconstruction).
function M.f32_from_bytes(b0, b1, b2, b3)
    return M.f32_from_bits(b0 + b1 * 256 + b2 * 65536 + b3 * 16777216)
end

-- 8 little-endian bytes of an fp64 (exact; used for string weight blobs).
function M.f64_to_bytes(x)
    if x ~= x then -- NaN canonical 0x7FF8000000000000
        return string.char(0, 0, 0, 0, 0, 0, 0xF8, 0x7F)
    end
    if x == math.huge or x == -math.huge then
        return string.char(0, 0, 0, 0, 0, 0, 0xF0, x < 0 and 0xFF or 0x7F)
    end
    if x == 0 then
        if 1 / x == -math.huge then
            return string.char(0, 0, 0, 0, 0, 0, 0, 0x80)
        end
        return string.char(0, 0, 0, 0, 0, 0, 0, 0)
    end
    local sign, e64, mant = extract_f64(x)
    -- Subnormal f64 (exp field would be <= 0): un-normalize the significand
    -- into the stored form (exp field 0, mantissa = value * 2^1074).
    if e64 <= 0 then
        mant = (mant + 2 ^ 52) * 2 ^ (e64 - 1)
        e64 = 0
    end
    -- mant < 2^52, exp < 2048: byte-split is exact.
    local b0 = mant % 256
    local b1 = floor(mant / 256) % 256
    local b2 = floor(mant / 65536) % 256
    local b3 = floor(mant / 16777216) % 256
    local b4 = floor(mant / 4294967296) % 256
    local b5 = floor(mant / 1099511627776) % 256
    local b6 = (floor(mant / 281474976710656) % 16) + (e64 % 16) * 16
    local b7 = floor(e64 / 16) % 128 + sign * 128
    return string.char(b0, b1, b2, b3, b4, b5, b6, b7)
end

-- fp64 from 8 little-endian bytes (exact reconstruction).
function M.f64_from_bytes(s, off)
    local b0 = s:byte(off)
    local b1 = s:byte(off + 1)
    local b2 = s:byte(off + 2)
    local b3 = s:byte(off + 3)
    local b4 = s:byte(off + 4)
    local b5 = s:byte(off + 5)
    local b6 = s:byte(off + 6)
    local b7 = s:byte(off + 7)
    local mant_low = b0 + b1 * 256 + b2 * 65536 + b3 * 16777216
    local mant_mid = b4 + b5 * 256 -- mant bits 32..47
    local mant_top = b6 % 16 -- mant bits 48..51 (low nibble of byte 6)
    local mant = mant_low + mant_mid * 4294967296 + mant_top * 281474976710656
    -- byte6 high nibble = exp bits 0..3; byte7 low 7 bits = exp bits 4..10.
    local e64 = (b7 % 128) * 16 + math.floor(b6 / 16)
    local sign = b7 >= 128 and -1 or 1
    if e64 == 0 then
        if mant == 0 then return sign == -1 and -0.0 or 0.0 end
        return sign * mant * 2 ^ -1074
    elseif e64 == 2047 then
        if mant == 0 then return sign * math.huge end
        return 0 / 0
    end
    return sign * (1 + mant * 2 ^ -52) * 2 ^ (e64 - 1023)
end

-- fp64 -> fp16 rounded value as an exact fp64 (used by the fp16 gate).
function M.round_fp16(x)
    local bits = M.f64_to_float_bits(x, 11, 5)
    -- Reconstruct: sign, e (5 bits), m (10 bits).
    local sign = bits >= 0x8000 and -1 or 1
    local e = floor(bits / 0x400) % 32
    local m = bits % 0x400
    if e == 0 then
        if m == 0 then return sign == -1 and -0.0 or 0.0 end
        return sign * m * 2 ^ -24
    elseif e == 31 then
        if m == 0 then return sign * math.huge end
        return 0 / 0
    end
    return sign * (1 + m * 2 ^ -10) * 2 ^ (e - 15)
end

-- ------------------------------------------------------------ quantization --

local BLOCK_SIZES = { 8, 16, 32, 64 }
M.BLOCK_SIZES = BLOCK_SIZES
M.DEFAULT_BLOCK_SIZE = 16

function M.is_valid_block_size(b)
    for _, s in ipairs(BLOCK_SIZES) do
        if s == b then return true end
    end
    return false
end

-- ceil(logical / block).
function M.block_count(logical, block)
    return floor((logical + block - 1) / block)
end

-- fp8: symmetric signed int8 codes q in [-127,127], stored byte = q+128.
-- Zero block short-circuits BEFORE division: scale 0, q 0, stored byte 128.
function M.encode_fp8_block(values, block_size)
    local n = #values
    local absmax = 0
    for i = 1, n do
        local a = math.abs(values[i])
        if a > absmax then absmax = a end
    end
    if absmax == 0 then
        local codes = {}
        for i = 1, n do codes[i] = 0 end
        return { scale = 0.0, codes = codes, zero_block = true }
    end
    local codes = {}
    for i = 1, n do
        local q = M.round_even(values[i] * 127.0 / absmax) -- fp64, no fp32 scale
        if q > 127 then q = 127 elseif q < -127 then q = -127 end
        codes[i] = q
    end
    return { scale = absmax / 127.0, codes = codes }
end

-- fp4/fp2: affine unsigned codes q in [0, max_code] (max_code 15 or 3).
-- Constant block short-circuits BEFORE division: offset=min, scale=0, q=0.
function M.encode_affine_block(values, max_code)
    local n = #values
    local mn, mx = values[1], values[1]
    for i = 2, n do
        local v = values[i]
        if v < mn then mn = v end
        if v > mx then mx = v end
    end
    if mx == mn then
        local codes = {}
        for i = 1, n do codes[i] = 0 end
        return { scale = 0.0, offset = mn, codes = codes, constant = true }
    end
    local scale = (mx - mn) / max_code
    local codes = {}
    for i = 1, n do
        local q = M.round_even((values[i] - mn) / scale)
        if q < 0 then q = 0 elseif q > max_code then q = max_code end
        codes[i] = q
    end
    return { scale = scale, offset = mn, codes = codes }
end

-- fp8 payload: stored byte = q + 128 (valid 1..255; byte 0 reserved). Padding
-- byte 128 (q == 0). `padded_count` is the block-aligned length.
function M.pack_fp8(codes, padded_count)
    local out = {}
    for i = 1, padded_count do
        local q = codes[i]
        if q == nil then q = 0 end
        out[i] = string.char(q + 128)
    end
    return table.concat(out)
end

-- fp4 payload: two codes per byte, low nibble first. Padding code 0.
function M.pack_fp4(codes, padded_count)
    local out, nb = {}, padded_count / 2
    for i = 1, nb do
        local c0 = codes[(i - 1) * 2 + 1] or 0
        local c1 = codes[(i - 1) * 2 + 2] or 0
        out[i] = string.char(c0 + c1 * 16)
    end
    return table.concat(out)
end

-- fp2 payload: four codes per byte, bits 0-1 first. Padding code 0.
function M.pack_fp2(codes, padded_count)
    local out, nb = {}, padded_count / 4
    for i = 1, nb do
        local c0 = codes[(i - 1) * 4 + 1] or 0
        local c1 = codes[(i - 1) * 4 + 2] or 0
        local c2 = codes[(i - 1) * 4 + 3] or 0
        local c3 = codes[(i - 1) * 4 + 4] or 0
        out[i] = string.char(c0 + c1 * 4 + c2 * 16 + c3 * 64)
    end
    return table.concat(out)
end

-- Decode one fp8 block (block index `bi`, 0-based). Returns fp64 values.
-- scale is the fp32-rounded scale from the scale tensor.
function M.decode_fp8_block(payload, base, bi, block_size, scale)
    local out = {}
    local byte_base = base + bi * block_size
    for i = 1, block_size do
        local q = payload:byte(byte_base + i) - 128
        out[i] = scale * q
    end
    return out
end

-- Decode one affine block (fp4 max_code 15 / fp2 max_code 3).
function M.decode_affine_block(payload, base, bi, block_size, max_code, offset, scale)
    local out = {}
    local bits = max_code == 15 and 4 or 2
    local byte_base = base + bi * block_size * bits / 8
    for i = 1, block_size do
        local byte = payload:byte(byte_base + floor((i - 1) * bits / 8) + 1)
        local shift = (i - 1) * bits % 8
        local q = floor(byte / 2 ^ shift) % (max_code + 1)
        out[i] = offset + scale * q
    end
    return out
end

-- Decode a full padded matrix (payload + per-block scales/offsets) into
-- logical fp64 values. max_code: nil (fp8) or 15/3 (fp4/fp2).
function M.decode_matrix(payload, payload_base, logical, block_size, scales, offsets, max_code)
    local blocks = M.block_count(logical, block_size)
    local padded = blocks * block_size
    local out = {}
    for i = 1, logical do
        local bi = floor((i - 1) / block_size)
        local in_block = (i - 1) % block_size + 1
        local scale = scales[bi + 1]
        if max_code == nil then
            local q = payload:byte(payload_base + bi * block_size + in_block) - 128
            out[i] = scale * q
        else
            local offset = offsets[bi + 1]
            local bits = max_code == 15 and 4 or 2
            local byte_idx = payload_base + bi * block_size * bits / 8 + floor((in_block - 1) * bits / 8)
            local byte = payload:byte(byte_idx + 1)
            local shift = (in_block - 1) * bits % 8
            local q = floor(byte / 2 ^ shift) % (max_code + 1)
            out[i] = offset + scale * q
        end
    end
    return out
end

return M
