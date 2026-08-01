# Life Simulator 5

A simple simulator of cellular natural selection.

## Running the simulation

To run the simulation, you will need to download the [Love2D](https://love2d.org/) framework.
Add Love2D to PATH, type "love" in console and paste the path to the simulation repository.

Example:
>love C:\Users\User\Desktop\Life-Simulator-5

## Keybinds

- `Space` — Pause/Unpause
- `Up` / `Down` — Increase/decrease simulation speed
- `U` — Hide/show interface
- `E` — Next view mode (Currently, only normal and map minerals viewing modes)
- `R` — Regenerate map

## Configuration

The `shares.lua` file contains all the necessary properties: map size and day
cycle, sun/mineral ranges, cell costs/colors/lifespans, the AI network
topologies (`AI_LAYERS_*`), genome lengths (`AI_LEN_*`) and per-network offsets
(`AI_OFFSET_*`), plus the shared map/cell/genome state tables.

## Gameplay overview

- The world is a grid of cells (Leaf, Root, Stem, Seed, Spore, Sprout) that
  harvest energy and minerals, grow, and reproduce — natural selection at the
  cellular level.
- Every AI cell carries a **per-cell genome**: a flat stream of IEEE-754
  doubles. The three networks are concatenated into one common genome of
  `AI_LEN_COMMON = 1003` doubles, with each network occupying its own slice
  (`AI_OFFSET_SEED/SPORE/SPROUT`).
- Network topologies (layer counts, in `shares.lua`):
  - `seed`   — `9-12-1`
  - `spore`  — `6-16-1`
  - `sprout` — `9-18-16-3`
- Genome layout per node: `(bias, threshold, dead_value, weights to next layer)`.
  Activation semantics for every non-input node:

  ```
  value = value + bias
  if value <= threshold then value = dead_value
  ```

  The comparison is inclusive (`<=`), and NaN propagates (`NaN <= t` is false).

## The `nn/` module

`nn/` is a standalone, portable neural-network package with **no LÖVE or game
dependencies**. It implements the game's networks (topology mirrored as data in
`nn/format.lua`, pinned against `shares.lua` by the tests) with dynamic
quantization, versioned `.nnw` serialization, a determinism/corpus-gate model,
and an optional Vulkan GPU backend. `require('nn')` resolves through
`nn/init.lua`.

### Backends

One backend is resolved **once** at init — there is no hybrid mode and no
switching afterwards (`runtime.backend` is always `"cpu"` or `"gpu"`):

- `cpu` — LuaJIT FFI `double` path; bit-identical to the pure-Lua reference.
- `reference` — pure Lua; validation only, never chosen by `auto`.
- `gpu` — optional Vulkan (Windows `vulkan-1.dll`, Linux `libvulkan.so.1`).

Resolution rules (`backend` option is `"cpu" | "gpu" | "auto"`):

- `deterministic = true` forces `"cpu"` (`auto` → cpu; explicit `gpu` is an init error).
- `deterministic = false` + `auto` → `gpu` if Vulkan initializes, otherwise `cpu`.
- Explicit `gpu` with no Vulkan support → init error.

After init there is **never** a silent CPU fallback: a post-init `DEVICE_LOST`
is a hard structured error.

### Dynamic quantization

Precisions `fp32`, `fp16`, `fp8`, `fp4`, `fp2` with **block quantization**
(block sizes 8/16/32/64, default 16):

- `fp8` — symmetric int8-code (`byte = q + 128`; byte 0 is reserved; padding
  byte is 128).
- `fp4` / `fp2` — affine offset + scale (padding code 0).
- Rounding is round-half-to-even; code selection happens in fp64 and
  scales/offsets are then serialized as fp32.
- The quantizer is **CPU-authoritative**: the GPU only decodes canonical packed
  data.
- Pure-Lua f64→f32 serialization is used throughout (LuaJIT has no
  `string.pack('<f')`).

### `.nnw` serialization

`nn.save` / `nn.load` read and write a versioned binary format: magic `"NNW\0"`,
an endianness marker, and `Tensor` / `BlockQuant` / `Network` / `Layer` records.
Round trips are byte-stable — `save(model)` twice yields identical bytes, and
`load` validates magic, version, endianness, truncation, and cross-record
references.

### Determinism & corpus gates

- The pure-Lua reference implementation is authoritative.
- Determinism is verified against a **recorded, frozen, hashed corpus** — never
  a per-runtime seeded PRNG (PUC-Lua and LuaJIT RNGs differ).
- Action-agreement gates per precision: `fp16 ≥ 99%`, `fp8 ≥ 97%`,
  `fp4 ≥ 90%`, `fp2 ≥ 80%`, plus bounded-error limits.
- Epsilon boundary buckets: nodes with `|value − threshold| ≤ ε` are reported
  separately and **excluded** from the primary gate, so they cannot hide
  catastrophic non-boundary errors.
- NaN propagates (`NaN <= t` is false); the threshold comparison is inclusive
  (`<=`).

### `set_precision` gating

- `fp16` / `fp8` require a passing corpus gate for that precision.
- `fp4` / `fp2` require `experimental = true` in `set_precision` — a passing
  corpus gate alone does **not** unlock them (selection is experimental-only;
  without the flag `CORPUS_GATE_FAILED` is returned).
- `fp32` is always allowed.
- The default profile is all-`fp8` / block-16 with fp32 scales and fp32 specials.

### Vulkan backend

- Mandatory word-addressed `uint[]` byte loading (`uint8` SSBO is not core;
  when present, the 8-bit path must produce identical bytes).
- Uniform per-network profiles: one format + one block size per network.
  One forward pass = one dispatch **per layer** (the packed shader is a dense
  single-layer model; intermediate activations round-trip through the worker
  readback between the `#layers - 1` dispatch ticks). Non-uniform per-layer
  maps are runtime-rejected unless an explicit experimental per-layer dispatch
  flag is set.
- The GPU path is **semantically equivalent to the CPU/reference path**: the
  shader applies the exact reference activation (bias/threshold/dead, inclusive
  `<=`) to each computed layer's output nodes and — on the first-layer
  dispatch — to the input nodes too. The only residual divergence is
  fp32 (GPU) vs fp64 (CPU) accumulation precision, within the accepted
  tolerance.
- Separate `VkBuffer`s per arena (`payload` / `scale` / `offset` / `special`),
  each bound at offset 0; per-entry 4-byte alignment.
- Lazy init; precompiled SPIR-V lives in `nn/shaders/spv/` (the `.comp` files
  are dev-only sources — see `nn/shaders/spv/README.md`).
- GPU worker thread (love.thread when available, same-thread passthrough
  otherwise): plain io paths only, **no FFI cdata across threads**, fence drain
  + `vkDeviceWaitIdle` on shutdown.
- FFI ABI golden checks pin struct sizes/offsets to the Vulkan headers
  (`ABI_MISMATCH` instead of UB on a toolchain mismatch).

### Game integration: `nn/compat_ai_module.lua`

An adapter exposing the game's `ai_module` surface (`genWeights`,
`mutateWeights`, `run`, `acquire`, `release`) implemented on top of `nn/`
(cpu/reference backends only). It is behaviorally equivalent (bit-exact fp32)
to the existing `ai_module.lua`, purely additive, and does **not** modify the
game's current AI path.

### Usage

```lua
local nn = require('nn')

-- One backend is resolved once at init: "cpu" | "gpu" (never "auto").
local runtime, err = nn.new({
    backend       = 'auto',    -- "cpu" | "gpu" | "auto"
    deterministic = false,     -- true forces the cpu backend
    precision     = 'fp8',     -- fp32 | fp16 | fp8 | fp4 | fp2
    block_size    = 16,        -- 8 | 16 | 32 | 64
    -- topology = ...          -- optional; must match the game's seed/spore/sprout
})
if not runtime then error(err.message) end

-- Forward one genome (weights) through a network. Returns out, or nil, err.
local out = runtime:forward('seed', genome, inputs)

-- Caller-owned output buffer: no allocation, no retention.
runtime:forward_into('seed', genome, inputs, out)

-- Production batch path (writes into out_desc.buffer, caller-owned).
runtime:forward_batch('seed', batch_items, in_desc, out_desc)

-- Precision switch (gated; see the set_precision rules above).
local ok, err = runtime:set_precision('fp16', { experimental = false })

local caps = runtime:capabilities()   -- backend feature descriptor
runtime:shutdown()

-- .nnw serialization (plain io; no love.filesystem).
nn.save('model.nnw', model)
local model = nn.load('model.nnw', runtime)
```

## Tests

From the project root:

```
lua tests/test_runner.lua        # or: luajit tests/test_runner.lua
```

This runs the full suite (the runner plus auto-discovered `tests/test_*.lua`).
Expect `90/90 passed` and exit code 0. The runner is pure Lua and works under
both PUC-Lua and LuaJIT with no LÖVE dependency.
