# Life Simulator 5

A simple simulator of cellular natural selection.

## Running

Download [Love2D](https://love2d.org/), add it to PATH, then run `love` on the repo.

## Keybinds

- `Space` — Pause/Unpause
- `Up` / `Down` — Increase/decrease simulation speed
- `U` — Hide/show interface
- `E` — Next view mode (Normal / Map Minerals)
- `R` — Regenerate map

## Configuration

All simulation properties live in `shares.lua`: map size and day cycle,
sun/mineral ranges, cell costs/colors/lifespans, AI topologies (`AI_LAYERS_*`),
genome lengths (`AI_LEN_*`), per-network offsets (`AI_OFFSET_*`), and the
shared map/cell/genome state tables.

## Gameplay

A grid of cells (Leaf, Root, Stem, Seed, Spore, Sprout) harvest energy and
minerals, grow, and reproduce. AI cells carry a per-cell genome — the three
networks concatenated into one common genome of `AI_LEN_COMMON = 1003` doubles,
each network occupying its slice (`AI_OFFSET_SEED/SPORE/SPROUT`). Topologies:
seed `9-12-1`, spore `6-16-1`, sprout `9-18-16-3`.

Activation per non-input node: `value = value + bias; if value <= threshold
then value = dead_value` (inclusive `<=`; NaN propagates).

## Neural network (`nn/`)

`nn/` is a standalone neural package (no LÖVE/game deps) implementing the
game's networks with dynamic quantization, versioned `.nnw` serialization, a
determinism/corpus-gate model, and an optional Vulkan GPU backend. One backend
resolves once at init: `"cpu" | "gpu"`. It is reached only by the test suite;
the game's AI path is `ai_module.lua`. The full backend/quantization/
serialization/Vulkan contract is documented in the source comments of
`nn/` (read `nn/api.lua` first).

## Tests

From the project root:

```
lua tests/test_runner.lua        # or: luajit tests/test_runner.lua
```

Expect `98/98 passed` and exit code 0. Pure Lua, no LÖVE dependency.