# nn/shaders/spv — compiled SPIR-V binaries (build artifacts)

The `.spv` files in this directory are **generated build artifacts**, not source.
They are produced from the GLSL in `../` (e.g. `forward_packed.comp`) and are
safe to delete and regenerate. Do not hand-edit them.

## Recompile command

```sh
# from the repo root (or any directory with the shaders + this dir):
glslangValidator -V --target-env vulkan1.2 -o nn/shaders/spv/forward_packed.spv nn/shaders/forward_packed.comp
glslangValidator -V --target-env vulkan1.2 -o nn/shaders/spv/forward_fp32.spv  nn/shaders/forward_fp32.comp
glslangValidator -V --target-env vulkan1.2 -o nn/shaders/spv/forward_fp16.spv  nn/shaders/forward_fp16.comp
```

- Target: **SPIR-V 1.5**, Vulkan environment `vulkan1.2` (`--target-env vulkan1.2`).
- The fp16 variant additionally requires `GL_EXT_shader_explicit_arithmetic_types_float16`
  and the runtime extension `VK_KHR_shader_float16_int8`; it is **not enabled by
  default** (see `nn/vulkan.lua` capabilities: `f16_storage` vs `f16_arithmetic`).

## Runtime behavior

`nn/vulkan_pipelines.lua` loads these files at pipeline-build time
(`load_spv`). If a `.spv` is missing, pipeline creation fails with a structured
`VULKAN_PIPELINE_FAILED` error — it never crashes and never silently falls back
to a different shader (a `shader_abi` mismatch is likewise rejected).

## ABI contract

Each `.comp` carries a `shader_abi:` version in its header comment. Pipelines
are keyed by the quant ABI version in the profile; bump the version on any
breaking change to the binding table, config SSBO layout, or decode semantics.
