# Third-party code

Rexcomp and the AArch64 kit are by NotRexed. This repository includes:

| What | Where | License |
|---|---|---|
| suyu's AArch64 -> C translator (`arm64_to_c.h`, unmodified), from the mk8-recomp fork of suyu (`dougchansan/suyu-v0.0.4`, branch `mk8-recomp`, commit `89c5e72412b28c788557991a86f471ea3332f62a`) | `a64-kit/lift/suyu/` | GPL-2.0-or-later (texts next to it) |
| suyu / yuzu shader recompiler, common and video_core parts, with Rexcomp's WGSL backend and changes | `a64-kit/gpu/third_party/suyu/` | GPL-2.0-or-later and GPL-3.0-or-later per file (SPDX headers; texts next to it) |
| Small stand-ins for boost, fmt and ankerl::unordered_dense headers | `a64-kit/gpu/compat/` | see each file |
| wgpu-native v29.0.1.1, Windows x64 release | `third_party/wgpu-native/` | MIT or Apache-2.0 (`LICENSE.md` there) |
| SDL2 2.32.10 Windows x64 runtime | `third_party/sdl2/` | zlib (`LICENSE.txt` there) |
| NVN API types reconstructed from Darksiders II debug information by h1k421 (as adapted for TOTK by jhaizhou) | `a64-kit/gpu/render/nvn_api.h` | MIT (text in the file) |
| The native NVN layer's design, object layouts, command records and TOTK findings, after the NVN layer of d2r-mobile / MacNativeGames by jhaizhou (github.com/jhaizhou-ops/d2r-mobile, github.com/ly05010419/MacNativeGames) | `a64-kit/hos/nvn_native.c`, `a64-kit/gpu/render/rx_nvn.*` | MIT |
| NVN texture storage layout, from deko3d `source/dk_image.cpp` by fincs (via the layer above) | `a64-kit/hos/nvn_layout.c` | zlib |
| AMD FidelityFX Super Resolution 1 (EASU, RCAS), ported to WGSL | `a64-kit/gpu/render/rx_present_fx.cpp` | MIT (text in the file) |
| RenderDoc's in-application API header (`renderdoc_app.h`, RenderDoc 1.46, unmodified) | `a64-kit/gpu/render/renderdoc_app.h` | MIT (text in the file) |
| Zstandard 1.5.6 decoder (the single-file `zstddeclib.c` its release makes) | `third_party/zstd/` | BSD (`LICENSE` there) |
| Where TOTK v1.0.0 sets its camera aspect and HUD scale, after TOTK-Ultrawide by Fruithapje21 (github.com/Fruithapje21/TOTK-Ultrawide) | `a64-kit/hos/totk_ultrawide*.c` | GPL-3.0 (locations only; the code is ours) |

Because of the GPL parts, `a64emit.dll`, the generated runtime, the lifted C and every game build made with this kit
are GPL. Game data and code are Nintendo's and are not included.

- `third_party/opus`: libopus 1.5.2 (https://opus-codec.org), BSD 3-clause; see third_party/opus/COPYING.
