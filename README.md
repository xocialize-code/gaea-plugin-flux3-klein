# gaea-plugin-flux3-klein

Black Forest Labs' **FLUX 3 Image Klein** (a 5.55B flow-map DiT with Qwen3-VL-4B conditioning, under the **FLUX
Kommunity licence**) as a [Gaea](https://github.com/xocialize/gaea) research plugin. It is built on
[GaeaPluginKit](https://github.com/xocialize/gaea-plugin-kit) over
[flux3-klein-swift](https://github.com/xocialize/flux3-klein-swift), runs in its own process, and speaks the Gaea
Plugin Protocol v1. Add it to Gaea by its repo URL:

```bash
gaea plugins add https://github.com/xocialize/gaea-plugin-flux3-klein
```

> **Licence first.** The weights are not open: non-commercial use, and commercial use of *outputs* only for Qualifying
> Users (under US$5M revenue) who filter content and disclose AI generation. Outputs must never train another image
> model. Gaea installs the model **off and named-only**: an operator turns it on, and a request runs it only by naming
> it. See [Licence](#licence).

## What it offers

| Model | MES operation | Quality tiers | Notes |
|---|---|---|---|
| `flux-3-klein` | `image.generate` | standard, best | 16 steps without CFG: a negative prompt does nothing and is reported `negative_prompt_ignored`. Strong English text rendering and bounding-box layout. |
| `flux-3-klein` | `image.edit` | standard, best | Editing and composition from 1–10 images, named in the instruction in order ("the second image"). Describe the whole result with the edit folded in. Drawn from noise at the first image's size, so `strength` does nothing (`param_ignored`). |

Both offers are named-only and opt-in: the normalizer never picks FLUX 3 Klein by default.

- **Prompts.** The model wants a full scene caption: medium and style, the scene, lighting and camera, and every
  subject with its colours, materials and place in the frame. Short requests are fine, since BFL's prompt upsampler
  expands each one first (about 10 s). Put text to render in quotes, e.g. a sign reading "OPEN".
- **Sizes.** Sides of at least 64 px in multiples of 16, at most 2048² in area (the area the package's footprints were
  measured at). Anything else is fitted, keeping the aspect, and reported `size_snapped`. `aspect` with `megapixels`
  works too. The default is 1024². On `image.edit` the package fits one reference under 2048² and several each under
  1 MP itself.
- **`count`** on `image.generate`: 1–8 images, each with `seed + i`. A request without a seed gets one, reported in
  `effective`. `image.edit` has one output. The package keeps a seed's low 32 bits.
- **Options**, as MES extensions:
  - `x-flux3.steps`: 1–100, default 16.
  - `x-flux3.upsample`: `false` sends the prompt as typed, for a caller that wrote the full caption. Absent or `true`,
    the upsampler runs. It goes to the package as `metaData["upsample"]`, as mlx-image-runtime's request key does.
  - `x-flux3.boxes`: layout boxes on 0–1 coordinates, `{id, bbox: [y_min, x_min, y_max, x_max], desc}`, or box edits
    `{id, from: ref_image_n | target, src_bbox, tgt_bbox (null removes), desc}`. They go to `metaData["boxes"]`.

### The upsampler and the safety checks

Each lane is configured as mlx-image-runtime's catalog does by default:

- **Prompt upsampler: int8** (BFL's `flux-3-image-klein-pup`, a Qwen3.5-9B VL fine-tune, Apache-2.0). It loads per
  request and is released before the encoder. Its 11.1 GB transient stays under the declared activation.
- **Safety checks: off**, as BFL ships them and as the catalog. On, they would check the prompt, each reference and
  the output for copyrighted characters, brands, public figures and adult content. They are the licence's §2(e)
  content-filter route for a Qualifying User's commercial outputs; reviewing each output before it is shared is the
  other route. This plugin does not offer them yet.

## Lanes and memory

Three lanes, one per DiT quant, all made at load from the bf16 snapshot. Footprints are the package's, measured at the
2048² envelope's worst request (ten references into 2048²):

| Lane | Resident | Peak while running | Gaea schedules |
|---|---|---|---|
| `bf16` (default) | 11.7 GB | 12.0 GB | 23.7 GB |
| `int8` | 6.7 GB | 12.0 GB | 18.7 GB |
| `int4` (8-bit attention) | 4.9 GB | 12.0 GB | 16.9 GB |

bf16 is the default, the catalog's choice from 48 GB. Every lane's text is within the pre-registered OCR tolerance of
bf16; int8 draws the same image as bf16 at a seed, int4 a different one. When bf16 doesn't fit, Gaea loads a smaller
lane and warns (`lane_fallback`, G-23); `gaea models lane flux-3-klein int4` makes another lane the default. A request
naming a lane (`x-gaea.lane`) never falls back.

## Weights

Gaea sends its models root in `configure` (MLXEngine's store by default, `gaea models root` to change it). The worker
looks for these there, and downloads a missing snapshot on `prepare` or `materialize`:

- `mlx-community/flux-3-image-klein-bf16`: BFL's bf16 package, byte for byte, pinned by commit. It is **gated** like
  upstream. A first download needs a Hugging Face token whose account has accepted the FLUX Kommunity licence; the
  engine reads `HF_TOKEN`, then the Keychain, then the CLI's token file.
- `mlx-community/flux-3-image-klein-pup-8bit`: the int8 prompt upsampler, public, Apache-2.0.

## Build and check locally

```bash
scripts/build-plugin.sh
```

```bash
scripts/check-plugin.sh
```

`check-plugin.sh` generates one 512² image with the int4 lane and `x-flux3.upsample: false` (the upsampler never
loads) against the weights in MLXEngine's store (`CHECK_MODELS_ROOT` moves it), with every `.build` folder hidden so
MLX can find its metallib only inside the bundle. A control then shows that the bare worker, outside the bundle, can't.
`scripts/release.sh` builds, checks, notarizes with the fleet's `mlx-notary` profile, and publishes the GitHub release.
See the [demo plugin](https://github.com/xocialize/gaea-plugin-demo) for signing, notarization and CI secrets.

`flux3-inproc` (`swift run -c release flux3-inproc`; its header lists the options) is not part of the plugin. It runs
the same lane in-process through MLXServeEngine, the reference that Gaea's runs are compared against byte for byte.
It builds its request with the same library functions as the worker: `--width`/`--height` are snapped the same way,
and `--no-upsample` is `x-flux3.upsample: false`.

## Licence

- **The weights: FLUX Kommunity License v1.0** (Black Forest Labs, `LicenseRef-FLUX-Kommunity-1.0`,
  [terms](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)).
  - Non-commercial use is allowed.
  - Commercial use of **outputs** is allowed only for Qualifying Users: under US$5M gross annualized revenue,
    affiliates included. They must filter content or review each output before sharing it, and disclose AI generation
    where the law requires it.
  - **Never use outputs to train or improve another image model** (§4(b)).
  - The model itself may not go into a sold product. That needs a licence from BFL.
- **In Gaea.** The offers say `commercial_use: false` and carry the Kommunity notice, which names the Qualifying-User
  exception: Gaea can't tell whether its operator qualifies (AB-D-0151). Because the licence is not permissive, Gaea
  makes the model **opt-in and named-only**. What a user does with the outputs, including whether they qualify, is
  theirs to decide.
- **The plugin code** is Apache-2.0 (Gaea G-21), and flux3-klein-swift is MIT. The prompt upsampler and the text
  encoder (Qwen3-VL-4B) are Apache-2.0. This plugin doesn't redistribute any weights: the worker downloads them from
  Hugging Face.
