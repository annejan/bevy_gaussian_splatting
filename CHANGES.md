# martin branch — change log

This is the **`martin` branch** of `annejan/bevy_gaussian_splatting`: upstream
**8.0.0 (Bevy 0.19)** plus the edits below. The [martin](https://github.com/annejan/martin)
demo engine consumes this branch as a git dependency via `[patch.crates-io]` in its
`Cargo.toml` (`Cargo.lock` pins the exact commit).

Rebased 7.1.0 → 8.0.0 (Bevy 0.18 → **0.19**) on 2026-06-25: upstream's #230 (Bevy 0.19)
+ #235 landed; all four of our edit commits replayed onto upstream/main with **zero
conflicts** — the edits are WGSL shaders (`gaussian.wgsl`/`bindings.wgsl`/`interpolate.wgsl`,
version-independent) plus small additive Rust hunks in `settings.rs`/`io/scene.rs`/
`render/mod.rs` that 0.19 didn't touch.

Rebased 7.0.2 → 7.1.0 on 2026-06-22: §3 (the radix sort speedup) is now UPSTREAM
(merged as #229), so it is no longer a fork edit — the conflicting radix hunks were
resolved in favour of upstream (which also gained configurable depth precision on
top of our change). The remaining edits (§1–2, §4–8) replay cleanly.

This file lists every edit on top of upstream, so the set can be re-applied (or
dropped) when rebasing onto a new upstream release. Keep it current when you touch
this branch. To see the raw diff against upstream:

```bash
git diff main   # main tracks mosure/bevy_gaussian_splatting
```

Feature selection (e.g. `sh0`) is **not** a fork edit — it's set in martin's
`Cargo.toml` (`default-features = false, features = [...]`). This branch's
`Cargo.toml` is upstream.

---

## 1. Custom render-shader effects  (`src/render/gaussian.wgsl`)

Injected after `var position = get_position(...)`:

- **Explode displacement** — closed-form ballistic per-gaussian offset driven by
  `gaussian_uniforms.time`, gated `time != 0 && !interp_active` (so it only fires for
  plain explode-mode clouds, never a morph output).
- **Morph "ball cloud" pulse** — when `interp_active` (an interpolation range is set) and
  `gaussian_uniforms.bulge > 0`, routes each gaussian onto a fuzzy sphere by
  `sin(pi*t)` (zero at t=0/1, max mid) so a morph disperses into a ball and reassembles.

Both rely on `explode_hash3(splat_index)` (also added here).

*Why a fork:* the crate loads shaders by hardcoded UUID (`load_internal_asset!`) — there
is no app-side hook to extend the WGSL. This is the core reason the fork exists.

## 2. New uniform field `bulge`  (4 spots — must stay in sync)

Drives the ball-pulse amplitude per cloud. `encase`/std140 layout couples these:

- `src/render/bindings.wgsl` — `bulge: f32` in `struct GaussianUniforms` (after `time_stop`).
- `src/render/mod.rs` — `bulge: f32` in `pub struct CloudUniform` (after `time_stop`) **and**
  `bulge: settings.bulge` in its construction.
- `src/gaussian/settings.rs` — `pub bulge: f32` in `CloudSettings` + `bulge: 0.0` in `Default`.

> Fragile on upgrade: if upstream changes `CloudUniform`'s field order, re-check alignment.

## 3. Sort optimizations  (`src/sort/radix.{rs,wgsl}`, `src/render/mod.rs`) — ✅ MERGED UPSTREAM, NO LONGER A FORK EDIT

~2.4× faster radix sort on the iGPU, correctness preserved (LSD-stable, reads live GPU
positions → no holes). **Merged upstream as mosure/bevy_gaussian_splatting#229** and shipped in
7.1.0 (upstream then layered configurable depth precision on top). As of the 7.1.0 rebase this is
upstream code, not a fork edit — kept here only for the historical record. Original notes:

- `render/mod.rs` `ShaderDefines::default()` — `radix_digit_places = 2` (was `32/bits` = 4):
  16-bit depth key → halves the radix C-pass cost (65536 buckets is ample).
- `sort/radix.wgsl` `radix_sort_a` — store `key = key >> 16u` (sort the high 16 bits, to
  match the 2 passes).
- `sort/radix.wgsl` `radix_sort_c_scan_tiles` — `@workgroup_size(RADIX_BASE)`, lane = digit
  (was `@workgroup_size(1)` × RADIX_BASE single-lane workgroups, 1/64 wave occupancy).
- `sort/radix.wgsl` `radix_sort_c_scatter` — per-digit COUNT parallelized across all lanes
  (atomic into `tile_digit_counts`); the stable placement stays serial (LSD requires it).
- `sort/radix.rs` — `radix_sort_c_scan` dispatch `(1,1,1)` (was `(1, radix_base, 1)`);
  dropped the now-unused `radix_base` local.

## 4. Per-particle transition phase  (`gaussian.wgsl` + 4 uniform spots — opt-in, default-off)

Enables staggered *per-particle* transitions a single global `time` can't express
(typewriter, slither, sparkle, true vortex, hard wipe, **shockwave**). **`transition_mode == 0` is the
default and is byte-identical to upstream.** Append-only + default-off ⇒ a clean candidate to
**upstream as a PR**. Full reference: `SHADER-BLUEPRINT.md` in the martin repo.

Modes: 1 typewriter · 2 slither · 3 sparkle-in · 4 spark-out · 5 vortex · 6 directional-wipe ·
7 pen-write · **8 shockwave** (phase = `radial` distance from the centre → an expanding blast-front
reveal, so a shape materialises as a ring sweeping outward instead of a uniform converge). Adding a
mode = one `transition_phase` arm + one `tx_reveal`/position arm; mode 8 reuses the existing `radial`
and the `tx_reveal = local` opacity sink (no new math, no new uniform).

- `src/render/gaussian.wgsl` — `transition_phase(index, position) -> f32` helper (after
  `explode_hash3`); a gated branch in `vs_points` (after the ball-pulse) computing a moving
  window `local = saturate((gt*(1+softness) - phase)/softness)` → `tx_reveal` (opacity sink)
  or a position nudge (slither / vortex); and one factor at the color finalize,
  `opacity * global_opacity * tx_reveal` (`* 1.0` ⇒ no-op in mode 0). if/else-if, not switch (RADV).
- New uniform group (4 spots, like `bulge`): `transition_mode: u32`, `transition_softness: f32`,
  `transition_axis: u32`, `_transition_pad: u32` — appended **after `max: vec4`** (the true
  struct end, **NOT** after `bulge`) in `bindings.wgsl` `GaussianUniforms`, `mod.rs`
  `CloudUniform` (+ its construction), and `gaussian/settings.rs` `CloudSettings` (+ `Default`:
  mode 0 / softness 0.15 / axis 0).

> Lives in `vs_points` (runs *after* the sort) → invisible to the radix sort; do **not** move
> it into `interpolate.wgsl` (that buffer is what the sort reads). Pure fn of `splat_index` +
> position + uniforms ⇒ deterministic in record mode.

## 5. Persistent vertex deform  (`gaussian.wgsl` + 4 uniform spots — opt-in, default-off)

A continuous, *non-morph* displacement so a held shape keeps moving (a waving "wall of text":
wave/cloth/ripple/twist/**wind**). Unlike §4 (gated to `interp_active`, plays once over a morph), this is
driven by `deform_time` and runs **every frame**. **`deform_mode == 0` is the default and is
byte-identical to upstream.** Append-only + default-off ⇒ also a clean candidate to upstream.

- `src/render/gaussian.wgsl` — a gated branch in `vs_points` **after** the transition block and
  **before** `transformed_position` (so the deform is in object space, pre-transform). Displaces
  `position` by a per-mode function of its centred coords + `deform_time`. if/else-if, not switch.
  Modes 1-4 (wave/cloth/ripple/twist); **5** wind (gusting sway), **6** turbulence (churning 3D
  field), **7** pulse (breathe in/out about the centre), **8** jitter (fast per-particle shake), **9**
  spiral (radial pinwheel about the vertical axis). All append-only `else if` branches gated by
  `deform_mode != 0u`, so mode 0 stays byte-identical to upstream.
- New uniform group (4 spots, like §2/§4): `deform_mode: u32`, `deform_amp: f32`,
  `deform_freq: f32`, `deform_time: f32` — a **second 16-byte block appended after the transition
  group** (`_transition_pad`) in `bindings.wgsl` `GaussianUniforms`, `mod.rs` `CloudUniform`
  (+ its construction), and `gaussian/settings.rs` `CloudSettings` (+ `Default`: all 0).

> Same sort caveat as §4: lives in `vs_points`, invisible to the radix sort. Amplitudes are small
> so the depth-sort error is negligible. The app advances `deform_time` from the show clock.

## 6. Swarm morph detour  (`gaussian.wgsl` + 4 uniform spots — opt-in, default-off)

`swarm: f32` (0 = off → byte-identical). During a morph (`interp_active`) each gaussian curls along
its own pseudo-random + tangential (about the vertical axis) direction by `sin(pi*t)*swarm` — peaks
mid-transition, **exactly zero at t=0/t=1** so both endpoints stay pixel-exact, like §2's ball-pulse.
The effect: a shape→shape morph *flocks/swarms* between the two scenes instead of lerping straight
(the app's `~swarm` transition). Same injection point as §1/§2, right after the ball-pulse block.

- New uniform group (4 spots, like §2/§4/§5): `swarm: f32` + three `_swarm_pad` f32 — a **third
  16-byte block appended after the deform group** in `bindings.wgsl` `GaussianUniforms`, `mod.rs`
  `CloudUniform` (+ its construction), and `gaussian/settings.rs` `CloudSettings` (+ `Default`: 0).

> Same sort caveat as §2/§4. Driven by the morph `time`, so it self-cancels at the endpoints.

## 7. Don't claim gltf/glb extensions in the scene loader  (`src/io/scene.rs` — martin app need)

`GaussianSceneLoader` (KHR_gaussian_splatting glTF) claimed `["gltf","glb"]`, shadowing Bevy's native
`GltfLoader`. martin loads its splats from `.ply` and wants Bevy's glTF loader free so it can render
**real PBR meshes alongside the splats** (the app's `model:` compose source — meshes + splats share
the camera + the depth test, so they composite). Changed `extensions()` to return `&[]`; the loader
still exists for explicit type-loads, it just no longer grabs the extension. (One-line, reversible.)

## 8. Staggered morph timing  (`morph/interpolate.wgsl` + 1 uniform spot — opt-in, default-off)

`morph_stagger: f32` (0 = off → byte-identical). The interpolate compute shader lerped every gaussian
with one global blend factor `t`, so a morph slid the whole cloud as a synchronized block — which reads
as straight-line **streaks**. With stagger > 0, each gaussian morphs over its own sub-window of the
factor: `offset = hash(index)*stagger`, `width = 1-stagger`, `t_i = smoothstep(clamp((t-offset)/width))`.
Early-offset splats arrive while late ones haven't left, so the cloud **dissolves + reforms** instead of
sliding (a soft "cloudy" transition). `stagger == 0` short-circuits to the plain global `t` → identical
to upstream. Endpoints stay exact (t=0 → all at lhs, t=1 → all at rhs).

- Reuses the §6 swarm group's first pad slot — `_swarm_pad0` → `morph_stagger` in `bindings.wgsl`,
  `mod.rs` `CloudUniform`, and `settings.rs` `CloudSettings`. **No new 16-byte block, no offset moves.**

> Same sort caveat as §2/§6 (lives in the interpolate pass, invisible to the radix sort); the per-splat
> timing spread is bounded so the depth-sort error stays small. The app drives it from a show knob
> (`MARTIN_MORPH_STAGGER` / `.show` `morph_stagger =`).

## 9. Tighter splat-quad extent for synthetic content — 3.0σ → 2.4σ, **gated on `SH_DEGREE`**  (`render/gaussian.wgsl`)

A fill-rate win for overdraw-bound GPUs. The vertex shader sizes each splat's quad to a `cutoff`-σ
radius: `sqrt(9.0 + 2·log(opacity))` (= **3.0σ** at full opacity; `9.0 = 3²`) with `OPACITY_ADAPTIVE_RADIUS`,
else a flat `3.0`. The 2.4–3.0σ ring is near-zero alpha — `exp(-0.5·2.4²) ≈ 5.6 %` of peak, ×opacity —
so for **synthetic** content (text/morph/procedural) it's rasterised + alpha-blended for almost nothing.
Tightening to **2.4σ** (`9.0 → 5.76 = 2.4²`, flat `3.0 → 2.4`) cuts the quad **pixel area ~(2.4/3)² ≈ 36 %**.

**Gated `#if SH_DEGREE > 0`** (the sh3 build = real captures) → those keep the full **3.0σ**, because
anisotropic photogrammetry splats lean on the wider tails for smooth inter-splat blending: A/B on an
aerial city capture showed 2.4σ visibly **thinned** it (sparser foliage, eroded footprint). So the
default (sh3/captures) == upstream behaviour; only the synthetic sh0 build opts into the tighter quad.

- Measured on the martin "PonyCamp" sh0 stage (AMD 860M iGPU): climax **720p 30→40 fps (+32 %)**,
  854×480 44.6→54.4. No visible change on synthetic content (text/props/fire). Unlike shrinking
  `global_scale`, it does **not** shrink the gaussians, so no thinning/gaps.
- 2.2σ (`4.84`) was also clean on synthetic and ~+10 % more, but 2.4σ keeps a brand-text margin.

> No sort interaction (vertex-stage quad size only). The fragment-side ellipse test (`gaussian_2d.wgsl`)
> uses the same `cutoff`, so the discard boundary tightens consistently — no half-drawn splats.
> For a future upstream PR this should be a `CloudSettings` knob (`quad_cutoff: f32`, default 3.0) rather
> than a compile-time SH gate — then a single show could mix full-σ captures + tight-σ synthetic props.

---

## 10. Additive/emissive blend mode  (`gaussian/settings.rs` + `render/mod.rs` — opt-in, default-off)

A per-cloud `CloudSettings.additive: bool` (default `false` → **byte-identical** to upstream). When
`true`, the render pipeline composites with `One + One` (add) instead of `PREMULTIPLIED_ALPHA_BLENDING`:
overlapping (premultiplied) fragments **accumulate light** rather than alpha-over-saturating, so a cloud
of translucent gaussians **glows** on a dark background — the demoscene nebula/neon look — instead of
reading as a solid opaque blob. There is no occlusion in additive mode (everything shows through), so
it's meant for glow content on black; solid captures should stay `false`.

Wiring (3 spots, all additive, no behaviour change when off):
- `CloudSettings` gains `pub additive: bool` (+ `false` in `Default`).
- `CloudPipelineKey` gains `pub additive: bool` — so the two blend variants specialize to distinct
  pipelines (blend is fixed-function pipeline state, not shader-selectable).
- `specialize()` picks the `BlendState` on `key.additive`; the render-queue key reads `settings.additive`.
  (The compute/interpolate key uses `..Default::default()` → `false`; blend is irrelevant to a compute pass.)

A/B on the procedural `galaxy` (martin, 120k splats, on black + bloom): alpha-over rendered a flat muddy
opaque disk; additive rendered a glowing spiral with visible arms + a white-hot core. Consumed by martin
via `MARTIN_ADDITIVE` / a `.show [settings] additive = 1`.

> For a future upstream PR: `additive` could generalise to a `blend_mode` enum (AlphaOver | Additive |
> …) on `CloudSettings`, same specialization pattern.

---

## Not a fork edit (for reference)
- `sh0` vs `sh3`: feature selection in martin's `Cargo.toml`.
- `assets/font.ttf`, `build_text_gaussians`, the `GaussianInterpolate` morph, the
  `MARTIN_SEQ` timeline: all live in the **app** (martin's `src/`) and use the crate's
  public API — zero crate changes.
