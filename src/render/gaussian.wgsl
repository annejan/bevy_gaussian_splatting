#import bevy_gaussian_splatting::bindings::{
    view,
    gaussian_uniforms,
    Entry,
}
#import bevy_gaussian_splatting::classification::class_to_rgb
#import bevy_gaussian_splatting::depth::depth_to_rgb
#import bevy_gaussian_splatting::optical_flow::{
    calculate_motion_vector,
    optical_flow_to_rgb,
}
#import bevy_gaussian_splatting::helpers::{
    get_rotation_matrix,
    get_scale_matrix,
}
#import bevy_gaussian_splatting::transform::{
    world_to_clip,
    in_frustum,
}

#ifdef GAUSSIAN_2D
    #import bevy_gaussian_splatting::gaussian_2d::{
        compute_cov2d_surfel,
        get_bounding_box_cov2d,
        surfel_fragment_power,
    }
#else ifdef GAUSSIAN_3D
    #import bevy_gaussian_splatting::gaussian_3d::{
        compute_cov2d_3dgs,
    }
    #import bevy_gaussian_splatting::helpers::{
        get_bounding_box_clip,
    }
#else ifdef GAUSSIAN_4D
    #import bevy_gaussian_splatting::gaussian_4d::{
        conditional_cov3d,
    }
    #import bevy_gaussian_splatting::helpers::{
        cov2d,
        get_bounding_box_clip,
    }
#endif

#ifdef PACKED
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::packed::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::packed::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_STORAGE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_cov3d,
        }
    #else
        #import bevy_gaussian_splatting::planar::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
        }
    #endif
#else ifdef BUFFER_TEXTURE
    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::texture::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_cov3d,
            location,
        }
    #else
        #import bevy_gaussian_splatting::texture::{
            get_position,
            get_color,
            get_visibility,
            get_opacity,
            get_rotation,
            get_scale,
            location,
        }
    #endif
#endif

#ifdef BUFFER_STORAGE
    @group(3) @binding(0) var<storage, read> sorted_entries: array<Entry>;
    fn get_entry(index: u32) -> Entry {
        return sorted_entries[index];
    }
#else ifdef BUFFER_TEXTURE
    @group(3) @binding(0) var sorted_entries: texture_2d<u32>;
    fn get_entry(index: u32) -> Entry {
        let sample = textureLoad(
            sorted_entries,
            location(index),
            0,
        );

        return Entry(
            sample.r,
            sample.g,
        );
    }
#endif

#ifdef WEBGL2
    struct GaussianVertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
        @location(1) uv: vec2<f32>,
    #ifdef GAUSSIAN_2D
        @location(2) local_to_pixel_u: vec3<f32>,
        @location(3) local_to_pixel_v: vec3<f32>,
        @location(4) local_to_pixel_w: vec3<f32>,
        @location(5) mean_2d: vec2<f32>,
        @location(6) radius: vec2<f32>,
    #else #ifdef GAUSSIAN_3D
        @location(2) conic: vec3<f32>,
        @location(3) major_minor: vec2<f32>,
    #else #ifdef GAUSSIAN_4D
        @location(2) conic: vec3<f32>,
        @location(3) major_minor: vec2<f32>,
    #endif
    };
#else
    struct GaussianVertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) @interpolate(flat) color: vec4<f32>,
        @location(1) @interpolate(linear) uv: vec2<f32>,
    #ifdef GAUSSIAN_2D
        @location(2) @interpolate(flat) local_to_pixel_u: vec3<f32>,
        @location(3) @interpolate(flat) local_to_pixel_v: vec3<f32>,
        @location(4) @interpolate(flat) local_to_pixel_w: vec3<f32>,
        @location(5) @interpolate(flat) mean_2d: vec2<f32>,
        @location(6) @interpolate(flat) radius: vec2<f32>,
    #else ifdef GAUSSIAN_3D
        @location(2) @interpolate(flat) conic: vec3<f32>,
        @location(3) @interpolate(linear) major_minor: vec2<f32>,
    #else ifdef GAUSSIAN_4D
        @location(2) @interpolate(flat) conic: vec3<f32>,
        @location(3) @interpolate(linear) major_minor: vec2<f32>,
    #endif
    };
#endif

fn world_to_local_direction(ray_direction_world: vec3<f32>, transform: mat4x4<f32>) -> vec3<f32> {
    let basis = mat3x3<f32>(
        transform[0].xyz,
        transform[1].xyz,
        transform[2].xyz,
    );
    let basis_x = normalize(basis[0]);
    let basis_y = normalize(basis[1]);
    let basis_z = normalize(basis[2]);

    let local = vec3<f32>(
        dot(basis_x, ray_direction_world),
        dot(basis_y, ray_direction_world),
        dot(basis_z, ray_direction_world),
    );

    return normalize(local);
}

// per-splat pseudo-random vec3 in [0,1)^3 (for varied explosion speed + noise)
fn explode_hash3(i: u32) -> vec3<f32> {
    var n = i * 1664525u + 1013904223u;
    n = (n ^ (n >> 16u)) * 2246822519u;
    let x = f32(n & 0x3FFu) / 1023.0;
    n = (n ^ (n >> 13u)) * 3266489917u;
    let y = f32(n & 0x3FFu) / 1023.0;
    n = (n ^ (n >> 16u)) * 668265263u;
    let z = f32(n & 0x3FFu) / 1023.0;
    return vec3<f32>(x, y, z);
}

// --- per-particle transition phase in [0,1] for staggered transitions (typewriter,
//     slither, sparkle, vortex, directional-wipe). A PURE function of splat_index +
//     position + uniforms (no wall-clock, no RNG state), so it is deterministic in
//     record mode. Never called in mode 0 (the caller guards transition_mode != 0u), so
//     mode 0 stays byte-identical to upstream. if/else-if, NOT switch (RADV). ---
fn transition_phase(index: u32, position: vec3<f32>) -> f32 {
    let mode = gaussian_uniforms.transition_mode;
    let center = (gaussian_uniforms.min.xyz + gaussian_uniforms.max.xyz) * 0.5;
    let extent = max(gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz, vec3<f32>(1e-6));
    let axis = gaussian_uniforms.transition_axis;          // 0=x, 1=y, 2=z
    var norm_axis = (position.x - gaussian_uniforms.min.x) / extent.x;
    if (axis == 1u) { norm_axis = (position.y - gaussian_uniforms.min.y) / extent.y; }
    else if (axis == 2u) { norm_axis = (position.z - gaussian_uniforms.min.z) / extent.z; }
    let radius = max(length(extent) * 0.5, 1e-4);
    let radial = clamp(length(position - center) / radius, 0.0, 1.0);
    let hashed = explode_hash3(index).x;
    if (mode == 1u) { return clamp(norm_axis, 0.0, 1.0); }       // typewriter (axis reveal)
    else if (mode == 2u) { return clamp(norm_axis, 0.0, 1.0); }  // slither (staggered along axis)
    else if (mode == 3u) { return hashed; }                      // sparkle-in
    else if (mode == 4u) { return hashed; }                      // spark-out (inverted at the sink)
    else if (mode == 5u) { return radial; }                      // vortex-true (unwind by radius)
    else if (mode == 6u) { return clamp(norm_axis, 0.0, 1.0); }  // directional-wipe HARD
    else if (mode == 7u) { return clamp(position.z, 0.0, 1.0); } // pen-write (baked z; see blueprint §9)
    return 0.0;
}

@vertex
fn vs_points(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
) -> GaussianVertexOutput {
    var output: GaussianVertexOutput;

    let entry = get_entry(instance_index);
    let splat_index = entry.value;

    var discard_quad = false;

    discard_quad |= entry.key == 0xFFFFFFFFu; // || splat_index == 0u;

    var position = vec4<f32>(get_position(splat_index), 1.0);

    // --- explosion: closed-form ballistic displacement in LOCAL space, object-relative,
    //     no-op at time == 0 (exact reset). Driven by gaussian_uniforms.time. ---
    // Skip when an interpolation range is set (time_stop > time_start): there `time` is a
    // GaussianInterpolate BLEND factor (morph output), not an explode clock — displacing
    // it would detonate the morph on top of the blend.
    let explode_t = gaussian_uniforms.time;
    let interp_active = gaussian_uniforms.time_stop > gaussian_uniforms.time_start;
    if (explode_t != 0.0 && !interp_active) {
        let center = (gaussian_uniforms.min.xyz + gaussian_uniforms.max.xyz) * 0.5;
        let radius = max(length(gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz) * 0.5, 1e-4);
        let rnd = explode_hash3(splat_index);
        let jitter = (rnd - vec3<f32>(0.5)) * radius * 0.6;
        let dir = normalize((position.xyz - center) + jitter + vec3<f32>(1e-5));
        let speed = radius * mix(0.7, 2.2, rnd.x);
        let noise = (rnd - vec3<f32>(0.5)) * radius * 0.7;
        // sign of time: + = outward (scatter, for the reformer's start state),
        //               - = inward (collapse/implode, for the Martins).
        var disp = dir * speed * explode_t + noise * explode_t;
        if (explode_t > 0.0) {
            let gravity = vec3<f32>(0.0, -radius * 0.2, 0.0);
            disp = disp + 0.5 * gravity * explode_t * explode_t;
        }
        position = vec4<f32>(position.xyz + disp, 1.0);
    }

    // --- morph through a BALL CLOUD: for a GaussianInterpolate output (interp_active),
    //     route each gaussian onto a fuzzy filled sphere by sin(pi*t) — peaks at the
    //     blend midpoint, EXACTLY zero at t=0/t=1 — so the shape disperses into a
    //     compact ball of particles then reassembles into the (already-interpolated)
    //     target. Kept within ~object radius (gaussian_uniforms.bulge) so it stays
    //     compact: far-flung splats spread over the whole screen and defeat the
    //     renderer's opacity early-out, which is what made the old radial blast slow. ---
    if (interp_active && gaussian_uniforms.bulge > 0.0) {
        let denom = max(gaussian_uniforms.time_stop - gaussian_uniforms.time_start, 1e-6);
        let mt = clamp((gaussian_uniforms.time - gaussian_uniforms.time_start) / denom, 0.0, 1.0);
        let pulse = sin(mt * 3.1415927);
        if (pulse > 0.0) {
            let center = (gaussian_uniforms.min.xyz + gaussian_uniforms.max.xyz) * 0.5;
            let radius = max(length(gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz) * 0.5, 1e-4);
            let rnd = explode_hash3(splat_index);
            // direction biased by the particle's own offset (coherent flow into the
            // ball) + jitter; radius varies per particle for a fuzzy FILLED ball.
            let dir = normalize((position.xyz - center) + (rnd - vec3<f32>(0.5)) * radius * 0.6 + vec3<f32>(1e-5));
            let ball_r = radius * gaussian_uniforms.bulge * mix(0.35, 1.0, rnd.x);
            let ball_pos = center + dir * ball_r;
            position = vec4<f32>(mix(position.xyz, ball_pos, pulse), 1.0);
        }
    }

    // --- SWARM (martin fork): a per-particle swirling detour during a morph. Like the ball-pulse
    //     it rides sin(pi*t) — EXACTLY zero at t=0/t=1, so both endpoints stay pixel-exact — but
    //     instead of collapsing to a ball each gaussian curls along its own pseudo-random +
    //     tangential (about the vertical axis) direction, so a shape→shape morph flocks/swarms
    //     between the two scenes. swarm == 0 skips the block (byte-identical to upstream). ---
    if (interp_active && gaussian_uniforms.swarm > 0.0) {
        let denom = max(gaussian_uniforms.time_stop - gaussian_uniforms.time_start, 1e-6);
        let mt = clamp((gaussian_uniforms.time - gaussian_uniforms.time_start) / denom, 0.0, 1.0);
        let pulse = sin(mt * 3.1415927);
        if (pulse > 0.0) {
            let radius = max(length(gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz) * 0.5, 1e-4);
            let rnd = explode_hash3(splat_index);
            let jitter = (rnd - vec3<f32>(0.5)) * 2.0;
            // tangential swirl about the vertical (Y) axis → coherent flocking, not pure noise.
            let swirl = normalize(vec3<f32>(-position.z, 0.0, position.x) + vec3<f32>(1e-5));
            let dir = normalize(jitter + swirl * 1.5 + vec3<f32>(1e-5));
            let amp = radius * gaussian_uniforms.swarm * mix(0.5, 1.5, rnd.y);
            position = vec4<f32>(position.xyz + dir * amp * pulse, 1.0);
        }
    }

    // --- per-particle TRANSITION phase (martin fork). Off by default: transition_mode == 0u
    //     skips the whole block, so mode 0 is byte-identical to upstream. Active only during a
    //     morph (interp_active), exactly like the ball-pulse above. Produces tx_reveal for the
    //     opacity sink (read at the finalize site below) and, for motion modes, nudges position.
    //     A moving window local = saturate((gt*(1+softness) - phase)/softness) sweeps the phase
    //     axis. if/else-if, NOT switch (RADV). ---
    var tx_reveal = 1.0;
    if (interp_active && gaussian_uniforms.transition_mode != 0u) {
        let denom = max(gaussian_uniforms.time_stop - gaussian_uniforms.time_start, 1e-6);
        let gt = clamp((gaussian_uniforms.time - gaussian_uniforms.time_start) / denom, 0.0, 1.0);
        let softness = max(gaussian_uniforms.transition_softness, 1e-4);
        let mode = gaussian_uniforms.transition_mode;
        // pen-write (mode 7) reads its per-particle phase from the visibility channel (cumulative
        // pen-distance baked by build_text_pen_gaussians); the rest derive it from position.
        var phase = transition_phase(splat_index, position.xyz);
        if (mode == 7u) { phase = get_visibility(splat_index); }
        let local = clamp((gt * (1.0 + softness) - phase) / softness, 0.0, 1.0);
        if (mode == 1u || mode == 6u) {
            tx_reveal = local;                  // typewriter / directional-wipe HARD
        } else if (mode == 2u) {
            // slither: lateral sine that dies as the particle settles (local -> 1).
            let amp = (1.0 - local) * length(gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz) * 0.04;
            let wobble = sin(phase * 18.0 + gt * 6.2831853);
            position = vec4<f32>(position.x, position.y + amp * wobble, position.z, 1.0);
        } else if (mode == 3u) {
            tx_reveal = local;                  // sparkle-in (hashed reveal; HDR Bloom twinkles)
        } else if (mode == 4u) {
            tx_reveal = 1.0 - local;            // spark-out (reversed reveal)
        } else if (mode == 5u) {
            // vortex: a turntable spin that DECELERATES into place. Angle is driven by gt
            // (uniform timing — the whole cloud settles together, no per-radius timing shear)
            // with a quadratic ease-out (1-gt)^2 so it slows as it lands, and only a gentle
            // radial gradient (0.8..1.0) so outer splats trail slightly without tearing.
            let center = (gaussian_uniforms.min.xyz + gaussian_uniforms.max.xyz) * 0.5;
            let half = max(length(gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz) * 0.5, 1e-4);
            let p = position.xyz - center;
            let rr = clamp(length(p.xz) / half, 0.0, 1.0);
            let ease_out = (1.0 - gt) * (1.0 - gt);
            let ang = ease_out * 1.25 * 6.2831853 * (0.8 + 0.2 * rr);
            let c = cos(ang); let s = sin(ang);
            let rp = vec3<f32>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
            position = vec4<f32>(center + rp, 1.0);
        } else if (mode == 7u) {
            tx_reveal = local;                  // pen-write (phase = baked pen distance)
        }
    }

    // --- persistent vertex DEFORM (martin fork). Off by default: deform_mode == 0u skips the
    //     whole block → byte-identical to upstream. NOT gated to a morph (unlike the transition
    //     above): driven by deform_time it animates every frame, so a held shape keeps moving.
    //     Displaces the OBJECT-space position before the world transform. if/else-if (RADV-safe). ---
    if (gaussian_uniforms.deform_mode != 0u) {
        let dcenter = (gaussian_uniforms.min.xyz + gaussian_uniforms.max.xyz) * 0.5;
        let dp = position.xyz - dcenter;
        let damp = gaussian_uniforms.deform_amp;
        let dfreq = gaussian_uniforms.deform_freq;
        let dtt = gaussian_uniforms.deform_time;
        let dmode = gaussian_uniforms.deform_mode;
        if (dmode == 1u) {
            // wave (flag): z displaced by a sine travelling across x
            position = vec4<f32>(position.x, position.y, position.z + damp * sin(dp.x * dfreq + dtt), 1.0);
        } else if (dmode == 2u) {
            // cloth (billow): 2D undulation, x and y out of phase
            let d = damp * sin(dp.x * dfreq + dtt) * cos(dp.y * dfreq * 0.7 + dtt * 0.8);
            position = vec4<f32>(position.x, position.y, position.z + d, 1.0);
        } else if (dmode == 3u) {
            // ripple (radial): concentric waves from the centre outward
            let rr = length(dp.xy);
            position = vec4<f32>(position.x, position.y, position.z + damp * sin(rr * dfreq - dtt), 1.0);
        } else if (dmode == 4u) {
            // twist / curl: rotate the x-z plane by an angle that varies with height + time
            let ang = damp * sin(dp.y * dfreq + dtt);
            let cs = cos(ang); let sn = sin(ang);
            position = vec4<f32>(dcenter.x + cs * dp.x + sn * dp.z, position.y, dcenter.z - sn * dp.x + cs * dp.z, 1.0);
        } else if (dmode == 5u) {
            // wind: a gusting sideways (+x) sway with particles lagging by position, plus spatial
            // turbulence in y/z — the cloud flutters and streams in the wind (sways around 0, no drift)
            let phase = dp.x * dfreq * 0.3 + dp.y * dfreq * 0.5;
            let gust = 0.6 + 0.4 * sin(dtt * 0.5);                 // slow gust swell
            let swayx = damp * gust * sin(dtt * 1.2 + phase);
            let fly = damp * 0.4 * sin(dp.x * dfreq + dtt * 1.7);
            let flz = damp * 0.5 * cos(dp.y * dfreq * 1.1 - dtt * 1.4);
            position = vec4<f32>(position.x + swayx, position.y + fly, position.z + flz, 1.0);
        } else if (dmode == 6u) {
            // turbulence: a churning 3D field — each axis pushed by sines of the OTHER axes + time,
            // so particles swirl/boil in place (a turbulent force field; sways around 0, no drift).
            let tx = sin(dp.y * dfreq + dtt) + 0.5 * sin(dp.z * dfreq * 1.7 - dtt * 1.3);
            let ty = sin(dp.z * dfreq * 1.1 + dtt * 1.2) + 0.5 * sin(dp.x * dfreq * 1.5 - dtt);
            let tz = sin(dp.x * dfreq * 0.9 - dtt) + 0.5 * sin(dp.y * dfreq * 1.3 + dtt * 0.8);
            position = vec4<f32>(position.x + damp * tx, position.y + damp * ty, position.z + damp * tz, 1.0);
        } else if (dmode == 7u) {
            // pulse: the whole shape breathes — scaled in/out about its centre by a sine (damp = the
            // fraction, e.g. 0.1 = ±10%). dfreq unused; the rate is deform_time's own speed.
            let s = 1.0 + damp * sin(dtt * 1.5);
            position = vec4<f32>(dcenter + dp * s, 1.0);
        } else if (dmode == 8u) {
            // jitter: a fast per-particle shake — each axis offset by a sine at a per-position phase,
            // so every splat trembles independently (a nervous, glitchy energy). damp small (~0.04).
            let ph = dp.x * 7.0 + dp.y * 13.0 + dp.z * 17.0;
            let jx = sin(dtt * 18.0 + ph);
            let jy = sin(dtt * 19.0 + ph * 1.3);
            let jz = sin(dtt * 17.0 + ph * 0.7);
            position = vec4<f32>(position.x + damp * jx, position.y + damp * jy, position.z + damp * jz, 1.0);
        } else if (dmode == 9u) {
            // spiral: a radial pinwheel — rotate each point about the vertical axis by an angle that
            // grows with its radius and time, so the shape swirls/curls outward (damp = radians scale).
            let r = length(dp.xz);
            let ang = damp * sin(r * dfreq - dtt);
            let ca = cos(ang);
            let sa = sin(ang);
            position = vec4<f32>(dcenter.x + dp.x * ca - dp.z * sa, position.y, dcenter.z + dp.x * sa + dp.z * ca, 1.0);
        }
    }

    var transformed_position = (gaussian_uniforms.transform * position).xyz;
    var previous_transformed_position = transformed_position;

#ifdef DRAW_SELECTED
    discard_quad |= get_visibility(splat_index) < 0.5;
#endif

#ifdef GAUSSIAN_4D
#else
    let projected_position = world_to_clip(transformed_position);
    discard_quad |= !in_frustum(projected_position.xyz);
#endif

    if (discard_quad) {
        output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return output;
    }

    var quad_vertices = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    let quad_index = vertex_index % 4u;
    let quad_offset = quad_vertices[quad_index];

    var opacity = get_opacity(splat_index);

#ifdef OPACITY_ADAPTIVE_RADIUS
    let cutoff = sqrt(max(9.0 + 2.0 * log(opacity), 0.000001));
#else
    let cutoff = 3.0;
#endif

#ifdef GAUSSIAN_2D
    let surfel = compute_cov2d_surfel(
        transformed_position,
        splat_index,
        cutoff,
    );

    output.local_to_pixel_u = surfel.local_to_pixel[0];
    output.local_to_pixel_v = surfel.local_to_pixel[1];
    output.local_to_pixel_w = surfel.local_to_pixel[2];
    output.mean_2d = surfel.mean_2d;

    let bb = get_bounding_box_cov2d(
        surfel.extent,
        quad_offset,
        cutoff,
    );
    output.radius = bb.zw;
#else
    #ifdef GAUSSIAN_3D
        let gaussian_cov2d = compute_cov2d_3dgs(
            transformed_position,
            splat_index,
        );
    #else ifdef GAUSSIAN_4D
        let gaussian_4d = conditional_cov3d(
            transformed_position,
            splat_index,
            gaussian_uniforms.time,
        );

        if !gaussian_4d.mask {
            output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            return output;
        }

        let position_t = vec4<f32>(position.xyz + gaussian_4d.delta_mean, 1.0);
        transformed_position = (gaussian_uniforms.transform * position_t).xyz;
        // TODO: set previous_transformed_position based on temporal position delta
        let projected_position = world_to_clip(transformed_position);

        if !in_frustum(projected_position.xyz) {
            output.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            return output;
        }

        opacity = opacity * gaussian_4d.opacity_modifier;

        let gaussian_cov2d = cov2d(
            transformed_position,
            gaussian_4d.cov3d,
        );
    #endif

    let bb = get_bounding_box_clip(
        gaussian_cov2d,
        quad_offset,
        cutoff,
    );

    #ifdef USE_AABB
        let det = gaussian_cov2d.x * gaussian_cov2d.z - gaussian_cov2d.y * gaussian_cov2d.y;
        let det_inv = 1.0 / det;
        let conic = vec3<f32>(
            gaussian_cov2d.z * det_inv,
            -gaussian_cov2d.y * det_inv,
            gaussian_cov2d.x * det_inv
        );
        output.conic = conic;
        output.major_minor = bb.zw;
    #endif
#endif

    var rgb = vec3<f32>(0.0);

// TODO: RASTERIZE_ACCELERATION
#ifdef RASTERIZE_CLASSIFICATION
    let ray_direction_world = normalize(transformed_position - view.world_position);
    let ray_direction_local = world_to_local_direction(ray_direction_world, gaussian_uniforms.transform);

    #ifdef GAUSSIAN_3D_STRUCTURE
        rgb = get_color(splat_index, ray_direction_local);
    #else ifdef GAUSSIAN_4D
        rgb = get_color(splat_index, gaussian_4d.dir_t, ray_direction_local);
    #endif

    rgb = class_to_rgb(
        get_visibility(splat_index),
        rgb,
    );
#else ifdef RASTERIZE_DEPTH
    // TODO: unbiased depth rendering, see: https://zju3dv.github.io/pgsr/
    let first_position = vec4<f32>(get_position(get_entry(1u).value), 1.0);
    let last_position = vec4<f32>(get_position(get_entry(gaussian_uniforms.count - 1u).value), 1.0);

    let min_position = (gaussian_uniforms.transform * last_position).xyz;
    let max_position = (gaussian_uniforms.transform * first_position).xyz;

    let camera_position = view.world_position;

    let min_distance = length(min_position - camera_position);
    let max_distance = length(max_position - camera_position);

    let depth = length(transformed_position - camera_position);
    rgb = depth_to_rgb(
        depth,
        min_distance,
        max_distance,
    );
#else ifdef RASTERIZE_NORMAL
    // TODO: support rotation decomposition for 4d gaussians
    let R = get_rotation_matrix(get_rotation(splat_index));
    let S = get_scale_matrix(get_scale(splat_index));
    let T = mat3x3<f32>(
        gaussian_uniforms.transform[0].xyz,
        gaussian_uniforms.transform[1].xyz,
        gaussian_uniforms.transform[2].xyz,
    );
    let L = T * S * R;

    let local_normal = vec4<f32>(L[2], 0.0);
    let world_normal = view.view_from_world * local_normal;

    let t = normalize(world_normal);

    rgb = vec3<f32>(
        0.5 * (t.x + 1.0),
        0.5 * (t.y + 1.0),
        0.5 * (t.z + 1.0)
    );
#else ifdef RASTERIZE_OPTICAL_FLOW
    let motion_vector = calculate_motion_vector(
        transformed_position,
        previous_transformed_position,
    );

    rgb = optical_flow_to_rgb(motion_vector);
#else ifdef RASTERIZE_POSITION
    rgb = (transformed_position - gaussian_uniforms.min.xyz) / (gaussian_uniforms.max.xyz - gaussian_uniforms.min.xyz);
#else ifdef RASTERIZE_VELOCITY
    let time_delta = 1e-3;
    let future_gaussian_4d = conditional_cov3d(
        transformed_position,
        splat_index,
        gaussian_uniforms.time + time_delta,
    );
    let position_delta = future_gaussian_4d.delta_mean - gaussian_4d.delta_mean;
    let velocity = position_delta / time_delta;
    let velocity_magnitude = length(velocity);
    let velocity_normalized = normalize(velocity);

    // TODO: magnitude normalization
    let min_magnitude = 1.0;
    let max_magnitude = 2.0;

    let scaled_mag = clamp(
        (velocity_magnitude - min_magnitude) / (max_magnitude - min_magnitude),
        0.0,
        1.0
    );

    if scaled_mag < 1e-2 {
        opacity = 0.0;
    }

    let base_color = 0.5 * (velocity_normalized + vec3<f32>(1.0, 1.0, 1.0));
    rgb = base_color * scaled_mag;
#else ifdef RASTERIZE_COLOR
    // TODO: verify color benefit for ray_direction computed at quad verticies instead of gaussian center (same as current complexity)
    let ray_direction_world = normalize(transformed_position - view.world_position);
    let ray_direction_local = world_to_local_direction(ray_direction_world, gaussian_uniforms.transform);

    #ifdef GAUSSIAN_3D_STRUCTURE
        rgb = get_color(splat_index, ray_direction_local);
    #else ifdef GAUSSIAN_4D
        rgb = get_color(splat_index, gaussian_4d.dir_t, ray_direction_local);
    #endif
#endif

    output.color = vec4<f32>(
        rgb,
        opacity * gaussian_uniforms.global_opacity * tx_reveal,
    );

#ifdef HIGHLIGHT_SELECTED
    if (get_visibility(splat_index) > 0.5) {
        output.color = vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

    output.uv = quad_offset;
    output.position = vec4<f32>(
        projected_position.xy + bb.xy,
        projected_position.zw,
    );

    return output;
}

@fragment
fn fs_main(input: GaussianVertexOutput) -> @location(0) vec4<f32> {
#ifdef USE_AABB
#ifdef GAUSSIAN_2D
    let radius = input.radius;
    let mean_2d = input.mean_2d;
    let aspect = vec2<f32>(
        1.0,
        view.viewport.z / view.viewport.w,
    );
    let pixel_coord = input.uv * radius * aspect + mean_2d;

    let power = surfel_fragment_power(
        mat3x3<f32>(
            input.local_to_pixel_u,
            input.local_to_pixel_v,
            input.local_to_pixel_w,
        ),
        pixel_coord,
        mean_2d,
    );
#else ifdef GAUSSIAN_3D
    let d = -input.major_minor;
    let conic = input.conic;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
#else ifdef GAUSSIAN_4D
    let d = -input.major_minor;
    let conic = input.conic;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
#endif

    if (power > 0.0) {
        discard;
    }
#endif

#ifdef USE_OBB
    let sigma = 1.0 / 3.0;
    let sigma_squared = 2.0 * sigma * sigma;
    let distance_squared = dot(input.uv, input.uv);

    let power = -distance_squared / sigma_squared;

    if (distance_squared > 3.0 * 3.0) {
        discard;
    }
#endif

#ifdef VISUALIZE_BOUNDING_BOX
    let uv = input.uv * 0.5 + 0.5;
    let edge_width = 0.08;
    if (
        (uv.x < edge_width || uv.x > 1.0 - edge_width) ||
        (uv.y < edge_width || uv.y > 1.0 - edge_width)
    ) {
        return vec4<f32>(0.3, 1.0, 0.1, 1.0);
    }
#endif

    let alpha = min(exp(power) * input.color.a, 0.999);

    // TODO: round alpha to terminate depth test?

    return vec4<f32>(
        input.color.rgb * alpha,
        alpha,
    );
}
