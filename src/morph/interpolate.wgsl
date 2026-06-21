#define_import_path bevy_gaussian_splatting::morph::interpolate

#import bevy_gaussian_splatting::bindings::gaussian_uniforms

#ifdef PACKED_F32
    #import bevy_gaussian_splatting::packed::{
        get_opacity,
        get_position,
        get_rotation,
        get_scale,
        get_spherical_harmonics,
        get_visibility,
        get_rhs_opacity,
        get_rhs_position,
        get_rhs_rotation,
        get_rhs_scale,
        get_rhs_spherical_harmonics,
        get_rhs_visibility,
        set_output_position_visibility,
        set_output_spherical_harmonics,
        set_output_transform,
    };
#else
    #import bevy_gaussian_splatting::planar::{
        get_opacity,
        get_position,
        get_rotation,
        get_scale,
        get_spherical_harmonics,
        get_visibility,
        get_rhs_opacity,
        get_rhs_position,
        get_rhs_rotation,
        get_rhs_scale,
        get_rhs_spherical_harmonics,
        get_rhs_visibility,
        set_output_position_visibility,
        set_output_spherical_harmonics,
        set_output_transform,
    };

    #ifdef PRECOMPUTE_COVARIANCE_3D
        #import bevy_gaussian_splatting::planar::{
            get_cov3d,
            get_rhs_cov3d,
            set_output_covariance,
        };
    #endif
#endif

fn interpolation_factor() -> f32 {
    let duration = gaussian_uniforms.time_stop - gaussian_uniforms.time_start;
    if abs(duration) < 1e-6 {
        return select(0.0, 1.0, gaussian_uniforms.time >= gaussian_uniforms.time_stop);
    }
    return clamp((gaussian_uniforms.time - gaussian_uniforms.time_start) / duration, 0.0, 1.0);
}

// Per-splat pseudo-random in [0,1) (PCG-style integer hash) — used to stagger each splat's morph start.
fn hash_u32(x: u32) -> f32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return f32(h) / 4294967295.0;
}

fn normalize_quaternion(q: vec4<f32>) -> vec4<f32> {
    let length_squared = dot(q, q);
    if length_squared <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return q / sqrt(length_squared);
}

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn interpolate_gaussians(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let index = global_id.x;
    if index >= gaussian_uniforms.count {
        return;
    }

    let t_global = interpolation_factor();
    // Per-splat STAGGERED timing: each splat morphs over its own sub-window [offset, offset+width] of
    // the global factor, so the cloud DISSOLVES + reforms instead of sliding as one coherent block —
    // which is what made the morph read as straight-line streaks. Driven by the `morph_stagger` uniform;
    // stagger 0 short-circuits to the plain global factor → SYNCHRONIZED, byte-identical to upstream.
    let stagger = clamp(gaussian_uniforms.morph_stagger, 0.0, 0.98);
    var t = t_global;
    if stagger > 0.0 {
        let offset = hash_u32(index) * stagger;
        let width = max(1.0 - stagger, 1e-3);
        let tu = clamp((t_global - offset) / width, 0.0, 1.0);
        t = tu * tu * (3.0 - 2.0 * tu); // smooth per-splat ease
    }
    let position_t = vec3<f32>(t);
    let rotation_t = vec4<f32>(t);

    let lhs_position = get_position(index);
    let rhs_position = get_rhs_position(index);
    let lhs_visibility = get_visibility(index);
    let rhs_visibility = get_rhs_visibility(index);

    let position = mix(lhs_position, rhs_position, position_t);
    let visibility = mix(lhs_visibility, rhs_visibility, t);
    set_output_position_visibility(index, position, visibility);

    var sh = get_spherical_harmonics(index);
    let rhs_sh = get_rhs_spherical_harmonics(index);
    for (var i = 0u; i < #{SH_COEFF_COUNT}; i = i + 1u) {
        sh[i] = mix(sh[i], rhs_sh[i], t);
    }
    set_output_spherical_harmonics(index, sh);

#ifdef PRECOMPUTE_COVARIANCE_3D
    var cov = get_cov3d(index);
    let rhs_cov = get_rhs_cov3d(index);
    for (var i = 0u; i < 6u; i = i + 1u) {
        cov[i] = mix(cov[i], rhs_cov[i], t);
    }
    let opacity = mix(get_opacity(index), get_rhs_opacity(index), t);
    set_output_covariance(index, cov, opacity);
#else
    let rotation = normalize_quaternion(
        mix(
            get_rotation(index),
            get_rhs_rotation(index),
            rotation_t,
        ),
    );

    let scale = mix(get_scale(index), get_rhs_scale(index), position_t);
    let opacity = mix(get_opacity(index), get_rhs_opacity(index), t);
    set_output_transform(index, rotation, scale, opacity);
#endif
}
