use bevy::prelude::*;
use bevy_args::{Deserialize, Serialize, ValueEnum};

use crate::sort::SortMode;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
pub enum DrawMode {
    #[default]
    All,
    Selected,
    HighlightSelected,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum GaussianMode {
    Gaussian2d,
    #[default]
    Gaussian3d,
    Gaussian4d,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum PlaybackMode {
    Loop,
    Once,
    Sin,
    #[default]
    Still,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum RasterizeMode {
    Classification,
    #[default]
    Color,
    Depth,
    Normal,
    OpticalFlow,
    Position,
    Velocity,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize, ValueEnum,
)]
pub enum RadixSortDepthBits {
    Bits16,
    Bits24,
    #[default]
    Bits32,
}

impl RadixSortDepthBits {
    pub const VARIANTS: [Self; 3] = [Self::Bits16, Self::Bits24, Self::Bits32];

    pub const fn bits(self) -> u32 {
        match self {
            Self::Bits16 => 16,
            Self::Bits24 => 24,
            Self::Bits32 => 32,
        }
    }

    pub const fn pipeline_index(self) -> usize {
        match self {
            Self::Bits16 => 0,
            Self::Bits24 => 1,
            Self::Bits32 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, Reflect, Serialize, Deserialize)]
pub enum GaussianColorSpace {
    #[default]
    SrgbRec709Display,
    LinRec709Display,
}

// TODO: breakdown into components
#[derive(Component, Clone, Debug, Reflect, Serialize, Deserialize)]
#[reflect(Component)]
#[serde(default)]
pub struct CloudSettings {
    pub aabb: bool,
    pub global_opacity: f32,
    pub global_scale: f32,
    pub opacity_adaptive_radius: bool,
    pub visualize_bounding_box: bool,
    pub sort_mode: SortMode,
    pub radix_sort_depth_bits: RadixSortDepthBits,
    pub draw_mode: DrawMode,
    pub gaussian_mode: GaussianMode,
    pub playback_mode: PlaybackMode,
    pub rasterize_mode: RasterizeMode,
    pub color_space: GaussianColorSpace,
    pub num_classes: usize,
    pub time: f32,
    pub time_scale: f32,
    pub time_start: f32,
    pub time_stop: f32,
    /// Midpoint explosive-bulge amplitude for GaussianInterpolate morphs (0 = none).
    /// The renderer scatters each gaussian radially by `sin(pi*t)*bulge` (peaks at the
    /// blend midpoint, zero at both ends) so a morph blows apart then reassembles.
    pub bulge: f32,
    /// Per-particle transition mode (0 = off/identity → byte-identical to upstream).
    /// Selects a staggered reveal/motion effect in vs_points. See SHADER-BLUEPRINT.md.
    pub transition_mode: u32,
    /// Moving-window ramp width for the transition. Small ≈ hard edge (typewriter/wipe),
    /// larger → soft dissolve. Ignored when transition_mode == 0.
    pub transition_softness: f32,
    /// Axis for axis/wipe/vortex transition modes: 0 = x, 1 = y, 2 = z.
    pub transition_axis: u32,
    /// Persistent vertex deform (0 = off → byte-identical to upstream): 1 wave, 2 cloth,
    /// 3 ripple, 4 twist. Unlike the transition it runs *every frame* (driven by `deform_time`),
    /// so a held shape keeps moving. See gaussian.wgsl / SHADER-BLUEPRINT.md.
    pub deform_mode: u32,
    /// Deform displacement amplitude (object units; radians for twist). Ignored when mode == 0.
    pub deform_amp: f32,
    /// Deform spatial frequency. Ignored when mode == 0.
    pub deform_freq: f32,
    /// Deform animation phase in seconds (the app advances it from the show clock).
    pub deform_time: f32,
    /// Per-particle swarm detour during a GaussianInterpolate morph (0 = off → byte-identical to
    /// upstream). Each gaussian swirls along a pseudo-random + tangential curl by `sin(pi*t)*swarm`
    /// (peaks mid-morph, zero at both ends), so a shape→shape morph *flocks/swarms* between the two
    /// scenes instead of lerping straight. See gaussian.wgsl / SHADER-BLUEPRINT.md.
    pub swarm: f32,
    /// Per-particle STAGGERED morph timing (0 = off → synchronized, byte-identical to upstream). Each
    /// gaussian morphs over its own sub-window `[offset, offset+(1-stagger)]` of the blend factor, with
    /// `offset = hash(index)*stagger`, so the cloud DISSOLVES + reforms instead of sliding as one block
    /// (the latter reads as straight-line streaks). ~0.6 = a strong, "cloudy" dissolve. See
    /// morph/interpolate.wgsl.
    pub morph_stagger: f32,
    /// Additive/emissive blend (false = premultiplied alpha-over → byte-identical to upstream). When
    /// true the cloud composites One+One (light ACCUMULATES) so overlapping translucent gaussians GLOW
    /// on black instead of alpha-saturating to a solid blob — the demoscene nebula/neon look. Meant for
    /// glow content on a dark background; solid captures (real scenes) should stay false (additive has
    /// no occlusion — everything shows through everything).
    pub additive: bool,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            aabb: false,
            global_opacity: 1.0,
            global_scale: 1.0,
            opacity_adaptive_radius: true,
            visualize_bounding_box: false,
            sort_mode: SortMode::default(),
            radix_sort_depth_bits: RadixSortDepthBits::default(),
            draw_mode: DrawMode::default(),
            gaussian_mode: GaussianMode::default(),
            rasterize_mode: RasterizeMode::default(),
            color_space: GaussianColorSpace::default(),
            num_classes: 1,
            playback_mode: PlaybackMode::default(),
            time: 0.0,
            time_scale: 1.0,
            time_start: 0.0,
            time_stop: 1.0,
            bulge: 0.0,
            transition_mode: 0,        // off → byte-identical to upstream
            transition_softness: 0.15, // sensible ramp; only used when mode != 0
            transition_axis: 0,        // x
            deform_mode: 0,            // off → byte-identical to upstream
            deform_amp: 0.0,
            deform_freq: 0.0,
            deform_time: 0.0,
            swarm: 0.0,        // off → byte-identical to upstream
            morph_stagger: 0.0, // off → synchronized, byte-identical to upstream
            additive: false,    // off → premultiplied alpha-over, byte-identical to upstream
        }
    }
}

#[derive(Default)]
pub struct SettingsPlugin;
impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<CloudSettings>();
        app.register_type::<RadixSortDepthBits>();

        app.add_systems(Update, (playback_update,));
    }
}

fn playback_update(time: Res<Time>, mut query: Query<(&mut CloudSettings,)>) {
    for (mut settings,) in query.iter_mut() {
        if settings.time_scale == 0.0 {
            continue;
        }

        // bail condition
        match settings.playback_mode {
            PlaybackMode::Loop => {}
            PlaybackMode::Once => {
                if settings.time >= settings.time_stop {
                    continue;
                }
            }
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {
                continue;
            }
        }

        // forward condition
        match settings.playback_mode {
            PlaybackMode::Loop | PlaybackMode::Once => {
                settings.time += time.delta_secs() * settings.time_scale;
            }
            PlaybackMode::Sin => {
                let theta = settings.time_scale * time.elapsed_secs();
                let y = (theta * 2.0 * std::f32::consts::PI).sin();
                settings.time = settings.time_start
                    + (settings.time_stop - settings.time_start) * (y + 1.0) / 2.0;
            }
            PlaybackMode::Still => {}
        }

        // reset condition
        match settings.playback_mode {
            PlaybackMode::Loop => {
                if settings.time > settings.time_stop {
                    settings.time = settings.time_start;
                }
            }
            PlaybackMode::Once => {}
            PlaybackMode::Sin => {}
            PlaybackMode::Still => {}
        }
    }
}
