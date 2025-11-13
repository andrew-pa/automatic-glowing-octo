use glam::{Mat4, Vec2, Vec3};
use wgpu::BufferAddress;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

pub const GRAVITY_WELL_COUNT: usize = 4;
pub const DEFAULT_AUDIO_BAND_RANGES: [(f32, f32); GRAVITY_WELL_COUNT] = [
    (20.0, 200.0),
    (200.0, 800.0),
    (800.0, 3_000.0),
    (3_000.0, 12_000.0),
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 4],
    pub velocity: [f32; 4],
}

impl GpuParticle {
    pub const STRIDE: BufferAddress = std::mem::size_of::<GpuParticle>() as BufferAddress;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimUniform {
    pub integrator: [f32; 4], // dt, flow, damping, color_mix
    pub attractor: [f32; 4],  // a, b, c, d
    pub misc: [f32; 4],       // jitter, drive, unused, unused
    pub gravity: [[f32; 4]; GRAVITY_WELL_COUNT],
    pub counters: [u32; 4], // frame, reset, particle_count, unused
}

impl SimUniform {
    pub fn from_settings(
        settings: &SimSettings,
        wells: &[GravityWell; GRAVITY_WELL_COUNT],
        attractor: [f32; 4],
    ) -> Self {
        Self {
            integrator: [
                settings.dt,
                settings.flow,
                settings.damping,
                settings.color_mix,
            ],
            attractor,
            misc: [settings.jitter, settings.drive, 0.0, 0.0],
            gravity: wells.map(|well| well.as_vec4()),
            counters: [0, 1, settings.particle_count, 0],
        }
    }

    pub fn update(
        &mut self,
        dt: f32,
        frame: u64,
        reset: bool,
        wells: &[GravityWell; GRAVITY_WELL_COUNT],
        attractor: [f32; 4],
        settings: &SimSettings,
    ) {
        self.integrator = [dt, settings.flow, settings.damping, settings.color_mix];
        self.attractor = attractor;
        self.misc = [settings.jitter, settings.drive, 0.0, 0.0];
        self.gravity = wells.map(|well| well.as_vec4());
        self.counters[0] = frame as u32;
        self.counters[1] = if reset { 1 } else { 0 };
        self.counters[2] = settings.particle_count;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub params: [f32; 4], // point_size, exposure, viewport_width, viewport_height
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            eye: [0.0; 4],
            params: [2.0, 1.0, 1.0, 1.0],
        }
    }

    pub fn update(
        &mut self,
        view_proj: Mat4,
        eye: Vec3,
        point_size: f32,
        exposure: f32,
        viewport: Vec2,
    ) {
        self.view_proj = view_proj.to_cols_array_2d();
        self.eye = [eye.x, eye.y, eye.z, 1.0];
        self.params = [
            point_size,
            exposure,
            viewport.x.max(1.0),
            viewport.y.max(1.0),
        ];
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SimSettings {
    pub particle_count: u32,
    pub workgroup_size: u32,
    pub dt: f32,
    pub flow: f32,
    pub damping: f32,
    pub color_mix: f32,
    pub jitter: f32,
    pub drive: f32,
    pub attractor: [f32; 4],
    pub point_size: f32,
    pub exposure: f32,
    pub trail_decay: f32,
    pub trail_intensity: f32,
    pub audio: AudioSettings,
    pub gravity_wells: [GravityWell; GRAVITY_WELL_COUNT],
    pub time_scale: f32,
}

impl SimSettings {
    pub fn buffer_size(&self) -> BufferAddress {
        GpuParticle::STRIDE * self.particle_count as BufferAddress
    }

    pub fn dispatch_count(&self) -> u32 {
        (self.particle_count + self.workgroup_size - 1) / self.workgroup_size
    }
}

impl Default for SimSettings {
    fn default() -> Self {
        Self {
            particle_count: 5_000_000, //524_288,
            workgroup_size: 256,
            dt: 0.004,
            flow: 0.2,
            damping: 0.06,
            color_mix: 0.6,
            jitter: 0.0015,
            drive: 0.18,
            attractor: [-1.4, 1.6, 1.0, 0.7],
            point_size: 1.2,
            exposure: 0.4,
            trail_decay: 0.93,
            trail_intensity: 1.4,
            audio: AudioSettings::default(),
            gravity_wells: [
                GravityWell::new([6.0, 0.0, 0.0], 12.0, 0, 0.6, 0.3),
                GravityWell::new([-6.0, 0.0, 0.0], 12.0, 1, 0.6, 0.3),
                GravityWell::new([0.0, 6.0, 0.0], 8.0, 2, 0.5, 0.4),
                GravityWell::new([0.0, -6.0, 0.0], 8.0, 3, 0.5, 0.4),
            ],
            time_scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GravityWell {
    pub position: [f32; 3],
    pub strength: f32,
    pub audio_band: usize,
    pub strength_mod: f32,
    pub position_mod: f32,
}

impl GravityWell {
    pub const fn new(
        position: [f32; 3],
        strength: f32,
        audio_band: usize,
        strength_mod: f32,
        position_mod: f32,
    ) -> Self {
        Self {
            position,
            strength,
            audio_band,
            strength_mod,
            position_mod,
        }
    }

    pub fn as_vec4(self) -> [f32; 4] {
        [
            self.position[0],
            self.position[1],
            self.position[2],
            self.strength,
        ]
    }
}

impl Default for GravityWell {
    fn default() -> Self {
        Self::new([0.0; 3], 0.0, 0, 0.0, 0.0)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub enabled: bool,
    pub capture_sink: bool,
    pub device: String,
    pub gain: f32,
    pub gate: f32,
    pub smoothing: f32,
    pub modulate_strength: bool,
    pub modulate_position: bool,
    pub modulate_attractor: bool,
    pub strength_depth: f32,
    pub position_depth: f32,
    pub attractor_depths: [f32; 4],
    pub attractor_bands: [usize; 4],
    pub band_ranges: [(f32, f32); GRAVITY_WELL_COUNT],
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            capture_sink: false,
            device: String::new(),
            gain: 4.0,
            gate: 0.02,
            smoothing: 0.6,
            modulate_strength: true,
            modulate_position: true,
            modulate_attractor: false,
            strength_depth: 6.0,
            position_depth: 1.5,
            attractor_depths: [0.5; 4],
            attractor_bands: [0, 1, 2, 3],
            band_ranges: DEFAULT_AUDIO_BAND_RANGES,
        }
    }
}

impl SimSettings {
    pub fn modulated_wells(
        &self,
        bands: [f32; GRAVITY_WELL_COUNT],
    ) -> [GravityWell; GRAVITY_WELL_COUNT] {
        let mut wells = self.gravity_wells;
        if !self.audio.enabled {
            return wells;
        }

        for (idx, well) in wells.iter_mut().enumerate() {
            let base = self.gravity_wells[idx];
            let band_index = base.audio_band.min(GRAVITY_WELL_COUNT - 1);
            let signal = bands[band_index];
            if self.audio.modulate_strength {
                let depth = self.audio.strength_depth * base.strength_mod;
                well.strength = (base.strength + depth * signal).max(0.0);
            } else {
                well.strength = base.strength;
            }
            if self.audio.modulate_position {
                let depth = self.audio.position_depth * base.position_mod;
                let base_pos = Vec3::from_array(base.position);
                let axis = if base_pos.length_squared() > 1e-4 {
                    base_pos.normalize()
                } else {
                    Vec3::Y
                };
                let new_pos = base_pos + axis * depth * signal;
                well.position = [new_pos.x, new_pos.y, new_pos.z];
            } else {
                well.position = base.position;
            }
            well.audio_band = base.audio_band;
            well.strength_mod = base.strength_mod;
            well.position_mod = base.position_mod;
        }

        wells
    }

    pub fn modulated_attractor(&self, bands: [f32; GRAVITY_WELL_COUNT]) -> [f32; 4] {
        if !self.audio.enabled || !self.audio.modulate_attractor {
            return self.attractor;
        }

        let mut result = self.attractor;
        for (idx, value) in result.iter_mut().enumerate() {
            let band_index = self.audio.attractor_bands[idx].min(GRAVITY_WELL_COUNT - 1);
            let signal = bands[band_index].clamp(0.0, 1.0);
            let depth = self.audio.attractor_depths[idx].max(0.0);
            if depth == 0.0 {
                continue;
            }
            let offset = (signal * 2.0 - 1.0) * depth;
            *value = (self.attractor[idx] + offset).clamp(-15.0, 15.0);
        }
        result
    }
}
