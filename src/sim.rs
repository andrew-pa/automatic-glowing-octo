use glam::{Mat4, Vec2, Vec3};
use wgpu::BufferAddress;

use bytemuck::{Pod, Zeroable};

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
    pub counters: [u32; 4],   // frame, reset, particle_count, unused
}

impl SimUniform {
    pub fn from_settings(settings: &SimSettings) -> Self {
        Self {
            integrator: [
                settings.dt,
                settings.flow,
                settings.damping,
                settings.color_mix,
            ],
            attractor: settings.attractor,
            misc: [settings.jitter, settings.drive, 0.0, 0.0],
            counters: [0, 1, settings.particle_count, 0],
        }
    }

    pub fn update(&mut self, dt: f32, frame: u64, reset: bool, particle_count: u32) {
        self.integrator[0] = dt;
        self.counters[0] = frame as u32;
        self.counters[1] = if reset { 1 } else { 0 };
        self.counters[2] = particle_count;
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

#[derive(Clone)]
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
            particle_count: 1_000_000, //524_288,
            workgroup_size: 256,
            dt: 0.004,
            flow: 0.9,
            damping: 0.08,
            color_mix: 0.1,
            jitter: 0.0015,
            drive: 0.18,
            attractor: [-1.4, 1.6, 1.0, 0.7],
            point_size: 2.2,
            exposure: 1.0,
            time_scale: 1.0,
        }
    }
}
