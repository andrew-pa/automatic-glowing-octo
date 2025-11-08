use glam::{Mat4, Vec2, Vec3};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

pub struct OrbitCamera {
    target: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, radius: f32, aspect: f32) -> Self {
        let mut cam = Self {
            target,
            radius,
            yaw: -1.2,
            pitch: 0.9,
            fov_y: 45.0_f32.to_radians(),
            aspect: aspect.max(0.1),
            near: 0.1,
            far: 250.0,
        };
        cam.radius = cam.radius.clamp(2.0, 120.0);
        cam
    }

    pub fn eye(&self) -> Vec3 {
        let sin_yaw = self.yaw.sin();
        let cos_yaw = self.yaw.cos();
        let sin_pitch = self.pitch.sin();
        let cos_pitch = self.pitch.cos();
        let dir = Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw);
        self.target + dir * self.radius
    }

    pub fn view_proj(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect.max(0.01), self.near, self.far);
        proj * view
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect.max(0.1);
    }

    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw -= delta.x;
        self.pitch = (self.pitch - delta.y).clamp(-1.4, 1.4);
    }

    pub fn zoom(&mut self, amount: f32) {
        let factor = (1.0 + amount).clamp(0.1, 3.0);
        self.radius = (self.radius * factor).clamp(2.0, 140.0);
    }

    pub fn pan(&mut self, delta: Vec2) {
        let eye_dir = (self.eye() - self.target).normalize();
        let right = eye_dir.cross(Vec3::Y).normalize_or_zero();
        let up = Vec3::Y;
        self.target += (-right * delta.x + up * delta.y) * self.radius * 0.8;
    }
}

pub struct OrbitController {
    rotate_button: MouseButton,
    pan_button: MouseButton,
    rotating: bool,
    panning: bool,
    last_cursor: Option<Vec2>,
    pub rotate_speed: f32,
    pub pan_speed: f32,
    pub zoom_speed: f32,
}

impl Default for OrbitController {
    fn default() -> Self {
        Self {
            rotate_button: MouseButton::Right,
            pan_button: MouseButton::Middle,
            rotating: false,
            panning: false,
            last_cursor: None,
            rotate_speed: 1.2,
            pan_speed: 1.5,
            zoom_speed: 0.1,
        }
    }
}

impl OrbitController {
    pub fn process(
        &mut self,
        camera: &mut OrbitCamera,
        event: &WindowEvent,
        size: PhysicalSize<u32>,
    ) -> bool {
        let viewport = Vec2::new(size.width.max(1) as f32, size.height.max(1) as f32);
        match event {
            WindowEvent::MouseInput { state, button, .. } if *button == self.rotate_button => {
                self.rotating = *state == ElementState::Pressed;
                self.last_cursor = None;
                true
            }
            WindowEvent::MouseInput { state, button, .. } if *button == self.pan_button => {
                self.panning = *state == ElementState::Pressed;
                self.last_cursor = None;
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if let Some(prev) = self.last_cursor {
                    let delta = (pos - prev) / viewport;
                    if self.rotating {
                        camera.orbit(delta * self.rotate_speed * std::f32::consts::TAU);
                    } else if self.panning {
                        camera.pan(delta * self.pan_speed);
                    }
                }
                self.last_cursor = Some(pos);
                self.rotating || self.panning
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -*y * self.zoom_speed,
                    MouseScrollDelta::PixelDelta(delta) => {
                        -(delta.y as f32) * 0.001 * self.zoom_speed
                    }
                };
                camera.zoom(scroll);
                true
            }
            _ => false,
        }
    }
}
