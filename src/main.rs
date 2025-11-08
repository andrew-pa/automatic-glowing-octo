mod audio;
mod camera;
mod fx;
mod sim;
mod ui;
mod ui_panels;

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as AnyhowContext, Result};
use audio::{AudioEngine, AudioTarget};
use camera::{OrbitCamera, OrbitController};
use egui::{CollapsingHeader, Context};
use fx::{TRAIL_FORMAT, TrailComposer};
use glam::{Vec2, Vec3};
use log::{error, warn};
use sim::{CameraUniform, GRAVITY_WELL_COUNT, GpuParticle, GravityWell, SimSettings, SimUniform};
use ui::{UiFrameOutput, UiLayer};
use ui_panels::{audio_window, gravity_window};
use wgpu::SurfaceError;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const COMPUTE_SHADER: &str = include_str!("../shaders/attractor.comp.wgsl");
const RENDER_SHADER: &str = include_str!("../shaders/attractor.render.wgsl");
const QUAD_VERTICES: [[f32; 2]; 4] = [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]];

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = DustApp::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct DustApp {
    state: Option<State>,
}

impl ApplicationHandler for DustApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Dust — Strange Attractor")
            .with_inner_size(LogicalSize::new(1400.0, 900.0))
            .with_visible(true);

        match event_loop.create_window(attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                match pollster::block_on(State::new(window)) {
                    Ok(state) => self.state = Some(state),
                    Err(err) => {
                        error!("failed to initialize GPU state: {err:?}");
                        event_loop.exit();
                    }
                }
            }
            Err(err) => {
                error!("failed to create window: {err:?}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let mut should_shutdown = false;

        {
            let Some(state) = self.state.as_mut() else {
                return;
            };
            if window_id != state.window_id() {
                return;
            }

            let ui_response = state.ui_layer.handle_event(state.window.as_ref(), &event);
            if ui_response.repaint {
                state.window.request_redraw();
            }

            if !ui_response.consumed
                && state
                    .camera_controller
                    .process(&mut state.camera, &event, state.size)
            {
                return;
            }

            match event {
                WindowEvent::CloseRequested => should_shutdown = true,
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::ScaleFactorChanged { .. } => state.resize(state.window.inner_size()),
                WindowEvent::RedrawRequested => {
                    state.update();
                    if let Err(err) = state.render() {
                        if state.handle_surface_error(err) {
                            should_shutdown = true;
                        }
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat =>
                {
                    match &event.logical_key {
                        Key::Named(NamedKey::Escape) => should_shutdown = true,
                        Key::Named(NamedKey::Space) if !ui_response.consumed => {
                            state.toggle_pause()
                        }
                        Key::Character(ch)
                            if !ui_response.consumed && ch.eq_ignore_ascii_case("r") =>
                        {
                            state.queue_reset()
                        }
                        Key::Character(ch) if ch.eq_ignore_ascii_case("q") => {
                            should_shutdown = true
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        if should_shutdown {
            self.shutdown(event_loop);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl DustApp {
    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        self.state = None;
        event_loop.exit();
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    settings: SimSettings,
    sim_uniform: SimUniform,
    sim_buffer: wgpu::Buffer,
    sim_bind_group: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,
    dispatch_count: u32,
    particle_buffer: wgpu::Buffer,
    quad_vertex_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    trail: TrailComposer,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera: OrbitCamera,
    camera_controller: OrbitController,
    ui_layer: UiLayer,
    ui_output: Option<UiFrameOutput>,
    audio_engine: AudioEngine,
    audio_bands: [f32; GRAVITY_WELL_COUNT],
    modulated_wells: [GravityWell; GRAVITY_WELL_COUNT],
    audio_targets: Vec<AudioTarget>,
    last_target_refresh: Instant,
    pending_reset: bool,
    paused: bool,
    last_frame: Instant,
    frame_time_smooth: f32,
    frame_id: u64,
    clear_color: wgpu::Color,
}

impl State {
    async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create WGPU surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .context("no suitable GPU adapters found")?;

        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Dust Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::default(),
            })
            .await?;

        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(capabilities.formats[0]);
        let present_mode = if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::Mailbox)
        {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let settings = SimSettings::default();
        let dispatch_count = settings.dispatch_count();
        let modulated_wells = settings.gravity_wells;
        let sim_uniform = SimUniform::from_settings(&settings, &modulated_wells);
        let camera_uniform = CameraUniform::new();
        let mut audio_engine = AudioEngine::new();
        audio_engine.refresh(&settings.audio);
        audio_engine.refresh_targets();
        let audio_bands = audio_engine.bands();
        let audio_targets = audio_engine.targets();

        let sim_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Uniform"),
            contents: bytemuck::bytes_of(&sim_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particles"),
            size: settings.buffer_size(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Vertices"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let sim_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sim BindGroup Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let sim_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sim BindGroup"),
            layout: &sim_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera BindGroup Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BindGroup"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&sim_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Strange Attractor Compute"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: GpuParticle::STRIDE,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: TRAIL_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: None,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let trail = TrailComposer::new(&device, &config);
        trail.clear(&device, &queue);
        let camera = OrbitCamera::new(Vec3::ZERO, 18.0, config.width as f32 / config.height as f32);
        let camera_controller = OrbitController::default();
        let ui_layer = UiLayer::new(window.as_ref(), &device, format);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            settings,
            sim_uniform,
            sim_buffer,
            sim_bind_group,
            compute_pipeline,
            dispatch_count,
            particle_buffer,
            quad_vertex_buffer,
            render_pipeline,
            trail,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera,
            camera_controller,
            ui_layer,
            ui_output: None,
            audio_engine,
            audio_bands,
            modulated_wells,
            audio_targets,
            last_target_refresh: Instant::now(),
            pending_reset: true,
            paused: false,
            last_frame: Instant::now(),
            frame_time_smooth: 1.0 / 60.0,
            frame_id: 0,
            clear_color: wgpu::Color {
                r: 0.01,
                g: 0.01,
                b: 0.017,
                a: 1.0,
            },
        })
    }

    fn window_id(&self) -> WindowId {
        self.window.id()
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.camera
            .set_aspect(self.config.width as f32 / self.config.height as f32);
        self.trail.resize(
            &self.device,
            self.config.width.max(1),
            self.config.height.max(1),
        );
        self.trail.clear(&self.device, &self.queue);
    }

    fn update(&mut self) {
        let now = Instant::now();
        let frame_dt = (now - self.last_frame).as_secs_f32().max(1e-6);
        self.last_frame = now;
        self.frame_time_smooth = self.frame_time_smooth * 0.9 + frame_dt * 0.1;

        self.run_ui();
        self.audio_engine.refresh(&self.settings.audio);
        self.audio_bands = self.audio_engine.bands();
        self.modulated_wells = self.settings.modulated_wells(self.audio_bands);

        let sim_dt = if self.paused {
            0.0
        } else {
            (frame_dt * self.settings.time_scale).clamp(0.0001, self.settings.dt)
        };

        self.dispatch_count = self.settings.dispatch_count();

        self.sim_uniform.update(
            sim_dt,
            self.frame_id,
            std::mem::take(&mut self.pending_reset),
            &self.modulated_wells,
            &self.settings,
        );
        self.queue
            .write_buffer(&self.sim_buffer, 0, bytemuck::bytes_of(&self.sim_uniform));

        self.camera_uniform.update(
            self.camera.view_proj(),
            self.camera.eye(),
            self.settings.point_size,
            self.settings.exposure,
            Vec2::new(self.size.width as f32, self.size.height as f32),
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&self.camera_uniform),
        );
        self.trail.update_uniforms(
            &self.queue,
            self.settings.trail_decay,
            self.settings.trail_intensity,
            self.settings.exposure,
        );
    }

    fn run_ui(&mut self) {
        let window = self.window.clone();
        let window_ref = window.as_ref();
        let raw_input = self.ui_layer.take_input(window_ref, self.size);
        let egui_ctx = self.ui_layer.context().clone();
        if self.last_target_refresh.elapsed() > Duration::from_secs(3) {
            self.audio_engine.refresh_targets();
            self.audio_targets = self.audio_engine.targets();
            self.last_target_refresh = Instant::now();
        }
        let full_output = egui_ctx.run(raw_input, |ctx| self.build_ui(ctx));
        self.ui_output = Some(self.ui_layer.process_output(window_ref, full_output));
    }

    fn build_ui(&mut self, ctx: &Context) {
        use egui::{Slider, pos2};

        let fps = 1.0 / self.frame_time_smooth.max(1e-6);
        let dispatch_preview = self.settings.dispatch_count();

        egui::Window::new("Simulation Controls")
            .default_width(320.0)
            .default_pos(pos2(16.0, 16.0))
            .show(ctx, |ui| {
                ui.label("Tweak the integrator and tone map settings in real time.");
                ui.horizontal(|ui| {
                    if ui
                        .button(if self.paused { "Resume" } else { "Pause" })
                        .clicked()
                    {
                        self.toggle_pause();
                    }
                    if ui.button("Reset").clicked() {
                        self.queue_reset();
                    }
                });
                ui.separator();
                ui.label("Playback");
                ui.add(Slider::new(&mut self.settings.time_scale, 0.1..=4.0).text("time scale"));
                ui.add(Slider::new(&mut self.settings.dt, 0.0005..=0.02).text("target dt"));
                ui.separator();
                ui.label("Flow Field");
                ui.add(Slider::new(&mut self.settings.flow, 0.001..=20.0).text("flow"));
                ui.add(Slider::new(&mut self.settings.damping, 0.0..=0.99).text("damping"));
                ui.add(Slider::new(&mut self.settings.color_mix, 0.01..=1.0).text("color mix"));
                ui.add(Slider::new(&mut self.settings.jitter, 0.0..=0.01).text("jitter"));
                ui.add(Slider::new(&mut self.settings.drive, 0.0..=0.5).text("drive"));
                CollapsingHeader::new("Attractor coefficients")
                    .default_open(true)
                    .show(ui, |ui| {
                        let labels = ["a", "b", "c", "d"];
                        for (idx, label) in labels.iter().enumerate() {
                            ui.add(
                                Slider::new(&mut self.settings.attractor[idx], -30.0..=30.0)
                                    .text(*label),
                            );
                        }
                    });
                ui.separator();
                ui.label("Rendering");
                ui.add(Slider::new(&mut self.settings.point_size, 0.1..=12.0).text("point size"));
                ui.add(
                    Slider::new(&mut self.settings.exposure, 0.0..=3.0)
                        .step_by(0.00001)
                        .text("exposure"),
                );
                ui.add(
                    Slider::new(&mut self.settings.trail_decay, 0.5..=0.999)
                        .logarithmic(true)
                        .text("trail decay"),
                );
                ui.add(
                    Slider::new(&mut self.settings.trail_intensity, 0.1..=5.0).text("trail gain"),
                );
            });

        gravity_window(ctx, &mut self.settings, self.audio_bands);
        audio_window(ctx, &mut self.settings, &self.audio_targets);

        egui::TopBottomPanel::bottom("status_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Particles: {} · Dispatch: {}",
                        self.settings.particle_count, dispatch_preview
                    ));
                    ui.separator();
                    ui.label(format!(
                        "#{} FPS: {fps:4.1} dt: {:2.8}",
                        self.frame_id, self.frame_time_smooth
                    ));
                    ui.separator();
                    ui.label("Hold right mouse to orbit, middle to pan, scroll to zoom");
                });
            });
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let suboptimal = frame.suboptimal;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });

        if !self.paused {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attractor Compute"),
                ..Default::default()
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.sim_bind_group, &[]);
            compute_pass.dispatch_workgroups(self.dispatch_count, 1, 1);
        }

        let (trail_read_idx, trail_write_idx) = self.trail.prepare_indices();
        self.trail
            .run_decay(&mut encoder, trail_read_idx, trail_write_idx);

        {
            let trail_view = self.trail.particle_target(trail_write_idx);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Attractor Render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: trail_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.particle_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.quad_vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..self.settings.particle_count);
        }

        self.trail
            .present(&mut encoder, trail_write_idx, &view, self.clear_color);
        self.trail.advance(trail_write_idx);

        if let Some(output) = self.ui_output.take() {
            self.ui_layer
                .paint(&self.device, &self.queue, &mut encoder, &view, output);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        if suboptimal {
            self.resize(self.window.inner_size());
        }
        self.frame_id = self.frame_id.wrapping_add(1);
        Ok(())
    }

    fn handle_surface_error(&mut self, error: SurfaceError) -> bool {
        match error {
            SurfaceError::Lost => {
                self.resize(self.size);
                false
            }
            SurfaceError::Outdated => {
                self.resize(self.size);
                false
            }
            SurfaceError::Timeout => {
                warn!("surface timeout");
                false
            }
            SurfaceError::Other => {
                warn!("surface error: other");
                false
            }
            SurfaceError::OutOfMemory => {
                error!("GPU ran out of memory");
                true
            }
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    fn queue_reset(&mut self) {
        self.pending_reset = true;
    }
}
