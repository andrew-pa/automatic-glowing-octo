use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const TRAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const TRAIL_SHADER: &str = include_str!("../shaders/trails.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TrailUniform {
    pub decay: f32,
    pub intensity: f32,
    pub exposure: f32,
    pub padding: f32,
}

impl TrailUniform {
    pub fn new() -> Self {
        Self {
            decay: 0.92,
            intensity: 1.0,
            exposure: 0.6,
            padding: 0.0,
        }
    }

    pub fn update(&mut self, decay: f32, intensity: f32, exposure: f32) {
        self.decay = decay.clamp(0.0, 1.0);
        self.intensity = intensity.max(0.0);
        self.exposure = exposure.max(0.0);
    }
}

pub struct TrailComposer {
    accum_format: wgpu::TextureFormat,
    uniform: TrailUniform,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    decay_pipeline: wgpu::RenderPipeline,
    present_pipeline: wgpu::RenderPipeline,
    targets: [TrailTarget; 2],
    front: usize,
}

struct TrailTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl TrailComposer {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let accum_format = TRAIL_FORMAT;
        let surface_format = config.format;

        let uniform = TrailUniform::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Trail Uniform"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Trail Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Trail BindGroup Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Trail Shader"),
            source: wgpu::ShaderSource::Wgsl(TRAIL_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Trail Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let decay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Trail Decay Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_decay"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: accum_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Trail Present Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_present"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let targets = [
            TrailTarget::new(
                device,
                accum_format,
                config.width.max(1),
                config.height.max(1),
                "Trail Target A",
                &bind_group_layout,
                &uniform_buffer,
                &sampler,
            ),
            TrailTarget::new(
                device,
                accum_format,
                config.width.max(1),
                config.height.max(1),
                "Trail Target B",
                &bind_group_layout,
                &uniform_buffer,
                &sampler,
            ),
        ];

        Self {
            accum_format,
            uniform,
            uniform_buffer,
            sampler,
            bind_group_layout,
            decay_pipeline,
            present_pipeline,
            targets,
            front: 0,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.targets = [
            TrailTarget::new(
                device,
                self.accum_format,
                width,
                height,
                "Trail Target A",
                &self.bind_group_layout,
                &self.uniform_buffer,
                &self.sampler,
            ),
            TrailTarget::new(
                device,
                self.accum_format,
                width,
                height,
                "Trail Target B",
                &self.bind_group_layout,
                &self.uniform_buffer,
                &self.sampler,
            ),
        ];
        self.front = 0;
    }

    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        decay: f32,
        intensity: f32,
        exposure: f32,
    ) {
        self.uniform.update(decay, intensity, exposure);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
    }

    pub fn prepare_indices(&self) -> (usize, usize) {
        let read = self.front;
        let write = 1 - self.front;
        (read, write)
    }

    pub fn run_decay(&self, encoder: &mut wgpu::CommandEncoder, read_idx: usize, write_idx: usize) {
        let bind_group = self.targets[read_idx].bind_group();
        let view = self.targets[write_idx].view();
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Trail Decay Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        rp.set_pipeline(&self.decay_pipeline);
        rp.set_bind_group(0, bind_group, &[]);
        rp.draw(0..3, 0..1);
    }

    pub fn present(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_idx: usize,
        target_view: &wgpu::TextureView,
        clear_color: wgpu::Color,
    ) {
        let bind_group = self.targets[source_idx].bind_group();
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Trail Present Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        rp.set_pipeline(&self.present_pipeline);
        rp.set_bind_group(0, bind_group, &[]);
        rp.draw(0..3, 0..1);
    }

    pub fn particle_target(&self, write_idx: usize) -> &wgpu::TextureView {
        self.targets[write_idx].view()
    }

    pub fn advance(&mut self, write_idx: usize) {
        self.front = write_idx;
    }

    pub fn clear(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Trail Clear Encoder"),
        });
        for target in &self.targets {
            let view = target.view();
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Trail Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}

impl TrailTarget {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let extent = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        Self {
            _texture: texture,
            view,
            bind_group,
        }
    }

    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
