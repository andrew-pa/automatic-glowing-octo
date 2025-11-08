use egui::{self, FullOutput, RawInput};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::{self, EventResponse, State as EguiWinitState};
use wgpu::{CommandEncoder, Device, Queue, TextureFormat, TextureView};
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::window::Window;

pub struct UiLayer {
    ctx: egui::Context,
    winit_state: EguiWinitState,
    renderer: Renderer,
    screen_desc: ScreenDescriptor,
}

pub struct UiFrameOutput {
    pub shapes: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
}

impl UiLayer {
    pub fn new(window: &Window, device: &Device, format: TextureFormat) -> Self {
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());
        let viewport = egui::ViewportId::ROOT;
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            viewport,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = Renderer::new(device, format, RendererOptions::default());
        let size = window.inner_size();
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [size.width.max(1), size.height.max(1)],
            pixels_per_point: egui_winit::pixels_per_point(&ctx, window),
        };
        Self {
            ctx,
            winit_state,
            renderer,
            screen_desc,
        }
    }

    pub fn context(&self) -> &egui::Context {
        &self.ctx
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    pub fn take_input(&mut self, window: &Window, size: PhysicalSize<u32>) -> RawInput {
        self.screen_desc.size_in_pixels = [size.width.max(1), size.height.max(1)];
        self.screen_desc.pixels_per_point = egui_winit::pixels_per_point(&self.ctx, window);
        self.winit_state.take_egui_input(window)
    }

    pub fn process_output(&mut self, window: &Window, output: FullOutput) -> UiFrameOutput {
        let platform_output = output.platform_output;
        self.winit_state
            .handle_platform_output(window, platform_output);
        let shapes = self
            .ctx
            .tessellate(output.shapes, self.screen_desc.pixels_per_point);
        UiFrameOutput {
            shapes,
            textures_delta: output.textures_delta,
        }
    }

    pub fn paint(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        mut output: UiFrameOutput,
    ) {
        for (id, delta) in output.textures_delta.set.drain(..) {
            self.renderer.update_texture(device, queue, id, &delta);
        }

        let user_commands =
            self.renderer
                .update_buffers(device, queue, encoder, &output.shapes, &self.screen_desc);
        if !user_commands.is_empty() {
            queue.submit(user_commands);
        }

        if !output.shapes.is_empty() {
            let pass_desc = wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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
            };
            let render_pass = encoder.begin_render_pass(&pass_desc);
            let mut render_pass = render_pass.forget_lifetime();
            self.renderer
                .render(&mut render_pass, &output.shapes, &self.screen_desc);
        }

        for id in output.textures_delta.free {
            self.renderer.free_texture(&id);
        }
    }
}
