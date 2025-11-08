struct CameraUniform {
    view_proj : mat4x4<f32>,
    eye : vec4<f32>,
    params : vec4<f32>, // point_size, exposure, viewport_width, viewport_height
};

@group(0) @binding(0) var<uniform> camera : CameraUniform;

struct InstanceInput {
    @location(0) position : vec4<f32>,
    @location(1) velocity : vec4<f32>,
};

struct CornerInput {
    @location(2) corner : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) color : vec3<f32>,
    @location(1) sparkle : f32,
    @location(2) corner : vec2<f32>,
};

fn palette(speed : f32, dir : vec3<f32>) -> vec3<f32> {
    let base = vec3<f32>(0.1, 0.2, 0.8);
    let tint = normalize(abs(dir) + 0.001);
    let strength = clamp(speed * 0.35, 0.0, 1.0);
    return mix(base, tint, strength);
}

@vertex
fn vs_main(instance : InstanceInput, quad : CornerInput) -> VertexOutput {
    var out : VertexOutput;
    let world = vec4<f32>(instance.position.xyz, 1.0);
    let clip = camera.view_proj * world;

    let viewport = max(vec2<f32>(camera.params.z, camera.params.w), vec2<f32>(1.0));
    let pixel_extent = quad.corner * camera.params.x;
    let ndc_delta = vec2<f32>(
        pixel_extent.x * (2.0 / viewport.x),
        -pixel_extent.y * (2.0 / viewport.y)
    );
    let clip_offset = vec4<f32>(ndc_delta * clip.w, 0.0, 0.0);
    out.clip_position = clip + clip_offset;

    let vel_dir = normalize(instance.velocity.xyz + vec3<f32>(1e-4));
    let speed = max(instance.position.w, 1e-4);
    out.color = palette(speed, vel_dir);
    out.sparkle = speed;
    out.corner = quad.corner * 2.0;
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    let exposure = camera.params.y;
    let glow = pow(clamp(in.sparkle * exposure, 0.0, 8.0), 0.65);
    let mask = clamp(1.0 - length(in.corner), 0.0, 1.0);
    let alpha = glow * mask;
    let color = in.color * alpha;
    return vec4<f32>(color, alpha);
}
