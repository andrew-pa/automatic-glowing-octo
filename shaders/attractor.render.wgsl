struct CameraUniform {
    view_proj : mat4x4<f32>;
    eye : vec4<f32>;
    tonemapping : vec4<f32>;
};

@group(0) @binding(0) var<uniform> camera : CameraUniform;

struct VertexInput {
    @location(0) position : vec4<f32>;
    @location(1) velocity : vec4<f32>;
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>;
    @builtin(point_size) point_size : f32;
    @location(0) color : vec3<f32>;
    @location(1) sparkle : f32;
};

fn palette(speed : f32, dir : vec3<f32>) -> vec3<f32> {
    let base = vec3<f32>(0.1, 0.2, 0.8);
    let tint = normalize(abs(dir) + 0.001);
    let strength = clamp(speed * 0.35, 0.0, 1.0);
    return mix(base, tint, strength);
}

@vertex
fn vs_main(in : VertexInput) -> VertexOutput {
    var out : VertexOutput;
    let world = vec4<f32>(in.position.xyz, 1.0);
    out.clip_position = camera.view_proj * world;
    let vel_dir = normalize(in.velocity.xyz + vec3<f32>(1e-4));
    let speed = max(in.position.w, 1e-4);
    out.color = palette(speed, vel_dir);
    out.sparkle = speed;
    out.point_size = camera.tonemapping.x;
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
    let exposure = camera.tonemapping.y;
    let glow = pow(clamp(in.sparkle * exposure, 0.0, 8.0), 0.65);
    let color = in.color * glow;
    return vec4<f32>(color, glow);
}
