struct TrailUniform {
    decay : f32,
    intensity : f32,
    exposure : f32,
    padding : f32,
};

@group(0) @binding(0) var<uniform> trail : TrailUniform;
@group(0) @binding(1) var trail_sampler : sampler;
@group(0) @binding(2) var trail_texture : texture_2d<f32>;

struct VsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index : u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let pos = positions[vertex_index];
    var out : VsOut;
    out.clip_pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv = (pos * vec2<f32>(0.5, -0.5)) + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_decay(input : VsOut) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(trail_texture, trail_sampler, input.uv, 0.0);
    let decay = trail.decay;
    return vec4<f32>(color.rgb * decay, color.a * decay);
}

@fragment
fn fs_present(input : VsOut) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(trail_texture, trail_sampler, input.uv, 0.0).rgb;
    let boosted = color * max(trail.intensity, 0.0);
    let mapped = vec3<f32>(1.0) - exp(-boosted * max(trail.exposure, 1e-5));
    return vec4<f32>(mapped, 1.0);
}
