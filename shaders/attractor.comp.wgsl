struct SimUniform {
    integrator : vec4<f32>;
    attractor : vec4<f32>;
    misc : vec4<f32>;
    counters : vec4<u32>;
};

struct Particle {
    position : vec4<f32>;
    velocity : vec4<f32>;
};

@group(0) @binding(0) var<uniform> params : SimUniform;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;

const PI2 : f32 = 6.28318530718;

fn hash32(x_in : u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn hash11(seed : u32) -> f32 {
    return f32(hash32(seed)) / 4294967295.0;
}

fn hash33(seed : u32) -> vec3<f32> {
    return vec3<f32>(
        hash11(seed),
        hash11(seed ^ 0x68bc21u),
        hash11(seed ^ 0x02a2e9u)
    );
}

fn spawn_particle(index : u32, frame : u32) -> Particle {
    let rnd = hash33(index ^ frame);
    let radius = pow(rnd.x, 0.35) * 4.5;
    let theta = rnd.y * PI2;
    let phi = acos(clamp(rnd.z * 2.0 - 1.0, -1.0, 1.0));
    let pos = vec3<f32>(
        radius * sin(phi) * cos(theta),
        radius * cos(phi),
        radius * sin(phi) * sin(theta)
    );
    var p : Particle;
    p.position = vec4<f32>(pos, 0.0);
    p.velocity = vec4<f32>(vec3<f32>(0.0), 0.0);
    return p;
}

fn strange_field(pos : vec3<f32>) -> vec3<f32> {
    let a = params.attractor.x;
    let b = params.attractor.y;
    let c = params.attractor.z;
    let d = params.attractor.w;

    let clifford = vec3<f32>(
        sin(a * pos.y) + params.misc.x * cos(b * pos.x),
        sin(c * pos.x) - params.misc.x * cos(d * pos.z),
        sin(d * pos.z) - params.misc.x * cos(a * pos.y)
    );

    let thomas = vec3<f32>(
        sin(pos.y) - params.misc.y * pos.x,
        sin(pos.z) - params.misc.y * pos.y,
        sin(pos.x) - params.misc.y * pos.z
    );

    return mix(clifford, thomas, vec3<f32>(0.5));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let index = id.x;
    if (index >= params.counters.z) {
        return;
    }

    var particle = particles[index];
    if (params.counters.y != 0u) {
        particle = spawn_particle(index, params.counters.x);
    }

    let dt = params.integrator.x;
    let drive = params.integrator.y;
    let damping = params.integrator.z;
    let color_mix = params.integrator.w;

    var pos = particle.position.xyz;
    var vel = particle.velocity.xyz;

    if (dt > 0.0) {
        let accel = drive * strange_field(pos);
        vel = vel + dt * (accel - damping * vel);
        pos = pos + dt * vel;
        let jitter = (hash33(index * 3u + params.counters.x) - vec3<f32>(0.5)) * params.misc.x;
        pos += jitter;
    }

    let speed = length(vel) + 1e-5;
    let smoothed = particle.position.w + (speed - particle.position.w) * color_mix;
    particle.position = vec4<f32>(pos, smoothed);
    particle.velocity = vec4<f32>(vel, 0.0);
    particles[index] = particle;
}
