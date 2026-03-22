/**
 * @file volume.frag
 * @brief Multi-field volumetric raymarch shader for tornado visualization.
 *
 * Intersects camera rays with a unit volume, samples normalized density fields,
 * and accumulates physically motivated in-scatter/transmittance terms. Supports
 * isolated/cycled component rendering or blended supercell composites with
 * per-field palettes and optional cinematic black-and-white grading.
 */

#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

const int MAX_COMPONENTS = 8;

layout(set = 0, binding = 0) uniform sampler3D u_volume[MAX_COMPONENTS];

layout(push_constant) uniform PushConstants {
    vec4 camera_pos;      // xyz used
    vec4 camera_forward;  // xyz used, w=render mode (0=supercell, 1=composite, 2=isolated/cycle)
    vec4 camera_right;    // xyz used, w=selected component index
    vec4 camera_up;       // xyz used, w=active component count
    vec4 render0;         // x=aspect, y=tanHalfFov, z=opacityScale, w=threshold
    vec4 render1;         // x=brightness, y=ambient, z=anisotropy, w=maxDistance
    vec4 render2;         // xyz=sun direction, w=mode flags (bit0=supercell, bit1=cinematic-bw, bit2=natural-detail)
    vec4 volume_dims;     // xyz=texture dimensions, w=step count
} pc;

const float PI = 3.14159265359;

/** @brief Ray-box intersection for unit-volume bounds testing. */
bool intersect_aabb(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float t_near, out float t_far) {
    vec3 inv_dir = 1.0 / rd;
    vec3 t0 = (bmin - ro) * inv_dir;
    vec3 t1 = (bmax - ro) * inv_dir;

    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);

    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far = min(min(tmax.x, tmax.y), tmax.z);

    return t_far > max(t_near, 0.0);
}

/** @brief Convert normalized sample value into opacity-driving density. */
float density_from_sample(float sample_value, float threshold) {
    float d = smoothstep(threshold, 1.0, sample_value);
    return d * d;
}

/** @brief Stable world-space pseudo-turbulence used to break synthetic smoothness. */
float procedural_detail(vec3 uvw, float seed) {
    vec3 q = uvw * 15.0 + vec3(seed, seed * 1.41, seed * 2.07);
    float n0 = sin(dot(q, vec3(11.73, 19.91, 7.27)));
    float n1 = sin(dot(q * 1.87 + vec3(0.37, 0.91, 0.53), vec3(17.13, 5.77, 13.39)));
    return clamp(0.5 + 0.28 * n0 + 0.22 * n1, 0.0, 1.0);
}

/** @brief Apply small-scale heterogeneity while keeping the macro flow intact. */
float apply_texture_detail(float density, vec3 uvw, float seed, bool natural_texture) {
    if (!natural_texture) {
        return density;
    }

    float d = procedural_detail(uvw, seed);
    float modulation = mix(0.78, 1.24, d);
    return clamp(density * modulation, 0.0, 1.0);
}

/** @brief Clamp and fetch one component sample from descriptor array. */
float sample_component(int component_index, vec3 uvw) {
    int idx = clamp(component_index, 0, MAX_COMPONENTS - 1);
    return texture(u_volume[idx], clamp(uvw, 0.0, 1.0)).r;
}

/** @brief Return palette seed color for the requested atmospheric component index. */
vec3 component_palette(int component_index, bool cinematic_bw) {
    if (cinematic_bw) {
        float tone = clamp(0.22 + 0.09 * float(component_index), 0.15, 0.92);
        return vec3(tone);
    }

    const vec3 palette[MAX_COMPONENTS] = vec3[](
        vec3(0.95, 0.57, 0.24),
        vec3(0.28, 0.75, 0.98),
        vec3(0.42, 0.83, 0.53),
        vec3(0.23, 0.43, 0.92),
        vec3(0.84, 0.95, 1.00),
        vec3(0.73, 0.56, 0.94),
        vec3(0.97, 0.83, 0.36),
        vec3(0.94, 0.42, 0.64));
    return palette[clamp(component_index, 0, MAX_COMPONENTS - 1)];
}

/** @brief Derive display color for one component sample and altitude. */
vec3 component_color(int component_index, float sample_value, float height, bool cinematic_bw) {
    vec3 base = component_palette(component_index, cinematic_bw);
    vec3 high = cinematic_bw ? vec3(0.95) : mix(base, vec3(0.94, 0.97, 1.00), 0.45);
    vec3 shaded = mix(base * 0.55, high, clamp(0.22 + 0.78 * height, 0.0, 1.0));

    float energized = smoothstep(0.60, 1.0, sample_value);
    vec3 spark = cinematic_bw ? vec3(0.98) : vec3(1.00, 0.91, 0.74);
    return mix(shaded, spark, energized * 0.28);
}

/** @brief Sample the lighting-normal field used for gradient estimation. */
float lighting_sample(vec3 uvw, int render_mode, int selected_component) {
    if (render_mode == 2) {
        return sample_component(selected_component, uvw);
    }
    return sample_component(0, uvw);
}

/** @brief Estimate local density gradient via central differences in texture space. */
vec3 gradient_from_volume(vec3 uvw, vec3 texel, int render_mode, int selected_component) {
    vec3 dx = vec3(texel.x, 0.0, 0.0);
    vec3 dy = vec3(0.0, texel.y, 0.0);
    vec3 dz = vec3(0.0, 0.0, texel.z);

    float px = lighting_sample(clamp(uvw + dx, 0.0, 1.0), render_mode, selected_component);
    float nx = lighting_sample(clamp(uvw - dx, 0.0, 1.0), render_mode, selected_component);
    float py = lighting_sample(clamp(uvw + dy, 0.0, 1.0), render_mode, selected_component);
    float ny = lighting_sample(clamp(uvw - dy, 0.0, 1.0), render_mode, selected_component);
    float pz = lighting_sample(clamp(uvw + dz, 0.0, 1.0), render_mode, selected_component);
    float nz = lighting_sample(clamp(uvw - dz, 0.0, 1.0), render_mode, selected_component);

    return vec3(px - nx, py - ny, pz - nz);
}

/** @brief Compute Henyey-Greenstein phase response for anisotropic scattering. */
float henyey_greenstein(float cos_theta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-3);
    return (1.0 - g2) / (4.0 * PI * pow(denom, 1.5));
}

/** @brief Build procedural sky color used behind and through the volume. */
vec3 sky_color(vec3 ray_dir, vec3 light_dir, bool cinematic_bw) {
    float h = clamp(ray_dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 horizon = cinematic_bw ? vec3(0.20) : vec3(0.64, 0.73, 0.84);
    vec3 zenith = cinematic_bw ? vec3(0.05) : vec3(0.20, 0.33, 0.57);
    vec3 sky = mix(horizon, zenith, pow(h, 0.7));

    float sun = pow(max(dot(ray_dir, light_dir), 0.0), 256.0);
    vec3 sun_tint = cinematic_bw ? vec3(1.0) : vec3(1.0, 0.92, 0.78);
    sky += sun * sun_tint * 0.35;
    return sky;
}

/** @brief Sample one or many components and return density/color controls. */
void sample_multi_field(vec3 uvw,
                        int render_mode,
                        int selected_component,
                        int active_components,
                        bool supercell_mode,
                        bool cinematic_bw,
                        bool natural_texture,
                        out float out_density,
                        out float out_peak_sample,
                        out vec3 out_color) {
    out_density = 0.0;
    out_peak_sample = 0.0;
    out_color = component_color(0, 0.0, uvw.z, cinematic_bw);

    if (render_mode == 2) {
        int idx = clamp(selected_component, 0, active_components - 1);
        float sample_value = sample_component(idx, uvw);
        out_density = density_from_sample(sample_value, pc.render0.w);
        out_density = apply_texture_detail(out_density, uvw, float(idx) * 1.73, natural_texture);
        out_peak_sample = sample_value;
        out_color = component_color(idx, sample_value, uvw.z, cinematic_bw);
        return;
    }

    float weighted_density = 0.0;
    float weight_sum = 0.0;
    vec3 weighted_color = vec3(0.0);

    for (int i = 0; i < MAX_COMPONENTS; ++i) {
        if (i >= active_components) {
            break;
        }

        float sample_value = sample_component(i, uvw);
        float density = density_from_sample(sample_value, pc.render0.w);
        density = apply_texture_detail(density, uvw, float(i) * 1.73, natural_texture);

        float weight = (i == 0) ? 1.0 : 0.82;
        // Keep hydrometeor bands from overpowering the thermodynamic structure in supercell mode.
        if (supercell_mode && i >= 3) {
            weight *= 0.78;
        }

        weighted_density += density * weight;
        weight_sum += weight;
        weighted_color += component_color(i, sample_value, uvw.z, cinematic_bw) * density * weight;
        out_peak_sample = max(out_peak_sample, sample_value);
    }

    out_density = (weight_sum > 0.0) ? clamp(weighted_density / weight_sum, 0.0, 1.0) : 0.0;
    if (weighted_density > 1e-5) {
        out_color = weighted_color / weighted_density;
    }

    if (supercell_mode) {
        float anvil = smoothstep(0.58, 1.0, uvw.z) * smoothstep(0.30, 0.85, out_density);
        vec3 anvil_tint = cinematic_bw ? vec3(0.92) : vec3(0.84, 0.89, 0.98);
        out_color = mix(out_color, anvil_tint, anvil * 0.35);
    }
}

/** @brief Raymarch the 3D density texture and emit final composited radiance. */
void main() {
    vec2 ndc = v_uv * 2.0 - 1.0;

    vec3 ray_origin = pc.camera_pos.xyz;
    vec3 ray_dir = normalize(
        pc.camera_forward.xyz +
        ndc.x * pc.render0.x * pc.render0.y * pc.camera_right.xyz +
        ndc.y * pc.render0.y * pc.camera_up.xyz);

    vec3 light_dir = normalize(pc.render2.xyz);
    int render_mode = clamp(int(pc.camera_forward.w + 0.5), 0, 2);
    int active_components = clamp(int(pc.camera_up.w + 0.5), 1, MAX_COMPONENTS);
    int selected_component = clamp(int(pc.camera_right.w + 0.5), 0, active_components - 1);
    int mode_flags = int(pc.render2.w + 0.5);
    bool supercell_mode = (mode_flags & 1) != 0;
    bool cinematic_bw = (mode_flags & 2) != 0;
    bool natural_texture = (mode_flags & 4) != 0;

    float t_near = 0.0;
    float t_far = 0.0;
    if (!intersect_aabb(ray_origin, ray_dir, vec3(-0.5), vec3(0.5), t_near, t_far)) {
        out_color = vec4(sky_color(ray_dir, light_dir, cinematic_bw), 1.0);
        return;
    }

    t_near = max(t_near, 0.0);
    t_far = min(t_far, pc.render1.w);
    if (t_far <= t_near) {
        out_color = vec4(sky_color(ray_dir, light_dir, cinematic_bw), 1.0);
        return;
    }

    int steps = clamp(int(pc.volume_dims.w + 0.5), 32, 512);
    // Clamp march count so quality controls cannot trigger runaway GPU loops.
    float dt = (t_far - t_near) / float(steps);

    vec3 texel = 1.0 / max(pc.volume_dims.xyz, vec3(1.0));
    vec3 radiance = vec3(0.0);
    float transmittance = 1.0;
    float t = t_near;

    for (int i = 0; i < 512; ++i) {
        if (i >= steps || transmittance < 0.01) {
            break;
        }

        vec3 p = ray_origin + ray_dir * t;
        vec3 uvw = p + vec3(0.5);

        float density = 0.0;
        float peak_sample = 0.0;
        vec3 cloud_color = vec3(0.0);
        sample_multi_field(
            uvw,
            render_mode,
            selected_component,
            active_components,
            supercell_mode,
            cinematic_bw,
            natural_texture,
            density,
            peak_sample,
            cloud_color);

        if (density > 1e-4) {
            vec3 grad = gradient_from_volume(uvw, texel, render_mode, selected_component);
            // Epsilon avoids undefined normalize(0) when gradients flatten in quiescent regions.
            vec3 normal = normalize(grad + vec3(1e-5, 1e-5, 1e-5));

            float ndotl = max(dot(normal, light_dir), 0.0);
            float mu = dot(-ray_dir, light_dir);
            float phase = henyey_greenstein(mu, clamp(pc.render1.z, -0.85, 0.85));

            float height = clamp(uvw.z, 0.0, 1.0);

            float ambient = pc.render1.y * mix(0.35, 1.0, height);
            float diffuse = 0.2 + 0.8 * ndotl;
            vec3 ambient_tint = cinematic_bw ? vec3(0.72) : vec3(0.56, 0.63, 0.74);
            vec3 direct_tint = cinematic_bw ? vec3(1.0) : vec3(1.00, 0.96, 0.90);

            vec3 in_scatter = cloud_color * (
                ambient * ambient_tint +
                diffuse * direct_tint * (0.5 + 1.35 * phase));

            float core = smoothstep(0.75, 1.0, peak_sample);
            vec3 core_tint = cinematic_bw ? vec3(0.92) : vec3(1.0, 0.74, 0.50);
            in_scatter += core * core_tint * (cinematic_bw ? 0.03 : 0.08);

            float sigma_t = density * (2.3 * pc.render0.z);
            float alpha = clamp(1.0 - exp(-sigma_t * dt), 0.0, 1.0);

            // Beer-Lambert integration: accumulate in-scatter and decay residual throughput.
            radiance += transmittance * alpha * in_scatter * pc.render1.x;
            transmittance *= (1.0 - alpha);
        }

        t += dt;
    }

    vec3 sky = sky_color(ray_dir, light_dir, cinematic_bw);
    vec3 color = radiance + transmittance * sky;
    out_color = vec4(color, 1.0);
}
