/**
 * @file volume.vert
 * @brief Fullscreen-triangle vertex shader for volume raymarch rendering.
 *
 * Emits clip-space positions for a single fullscreen triangle and forwards
 * normalized UV coordinates to the fragment stage. This minimizes vertex state
 * and keeps the volume pass focused on fragment-side integration logic.
 */

#version 450

layout(location = 0) out vec2 v_uv;

void main() {
    // Oversized fullscreen triangle avoids shared-edge precision seams from two-triangle quads.
    const vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    const vec2 position = positions[gl_VertexIndex];
    v_uv = 0.5 * (position + 1.0);
    gl_Position = vec4(position, 0.0, 1.0);
}
