/**
 * @file volume_backend.cpp
 * @brief Multi-field volume rendering backend implementation for Vulkan.
 *
 * Resolves requested atmospheric fields from export datasets, normalizes
 * per-frame component data, uploads one 3D texture per field, and records
 * fullscreen raymarch passes. Also handles playback stepping and
 * swapchain-triggered pipeline recreation.
 */

#include "rendering/volume_backend.hpp"

#include "camera/camera_controller.hpp"
#include "core/context.hpp"
#include "data/shm_dataset.hpp"

#include "util/string_utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vkcpp 
{
namespace 
{

constexpr VkFormat kVolumeFormat = VK_FORMAT_R32_SFLOAT;
constexpr const char* kVertexShaderRelativePath = "shaders/rendering/volume.vert.spv";
constexpr const char* kFragmentShaderRelativePath = "shaders/rendering/volume.frag.spv";
constexpr float kPi = 3.14159265359f;


std::vector<std::string> split_csv_fields(const std::string& csv) 
{
    std::vector<std::string> out;

    std::size_t start = 0;
    while (start <= csv.size()) 
    {
        const std::size_t comma = csv.find(',', start);
        if (comma == std::string::npos) 
        {
            const std::string token = tmv::strutil::trim_copy(csv.substr(start));
            if (!token.empty()) {out.push_back(token);}
            break;
        }

        const std::string token = tmv::strutil::trim_copy(csv.substr(start, comma - start));
        if (!token.empty()) {out.push_back(token);}

        start = comma + 1;
    }

    return out;
}

std::vector<std::string> field_alias_candidates(const std::string& requested_field) 
{
    std::vector<std::string> candidates;
    std::unordered_set<std::string> seen;

    const auto push_unique = [&](const std::string& name)
    {
        if (name.empty()) {return;}
        if (seen.insert(name).second) {candidates.push_back(name);}
    };

    push_unique(requested_field);

    if (requested_field == "qr") 
    {
        push_unique("qrain");
        push_unique("rain");
    } 
    else if (requested_field == "qs") 
    {
        push_unique("q_snow");
        push_unique("snow");
        push_unique("qice");
    } 
    else if (requested_field == "qg") 
    {
        push_unique("q_graupel");
        push_unique("graupel");
        push_unique("qh");
        push_unique("hail");
    } 
    else if (requested_field == "qi") 
    {
        push_unique("q_ice");
        push_unique("ice");
        push_unique("qice");
        push_unique("qh");
        push_unique("hail");
    } 
    else if (requested_field == "qc") 
    {
        push_unique("q_cloud");
        push_unique("cloud");
    } 
    else if (requested_field == "w") 
    {
        push_unique("wz");
        push_unique("vertical_velocity");
        push_unique("updraft");
    } 
    else if (requested_field == "theta") 
    {
        push_unique("pt");
        push_unique("potential_temperature");
        push_unique("temperature");
    } 
    else if (requested_field == "vorticity" || requested_field == "zeta" || requested_field == "vertical_vorticity") 
    {
        push_unique("vorticity_z");
        push_unique("vorticity_magnitude");
        push_unique("omega");
    } 
    else if (requested_field == "vorticity_z") 
    {
        push_unique("zeta");
        push_unique("vorticity");
        push_unique("vertical_vorticity");
    } 
    else if (requested_field == "vorticity_magnitude") 
    {
        push_unique("vorticity");
        push_unique("zeta");
        push_unique("vorticity_z");
    }

    return candidates;
}

bool normalize_volume_mode(const std::string& input, std::string& normalized_mode) 
{
    const std::string mode = tmv::strutil::trim_and_lower(input);
    if (mode.empty() || mode == "supercell") 
    {
        normalized_mode = "supercell";
        return true;
    }
    if (mode == "composite" || mode == "together" || mode == "blend") 
    {
        normalized_mode = "composite";
        return true;
    }
    if (mode == "isolated" || mode == "isolate" || mode == "single") 
    {
        normalized_mode = "isolated";
        return true;
    }
    if (mode == "cycle" || mode == "animate" || mode == "animated") 
    {
        normalized_mode = "cycle";
        return true;
    }
    return false;
}

bool normalize_texture_mode(const std::string& input, std::string& normalized_mode) 
{
    const std::string mode = tmv::strutil::trim_and_lower(input);
    if (mode.empty() || mode == "natural" || mode == "realistic" || mode == "detailed") 
    {
        normalized_mode = "natural";
        return true;
    }
    if (mode == "smooth" || mode == "linear" || mode == "soft") 
    {
        normalized_mode = "smooth";
        return true;
    }
    return false;
}

bool resolve_field_index_by_alias(const std::vector<std::string>& resolved_field_names,
                                  const std::string& requested_field, std::size_t& out_index, std::string& out_name) 
{
    if (resolved_field_names.empty()) {return false;}

    const auto find_exact = [&](const std::string& name) -> bool 
    {
        for (std::size_t i = 0; i < resolved_field_names.size(); ++i) 
        {
            if (resolved_field_names[i] == name) 
            {
                out_index = i;
                out_name = resolved_field_names[i];
                return true;
            }
        }
        return false;
    };

    if (requested_field.empty()) {return false;}

    const std::vector<std::string> candidates = field_alias_candidates(requested_field);
    for (const std::string& candidate : candidates) 
    {
        if (find_exact(candidate)) {return true;}
    }

    return false;
}

std::string join_fields_csv(const std::vector<std::string>& fields) 
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < fields.size(); ++i) 
    {
        if (i > 0) { oss << ",";} oss << fields[i];
    }
    return oss.str();
}

bool supports_volume_format(VkPhysicalDevice physical_device) 
{
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(physical_device, kVolumeFormat, &properties);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

bool resolve_shader_path(const std::string& relative, std::filesystem::path& resolved_path) 
{
    const std::array<std::filesystem::path, 3> candidates = 
    {
        std::filesystem::path(relative),
        std::filesystem::path("vulkan") / relative,
        std::filesystem::path("..") / "vulkan" / relative,
    };

    for (const auto& candidate : candidates) 
    {
        if (std::filesystem::exists(candidate)) 
        {
            resolved_path = candidate;
            return true;
        }
    }

    resolved_path = candidates[1];
    return false;
}

bool read_binary_file(const std::filesystem::path& path, std::vector<char>& data, std::string& error) 
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) 
    {
        error = "failed to open shader file: " + path.string();
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) 
    {
        error = "shader file is empty: " + path.string();
        return false;
    }

    data.resize(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);

    if (!file.read(data.data(), size)) 
    {
        error = "failed to read shader file: " + path.string();
        return false;
    }

    return true;
}

bool create_shader_module(VkDevice device, const std::vector<char>& code, VkShaderModule& module, std::string& error) 
{
    if (code.size() % 4 != 0) 
    {
        error = "invalid SPIR-V bytecode size (must be 4-byte aligned)";
        return false;
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

    const VkResult result = vkCreateShaderModule(device, &create_info, nullptr, &module);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateShaderModule failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/**
 * @brief Compute safe voxel count for `(width,height,depth)` volume dimensions.
 */
bool checked_voxel_count(uint32_t width,
                         uint32_t height,
                         uint32_t depth,
                         std::size_t& out_voxels,
                         std::string& error) {
    if (width == 0 || height == 0 || depth == 0) 
    {
        error = "volume dimensions must be non-zero";
        return false;
    }

    const std::size_t sx = static_cast<std::size_t>(width);
    const std::size_t sy = static_cast<std::size_t>(height);
    const std::size_t sz = static_cast<std::size_t>(depth);
    if (sx > std::numeric_limits<std::size_t>::max() / sy) 
    {
        error = "volume dimensions overflow size_t (width*height)";
        return false;
    }
    const std::size_t xy = sx * sy;
    if (xy > std::numeric_limits<std::size_t>::max() / sz) 
    {
        error = "volume dimensions overflow size_t (width*height*depth)";
        return false;
    }

    out_voxels = xy * sz;
    return true;
}

}  // namespace

/** @brief Construct backend with default volume configuration. */
VolumeBackend::VolumeBackend()
    : VolumeBackend(Config{}) {}

/** @brief Construct backend with user-provided runtime configuration. */
VolumeBackend::VolumeBackend(Config config)
    : config_(std::move(config)) {}

/** @brief Initialize dataset state, GPU volume resources, and graphics pipeline. */
bool VolumeBackend::initialize(VulkanContext& context,
                               VkRenderPass render_pass,
                               VkExtent2D extent,
                               std::string& error) {
    shutdown(VK_NULL_HANDLE);

    device_ = context.device();
    render_pass_ = render_pass;
    extent_ = extent;
    pipeline_scaffold_ready_ = false;
    playback_clock_initialized_ = false;
    playback_accumulator_seconds_ = 0.0;

    if (!parse_requested_fields(error)) {return false;}

    if (!scan_datasets(error)) {return false;}

    // Pre-compute cross-frame normalization for each field.
    // This scans all frames to find the global absolute maximum so that
    // early weak signals don't self-normalize to fill the volume.
    for (auto& dataset : datasets_)
    {
        const float mag = dataset.compute_global_magnitude();
        if (mag > 0.0f)
        {
            dataset.set_global_magnitude(mag);
            std::cerr << "[vulkan][volume] " << dataset.field_name()
                      << ": cross-frame bipolar normalization, magnitude=" << mag << std::endl;
        }
        else
        {
            std::cerr << "[vulkan][volume] " << dataset.field_name()
                      << ": unipolar (magnitude=" << mag << ")" << std::endl;
        }
    }

    if (!load_initial_frame(error)) {return false;}

    if (!create_volume_resources(context, error)) {return false;}

    return ensure_pipeline_scaffold(error);
}

/** @brief Rebuild swapchain-dependent pipeline state after resize. */
bool VolumeBackend::on_swapchain_recreated(VkRenderPass render_pass, VkExtent2D extent, std::string& error) 
{
    render_pass_ = render_pass;
    extent_ = extent;

    pipeline_scaffold_ready_ = false;
    return ensure_pipeline_scaffold(error);
}

/** @brief Record one fullscreen volume raymarch draw into a command buffer. */
bool VolumeBackend::record(VkCommandBuffer command_buffer, VkFramebuffer framebuffer, VkExtent2D extent, std::string& error) 
{
    if (!pipeline_scaffold_ready_) 
    {
        if (!ensure_pipeline_scaffold(error)) {return false;}
    }

    if (!volume_resources_ready_) {error = "volume resources are not initialized";return false;}

    if (render_pass_ == VK_NULL_HANDLE || pipeline_ == VK_NULL_HANDLE || pipeline_layout_ == VK_NULL_HANDLE) 
    {
        error = "volume backend graphics pipeline is not initialized";
        return false;
    }

    if (extent.width == 0 || extent.height == 0) {extent = extent_;}

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    const VkResult begin_result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (begin_result != VK_SUCCESS) 
    {
        error = "vkBeginCommandBuffer failed with VkResult=" + std::to_string(static_cast<int>(begin_result));
        return false;
    }

    VkClearValue clear_color{};
    clear_color.color = clear_color_;

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = framebuffer;
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = extent;
    render_pass_info.clearValueCount = 1;
    render_pass_info.pClearValues = &clear_color;

    vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout_,
        0,
        1,
        &descriptor_set_,
        0,
        nullptr);

    const camera::Pose camera_pose = camera::build_pose(
        resolved_camera_mode_,
        camera_orbit_angle_rad_,
        config_.camera_distance,
        config_.camera_height,
        camera_position_,
        camera_yaw_rad_,
        camera_pitch_rad_);

    std::array<float, 3> sun_dir = config_.sun_dir;
    const float sun_len = std::sqrt(
        sun_dir[0] * sun_dir[0] +
        sun_dir[1] * sun_dir[1] +
        sun_dir[2] * sun_dir[2]);
    if (sun_len < 1e-6f) 
    {
        sun_dir = {0.66f, 0.34f, 0.67f};
    } 
    else 
    {
        sun_dir[0] /= sun_len;
        sun_dir[1] /= sun_len;
        sun_dir[2] /= sun_len;
    }

    const float aspect = extent.height > 0
                             ? static_cast<float>(extent.width) / static_cast<float>(extent.height)
                             : 1.0f;
    const float tan_half_fov = std::tan(config_.camera_fov_deg * 0.5f * (kPi / 180.0f));

    PushConstants push_constants{};
    push_constants.camera_pos = {camera_pose.position.x, camera_pose.position.y, camera_pose.position.z, 1.0f};
    int render_mode_id = 0;

    if (resolved_volume_mode_ == "composite") 
    {
        render_mode_id = 1;
    } 
    else if (resolved_volume_mode_ == "isolated" || resolved_volume_mode_ == "cycle") 
    {
        render_mode_id = 2;
    }
    const std::size_t selected_component =
        (resolved_volume_mode_ == "cycle") ? cycle_field_index_ : isolated_field_index_;
    // Pack render-mode metadata into otherwise-unused `.w` vector components.
    push_constants.camera_forward = 
    {
        camera_pose.forward.x,
        camera_pose.forward.y,
        camera_pose.forward.z,
        static_cast<float>(render_mode_id),
    };

    push_constants.camera_right = 
    {
        camera_pose.right.x,
        camera_pose.right.y,
        camera_pose.right.z,
        static_cast<float>(selected_component),
    };

    push_constants.camera_up = 
    {
        camera_pose.up.x,
        camera_pose.up.y,
        camera_pose.up.z,
        static_cast<float>(active_component_count_),
    };

    push_constants.render0 =
    {
        aspect,
        tan_half_fov,
        config_.ray_opacity,
        config_.ray_threshold,
    };

    push_constants.render1 = 
    {
        config_.ray_brightness,
        config_.ray_ambient,
        config_.ray_anisotropy,
        config_.ray_max_distance,
    };
    // Shader mode bits: bit0=supercell tinting, bit1=cinematic-bw, bit2=natural texture detail.
    const int mode_flags =
        (supercell_shading_enabled_ ? 1 : 0) |
        (config_.cinematic_bw ? 2 : 0) |
        (natural_texture_enabled_ ? 4 : 0);

    push_constants.render2 = 
    {
        sun_dir[0],
        sun_dir[1],
        sun_dir[2],
        static_cast<float>(mode_flags),
    };

    push_constants.volume_dims = 
    {
        static_cast<float>(volume_width_),
        static_cast<float>(volume_height_),
        static_cast<float>(volume_depth_),
        static_cast<float>(config_.ray_steps),
    };

    vkCmdPushConstants(
        command_buffer,
        pipeline_layout_,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        static_cast<uint32_t>(sizeof(PushConstants)),
        &push_constants);

    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(command_buffer);

    const VkResult end_result = vkEndCommandBuffer(command_buffer);
    if (end_result != VK_SUCCESS) 
    {
        error = "vkEndCommandBuffer failed with VkResult=" + std::to_string(static_cast<int>(end_result));
        return false;
    }

    return true;
}

/** @brief Advance playback clock and upload new frame data when needed. */
bool VolumeBackend::update_for_frame(VulkanContext& context, std::uint64_t, std::string& error) 
{
    if (!volume_resources_ready_) {return true;}

    const auto now = std::chrono::steady_clock::now();

    const float camera_dt = camera::compute_delta_seconds(
        camera_clock_initialized_,
        last_camera_update_time_,
        now,
        camera_input_.delta_seconds);

    // Scroll-wheel zoom: adjust camera distance in all modes
    if (std::abs(camera_input_.scroll_delta) > 0.01f)
    {
        const float zoom_speed = 0.15f;
        config_.camera_distance -= camera_input_.scroll_delta * zoom_speed;
        config_.camera_distance = std::clamp(config_.camera_distance, 0.80f, 10.0f);
    }

    // Left-mouse-drag orbit: rotate camera in orbit/fixed modes
    if (camera_input_.orbit_drag_active &&
        (resolved_camera_mode_ == "orbit" || resolved_camera_mode_ == "fixed"))
    {
        camera_orbit_angle_rad_ -= static_cast<double>(camera_input_.orbit_drag_delta_x) * 0.005;
        config_.camera_height -= camera_input_.orbit_drag_delta_y * 0.003f;
        config_.camera_height = std::clamp(config_.camera_height, -0.5f, 2.5f);
    }

    if (resolved_camera_mode_ == "orbit")
    {
        // Auto-orbit only when not manually dragging
        if (!camera_input_.orbit_drag_active)
        {
            camera::advance_orbit(camera_orbit_angle_rad_, config_.camera_orbit_fps, camera_dt);
        }
    }
    else if (resolved_camera_mode_ == "freefly") 
    {
        camera::update_freefly_pose(
            camera_pose_initialized_,
            camera_position_,
            camera_yaw_rad_,
            camera_pitch_rad_,
            camera_orbit_angle_rad_,
            config_.camera_distance,
            config_.camera_height,
            camera_input_,
            camera_dt);
    }

    // Poll SHM live sources for new simulation data (independent of disk playback).
    // When the simulation writes a new frame, the sequence number changes and
    // we reload + re-upload all SHM-connected fields.
    {
        bool any_new = false;
        for (std::size_t i = 0; i < shm_datasets_.size(); ++i)
        {
            if (shm_datasets_[i].has_new_frame())
            {
                any_new = true;
                break;
            }
        }
        if (any_new)
        {
            // Reload from SHM and re-upload to GPU. This bypasses
            // advance_to_next_frame() which early-returns when frame_count<=1.
            std::vector<FieldFrame> field_frames;
            std::string shm_reload_error;
            if (load_dataset_frame(0, field_frames, shm_reload_error))
            {
                current_field_frames_ = std::move(field_frames);
                if (!upload_volume_data(context, shm_reload_error))
                {
                    std::cerr << "[vulkan][volume] SHM upload error: " << shm_reload_error << "\n";
                }
            }
            else
            {
                std::cerr << "[vulkan][volume] SHM reload error: " << shm_reload_error << "\n";
            }
        }
    }

    if (frame_count_ > 1 && config_.playback_fps > 0.0f)
    {
        if (!playback_clock_initialized_) 
        {
            playback_clock_initialized_ = true;
            last_update_time_ = now;
            playback_accumulator_seconds_ = 0.0;
        } 
        else 
        {
            const std::chrono::duration<double> elapsed = now - last_update_time_;
            last_update_time_ = now;
            playback_accumulator_seconds_ += std::max(0.0, elapsed.count());

            const double step_period = 1.0 / static_cast<double>(config_.playback_fps);
            if (step_period > 0.0) 
            {
                int frame_updates = 0;
                while (playback_accumulator_seconds_ >= step_period) 
                {
                    playback_accumulator_seconds_ -= step_period;
                    ++frame_updates;

                    // Avoid an unbounded catch-up burst after stalls.
                    if (frame_updates >= 4) {break;}
                }

                for (int i = 0; i < frame_updates; ++i) 
                {
                    if (!advance_to_next_frame(context, error)) {return false;}
                }
            } 
            else 
            {
                playback_accumulator_seconds_ = 0.0;
            }
        }
    }

    if (resolved_volume_mode_ == "cycle" && field_names_.size() > 1 && config_.component_cycle_fps > 0.0f) 
    {
        if (!component_cycle_clock_initialized_) 
        {
            component_cycle_clock_initialized_ = true;
            last_component_update_time_ = now;
            component_cycle_accumulator_seconds_ = 0.0;
        }
        else 
        {
            const std::chrono::duration<double> elapsed = now - last_component_update_time_;
            last_component_update_time_ = now;
            component_cycle_accumulator_seconds_ += std::max(0.0, elapsed.count());

            const double cycle_period = 1.0 / static_cast<double>(config_.component_cycle_fps);
            if (cycle_period > 0.0) 
            {
                int cycle_updates = 0;
                while (component_cycle_accumulator_seconds_ >= cycle_period) 
                {
                    component_cycle_accumulator_seconds_ -= cycle_period;
                    cycle_field_index_ = (cycle_field_index_ + 1) % field_names_.size();

                    // Cap updates so long pauses do not create large upload bursts.
                    if (++cycle_updates >= 8) {break;}
                }
            } 
            else 
            {
                component_cycle_accumulator_seconds_ = 0.0;
            }
        }
    }

    return true;
}

/** @brief Cache latest interactive camera input snapshot from the window layer. */
void VolumeBackend::set_camera_input(const CameraInputState& input) {camera_input_ = input;}

std::string VolumeBackend::status_label() const
{
    if (convection_detected_) { return "Convection"; }
    if (grid_ready_)          { return "Grid Ready"; }
    return "Waiting for SHM";
}

/** @brief Destroy backend-owned Vulkan and dataset resources. */
void VolumeBackend::shutdown(VkDevice) 
{
    if (device_ != VK_NULL_HANDLE) 
    {
        if (pipeline_ != VK_NULL_HANDLE) 
        {
            vkDestroyPipeline(device_, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }

        if (pipeline_layout_ != VK_NULL_HANDLE) 
        {
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
        }

        if (descriptor_pool_ != VK_NULL_HANDLE) 
        {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            descriptor_pool_ = VK_NULL_HANDLE;
            descriptor_set_ = VK_NULL_HANDLE;
        }

        if (descriptor_set_layout_ != VK_NULL_HANDLE) 
        {
            vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
            descriptor_set_layout_ = VK_NULL_HANDLE;
        }
    }

    destroy_volume_resources();

    field_names_.clear();
    datasets_.clear();
    field_index_by_name_.clear();

    frame_count_ = 0;
    current_frame_index_ = 0;
    current_field_frames_.clear();
    playback_clock_initialized_ = false;
    playback_accumulator_seconds_ = 0.0;

    volume_width_ = 0;
    volume_height_ = 0;
    volume_depth_ = 0;
    volume_raw_min_ = 0.0f;
    volume_raw_max_ = 0.0f;
    volume_norm_low_ = 0.0f;
    volume_norm_high_ = 0.0f;

    composite_mode_ = false;
    supercell_shading_enabled_ = false;
    natural_texture_enabled_ = true;
    resolved_texture_mode_ = "natural";
    resolved_volume_mode_ = "supercell";
    resolved_camera_mode_ = "orbit";
    isolate_field_requested_.clear();
    isolate_field_resolved_.clear();
    isolated_field_index_ = 0;
    cycle_field_index_ = 0;
    component_cycle_clock_initialized_ = false;
    component_cycle_accumulator_seconds_ = 0.0;
    camera_clock_initialized_ = false;
    camera_orbit_angle_rad_ = 0.78;
    camera_input_ = {};
    camera_pose_initialized_ = false;
    camera_position_ = {1.7f, 0.85f, 1.6f};
    camera_yaw_rad_ = -2.4f;
    camera_pitch_rad_ = -0.28f;
    active_component_count_ = 0;

    pipeline_scaffold_ready_ = false;
    render_pass_ = VK_NULL_HANDLE;
    extent_ = {};
    device_ = VK_NULL_HANDLE;
}

/** @brief Ensure descriptor and pipeline resources are available for rendering. */
bool VolumeBackend::ensure_pipeline_scaffold(std::string& error) 
{
    if (device_ == VK_NULL_HANDLE) 
    {
        error = "volume backend is missing logical device";
        return false;
    }

    if (render_pass_ == VK_NULL_HANDLE) 
    {
        error = "volume backend is missing render pass";
        return false;
    }

    if (!volume_resources_ready_) 
    {
        error = "volume backend is missing uploaded 3D component textures";
        return false;
    }

    if (!create_descriptor_resources(error)) {return false;}

    if (!create_graphics_pipeline(error)) {return false;}

    pipeline_scaffold_ready_ = true;
    return true;
}

/** @brief Parse/normalize field, texture, and camera configuration state. */
bool VolumeBackend::parse_requested_fields(std::string& error) 
{
    field_names_.clear();

    if (!normalize_volume_mode(config_.volume_mode, resolved_volume_mode_)) 
    {
        error = "unsupported volume mode '" + config_.volume_mode +
                "' (expected supercell|composite|isolated|cycle)";
        return false;
    }
    if (!normalize_texture_mode(config_.texture_mode, resolved_texture_mode_)) 
    {
        error = "unsupported texture mode '" + config_.texture_mode +
                "' (expected smooth|natural)";
        return false;
    }
    if (!camera::normalize_mode(config_.camera_mode, resolved_camera_mode_)) 
    {
        error = "unsupported camera mode '" + config_.camera_mode +
                "' (expected fixed|orbit|freefly)";
        return false;
    }
    isolate_field_requested_ = tmv::strutil::trim_and_lower(config_.isolate_field);

    if (config_.component_cycle_fps <= 0.0f) 
    {
        error = "component cycle FPS must be > 0";
        return false;
    }
    if (config_.camera_orbit_fps < 0.0f) 
    {
        error = "camera orbit FPS must be >= 0";
        return false;
    }
    if (config_.camera_distance <= 0.0f) 
    {
        error = "camera distance must be > 0";
        return false;
    }
    if (config_.camera_fov_deg <= 1.0f || config_.camera_fov_deg >= 179.0f) 
    {
        error = "camera FOV must be in (1,179) degrees";
        return false;
    }

    supercell_shading_enabled_ = false;
    natural_texture_enabled_ = (resolved_texture_mode_ == "natural");
    isolate_field_resolved_.clear();
    isolated_field_index_ = 0;
    cycle_field_index_ = 0;
    component_cycle_clock_initialized_ = false;
    component_cycle_accumulator_seconds_ = 0.0;
    camera_clock_initialized_ = false;
    camera_pose_initialized_ = false;
    camera_input_ = {};

    std::vector<std::string> raw_fields;

    if (!config_.fields_csv.empty()) {raw_fields = split_csv_fields(config_.fields_csv);} else {raw_fields.push_back(config_.field);}

    if (raw_fields.empty()) {raw_fields.push_back("theta");}

    std::unordered_set<std::string> seen;
    for (std::string field : raw_fields) 
    {
        field = tmv::strutil::trim_and_lower(field);
        if (field.empty()) {continue;}

        if (seen.insert(field).second) {field_names_.push_back(field);}
    }

    if (field_names_.empty()) { error = "no valid field names provided"; return false;}

    composite_mode_ = field_names_.size() > 1;
    return true;
}

/** @brief Resolve field aliases and scan compatible datasets from disk or SHM. */
bool VolumeBackend::scan_datasets(std::string& error)
{
    const std::vector<std::string> requested_fields = field_names_;
    field_names_.clear();
    datasets_.clear();
    shm_datasets_.clear();
    field_index_by_name_.clear();
    nonfinite_warning_once_.clear();
    frame_count_ = 0;

    int expected_nx = 0;
    int expected_ny = 0;
    int expected_nz = 0;

    std::unordered_set<std::string> used_resolved_fields;

    for (const std::string& requested_field : requested_fields) 
    {
        bool resolved = false;
        std::string resolved_field_name;
        oglcpp::ExportDataset resolved_dataset(config_.input_dir, requested_field);
        std::string last_scan_error;

        const std::vector<std::string> candidates = field_alias_candidates(requested_field);
        for (const std::string& candidate : candidates) 
        {
            if (used_resolved_fields.find(candidate) != used_resolved_fields.end()) {continue;}

            oglcpp::ExportDataset dataset(config_.input_dir, candidate);
            std::string scan_error;

            if (!dataset.scan(scan_error)) {last_scan_error = scan_error; continue;}

            if (dataset.frame_count() == 0) {last_scan_error = "dataset has no frames"; continue;}

            resolved_field_name = candidate;
            resolved_dataset = std::move(dataset);
            resolved = true;
            break;
        }

        if (!resolved)
        {
            // What is SHM: A shared memory region that is written by the simulation and read by the viewer.
            // Disk resolution failed — check SHM before giving up.
            // SHM-only fields (e.g. qc before any disk export) are valid
            // visualization sources during live simulation. 
            const std::vector<std::string> shm_candidates = field_alias_candidates(requested_field);
            bool shm_resolved = false;
            for (const std::string& candidate : shm_candidates)
            {
                if (used_resolved_fields.count(candidate)) continue;
                oglcpp::ShmDataset shm_probe(candidate);
                std::string shm_error;
                if (shm_probe.scan(shm_error))
                {
                    resolved_field_name = candidate;
                    if (datasets_.empty() && shm_datasets_.empty())
                    {
                        expected_nx = shm_probe.nx();
                        expected_ny = shm_probe.ny();
                        expected_nz = shm_probe.nz();
                        frame_count_ = 1;
                        volume_width_ = static_cast<uint32_t>(expected_nx);
                        volume_height_ = static_cast<uint32_t>(expected_ny);
                        volume_depth_ = static_cast<uint32_t>(expected_nz);
                    }
                    std::cout << "[vulkan][volume] Live SHM source for '" << candidate << "'\n";
                    field_index_by_name_.emplace(candidate, datasets_.size());
                    field_names_.push_back(candidate);
                    // Empty disk dataset placeholder (SHM provides all data)
                    datasets_.push_back(oglcpp::ExportDataset(config_.input_dir, candidate));
                    shm_datasets_.push_back(std::move(shm_probe));
                    used_resolved_fields.insert(candidate);
                    shm_resolved = true;
                    break;
                }
            }
            if (!shm_resolved)
            {
                std::cout << "[vulkan][volume] Skipping field '" << requested_field << "'";
                if (!last_scan_error.empty()) {std::cout << " (" << last_scan_error << ")";}
                std::cout << "\n";
            }
            continue;
        }

        if (requested_field != resolved_field_name) 
        {
            std::cout << "[vulkan][volume] Field alias: '" << requested_field
                      << "' -> '" << resolved_field_name << "'\n";
        }

        if (datasets_.empty()) 
        {
            expected_nx = resolved_dataset.nx();
            expected_ny = resolved_dataset.ny();
            expected_nz = resolved_dataset.nz();
            frame_count_ = resolved_dataset.frame_count();
            volume_width_ = static_cast<uint32_t>(expected_nx);
            volume_height_ = static_cast<uint32_t>(expected_ny);
            volume_depth_ = static_cast<uint32_t>(expected_nz);
        } 
        else 
        {
            if (resolved_dataset.nx() != expected_nx ||
                resolved_dataset.ny() != expected_ny ||
                resolved_dataset.nz() != expected_nz) 
            {
                std::cout << "[vulkan][volume] Skipping field '" << resolved_field_name
                          << "' (dimension mismatch: expected "
                          << expected_nx << "x" << expected_ny << "x" << expected_nz
                          << ", got "
                          << resolved_dataset.nx() << "x" << resolved_dataset.ny() << "x" << resolved_dataset.nz()
                          << ")\n";
                continue;
            }

            if (resolved_dataset.frame_count() != frame_count_) 
            {
                std::cout << "[vulkan][volume] Field '" << resolved_field_name << "' has "
                          << resolved_dataset.frame_count()
                          << " frames; clipping playback to "
                          << std::min(frame_count_, resolved_dataset.frame_count())
                          << " shared frames.\n";
                frame_count_ = std::min(frame_count_, resolved_dataset.frame_count());
            }
        }

        // Try to attach a SHM live source for this field (preferred over disk)
        oglcpp::ShmDataset shm_ds(resolved_field_name);
        std::string shm_error;
        if (shm_ds.scan(shm_error))
        {
            // SHM source available — use its dimensions if this is the first field
            if (datasets_.empty() && shm_datasets_.empty())
            {
                expected_nx = shm_ds.nx();
                expected_ny = shm_ds.ny();
                expected_nz = shm_ds.nz();
                frame_count_ = std::max<std::size_t>(frame_count_, 1);
                volume_width_ = static_cast<uint32_t>(expected_nx);
                volume_height_ = static_cast<uint32_t>(expected_ny);
                volume_depth_ = static_cast<uint32_t>(expected_nz);
            }
            std::cout << "[vulkan][volume] Live SHM source for '" << resolved_field_name << "'\n";
        }

        field_index_by_name_.emplace(resolved_field_name, datasets_.size());
        field_names_.push_back(resolved_field_name);
        datasets_.push_back(std::move(resolved_dataset));
        shm_datasets_.push_back(std::move(shm_ds));
        used_resolved_fields.insert(resolved_field_name);
    }

    if (datasets_.empty() || frame_count_ == 0) 
    {
        error = "none of requested fields are available in " + config_.input_dir +
                " (requested: " + join_fields_csv(requested_fields) + ")";
        return false;
    }

    if (datasets_.size() > kMaxVolumeComponents) 
    {
        std::cout << "[vulkan][volume] resolved " << datasets_.size()
                  << " fields, but renderer supports up to " << kMaxVolumeComponents
                  << " components per pass; truncating extras\n";
        datasets_.erase(datasets_.begin() + static_cast<std::ptrdiff_t>(kMaxVolumeComponents), datasets_.end());
        field_names_.erase(field_names_.begin() + static_cast<std::ptrdiff_t>(kMaxVolumeComponents), field_names_.end());
        field_index_by_name_.clear();

        for (std::size_t i = 0; i < field_names_.size(); ++i) 
        {
            field_index_by_name_.emplace(field_names_[i], i);
        }
    }

    composite_mode_ = field_names_.size() > 1;

    if (resolved_volume_mode_ == "cycle" && field_names_.size() < 2) 
    {
        std::cout << "[vulkan][volume] cycle mode requested but only one field resolved; "
                  << "falling back to isolated mode\n";
        resolved_volume_mode_ = "isolated";
    }

    if (resolved_volume_mode_ == "isolated" || resolved_volume_mode_ == "cycle") 
    {
        if (!resolve_field_index_by_alias(field_names_, isolate_field_requested_, isolated_field_index_, isolate_field_resolved_)) 
        {
            isolated_field_index_ = 0;
            isolate_field_resolved_ = field_names_.front();
            if (!isolate_field_requested_.empty()) 
            {
                std::cout << "[vulkan][volume] isolate field '" << isolate_field_requested_
                          << "' was not found; using '" << isolate_field_resolved_ << "'\n";
            }
        }
        cycle_field_index_ = isolated_field_index_;
    } 
    else 
    {
        isolate_field_resolved_.clear();
        isolated_field_index_ = 0;
        cycle_field_index_ = 0;
    }

    supercell_shading_enabled_ = (resolved_volume_mode_ == "supercell" && composite_mode_);

    return true;
}

/** @brief Load one frame across all resolved datasets into memory. */
bool VolumeBackend::load_dataset_frame(std::size_t frame_idx, std::vector<FieldFrame>& out_field_frames, std::string& error) 
{
    if (frame_idx >= frame_count_) 
    {
        std::ostringstream oss;
        oss << "frame index " << frame_idx << " out of range [0, " << (frame_count_ - 1) << "]";
        error = oss.str();
        return false;
    }

    out_field_frames.clear();
    out_field_frames.reserve(datasets_.size());

    for (std::size_t i = 0; i < datasets_.size(); ++i)
    {
        oglcpp::VolumeFrame frame;
        std::string load_error;

        // Prefer SHM live source when available and has new data
        bool loaded = false;
        if (i < shm_datasets_.size() && shm_datasets_[i].frame_count() > 0)
        {
            loaded = shm_datasets_[i].load_frame(0, frame, load_error);
            if (loaded && !grid_ready_)
            {
                grid_ready_ = true;
                std::cerr << "[vulkan][status] Grid Ready\n";
            }
        }

        if (!loaded && !datasets_[i].load_frame(frame_idx, frame, load_error))
        {
            error = "failed loading field '" + field_names_[i] + "' frame " + std::to_string(frame_idx) +
                    ": " + load_error;
            return false;
        }

        if (static_cast<uint32_t>(frame.nx) != volume_width_ ||
            static_cast<uint32_t>(frame.ny) != volume_height_ ||
            static_cast<uint32_t>(frame.nz) != volume_depth_) 
        {
            std::ostringstream oss;
            oss << "frame dimension mismatch in field '" << field_names_[i]
                << "': expected " << volume_width_ << "x" << volume_height_ << "x" << volume_depth_
                << ", got " << frame.nx << "x" << frame.ny << "x" << frame.nz;
            error = oss.str();
            return false;
        }

        std::size_t expected_size = 0;
        if (!checked_voxel_count(volume_width_, volume_height_, volume_depth_, expected_size, error)) {return false;}

        if (frame.normalized.size() != expected_size) 
        {
            error = "field '" + field_names_[i] + "' frame payload size mismatch";
            return false;
        }

        const std::size_t nonfinite_count = frame.nan_count + frame.inf_count;
        if (nonfinite_count > 0) 
        {
            const std::string warning_key = field_names_[i] + "#" + std::to_string(frame_idx);
            if (nonfinite_warning_once_.insert(warning_key).second) 
            {
                std::cerr << "[vulkan][dataset] field '" << field_names_[i]
                          << "' frame " << frame_idx
                          << " had non-finite source samples (nan=" << frame.nan_count
                          << ", inf=" << frame.inf_count
                          << ", sanitized=" << frame.sanitized_nonfinite_count
                          << ")\n";
            }
        }

        if (i == 0) 
        {
            volume_raw_min_ = frame.raw_min;
            volume_raw_max_ = frame.raw_max;
            volume_norm_low_ = frame.norm_low;
            volume_norm_high_ = frame.norm_high;
        }

        if (!convection_detected_)
        {
            float field_max = 0.0f;
            for (float v : frame.normalized)
            {
                if (v > field_max) { field_max = v; }
            }
            if (field_max > 0.05f)
            {
                convection_detected_ = true;
                std::cerr << "[vulkan][status] Convection (field_max=" << field_max << ")\n";
            }
        }

        FieldFrame field_frame;
        field_frame.name = field_names_[i];
        field_frame.normalized = std::move(frame.normalized);
        out_field_frames.push_back(std::move(field_frame));
    }

    return true;
}

/** @brief Load and log the first available frame before rendering starts. */
bool VolumeBackend::load_initial_frame(std::string& error) 
{
    if (frame_count_ == 0) 
    {
        error = "no frames available for initial load";
        return false;
    }

    current_frame_index_ = 0;

    std::vector<FieldFrame> field_frames;
    if (!load_dataset_frame(current_frame_index_, field_frames, error)) {return false;}

    if (field_frames.empty()) {error = "initial volume frame is empty"; return false;}
    current_field_frames_ = std::move(field_frames);

    std::ostringstream fields_list;
    for (std::size_t i = 0; i < field_names_.size(); ++i) 
    {
        if (i > 0) {fields_list << ",";}

        fields_list << field_names_[i];
    }

    std::string mode_label = resolved_volume_mode_;
    if (resolved_volume_mode_ == "supercell" && !composite_mode_) {mode_label = "single-field";}

    if ((resolved_volume_mode_ == "isolated" || resolved_volume_mode_ == "cycle") &&!field_names_.empty()) 
    {
        const std::size_t selected =
            (resolved_volume_mode_ == "cycle") ? cycle_field_index_ : isolated_field_index_;
        mode_label += ":" + field_names_[selected % field_names_.size()];
    }

    std::cout << "[vulkan][volume] Mode=" << mode_label
              << " texture=" << resolved_texture_mode_
              << " camera=" << resolved_camera_mode_
              << " fields=[" << fields_list.str() << "]"
              << " frames=" << frame_count_
              << " dims=" << volume_width_ << "x" << volume_height_ << "x" << volume_depth_
              << " playback_fps=" << config_.playback_fps;
    if (resolved_volume_mode_ == "cycle") {
        std::cout << " component_cycle_fps=" << config_.component_cycle_fps;
    }
    std::cout << "\n";

    return true;
}

/** @brief Step to the next frame and upload refreshed volume texture data. */
bool VolumeBackend::advance_to_next_frame(VulkanContext& context, std::string& error) 
{
    if (frame_count_ <= 1) {return true;}

    const std::size_t next_frame = (current_frame_index_ + 1) % frame_count_;

    std::vector<FieldFrame> field_frames;
    if (!load_dataset_frame(next_frame, field_frames, error)) {return false;}

    current_frame_index_ = next_frame;
    current_field_frames_ = std::move(field_frames);

    return upload_volume_data(context, error);
}

/** @brief Create/update descriptor set state for 3D volume sampling. */
bool VolumeBackend::create_descriptor_resources(std::string& error) 
{
    if (component_image_views_.empty() || volume_sampler_ == VK_NULL_HANDLE) 
    {
        error = "component image resources are missing for descriptor creation";
        return false;
    }

    auto update_descriptor_binding = [&]() 
    {
        const VkImageView fallback_view = component_image_views_.front();
        std::array<VkDescriptorImageInfo, kMaxVolumeComponents> image_infos{};

        for (std::size_t i = 0; i < kMaxVolumeComponents; ++i) 
        {
            VkDescriptorImageInfo info{};
            info.sampler = volume_sampler_;
            info.imageView = (i < component_image_views_.size()) ? component_image_views_[i] : fallback_view;
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_infos[i] = info;
        }

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_set_;
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = static_cast<uint32_t>(kMaxVolumeComponents);
        write.pImageInfo = image_infos.data();

        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    };

    if (descriptor_set_layout_ != VK_NULL_HANDLE && descriptor_pool_ != VK_NULL_HANDLE &&
        descriptor_set_ != VK_NULL_HANDLE && pipeline_layout_ != VK_NULL_HANDLE) 
    {
        // If the volume image/view/sampler was recreated, refresh descriptor contents.
        update_descriptor_binding();
        return true;
    }

    if (pipeline_ != VK_NULL_HANDLE) {vkDestroyPipeline(device_, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE;}

    if (descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
    }

    if (descriptor_set_layout_ != VK_NULL_HANDLE) 
    {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }

    if (pipeline_layout_ != VK_NULL_HANDLE) 
    {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }

    VkDescriptorSetLayoutBinding volume_binding{};
    volume_binding.binding = 0;
    volume_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    volume_binding.descriptorCount = static_cast<uint32_t>(kMaxVolumeComponents);
    volume_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 1;
    set_layout_info.pBindings = &volume_binding;

    VkResult result = vkCreateDescriptorSetLayout(device_, &set_layout_info, nullptr, &descriptor_set_layout_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateDescriptorSetLayout failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = static_cast<uint32_t>(kMaxVolumeComponents);

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    result = vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateDescriptorPool failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = descriptor_pool_;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &descriptor_set_layout_;

    result = vkAllocateDescriptorSets(device_, &allocate_info, &descriptor_set_);
    if (result != VK_SUCCESS) 
    {
        error = "vkAllocateDescriptorSets failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    update_descriptor_binding();

    VkPushConstantRange push_constant_range{};
    push_constant_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = static_cast<uint32_t>(sizeof(PushConstants));

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant_range;

    result = vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_);
    if (result != VK_SUCCESS) {
        error = "vkCreatePipelineLayout failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/** @brief Build graphics pipeline used by fullscreen volume pass. */
bool VolumeBackend::create_graphics_pipeline(std::string& error) 
{
    if (pipeline_layout_ == VK_NULL_HANDLE) {error = "pipeline layout is not initialized"; return false;}

    if (render_pass_ == VK_NULL_HANDLE) {error = "render pass is not initialized"; return false;}

    std::filesystem::path vertex_shader_path;
    std::filesystem::path fragment_shader_path;

    if (!resolve_shader_path(kVertexShaderRelativePath, vertex_shader_path)) 
    {
        error = "vertex shader SPIR-V not found: " + std::string(kVertexShaderRelativePath) +
                " (run `make vulkan`)";
        return false;
    }

    if (!resolve_shader_path(kFragmentShaderRelativePath, fragment_shader_path)) 
    {
        error = "fragment shader SPIR-V not found: " + std::string(kFragmentShaderRelativePath) +
                " (run `make vulkan`)";
        return false;
    }

    std::vector<char> vertex_code;
    std::vector<char> fragment_code;
    if (!read_binary_file(vertex_shader_path, vertex_code, error) || !read_binary_file(fragment_shader_path, fragment_code, error))
    {
        return false;
    }

    VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
    VkShaderModule fragment_shader_module = VK_NULL_HANDLE;

    auto cleanup_modules = [&]() 
    {
        if (fragment_shader_module != VK_NULL_HANDLE) 
        {
            vkDestroyShaderModule(device_, fragment_shader_module, nullptr);
            fragment_shader_module = VK_NULL_HANDLE;
        }
        if (vertex_shader_module != VK_NULL_HANDLE) 
        {
            vkDestroyShaderModule(device_, vertex_shader_module, nullptr);
            vertex_shader_module = VK_NULL_HANDLE;
        }
    };

    if (!create_shader_module(device_, vertex_code, vertex_shader_module, error) ||
        !create_shader_module(device_, fragment_code, fragment_shader_module, error)) 
    {
        cleanup_modules();
        return false;
    }

    if (pipeline_ != VK_NULL_HANDLE) {vkDestroyPipeline(device_, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE;}

    VkPipelineShaderStageCreateInfo vertex_stage{};
    vertex_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertex_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertex_stage.module = vertex_shader_module;
    vertex_stage.pName = "main";

    VkPipelineShaderStageCreateInfo fragment_stage{};
    fragment_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragment_stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragment_stage.module = fragment_shader_module;
    fragment_stage.pName = "main";

    VkPipelineShaderStageCreateInfo shader_stages[] = {vertex_stage, fragment_stage};

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;

    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = static_cast<uint32_t>(std::size(dynamic_states));
    dynamic_state.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = static_cast<uint32_t>(std::size(shader_stages));
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = pipeline_layout_;
    pipeline_info.renderPass = render_pass_;
    pipeline_info.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
    cleanup_modules();

    if (result != VK_SUCCESS) 
    {
        error = "vkCreateGraphicsPipelines failed with VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    return true;
}

/** @brief Allocate one 3D texture per active field and upload initial frame data. */
bool VolumeBackend::create_volume_resources(VulkanContext& context, std::string& error) 
{
    if (volume_width_ == 0 || volume_height_ == 0 || volume_depth_ == 0) 
    {
        error = "volume dimensions are not initialized";
        return false;
    }

    if (!supports_volume_format(context.physical_device())) {
        error = "VK_FORMAT_R32_SFLOAT does not support sampled + transfer-dst optimal-tiling usage";
        return false;
    }

    destroy_volume_resources();

    if (current_field_frames_.empty()) 
    {
        error = "no field frame data available to initialize component textures";
        return false;
    }

    const std::size_t component_count =
        std::min<std::size_t>(current_field_frames_.size(), kMaxVolumeComponents);
    if (component_count == 0) 
    {
        error = "resolved component count is zero";
        return false;
    }

    active_component_count_ = static_cast<uint32_t>(component_count);
    component_images_.assign(component_count, VK_NULL_HANDLE);
    component_image_memories_.assign(component_count, VK_NULL_HANDLE);
    component_image_views_.assign(component_count, VK_NULL_HANDLE);
    component_image_layout_initialized_.assign(component_count, false);

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_3D;
    image_info.extent.width = volume_width_;
    image_info.extent.height = volume_height_;
    image_info.extent.depth = volume_depth_;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = kVolumeFormat;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;

    for (std::size_t i = 0; i < component_count; ++i) 
    {
        VkResult result = vkCreateImage(device_, &image_info, nullptr, &component_images_[i]);
        if (result != VK_SUCCESS) 
        {
            error = "vkCreateImage(component=" + std::to_string(i) + ") failed with VkResult=" +
                    std::to_string(static_cast<int>(result));
            destroy_volume_resources();
            return false;
        }

        VkMemoryRequirements image_memory_requirements{};
        vkGetImageMemoryRequirements(device_, component_images_[i], &image_memory_requirements);

        VkMemoryAllocateInfo image_alloc_info{};
        image_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        image_alloc_info.allocationSize = image_memory_requirements.size;
        std::string image_memory_type_error;
        image_alloc_info.memoryTypeIndex = find_memory_type( context, image_memory_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image_memory_type_error);

        if (!image_memory_type_error.empty()) 
        {
            error = std::move(image_memory_type_error);
            destroy_volume_resources();
            return false;
        }

        result = vkAllocateMemory(device_, &image_alloc_info, nullptr, &component_image_memories_[i]);
        if (result != VK_SUCCESS) 
        {
            error = "vkAllocateMemory(component=" + std::to_string(i) + ") failed with VkResult=" +
                    std::to_string(static_cast<int>(result));
            destroy_volume_resources();
            return false;
        }

        result = vkBindImageMemory(device_, component_images_[i], component_image_memories_[i], 0);
        if (result != VK_SUCCESS) 
        {
            error = "vkBindImageMemory(component=" + std::to_string(i) + ") failed with VkResult=" +
                    std::to_string(static_cast<int>(result));
            destroy_volume_resources();
            return false;
        }

        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = component_images_[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
        view_info.format = kVolumeFormat;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        result = vkCreateImageView(device_, &view_info, nullptr, &component_image_views_[i]);
        if (result != VK_SUCCESS) 
        {
            error = "vkCreateImageView(component=" + std::to_string(i) + ") failed with VkResult=" +
                    std::to_string(static_cast<int>(result));
            destroy_volume_resources();
            return false;
        }
    }

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;
    sampler_info.maxAnisotropy = 1.0f;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    const VkResult result = vkCreateSampler(device_, &sampler_info, nullptr, &volume_sampler_);
    if (result != VK_SUCCESS) {
        error = "vkCreateSampler(volume) failed with VkResult=" + std::to_string(static_cast<int>(result));
        destroy_volume_resources();
        return false;
    }

    if (!upload_volume_data(context, error)) {
        destroy_volume_resources();
        return false;
    }

    volume_resources_ready_ = true;
    return true;
}

/** @brief Upload all active field components to GPU using one staging transfer pass. */
bool VolumeBackend::upload_volume_data(VulkanContext& context, std::string& error) {
    if (active_component_count_ == 0) 
    {
        error = "no active volume components to upload";
        return false;
    }
    if (active_component_count_ > kMaxVolumeComponents) {
        error = "active component count exceeds shader descriptor array capacity";
        return false;
    }
    if (current_field_frames_.size() < active_component_count_) 
    {
        error = "field frame cache is smaller than active component count";
        return false;
    }
    if (component_images_.size() != active_component_count_ || component_image_memories_.size() != active_component_count_ ||
        component_image_views_.size() != active_component_count_ || component_image_layout_initialized_.size() != active_component_count_) 
    {
        error = "component image resource arrays are inconsistent";
        return false;
    }

    std::size_t expected_voxels = 0;
    if (!checked_voxel_count(volume_width_, volume_height_, volume_depth_, expected_voxels, error)) {return false;}

    for (uint32_t i = 0; i < active_component_count_; ++i) 
    {
        if (current_field_frames_[i].normalized.size() != expected_voxels) 
        {
            error = "component '" + current_field_frames_[i].name + "' data size does not match volume dimensions";
            return false;
        }
    }

    if (expected_voxels > std::numeric_limits<std::size_t>::max() / sizeof(float)) 
    {
        error = "volume upload byte size overflow (single component)";
        return false;
    }

    const std::size_t bytes_per_component = expected_voxels * sizeof(float);
    if (bytes_per_component == 0) 
    {
        error = "volume data is empty";
        return false;
    }

    if (bytes_per_component > std::numeric_limits<std::size_t>::max() / active_component_count_) 
    {
        error = "volume upload byte size overflow (all components)";
        return false;
    }
    const std::size_t total_upload_bytes = bytes_per_component * static_cast<std::size_t>(active_component_count_);
    const VkDeviceSize buffer_size = static_cast<VkDeviceSize>(total_upload_bytes);

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    VkCommandPool upload_command_pool = VK_NULL_HANDLE;
    VkCommandBuffer upload_command_buffer = VK_NULL_HANDLE;

    auto cleanup = [&]() 
    {
        if (upload_command_buffer != VK_NULL_HANDLE && upload_command_pool != VK_NULL_HANDLE) 
        {
            vkFreeCommandBuffers(device_, upload_command_pool, 1, &upload_command_buffer);
            upload_command_buffer = VK_NULL_HANDLE;
        }
        if (upload_command_pool != VK_NULL_HANDLE) 
        {
            vkDestroyCommandPool(device_, upload_command_pool, nullptr);
            upload_command_pool = VK_NULL_HANDLE;
        }
        if (staging_buffer != VK_NULL_HANDLE) 
        {
            vkDestroyBuffer(device_, staging_buffer, nullptr);
            staging_buffer = VK_NULL_HANDLE;
        }
        if (staging_memory != VK_NULL_HANDLE) 
        {
            vkFreeMemory(device_, staging_memory, nullptr);
            staging_memory = VK_NULL_HANDLE;
        }
    };

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(device_, &buffer_info, nullptr, &staging_buffer);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateBuffer(staging) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    VkMemoryRequirements staging_requirements{};
    vkGetBufferMemoryRequirements(device_, staging_buffer, &staging_requirements);

    VkMemoryAllocateInfo staging_alloc_info{};
    staging_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    staging_alloc_info.allocationSize = staging_requirements.size;
    std::string staging_memory_type_error;
    staging_alloc_info.memoryTypeIndex = find_memory_type(context, staging_requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_memory_type_error);

    if (!staging_memory_type_error.empty()) 
    {
        error = std::move(staging_memory_type_error);
        cleanup();
        return false;
    }

    result = vkAllocateMemory(device_, &staging_alloc_info, nullptr, &staging_memory);
    if (result != VK_SUCCESS) 
    {
        error = "vkAllocateMemory(staging) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    result = vkBindBufferMemory(device_, staging_buffer, staging_memory, 0);
    if (result != VK_SUCCESS) 
    {
        error = "vkBindBufferMemory(staging) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    void* mapped = nullptr;
    result = vkMapMemory(device_, staging_memory, 0, buffer_size, 0, &mapped);
    if (result != VK_SUCCESS || mapped == nullptr) 
    {
        error = "vkMapMemory(staging) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    auto* mapped_bytes = static_cast<std::uint8_t*>(mapped);
    for (uint32_t i = 0; i < active_component_count_; ++i) 
    {
        const std::size_t offset = bytes_per_component * static_cast<std::size_t>(i);
        std::memcpy(mapped_bytes + offset, current_field_frames_[i].normalized.data(), bytes_per_component);
    }
    vkUnmapMemory(device_, staging_memory);

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = context.graphics_queue_family_index();

    result = vkCreateCommandPool(device_, &pool_info, nullptr, &upload_command_pool);
    if (result != VK_SUCCESS) 
    {
        error = "vkCreateCommandPool(upload) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = upload_command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(device_, &alloc_info, &upload_command_buffer);
    if (result != VK_SUCCESS) 
    {
        error = "vkAllocateCommandBuffers(upload) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    result = vkBeginCommandBuffer(upload_command_buffer, &begin_info);
    if (result != VK_SUCCESS) 
    {
        error = "vkBeginCommandBuffer(upload) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    for (uint32_t i = 0; i < active_component_count_; ++i) 
    {
        VkImageLayout old_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags src_stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkAccessFlags src_access_mask = 0;

        if (component_image_layout_initialized_[i]) 
        {
            old_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            src_stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            src_access_mask = VK_ACCESS_SHADER_READ_BIT;
        }

        VkImageMemoryBarrier to_transfer{};
        to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_transfer.oldLayout = old_layout;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = component_images_[i];
        to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.baseMipLevel = 0;
        to_transfer.subresourceRange.levelCount = 1;
        to_transfer.subresourceRange.baseArrayLayer = 0;
        to_transfer.subresourceRange.layerCount = 1;
        to_transfer.srcAccessMask = src_access_mask;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(
            upload_command_buffer,
            src_stage_mask,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_transfer);

        VkBufferImageCopy copy_region{};
        // Staging is tightly packed: component `i` starts at `i * bytes_per_component`.
        copy_region.bufferOffset =
            static_cast<VkDeviceSize>(bytes_per_component * static_cast<std::size_t>(i));
        copy_region.bufferRowLength = 0;
        copy_region.bufferImageHeight = 0;
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.mipLevel = 0;
        copy_region.imageSubresource.baseArrayLayer = 0;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageOffset = {0, 0, 0};
        copy_region.imageExtent = {volume_width_, volume_height_, volume_depth_};

        vkCmdCopyBufferToImage(
            upload_command_buffer,
            staging_buffer,
            component_images_[i],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copy_region);

        VkImageMemoryBarrier to_shader_read{};
        to_shader_read.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        to_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.image = component_images_[i];
        to_shader_read.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_shader_read.subresourceRange.baseMipLevel = 0;
        to_shader_read.subresourceRange.levelCount = 1;
        to_shader_read.subresourceRange.baseArrayLayer = 0;
        to_shader_read.subresourceRange.layerCount = 1;
        to_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            upload_command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &to_shader_read);
    }

    result = vkEndCommandBuffer(upload_command_buffer);
    if (result != VK_SUCCESS) 
    {
        error = "vkEndCommandBuffer(upload) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &upload_command_buffer;

    result = vkQueueSubmit(context.graphics_queue(), 1, &submit_info, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) 
    {
        error = "vkQueueSubmit(upload) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    result = vkQueueWaitIdle(context.graphics_queue());
    if (result != VK_SUCCESS) 
    {
        error = "vkQueueWaitIdle(upload) failed with VkResult=" + std::to_string(static_cast<int>(result));
        cleanup();
        return false;
    }

    for (uint32_t i = 0; i < active_component_count_; ++i) {component_image_layout_initialized_[i] = true;}

    cleanup();
    return true;
}

/** @brief Find a Vulkan memory type index satisfying requested property flags. */
uint32_t VolumeBackend::find_memory_type(VulkanContext& context, uint32_t type_filter,
                                         VkMemoryPropertyFlags properties, std::string& error) const 
{
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(context.physical_device(), &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) 
    {
        const bool compatible_type = (type_filter & (1u << i)) != 0;
        const bool has_properties = (memory_properties.memoryTypes[i].propertyFlags & properties) == properties;

        if (compatible_type && has_properties) 
        {
            return i;
        }
    }

    error = "no suitable Vulkan memory type found";
    return 0;
}

/** @brief Destroy image/view/sampler resources used by component volume textures. */
void VolumeBackend::destroy_volume_resources() 
{
    if (device_ == VK_NULL_HANDLE) 
    {
        component_images_.clear();
        component_image_memories_.clear();
        component_image_views_.clear();
        component_image_layout_initialized_.clear();
        active_component_count_ = 0;
        volume_resources_ready_ = false;
        return;
    }

    if (volume_sampler_ != VK_NULL_HANDLE) 
    {
        vkDestroySampler(device_, volume_sampler_, nullptr);
        volume_sampler_ = VK_NULL_HANDLE;
    }

    for (VkImageView image_view : component_image_views_) 
    {
        if (image_view != VK_NULL_HANDLE) 
        {
            vkDestroyImageView(device_, image_view, nullptr);
        }
    }

    for (VkImage image : component_images_) 
    {
        if (image != VK_NULL_HANDLE) 
        {
            vkDestroyImage(device_, image, nullptr);
        }
    }

    for (VkDeviceMemory image_memory : component_image_memories_) 
    {
        if (image_memory != VK_NULL_HANDLE) 
        {
            vkFreeMemory(device_, image_memory, nullptr);
        }
    }

    component_images_.clear();
    component_image_memories_.clear();
    component_image_views_.clear();
    component_image_layout_initialized_.clear();
    active_component_count_ = 0;
    volume_resources_ready_ = false;
}

}  // namespace vkcpp
