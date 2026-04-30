/**
 * @file volume_backend.hpp
 * @brief GPU volume-render backend for tornado simulation exports.
 *
 * Loads one or more scalar fields from exported NPY datasets, builds normalized
 * per-component 3D textures, and records fullscreen raymarch commands. Handles
 * playback frame stepping and swapchain-dependent pipeline/descriptor resource
 * recreation.
 */

#pragma once

#include "data/export_dataset.hpp"
#include "data/shm_dataset.hpp"
#include "rendering/render_backend.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vkcpp 
{

/**
 * @brief Volume render backend with optional multi-field composite synthesis.
 */
class VolumeBackend : public RenderBackend 
{
public:
    /**
     * @brief Runtime parameters controlling dataset loading and raymarch tuning.
     */
    struct Config 
    {
        std::string input_dir = "data/exports";
        std::string field = "theta";
        std::string fields_csv;
        std::string volume_mode = "supercell";
        std::string isolate_field;
        float component_cycle_fps = 0.50f;
        bool cinematic_bw = false;
        std::string texture_mode = "natural";
        std::string camera_mode = "orbit";
        float camera_orbit_fps = 0.02f;
        float camera_distance = 2.25f;
        float camera_height = 0.85f;
        float camera_fov_deg = 55.0f;

        float playback_fps = 1.0f;

        int ray_steps = 192;
        float ray_threshold = 0.30f;
        float ray_opacity = 1.15f;
        float ray_brightness = 1.10f;
        float ray_ambient = 0.90f;
        float ray_anisotropy = 0.58f;
        float ray_max_distance = 5.0f;

        std::array<float, 3> sun_dir = {0.66f, 0.34f, 0.67f};
    };

    VolumeBackend();
    /**
     * @brief Construct backend with explicit configuration.
     * @param config Backend loading and rendering settings.
     */
    explicit VolumeBackend(Config config);

    /** @copydoc RenderBackend::initialize */
    bool initialize(VulkanContext& context,
                    VkRenderPass render_pass,
                    VkExtent2D extent,
                    std::string& error) override;

    /** @copydoc RenderBackend::on_swapchain_recreated */
    bool on_swapchain_recreated(VkRenderPass render_pass,
                                VkExtent2D extent,
                                std::string& error) override;

    /** @copydoc RenderBackend::record */
    bool record(VkCommandBuffer command_buffer,
                VkFramebuffer framebuffer,
                VkExtent2D extent,
                std::string& error) override;

    /** @copydoc RenderBackend::update_for_frame */
    bool update_for_frame(VulkanContext& context,
                          std::uint64_t frame_number,
                          std::string& error) override;

    /** @copydoc RenderBackend::set_camera_input */
    void set_camera_input(const CameraInputState& input) override;

    /** @copydoc RenderBackend::status_label */
    std::string status_label() const override;

    /** @copydoc RenderBackend::shutdown */
    void shutdown(VkDevice device) override;

private:
    /** @brief Fragment-shader push constants for camera and shading parameters. */
    struct PushConstants 
    {
        std::array<float, 4> camera_pos{};
        std::array<float, 4> camera_forward{};
        std::array<float, 4> camera_right{};
        std::array<float, 4> camera_up{};
        std::array<float, 4> render0{};
        std::array<float, 4> render1{};
        std::array<float, 4> render2{};
        std::array<float, 4> volume_dims{};
    };

    /*Structure to store the name of the field and the normalized data*/
    struct FieldFrame {std::string name; std::vector<float> normalized;};

    /** @brief Ensure descriptor/pipeline resources exist for current render pass. */
    bool ensure_pipeline_scaffold(std::string& error);

    /** @brief Parse and sanitize requested field list from config. */
    bool parse_requested_fields(std::string& error);

    /** @brief Discover and validate datasets for all requested fields. */
    bool scan_datasets(std::string& error);

    /** @brief Load one frame from every resolved dataset field. */
    bool load_dataset_frame(std::size_t frame_idx,
                            std::vector<FieldFrame>& out_field_frames,
                            std::string& error);
    /** @brief Load first frame and prime playback statistics/state. */
    bool load_initial_frame(std::string& error);

    /** @brief Advance playback to next frame and upload fresh component textures. */
    bool advance_to_next_frame(VulkanContext& context, std::string& error);

    /** @brief Create descriptor set layout/pool/set for volume sampler binding. */
    bool create_descriptor_resources(std::string& error);

    /** @brief Create graphics pipeline for fullscreen volume raymarch pass. */
    bool create_graphics_pipeline(std::string& error);

    /** @brief Create per-field 3D image/view resources and shared sampler state. */
    bool create_volume_resources(VulkanContext& context, std::string& error);

    /** @brief Upload current CPU component frames into GPU 3D textures. */
    bool upload_volume_data(VulkanContext& context, std::string& error);

    /** @brief Find suitable Vulkan memory type for requested properties. */
    uint32_t find_memory_type(VulkanContext& context,
                              uint32_t type_filter,
                              VkMemoryPropertyFlags properties,
                              std::string& error) const;

    /** @brief Destroy all GPU resources tied to uploaded component textures. */
    void destroy_volume_resources();

    /*==============================
    Public members for the VolumeBackend class
    ==============================*/
    Config config_{};

    std::vector<std::string> field_names_;
    std::vector<oglcpp::ExportDataset> datasets_;
    std::vector<oglcpp::ShmDataset> shm_datasets_;  ///< Parallel SHM sources (empty if not live)
    std::unordered_map<std::string, std::size_t> field_index_by_name_;
    std::unordered_set<std::string> nonfinite_warning_once_;
    bool composite_mode_ = false;
    bool supercell_shading_enabled_ = false;
    bool natural_texture_enabled_ = true;
    std::string resolved_texture_mode_ = "natural";
    std::string resolved_volume_mode_ = "supercell";
    std::string resolved_camera_mode_ = "orbit";
    std::string isolate_field_requested_;
    std::string isolate_field_resolved_;
    std::size_t isolated_field_index_ = 0;
    std::size_t cycle_field_index_ = 0;
    bool component_cycle_clock_initialized_ = false;
    std::chrono::steady_clock::time_point last_component_update_time_{};
    double component_cycle_accumulator_seconds_ = 0.0;

    std::size_t frame_count_ = 0;
    std::size_t current_frame_index_ = 0;
    std::vector<FieldFrame> current_field_frames_;

    bool playback_clock_initialized_ = false;
    std::chrono::steady_clock::time_point last_update_time_{};
    double playback_accumulator_seconds_ = 0.0;
    bool camera_clock_initialized_ = false;
    std::chrono::steady_clock::time_point last_camera_update_time_{};
    double camera_orbit_angle_rad_ = 0.78;
    CameraInputState camera_input_{};
    bool camera_pose_initialized_ = false;
    std::array<float, 3> camera_position_ = {1.7f, 0.85f, 1.6f};
    float camera_yaw_rad_ = -2.4f;
    float camera_pitch_rad_ = -0.28f;

    VkDevice device_ = VK_NULL_HANDLE;
    VkRenderPass render_pass_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};

    uint32_t volume_width_ = 0;
    uint32_t volume_height_ = 0;
    uint32_t volume_depth_ = 0;

    float volume_raw_min_ = 0.0f;
    float volume_raw_max_ = 0.0f;
    float volume_norm_low_ = 0.0f;
    float volume_norm_high_ = 0.0f;

    bool grid_ready_ = false;
    bool convection_detected_ = false;

    static constexpr std::size_t kMaxVolumeComponents = 8;
    std::vector<VkImage> component_images_;
    std::vector<VkDeviceMemory> component_image_memories_;
    std::vector<VkImageView> component_image_views_;
    VkSampler volume_sampler_ = VK_NULL_HANDLE;
    std::vector<bool> component_image_layout_initialized_;
    uint32_t active_component_count_ = 0;
    bool volume_resources_ready_ = false;

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    // Pipeline is intentionally minimal (single-pass fullscreen raymarch) so we
    // can iterate toward full supercell shading without churn in surrounding code.
    bool pipeline_scaffold_ready_ = false;

    VkClearColorValue clear_color_ = {{0.02f, 0.03f, 0.06f, 1.0f}};
};

}  // namespace vkcpp
