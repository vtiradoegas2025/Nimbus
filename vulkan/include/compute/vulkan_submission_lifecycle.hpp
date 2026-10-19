#pragma once

/**
 * @file vulkan_submission_lifecycle.hpp
 * @brief Pure state machine for synchronous Vulkan submission ownership.
 *
 * Nimbus currently records and submits one shared command buffer at a time.
 * A timeout (or an uncertain vkQueueSubmit failure) means that command buffer
 * and every resource referenced by it must remain pinned until device
 * shutdown.  This helper keeps that policy independent of Vulkan handles so it
 * can be fault-tested on hosts without a Vulkan device.
 */

#include <cstdint>

namespace tmv_vulkan
{

class SubmissionLifecycle
{
public:
    using Generation = std::uint64_t;

    Generation generation() const noexcept { return generation_; }

    bool can_dispatch() const noexcept
    {
        return !poisoned_ && !submission_may_be_in_flight_;
    }

    bool poisoned() const noexcept { return poisoned_; }

    bool submission_may_be_in_flight() const noexcept
    {
        return submission_may_be_in_flight_;
    }

    bool may_release_transients() const noexcept
    {
        return !submission_may_be_in_flight_;
    }

    bool may_destroy_child_resources() const noexcept
    {
        return !submission_may_be_in_flight_;
    }

    bool pipeline_is_ready(Generation owner_generation,
                           bool has_pipeline_handle) const noexcept
    {
        return has_pipeline_handle
               && owner_generation == generation_
               && can_dispatch();
    }

    /// A failure before vkQueueSubmit cannot leave this submission in flight.
    void poison_before_submit() noexcept
    {
        poisoned_ = true;
    }

    void note_submit_succeeded() noexcept
    {
        submission_may_be_in_flight_ = true;
    }

    /**
     * Conservatively treats a failed submit attempt as outcome-unknown.
     * This deliberately retains transient resources even when a particular
     * implementation would guarantee that an error queued no work.
     */
    void poison_after_submit_attempt() noexcept
    {
        poisoned_ = true;
        submission_may_be_in_flight_ = true;
    }

    void note_wait_succeeded() noexcept
    {
        submission_may_be_in_flight_ = false;
    }

    void poison_after_wait() noexcept
    {
        poisoned_ = true;
        submission_may_be_in_flight_ = true;
    }

    /// Called only after vkDeviceWaitIdle succeeds during shutdown.
    void note_device_quiesced() noexcept
    {
        submission_may_be_in_flight_ = false;
    }

    /**
     * Starts a fresh device generation.  Pipeline states carry the generation
     * in which they were constructed, so stale handles cannot report ready
     * after shutdown or a failed reinitialization.
     */
    void reset_for_new_device() noexcept
    {
        ++generation_;
        if (generation_ == 0)
        {
            ++generation_;
        }
        poisoned_ = false;
        submission_may_be_in_flight_ = false;
    }

private:
    Generation generation_ = 1;
    bool poisoned_ = false;
    bool submission_may_be_in_flight_ = false;
};

} // namespace tmv_vulkan
