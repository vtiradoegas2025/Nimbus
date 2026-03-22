Audit the Vulkan compute and rendering state:
1. Does `make vulkan` build cleanly?
2. Does `./bin/vulkan_viewer --dry-run` succeed?
3. Check compute dispatch state in `src/core/compute_kernel_template.cpp` — is `backend_dispatch_ready` true or false?
4. List any `.comp` shader files in `vulkan/shaders/`
5. Check `vulkan/src/compute_backend_vulkan.cpp` for actual compute pipeline code (VkPipeline, VkBuffer, dispatch calls)
6. Report: what's working (graphics rendering), what's stubbed (compute dispatch), what's missing (shaders, pipelines, buffers)