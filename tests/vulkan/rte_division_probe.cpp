#include <vulkan/vulkan.h>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static void check(VkResult result)
{
    if (result != VK_SUCCESS)
    {
        std::cerr << "Vulkan result " << result << '\n';
        std::exit(2);
    }
}

int main(int argc, char** argv)
{
    const bool product_error = argc == 3 && std::strcmp(argv[2], "--product-error") == 0;
    if (argc != 2 && !product_error)
        return 2;
    std::fesetround(FE_TONEAREST);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    check(vkCreateInstance(&ici, nullptr, &instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    for (auto device : devices)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (product_error ? physical == VK_NULL_HANDLE : properties.vendorID == 0x10de)
            physical = device;
    }
    if (!physical)
    {
        std::cerr << (product_error ? "Vulkan device required\n" : "NVIDIA Vulkan device required\n");
        return 3;
    }
    if (product_error)
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physical, &properties);
        std::cerr << "device=" << properties.deviceName << " vendor=" << properties.vendorID
                  << " driver=" << properties.driverVersion << '\n';
    }
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    uint32_t family = 0;
    for (; family < count; ++family)
        if (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)
            break;
    float                   priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures features{};
    features.shaderInt64         = product_error ? VK_FALSE : VK_TRUE;
    const char*        extension = VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.pEnabledFeatures        = &features;
    dci.enabledExtensionCount   = product_error ? 0 : 1;
    dci.ppEnabledExtensionNames = product_error ? nullptr : &extension;
    VkDevice device;
    check(vkCreateDevice(physical, &dci, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    const uint32_t     cells  = product_error ? 1u << 18 : 1u << 20;
    const uint32_t     passes = product_error ? 4 : 8;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = cells * (product_error ? 4 : 3) * sizeof(uint32_t);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buffer;
    check(vkCreateBuffer(device, &bci, nullptr, &buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t       memory_type = 0;
    constexpr auto flags       = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (; memory_type < memory.memoryTypeCount; ++memory_type)
        if ((requirements.memoryTypeBits & (1u << memory_type)) &&
            (memory.memoryTypes[memory_type].propertyFlags & flags) == flags)
            break;
    if (memory_type == memory.memoryTypeCount)
        return 4;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = requirements.size;
    mai.memoryTypeIndex = memory_type;
    VkDeviceMemory allocation;
    check(vkAllocateMemory(device, &mai, nullptr, &allocation));
    check(vkBindBufferMemory(device, buffer, allocation, 0));
    void* mapped = nullptr;
    check(vkMapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, &mapped));
    auto*                                       input  = static_cast<uint32_t*>(mapped);
    auto*                                       output = input + cells * 2;
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsci.bindingCount = bindings.size();
    dsci.pBindings    = bindings.data();
    VkDescriptorSetLayout ds;
    check(vkCreateDescriptorSetLayout(device, &dsci, nullptr, &ds));
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts    = &ds;
    VkPipelineLayout layout;
    check(vkCreatePipelineLayout(device, &plci, nullptr, &layout));
    VkDescriptorPoolSize       pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &pool_size;
    VkDescriptorPool pool;
    check(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &ds;
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(device, &dsai, &set));
    std::array<VkDescriptorBufferInfo, 2> infos{
        {{buffer, 0, cells * 2 * sizeof(uint32_t)},
         {buffer, cells * 2 * sizeof(uint32_t), cells * (product_error ? 2 : 1) * sizeof(uint32_t)}}};
    std::array<VkWriteDescriptorSet, 2> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i)
    {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file)
        return 5;
    auto size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    file.read(reinterpret_cast<char*>(words.data()), size);
    if (!file)
        return 5;
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = words.size() * 4;
    smci.pCode    = words.data();
    VkShaderModule shader;
    check(vkCreateShaderModule(device, &smci, nullptr, &shader));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.layout       = layout;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName  = "main";
    VkPipeline pipeline;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    VkCommandPool commands;
    check(vkCreateCommandPool(device, &cpi, nullptr, &commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = commands;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer command;
    check(vkAllocateCommandBuffers(device, &cai, &command));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(command, cells / 64, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         0,
                         1,
                         &barrier,
                         0,
                         nullptr,
                         0,
                         nullptr);
    check(vkEndCommandBuffer(command));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence           fence;
    check(vkCreateFence(device, &fci, nullptr, &fence));
    const std::array<uint32_t, 20> edge{{0,          1,          2,          3,          0x3fffff,
                                         0x400000,   0x7fffff,   0x800000,   0x800001,   0x3effffff,
                                         0x3f000000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x40000000,
                                         0x7f7ffffe, 0x7f7fffff, 0x7f800000, 0x7fc00000, 0x7fffffff}};
    uint32_t                       rng = 0x9e3779b9, failures = 0, product_failures = 0;
    auto                           random = [&]()
    {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };
    double                  elapsed_ms = 0;
    std::array<uint32_t, 5> total_kind{}, mismatch_kind{};
    for (uint32_t pass = 0; pass < passes; ++pass)
    {
        for (uint32_t n = 0; n < cells; ++n)
        {
            input[2 * n]     = random();
            input[2 * n + 1] = random();
            if (n < edge.size() * edge.size() * 4)
            {
                input[2 * n] = edge[n % edge.size()] | ((n / (edge.size() * edge.size())) & 1 ? 0x80000000u : 0);
                input[2 * n + 1] =
                    edge[n / edge.size() % edge.size()] | ((n / (edge.size() * edge.size())) & 2 ? 0x80000000u : 0);
            }
            output[product_error ? 2 * n : n] = 0xdeadbeef;
            if (product_error)
                output[2 * n + 1] = 0xdeadbeef;
        }
        check(vkResetFences(device, 1, &fence));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &command;
        const auto start          = std::chrono::steady_clock::now();
        check(vkQueueSubmit(queue, 1, &submit, fence));
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ULL));
        elapsed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (uint32_t n = 0; n < cells; ++n)
        {
            float a, b;
            std::memcpy(&a, &input[2 * n], 4);
            std::memcpy(&b, &input[2 * n + 1], 4);
            float expected;
            int   kind = 0;
            if (product_error)
            {
                volatile float product = a * b;
                expected               = std::fma(a, b, -product);
                const float rounded    = product;
                uint32_t    product_bits;
                std::memcpy(&product_bits, &rounded, 4);
                const uint32_t actual = output[2 * n + 1];
                if (std::isnan(rounded) ? (actual & 0x7fffffffu) <= 0x7f800000u : product_bits != actual)
                    ++product_failures;
                kind = !std::isfinite(a) || !std::isfinite(b)     ? 4
                       : std::isinf(product)                      ? 3
                       : product == 0                             ? 2
                       : std::fpclassify(product) == FP_SUBNORMAL ? 1
                                                                  : 0;
                ++total_kind[kind];
            }
            else
                expected = static_cast<float>(static_cast<long double>(a) / static_cast<long double>(b));
            uint32_t bits;
            std::memcpy(&bits, &expected, 4);
            const uint32_t actual = output[product_error ? 2 * n : n];
            const bool     nan    = (actual & 0x7fffffffu) > 0x7f800000u;
            if (std::isnan(expected) ? !nan : bits != actual)
            {
                if (failures < 8)
                    std::cerr << std::hex << input[2 * n] << '/' << input[2 * n + 1] << " expected " << bits << " got "
                              << actual << std::dec << '\n';
                ++failures;
                if (product_error)
                    ++mismatch_kind[kind];
            }
        }
    }
    std::cout << "{\"pairs\":" << cells * passes << ",\"product_mismatches\":" << product_failures
              << ",\"mismatches\":" << failures << ",\"dispatch_wall_ms\":" << elapsed_ms << "}\n";
    if (product_error)
        for (size_t kind = 0; kind < total_kind.size(); ++kind)
            std::cerr << "kind=" << kind << " mismatches=" << mismatch_kind[kind] << " total=" << total_kind[kind]
                      << '\n';
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commands, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorSetLayout(device, ds, nullptr);
    vkUnmapMemory(device, allocation);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, allocation, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return failures || product_failures ? 1 : 0;
}
