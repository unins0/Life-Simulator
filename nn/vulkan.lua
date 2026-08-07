-- nn/vulkan.lua — Vulkan loader + device context (lazy, optional).
--
-- A real LuaJIT FFI binding to the Vulkan 1.2 API, structured to run on
-- Windows (vulkan-1.dll) and Linux (libvulkan.so.1). The module IMPORT never
-- fails: with no loader/ICD, no FFI, or on unsupported platforms every runtime
-- entry point degrades to a structured error via nn/errors.lua classes
-- (VULKAN_UNSUPPORTED_PLATFORM, VULKAN_INITIALIZATION_FAILED, ...) and init()
-- returns nil + err instead of raising. The core `auto` backend calls this
-- module through pcall and falls back to CPU.
--
-- Extension gates (tracked in capabilities):
--   VK_KHR_16bit_storage          -> f16 STORAGE only
--   VK_KHR_shader_float16_int8    -> native f16 ARITHMETIC
--      (16bit_storage alone does NOT authorize fp16 arithmetic)
--   VK_KHR_shader_8bit_storage    -> optional byte-addressable SSBO path
--      (the mandatory primary path is word-addressed uint[] loadByte;
--       the 8-bit path, when available, must produce identical bytes)
--
-- ABI provenance: struct layouts mirror the official Vulkan headers (captured
-- from vulkan_core.h 1.4 on the reference box; LP64). The golden ABI test
-- (abi_check + nn/tests/test_vulkan_skeleton.lua) pins sizeof/offsetof so a
-- header/toolchain mismatch surfaces as ABI_MISMATCH instead of UB.

local _M = {
    _VERSION = '1.0.0',
    name = 'vulkan',
}

-- Structured error classes (canonical names shared with nn/errors.lua).
_M.ERRORS = {
    VULKAN_UNSUPPORTED_PLATFORM   = 'VULKAN_UNSUPPORTED_PLATFORM',
    VULKAN_EXTENSION_MISSING      = 'VULKAN_EXTENSION_MISSING',
    VULKAN_INITIALIZATION_FAILED  = 'VULKAN_INITIALIZATION_FAILED',
    VULKAN_PIPELINE_FAILED        = 'VULKAN_PIPELINE_FAILED',
    VULKAN_ARENA_EXHAUSTED        = 'VULKAN_ARENA_EXHAUSTED',
    DEVICE_LOST                   = 'DEVICE_LOST',
    ABI_MISMATCH                  = 'ABI_MISMATCH',
}

-- Compat shim for the core nn/errors.lua (created in parallel). If it exists,
-- prefer its constructors; otherwise build a minimal structured error table so
-- callers/tests can rely on {class=..., message=...} regardless.
local errors_mod
do
    local ok, mod = pcall(require, 'nn.errors')
    if ok and type(mod) == 'table' then errors_mod = mod end
end

local function mkerr(class, message)
    local e
    if errors_mod then
        local ctors = { errors_mod.new, errors_mod.error }
        for _, fn in ipairs(ctors) do
            if type(fn) == 'function' then
                local ok, r = pcall(fn, class, message)
                if ok and type(r) == 'table' then
                    e = r
                    break
                end
            end
        end
        if not e and type(errors_mod[class]) == 'function' then
            local ok, r = pcall(errors_mod[class], message)
            if ok and type(r) == 'table' then e = r end
        end
    end
    e = e or { nn_error = true }
    e.class = e.class or class
    e.message = e.message or message
    return e
end
_M.mkerr = mkerr

-- ---------------------------------------------------------------- platform --
local ffi_ok, ffi = pcall(require, 'ffi')
local ffi = ffi_ok and ffi or nil -- luacheck: ignore
_M.ffi = ffi

local function detect_os()
    if ffi and ffi.os then return ffi.os end
    local sep = package.config:sub(1, 1)
    if sep == '\\' then return 'Windows' end
    if sep == '/' then
        local p = io.popen('uname -s 2>/dev/null')
        if p then
            local s = p:read('*l')
            p:close()
            if s then
                if s:find('Linux') then return 'Linux' end
                if s:find('Darwin') then return 'OSX' end
            end
        end
        return 'Linux'
    end
    return 'Other'
end

local OS = detect_os()
_M.os = OS

local LOADER_NAMES = { Windows = 'vulkan-1', Linux = 'libvulkan.so.1' }

-- ------------------------------------------------------------------- FFI ----
local cdef_done = false
local function define_types()
    if cdef_done then return end
    cdef_done = true
    ffi.cdef[[
typedef int32_t VkResult;
typedef int32_t VkBool32;
typedef uint32_t VkFlags;
typedef uint64_t VkDeviceSize;
typedef size_t size_t;
typedef uint32_t VkSampleCountFlags;
typedef uint32_t VkMemoryPropertyFlags;
typedef uint32_t VkQueueFlags;
typedef uint32_t VkAccessFlags;
typedef uint32_t VkPipelineStageFlags;
typedef uint32_t VkShaderStageFlags;
typedef uint32_t VkBufferUsageFlags;
typedef uint32_t VkStructureType;
typedef int32_t VkPhysicalDeviceType;
typedef int32_t VkDescriptorType;
typedef int32_t VkPipelineBindPoint;
typedef int32_t VkCommandBufferLevel;
typedef int32_t VkSharingMode;

/* Handles: dispatchable are pointers; non-dispatchable are uint64 per the
 * official ABI (uint64_t on 64-bit targets; matches on LP64 and LP32). */
typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef void* VkDevice;
typedef void* VkQueue;
typedef void* VkCommandBuffer;
typedef uint64_t VkBuffer;
typedef uint64_t VkDeviceMemory;
typedef uint64_t VkShaderModule;
typedef uint64_t VkDescriptorSetLayout;
typedef uint64_t VkPipelineLayout;
typedef uint64_t VkPipeline;
typedef uint64_t VkDescriptorPool;
typedef uint64_t VkDescriptorSet;
typedef uint64_t VkFence;
typedef uint64_t VkSemaphore;
typedef uint64_t VkCommandPool;
typedef uint64_t VkPipelineCache;

typedef struct VkAllocationCallbacks VkAllocationCallbacks;
typedef struct VkCopyDescriptorSet VkCopyDescriptorSet;

typedef struct VkApplicationInfo {
    VkStructureType sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
} VkApplicationInfo;

typedef struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct VkPhysicalDeviceFeatures {
    VkBool32 robustBufferAccess;
    VkBool32 fullDrawIndexUint32;
    VkBool32 imageCubeArray;
    VkBool32 independentBlend;
    VkBool32 geometryShader;
    VkBool32 tessellationShader;
    VkBool32 sampleRateShading;
    VkBool32 dualSrcBlend;
    VkBool32 logicOp;
    VkBool32 multiDrawIndirect;
    VkBool32 drawIndirectFirstInstance;
    VkBool32 depthClamp;
    VkBool32 depthBiasClamp;
    VkBool32 fillModeNonSolid;
    VkBool32 depthBounds;
    VkBool32 wideLines;
    VkBool32 largePoints;
    VkBool32 alphaToOne;
    VkBool32 multiViewport;
    VkBool32 samplerAnisotropy;
    VkBool32 textureCompressionETC2;
    VkBool32 textureCompressionBC;
    VkBool32 occlusionQueryPrecise;
    VkBool32 pipelineStatisticsQuery;
    VkBool32 vertexPipelineStoresAndAtomics;
    VkBool32 fragmentStoresAndAtomics;
    VkBool32 shaderTessellationAndGeometryPointSize;
    VkBool32 shaderImageGatherExtended;
    VkBool32 shaderStorageImageExtendedFormats;
    VkBool32 shaderStorageImageMultisample;
    VkBool32 shaderStorageImageReadWithoutFormat;
    VkBool32 shaderStorageImageWriteWithoutFormat;
    VkBool32 shaderUniformBufferArrayDynamicIndexing;
    VkBool32 shaderSampledImageArrayDynamicIndexing;
    VkBool32 shaderStorageBufferArrayDynamicIndexing;
    VkBool32 shaderStorageImageArrayDynamicIndexing;
    VkBool32 shaderClipDistance;
    VkBool32 shaderCullDistance;
    VkBool32 shaderFloat64;
    VkBool32 shaderInt64;
    VkBool32 shaderInt16;
    VkBool32 shaderResourceResidency;
    VkBool32 shaderResourceMinLod;
    VkBool32 sparseBinding;
    VkBool32 sparseResidencyBuffer;
    VkBool32 sparseResidencyImage2D;
    VkBool32 sparseResidencyImage3D;
    VkBool32 sparseResidency2Samples;
    VkBool32 sparseResidency4Samples;
    VkBool32 sparseResidency8Samples;
    VkBool32 sparseResidency16Samples;
    VkBool32 sparseResidencyAliased;
    VkBool32 variableMultisampleRate;
    VkBool32 inheritedQueries;
} VkPhysicalDeviceFeatures;

typedef struct VkExtent3D {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
} VkExtent3D;

typedef struct VkQueueFamilyProperties {
    VkQueueFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
} VkQueueFamilyProperties;

typedef struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;

typedef struct VkPhysicalDeviceLimits {
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxTexelBufferElements;
    uint32_t maxUniformBufferRange;
    uint32_t maxStorageBufferRange;
    uint32_t maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount;
    uint32_t maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets;
    uint32_t maxPerStageDescriptorSamplers;
    uint32_t maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers;
    uint32_t maxPerStageDescriptorSampledImages;
    uint32_t maxPerStageDescriptorStorageImages;
    uint32_t maxPerStageDescriptorInputAttachments;
    uint32_t maxPerStageResources;
    uint32_t maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUniformBuffers;
    uint32_t maxDescriptorSetUniformBuffersDynamic;
    uint32_t maxDescriptorSetStorageBuffers;
    uint32_t maxDescriptorSetStorageBuffersDynamic;
    uint32_t maxDescriptorSetSampledImages;
    uint32_t maxDescriptorSetStorageImages;
    uint32_t maxDescriptorSetInputAttachments;
    uint32_t maxVertexInputAttributes;
    uint32_t maxVertexInputBindings;
    uint32_t maxVertexInputAttributeOffset;
    uint32_t maxVertexInputBindingStride;
    uint32_t maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel;
    uint32_t maxTessellationPatchSize;
    uint32_t maxTessellationControlPerVertexInputComponents;
    uint32_t maxTessellationControlPerVertexOutputComponents;
    uint32_t maxTessellationControlPerPatchOutputComponents;
    uint32_t maxTessellationControlTotalOutputComponents;
    uint32_t maxTessellationEvaluationInputComponents;
    uint32_t maxTessellationEvaluationOutputComponents;
    uint32_t maxGeometryShaderInvocations;
    uint32_t maxGeometryInputComponents;
    uint32_t maxGeometryOutputComponents;
    uint32_t maxGeometryOutputVertices;
    uint32_t maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents;
    uint32_t maxFragmentOutputAttachments;
    uint32_t maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t subPixelPrecisionBits;
    uint32_t subTexelPrecisionBits;
    uint32_t mipmapPrecisionBits;
    uint32_t maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount;
    float maxSamplerLodBias;
    float maxSamplerAnisotropy;
    uint32_t maxViewports;
    uint32_t maxViewportDimensions[2];
    float viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    int32_t minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    float minInterpolationOffset;
    float maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    VkSampleCountFlags framebufferColorSampleCounts;
    VkSampleCountFlags framebufferDepthSampleCounts;
    VkSampleCountFlags framebufferStencilSampleCounts;
    VkSampleCountFlags framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkSampleCountFlags sampledImageColorSampleCounts;
    VkSampleCountFlags sampledImageIntegerSampleCounts;
    VkSampleCountFlags sampledImageDepthSampleCounts;
    VkSampleCountFlags sampledImageStencilSampleCounts;
    VkSampleCountFlags storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
    uint32_t discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity;
    float lineWidthGranularity;
    VkBool32 strictLines;
    VkBool32 standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
} VkPhysicalDeviceLimits;

typedef struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    VkPhysicalDeviceType deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
} VkPhysicalDeviceProperties;

typedef struct VkMemoryType {
    VkMemoryPropertyFlags propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags flags;
} VkMemoryHeap;

typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct VkExtensionProperties {
    char extensionName[256];
    uint32_t specVersion;
} VkExtensionProperties;

typedef struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures* pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct VkBufferCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    VkSharingMode sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
} VkMemoryRequirements;

typedef struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct VkMappedMemoryRange {
    VkStructureType sType;
    const void* pNext;
    VkDeviceMemory memory;
    VkDeviceSize offset;
    VkDeviceSize size;
} VkMappedMemoryRange;

typedef struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t* pCode;
} VkShaderModuleCreateInfo;

typedef struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    VkDescriptorType descriptorType;
    uint32_t descriptorCount;
    VkShaderStageFlags stageFlags;
    const void* pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct VkPushConstantRange {
    VkShaderStageFlags stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange* pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct VkSpecializationMapEntry {
    uint32_t constantID;
    uint32_t offset;
    size_t size;
} VkSpecializationMapEntry;

typedef struct VkSpecializationInfo {
    uint32_t mapEntryCount;
    const VkSpecializationMapEntry* pMapEntries;
    size_t dataSize;
    const void* pData;
} VkSpecializationInfo;

typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkShaderStageFlags stage;
    VkShaderModule module;
    const char* pName;
    const VkSpecializationInfo* pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
} VkComputePipelineCreateInfo;

typedef struct VkDescriptorPoolSize {
    VkDescriptorType type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo;

typedef struct VkWriteDescriptorSet {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    VkDescriptorType descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandPool commandPool;
    VkCommandBufferLevel level;
    uint32_t commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct VkBufferCopy {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} VkBufferCopy;

typedef struct VkBufferMemoryBarrier {
    VkStructureType sType;
    const void* pNext;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
    uint32_t srcQueueFamilyIndex;
    uint32_t dstQueueFamilyIndex;
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
} VkBufferMemoryBarrier;

typedef struct VkMemoryBarrier {
    VkStructureType sType;
    const void* pNext;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
} VkMemoryBarrier;

typedef struct VkSubmitInfo {
    VkStructureType sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore* pWaitSemaphores;
    const VkPipelineStageFlags* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const VkSemaphore* pSignalSemaphores;
} VkSubmitInfo;

typedef struct VkFenceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
} VkFenceCreateInfo;

typedef struct VkSemaphoreCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
} VkSemaphoreCreateInfo;

/* ---- function pointer types (minimal: enumerate/instance/device/queue/
 *      buffer/memory/shader-module/descriptor-set/pipeline/command/fence/
 *      semaphore/DeviceWaitIdle families) ---- */

/* Loader-exported entry points (dlsym/GetProcAddress targets). */
void* vkGetInstanceProcAddr(void* instance, const char* pName);
void* vkGetDeviceProcAddr(void* device, const char* pName);

typedef void* (*PFN_vkGetInstanceProcAddr)(VkInstance instance, const char* pName);
typedef void* (*PFN_vkGetDeviceProcAddr)(VkDevice device, const char* pName);

typedef VkResult (*PFN_vkCreateInstance)(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance);
typedef void (*PFN_vkDestroyInstance)(VkInstance instance, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkEnumerateInstanceVersion)(uint32_t* pApiVersion);
typedef VkResult (*PFN_vkEnumeratePhysicalDevices)(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices);
typedef void (*PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties);
typedef void (*PFN_vkGetPhysicalDeviceFeatures)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures* pFeatures);
typedef void (*PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties);
typedef void (*PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties);
typedef VkResult (*PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties);
typedef VkResult (*PFN_vkCreateDevice)(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice);
typedef void (*PFN_vkDestroyDevice)(VkDevice device, const VkAllocationCallbacks* pAllocator);
typedef void (*PFN_vkGetDeviceQueue)(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue);

typedef VkResult (*PFN_vkCreateBuffer)(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer);
typedef void (*PFN_vkDestroyBuffer)(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator);
typedef void (*PFN_vkGetBufferMemoryRequirements)(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements);
typedef VkResult (*PFN_vkAllocateMemory)(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);
typedef void (*PFN_vkFreeMemory)(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkBindBufferMemory)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset);
typedef VkResult (*PFN_vkMapMemory)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkFlags flags, void** ppData);
typedef void (*PFN_vkUnmapMemory)(VkDevice device, VkDeviceMemory memory);
typedef VkResult (*PFN_vkFlushMappedMemoryRanges)(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges);
typedef VkResult (*PFN_vkInvalidateMappedMemoryRanges)(VkDevice device, uint32_t memoryRangeCount, const VkMappedMemoryRange* pMemoryRanges);

typedef VkResult (*PFN_vkCreateShaderModule)(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule);
typedef void (*PFN_vkDestroyShaderModule)(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkCreateDescriptorSetLayout)(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout);
typedef void (*PFN_vkDestroyDescriptorSetLayout)(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkCreatePipelineLayout)(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout);
typedef VkResult (*PFN_vkCreateComputePipelines)(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);
typedef void (*PFN_vkDestroyPipeline)(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator);
typedef void (*PFN_vkDestroyPipelineLayout)(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator);

typedef VkResult (*PFN_vkCreateDescriptorPool)(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool);
typedef void (*PFN_vkDestroyDescriptorPool)(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkAllocateDescriptorSets)(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets);
typedef void (*PFN_vkUpdateDescriptorSets)(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies);

typedef VkResult (*PFN_vkCreateCommandPool)(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool);
typedef void (*PFN_vkDestroyCommandPool)(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkAllocateCommandBuffers)(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers);
typedef VkResult (*PFN_vkBeginCommandBuffer)(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo);
typedef VkResult (*PFN_vkEndCommandBuffer)(VkCommandBuffer commandBuffer);
typedef void (*PFN_vkFreeCommandBuffers)(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers);

typedef VkResult (*PFN_vkCmdBindPipeline)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline);
typedef void (*PFN_vkCmdBindDescriptorSets)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets);
typedef void (*PFN_vkCmdPushConstants)(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues);
typedef void (*PFN_vkCmdDispatch)(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
typedef void (*PFN_vkCmdCopyBuffer)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions);
typedef void (*PFN_vkCmdPipelineBarrier)(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const void* pImageMemoryBarriers);

typedef VkResult (*PFN_vkQueueSubmit)(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
typedef VkResult (*PFN_vkWaitForFences)(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout);
typedef VkResult (*PFN_vkResetFences)(VkDevice device, uint32_t fenceCount, const VkFence* pFences);
typedef VkResult (*PFN_vkCreateFence)(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence);
typedef void (*PFN_vkDestroyFence)(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkCreateSemaphore)(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore);
typedef void (*PFN_vkDestroySemaphore)(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator);
typedef VkResult (*PFN_vkDeviceWaitIdle)(VkDevice device);
]]
end

-- --------------------------------------------------------------- constants --
-- Values captured from vulkan_core.h (Vulkan 1.2 / 1.4 headers).
local VK = {
    API_VERSION_1_0 = 0x00400000,
    API_VERSION_1_1 = 0x00401000,
    API_VERSION_1_2 = 0x00402000,

    SUCCESS = 0,
    ERROR_DEVICE_LOST = -4,
    ERROR_EXTENSION_NOT_PRESENT = -7,
    ERROR_INCOMPATIBLE_DRIVER = -9,

    STRUCTURE_TYPE_APPLICATION_INFO = 0,
    STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    STRUCTURE_TYPE_SUBMIT_INFO = 4,
    STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    STRUCTURE_TYPE_MAPPED_MEMORY_RANGE = 6,
    STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = 9,
    STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    STRUCTURE_TYPE_MEMORY_BARRIER = 43,
    STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44,

    QUEUE_GRAPHICS_BIT = 0x1,
    QUEUE_COMPUTE_BIT = 0x2,
    QUEUE_TRANSFER_BIT = 0x4,

    BUFFER_USAGE_TRANSFER_SRC_BIT = 0x1,
    BUFFER_USAGE_TRANSFER_DST_BIT = 0x2,
    BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x20,

    MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x1,
    MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x2,
    MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x4,

    DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    SHADER_STAGE_COMPUTE_BIT = 0x20,
    PIPELINE_BIND_POINT_COMPUTE = 1,
    COMMAND_BUFFER_LEVEL_PRIMARY = 0,
    COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x1,

    PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x800,
    PIPELINE_STAGE_TRANSFER_BIT = 0x1000,
    PIPELINE_STAGE_HOST_BIT = 0x4000,
    PIPELINE_STAGE_ALL_COMMANDS_BIT = 0x10000,

    ACCESS_SHADER_READ_BIT = 0x20,
    ACCESS_SHADER_WRITE_BIT = 0x40,
    ACCESS_TRANSFER_READ_BIT = 0x800,
    ACCESS_TRANSFER_WRITE_BIT = 0x1000,
    ACCESS_HOST_READ_BIT = 0x2000,
    ACCESS_HOST_WRITE_BIT = 0x4000,

    FENCE_CREATE_SIGNALED_BIT = 0x1,

    PHYSICAL_DEVICE_TYPE_OTHER = 0,
    PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
}
_M.VK = VK

local function make_api_version(major, minor, patch)
    return (major << 22) | (minor << 12) | patch
end
_M.make_api_version = make_api_version

-- VK_WHOLE_SIZE = ~0 (uint64). A bare 0xFFFFFFFFFFFFFFFF Lua literal is a
-- double and casts to 0 in uint64 — build it via uint64 arithmetic instead.
if ffi then
    VK.VK_WHOLE_SIZE = ffi.cast('uint64_t', 0) - 1
else
    VK.VK_WHOLE_SIZE = 0xFFFFFFFFFFFFFFFF -- placeholder; never used without ffi
end

-- ------------------------------------------------------------------ loader --
local _loader -- { lib, gipa, gdpa, os, loader_name }
local _load_error

local function load_loader()
    if _loader then return _loader end
    if _load_error then return nil, _load_error end

    if not LOADER_NAMES[OS] then
        _load_error = mkerr(_M.ERRORS.VULKAN_UNSUPPORTED_PLATFORM,
            'Vulkan is supported only on Windows/Linux, got platform: ' .. tostring(OS))
        return nil, _load_error
    end
    if not ffi then
        _load_error = mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED,
            'FFI unavailable (run under LuaJIT; plain Lua has no ffi module)')
        return nil, _load_error
    end

    define_types()
    local lib
    do
        local ok, l = pcall(ffi.load, LOADER_NAMES[OS])
        if not ok then
            _load_error = mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED,
                'cannot load Vulkan loader: ' .. LOADER_NAMES[OS])
            return nil, _load_error
        end
        lib = l
    end

    local gipa = ffi.cast('PFN_vkGetInstanceProcAddr', lib.vkGetInstanceProcAddr)
    local gdpa = ffi.cast('PFN_vkGetDeviceProcAddr', lib.vkGetDeviceProcAddr)
    if gipa == nil or gdpa == nil then
        _load_error = mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED,
            'loader missing vkGetInstanceProcAddr/vkGetDeviceProcAddr')
        return nil, _load_error
    end

    _loader = { lib = lib, gipa = gipa, gdpa = gdpa, os = OS, loader_name = LOADER_NAMES[OS] }
    return _loader
end

function _M.can_load()
    local loader = load_loader()
    return loader ~= nil
end

function _M.loader_info()
    local loader = load_loader()
    if not loader then return nil end
    return { loader_name = loader.loader_name, os = loader.os }
end

-- Resolve an entry point. scope: 'global' | 'instance' | 'device'.
local function resolve(loader, obj, scope, name, pfn_t)
    local p
    if scope == 'global' then
        p = loader.gipa(nil, name)
    elseif scope == 'instance' then
        p = loader.gipa(obj, name)
    else
        p = loader.gdpa(obj, name)
        if p == nil then p = loader.gipa(obj, name) end
    end
    if p == nil then return nil end
    return ffi.cast(pfn_t, p)
end
_M.resolve = resolve

-- NUL-terminated C string from a Lua string.
local function cstr(s)
    local p = ffi.new('char[?]', #s + 1)
    ffi.copy(p, s, #s + 1)
    return p
end
_M.cstr = cstr

-- ------------------------------------------------------------------- caps ---
-- Live capability table; reflects whatever the runtime probe found. Reset by
-- query_capabilities()/init(). Mirrors the fields the core nn/capabilities.lua
-- expects from a backend.
local capabilities = {
    vulkan = false,
    backend = 'gpu',
    loader = nil,
    loader_name = nil,
    api_version = 0,          -- instance api version (0 = unknown/no loader)
    target_api = '1.2',
    spv_target = 'SPIR-V 1.5',
    shader_abi = 2,
    device = nil,             -- { name, type, vendor, device, api_version } or nil
    queue_family = nil,
    extensions = {
        sixteen_bit_storage = false, -- VK_KHR_16bit_storage
        shader_float16_int8 = false, -- VK_KHR_shader_float16_int8
        shader_8bit_storage = false, -- VK_KHR_shader_8bit_storage
    },
    caps = {
        f16_storage = false,       -- fp16 SSBO storage allowed
        f16_arithmetic = false,    -- native fp16 arithmetic allowed
        i8_storage = false,        -- byte-addressable uint8 SSBO path
        word_addressed_payload = true, -- primary path: uint[] loadByte
    },
}
_M.capabilities = capabilities

function _M.query_capabilities()
    local loader = load_loader()
    if not loader then
        capabilities.vulkan = false
        capabilities.loader = nil
        return capabilities
    end
    return capabilities -- probe happens during init(); before that, vulkan=false
end

-- ------------------------------------------------------------------ init ---
local _ctx -- cached active context (idempotent init)

local function pick_memory_type(fn_mem_props, physical, type_bits, required)
    local props = ffi.new('VkPhysicalDeviceMemoryProperties')
    fn_mem_props(physical, props)
    for i = 0, props.memoryTypeCount - 1 do
        local mask = 1 << i
        if (type_bits & mask) ~= 0 then
            local t = props.memoryTypes[i]
            if (t.propertyFlags & required) == required then
                return i
            end
        end
    end
    return nil
end
_M.pick_memory_type = pick_memory_type

local function find_compute_queue_family(fn_queue_props, physical)
    local count = ffi.new('uint32_t[1]')
    fn_queue_props(physical, count, nil)
    if count[0] == 0 then return nil end
    local props = ffi.new('VkQueueFamilyProperties[?]', count[0])
    fn_queue_props(physical, count, props)
    -- Prefer a compute-capable family (compute or graphics implies compute).
    for i = 0, count[0] - 1 do
        if (props[i].queueFlags & VK.QUEUE_COMPUTE_BIT) ~= 0 then
            return i
        end
    end
    return nil
end

local function enumerate_available_extensions(fn_enum_ext, physical)
    local out = {}
    local count = ffi.new('uint32_t[1]')
    local res = fn_enum_ext(physical, nil, count, nil)
    if res ~= VK.SUCCESS or count[0] == 0 then return out end
    local exts = ffi.new('VkExtensionProperties[?]', count[0])
    res = fn_enum_ext(physical, nil, count, exts)
    if res ~= VK.SUCCESS then return out end
    for i = 0, count[0] - 1 do
        out[#out + 1] = ffi.string(exts[i].extensionName)
    end
    return out
end

-- init(opts) -> ctx, err, already_initialized
-- opts.require_extensions = { 'VK_KHR_16bit_storage', ... } -> VULKAN_EXTENSION_MISSING if absent
-- opts.api_version = 0x00402000 (default Vulkan 1.2)
function _M.init(opts)
    if _ctx then return _ctx, nil, true end
    opts = opts or {}

    local loader, lerr = load_loader()
    if not loader then return nil, lerr end

    -- Optional: vkEnumerateInstanceVersion.
    local instance_version = 0
    do
        local fn = resolve(loader, nil, 'global', 'vkEnumerateInstanceVersion', 'PFN_vkEnumerateInstanceVersion')
        if fn then
            local v = ffi.new('uint32_t[1]')
            if fn(v) == VK.SUCCESS then instance_version = v[0] end
        end
    end

    -- Create instance.
    local fnCreateInstance = resolve(loader, nil, 'global', 'vkCreateInstance', 'PFN_vkCreateInstance')
    if not fnCreateInstance then
        return nil, mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'vkCreateInstance not resolvable')
    end
    local app = ffi.new('VkApplicationInfo')
    app.sType = VK.STRUCTURE_TYPE_APPLICATION_INFO
    app.pApplicationName = cstr('life-simulator')
    app.applicationVersion = 1
    app.pEngineName = cstr('nn-vulkan')
    app.engineVersion = 1
    app.apiVersion = opts.api_version or VK.API_VERSION_1_2

    local ici = ffi.new('VkInstanceCreateInfo')
    ici.sType = VK.STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    ici.pApplicationInfo = app
    -- No instance layers/extensions for the skeleton.

    local inst = ffi.new('VkInstance[1]')
    local res = fnCreateInstance(ici, nil, inst)
    if res ~= VK.SUCCESS then
        if res == VK.ERROR_INCOMPATIBLE_DRIVER then
            return nil, mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED,
                'no compatible Vulkan ICD (vkCreateInstance = ' .. res .. ')')
        end
        return nil, mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'vkCreateInstance failed: ' .. res)
    end
    local instance = inst[0]

    local function fail(err)
        local d = resolve(loader, instance, 'instance', 'vkDestroyInstance', 'PFN_vkDestroyInstance')
        if d then d(instance, nil) end
        return nil, err
    end

    -- Enumerate physical devices.
    local fnEnumPhys = resolve(loader, instance, 'instance', 'vkEnumeratePhysicalDevices', 'PFN_vkEnumeratePhysicalDevices')
    local fnPhysProps = resolve(loader, instance, 'instance', 'vkGetPhysicalDeviceProperties', 'PFN_vkGetPhysicalDeviceProperties')
    local fnQueueProps = resolve(loader, instance, 'instance', 'vkGetPhysicalDeviceQueueFamilyProperties', 'PFN_vkGetPhysicalDeviceQueueFamilyProperties')
    local fnMemProps = resolve(loader, instance, 'instance', 'vkGetPhysicalDeviceMemoryProperties', 'PFN_vkGetPhysicalDeviceMemoryProperties')
    local fnEnumExt = resolve(loader, instance, 'instance', 'vkEnumerateDeviceExtensionProperties', 'PFN_vkEnumerateDeviceExtensionProperties')
    if not (fnEnumPhys and fnPhysProps and fnQueueProps and fnMemProps and fnEnumExt) then
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'required instance entry points not resolvable'))
    end

    local dev_count = ffi.new('uint32_t[1]')
    res = fnEnumPhys(instance, dev_count, nil)
    if res ~= VK.SUCCESS or dev_count[0] == 0 then
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'no physical devices (count=' .. dev_count[0] .. ')'))
    end
    local phys = ffi.new('VkPhysicalDevice[?]', dev_count[0])
    res = fnEnumPhys(instance, dev_count, phys)
    if res ~= VK.SUCCESS then
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'vkEnumeratePhysicalDevices failed: ' .. res))
    end

    -- Pick the first device with a compute-capable queue family.
    local physical, family
    for i = 0, dev_count[0] - 1 do
        local f = find_compute_queue_family(fnQueueProps, phys[i])
        if f then physical, family = phys[i], f break end
    end
    if not physical then
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'no physical device with a compute queue family'))
    end

    -- Device properties.
    local props = ffi.new('VkPhysicalDeviceProperties')
    fnPhysProps(physical, props)
    local device_name = ffi.string(props.deviceName)
    local device_type = props.deviceType

    -- Extension availability + gates.
    local available = enumerate_available_extensions(fnEnumExt, physical)
    local avail = {}
    for _, n in ipairs(available) do avail[n] = true end

    local have_16bit = avail['VK_KHR_16bit_storage'] == true
    local have_f16arith = avail['VK_KHR_shader_float16_int8'] == true
    local have_8bit = avail['VK_KHR_shader_8bit_storage'] == true

    -- User-requested extensions that are mandatory must be present.
    if opts.require_extensions then
        for _, name in ipairs(opts.require_extensions) do
            if not avail[name] then
                return fail(mkerr(_M.ERRORS.VULKAN_EXTENSION_MISSING, 'extension not supported by device: ' .. name))
            end
        end
    end

    -- Build the enabled-extension list: auto-enable the optional gates.
    local enabled_names = {}
    if have_16bit then enabled_names[#enabled_names + 1] = 'VK_KHR_16bit_storage' end
    if have_f16arith then enabled_names[#enabled_names + 1] = 'VK_KHR_shader_float16_int8' end
    if have_8bit then enabled_names[#enabled_names + 1] = 'VK_KHR_shader_8bit_storage' end

    -- Create device.
    local fnCreateDevice = resolve(loader, instance, 'instance', 'vkCreateDevice', 'PFN_vkCreateDevice')
    if not fnCreateDevice then
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'vkCreateDevice not resolvable'))
    end
    local qci = ffi.new('VkDeviceQueueCreateInfo')
    qci.sType = VK.STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
    qci.queueFamilyIndex = family
    qci.queueCount = 1
    local prio = ffi.new('float[1]', 1.0)
    qci.pQueuePriorities = prio

    local dci = ffi.new('VkDeviceCreateInfo')
    dci.sType = VK.STRUCTURE_TYPE_DEVICE_CREATE_INFO
    dci.queueCreateInfoCount = 1
    dci.pQueueCreateInfos = qci
    -- pEnabledFeatures = NULL: skeleton runs with default (disabled) features.
    -- Enabling fp16 SSBO storage would require VkPhysicalDevice16BitStorageFeatures
    -- in the pNext chain; not needed for the packed path.

    if #enabled_names > 0 then
        local names = ffi.new('const char*[?]', #enabled_names)
        for i, n in ipairs(enabled_names) do
            names[i - 1] = cstr(n)
        end
        dci.enabledExtensionCount = #enabled_names
        dci.ppEnabledExtensionNames = names
    end

    local dev = ffi.new('VkDevice[1]')
    res = fnCreateDevice(physical, dci, nil, dev)
    if res ~= VK.SUCCESS then
        local cls = res == VK.ERROR_DEVICE_LOST and _M.ERRORS.DEVICE_LOST or _M.ERRORS.VULKAN_INITIALIZATION_FAILED
        return fail(mkerr(cls, 'vkCreateDevice failed: ' .. res))
    end
    local device = dev[0]

    -- Bind all device-level functions.
    local fn = {}
    local device_bindings = {
        getDeviceQueue = 'vkGetDeviceQueue',
        createBuffer = 'vkCreateBuffer',
        destroyBuffer = 'vkDestroyBuffer',
        getBufferMemoryRequirements = 'vkGetBufferMemoryRequirements',
        allocateMemory = 'vkAllocateMemory',
        freeMemory = 'vkFreeMemory',
        bindBufferMemory = 'vkBindBufferMemory',
        mapMemory = 'vkMapMemory',
        unmapMemory = 'vkUnmapMemory',
        flushMappedMemoryRanges = 'vkFlushMappedMemoryRanges',
        invalidateMappedMemoryRanges = 'vkInvalidateMappedMemoryRanges',
        createShaderModule = 'vkCreateShaderModule',
        destroyShaderModule = 'vkDestroyShaderModule',
        createDescriptorSetLayout = 'vkCreateDescriptorSetLayout',
        destroyDescriptorSetLayout = 'vkDestroyDescriptorSetLayout',
        createPipelineLayout = 'vkCreatePipelineLayout',
        createComputePipelines = 'vkCreateComputePipelines',
        destroyPipeline = 'vkDestroyPipeline',
        destroyPipelineLayout = 'vkDestroyPipelineLayout',
        createDescriptorPool = 'vkCreateDescriptorPool',
        destroyDescriptorPool = 'vkDestroyDescriptorPool',
        allocateDescriptorSets = 'vkAllocateDescriptorSets',
        updateDescriptorSets = 'vkUpdateDescriptorSets',
        createCommandPool = 'vkCreateCommandPool',
        destroyCommandPool = 'vkDestroyCommandPool',
        allocateCommandBuffers = 'vkAllocateCommandBuffers',
        beginCommandBuffer = 'vkBeginCommandBuffer',
        endCommandBuffer = 'vkEndCommandBuffer',
        freeCommandBuffers = 'vkFreeCommandBuffers',
        cmdBindPipeline = 'vkCmdBindPipeline',
        cmdBindDescriptorSets = 'vkCmdBindDescriptorSets',
        cmdPushConstants = 'vkCmdPushConstants',
        cmdDispatch = 'vkCmdDispatch',
        cmdCopyBuffer = 'vkCmdCopyBuffer',
        cmdPipelineBarrier = 'vkCmdPipelineBarrier',
        queueSubmit = 'vkQueueSubmit',
        waitForFences = 'vkWaitForFences',
        resetFences = 'vkResetFences',
        createFence = 'vkCreateFence',
        destroyFence = 'vkDestroyFence',
        createSemaphore = 'vkCreateSemaphore',
        destroySemaphore = 'vkDestroySemaphore',
        deviceWaitIdle = 'vkDeviceWaitIdle',
        destroyDevice = 'vkDestroyDevice',
    }
    local ok, binderr = true, nil
    for key, name in pairs(device_bindings) do
        local pfn = resolve(loader, device, 'device', name, 'PFN_' .. name)
        if pfn then fn[key] = pfn else ok, binderr = false, name end
    end
    if not ok then
        local d = fn.destroyDevice or resolve(loader, device, 'device', 'vkDestroyDevice', 'PFN_vkDestroyDevice')
        if d then d(device, nil) end
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'device entry point not resolvable: ' .. binderr))
    end

    -- Command pool (one for the whole worker lifecycle).
    local pool = ffi.new('VkCommandPool[1]')
    local cpci = ffi.new('VkCommandPoolCreateInfo')
    cpci.sType = VK.STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
    cpci.queueFamilyIndex = family
    res = fn.createCommandPool(device, cpci, nil, pool)
    if res ~= VK.SUCCESS then
        return fail(mkerr(_M.ERRORS.VULKAN_INITIALIZATION_FAILED, 'vkCreateCommandPool failed: ' .. res))
    end

    local queue = ffi.new('VkQueue[1]')
    fn.getDeviceQueue(device, family, 0, queue)

    local instance_fn = {
        destroyInstance = resolve(loader, instance, 'instance', 'vkDestroyInstance', 'PFN_vkDestroyInstance'),
        enumeratePhysicalDevices = fnEnumPhys,
        getPhysicalDeviceProperties = fnPhysProps,
        getPhysicalDeviceQueueFamilyProperties = fnQueueProps,
        getPhysicalDeviceMemoryProperties = fnMemProps,
        enumerateDeviceExtensionProperties = fnEnumExt,
    }

    local ctx = {
        loader = loader,
        ffi = ffi,
        os = OS,
        instance = instance,
        physical = physical,
        device = device,
        queue_family = family,
        queue = queue[0],
        command_pool = pool[0],
        fn = { instance = instance_fn, device = fn, global = { createInstance = fnCreateInstance } },
        props = {
            name = device_name,
            type = device_type,
            api_version = props.apiVersion,
            driver_version = props.driverVersion,
            vendor = props.vendorID,
            device = props.deviceID,
        },
        extensions = {
            sixteen_bit_storage = have_16bit,
            shader_float16_int8 = have_f16arith,
            shader_8bit_storage = have_8bit,
        },
    }
    _ctx = ctx

    -- Update live capabilities.
    capabilities.vulkan = true
    capabilities.loader_name = loader.loader_name
    capabilities.loader = loader.loader_name
    capabilities.api_version = instance_version ~= 0 and instance_version or props.apiVersion
    capabilities.device = {
        name = device_name,
        type = device_type,
        vendor = props.vendorID,
        device = props.deviceID,
        api_version = props.apiVersion,
    }
    capabilities.queue_family = family
    capabilities.extensions.sixteen_bit_storage = have_16bit
    capabilities.extensions.shader_float16_int8 = have_f16arith
    capabilities.extensions.shader_8bit_storage = have_8bit
    capabilities.caps.f16_storage = have_16bit
    capabilities.caps.f16_arithmetic = have_f16arith
    capabilities.caps.i8_storage = have_8bit
    capabilities.caps.word_addressed_payload = true

    return ctx, nil, false
end

-- destroy(ctx) — drains and tears down the whole device context.
function _M.destroy(ctx)
    ctx = ctx or _ctx
    if not ctx then return end
    local fn = ctx.fn and ctx.fn.device
    if fn then
        if fn.deviceWaitIdle then
            -- DEVICE_LOST during teardown: still release handles.
            pcall(fn.deviceWaitIdle, ctx.device)
        end
        if ctx.command_pool then
            if fn.destroyCommandPool then
                fn.destroyCommandPool(ctx.device, ctx.command_pool, nil)
            end
        end
        fn.destroyDevice(ctx.device, nil)
    end
    local ifn = ctx.fn and ctx.fn.instance
    if ifn and ifn.destroyInstance then
        ifn.destroyInstance(ctx.instance, nil)
    end
    if _ctx == ctx then _ctx = nil end
    capabilities.vulkan = false
    capabilities.device = nil
    return true
end

function _M.ctx() return _ctx end

-- ------------------------------------------------------------------ ABI ----
-- Golden LP64 sizeof/offsetof expectations, captured from vulkan_core.h
-- (Vulkan 1.4 header on the reference box; identical layout for the whole
-- 1.0-1.4 range for these structs). On 32-bit targets only non-zero sizes and
-- monotonic field order are asserted (pointer widths differ).
local ABI_GOLDEN = {
    VkApplicationInfo = { 48, {
        sType = 0, pNext = 8, pApplicationName = 16, applicationVersion = 24,
        pEngineName = 32, engineVersion = 40, apiVersion = 44,
    } },
    VkInstanceCreateInfo = { 64, {
        sType = 0, pNext = 8, flags = 16, pApplicationInfo = 24, enabledLayerCount = 32,
        ppEnabledLayerNames = 40, enabledExtensionCount = 48, ppEnabledExtensionNames = 56,
    } },
    VkPhysicalDeviceFeatures = { 216, { robustBufferAccess = 0, fullDrawIndexUint32 = 4,
        imageCubeArray = 8, independentBlend = 12, geometryShader = 16, tessellationShader = 20,
        sampleRateShading = 24, dualSrcBlend = 28, logicOp = 32, multiDrawIndirect = 36,
        drawIndirectFirstInstance = 40, depthClamp = 44, depthBiasClamp = 48, fillModeNonSolid = 52,
        depthBounds = 56, wideLines = 60, largePoints = 64, alphaToOne = 68, multiViewport = 72,
        samplerAnisotropy = 76, textureCompressionETC2 = 80, textureCompressionBC = 84,
        occlusionQueryPrecise = 88, pipelineStatisticsQuery = 92,
        vertexPipelineStoresAndAtomics = 96, fragmentStoresAndAtomics = 100,
        shaderTessellationAndGeometryPointSize = 104, shaderImageGatherExtended = 108,
        shaderStorageImageExtendedFormats = 112, shaderStorageImageMultisample = 116,
        shaderStorageImageReadWithoutFormat = 120, shaderStorageImageWriteWithoutFormat = 124,
        shaderUniformBufferArrayDynamicIndexing = 128, shaderSampledImageArrayDynamicIndexing = 132,
        shaderStorageBufferArrayDynamicIndexing = 136, shaderStorageImageArrayDynamicIndexing = 140,
        shaderClipDistance = 144, shaderCullDistance = 148, shaderFloat64 = 152, shaderInt64 = 156,
        shaderInt16 = 160, shaderResourceResidency = 164, shaderResourceMinLod = 168,
        sparseBinding = 172, sparseResidencyBuffer = 176, sparseResidencyImage2D = 180,
        sparseResidencyImage3D = 184, sparseResidency2Samples = 188, sparseResidency4Samples = 192,
        sparseResidency8Samples = 196, sparseResidency16Samples = 200, sparseResidencyAliased = 204,
        variableMultisampleRate = 208, inheritedQueries = 212,
    } },
    VkDeviceQueueCreateInfo = { 40, { sType = 0, pNext = 8, flags = 16, queueFamilyIndex = 20, queueCount = 24, pQueuePriorities = 32 } },
    VkDeviceCreateInfo = { 72, { sType = 0, pNext = 8, flags = 16, queueCreateInfoCount = 20, pQueueCreateInfos = 24, enabledLayerCount = 32, ppEnabledLayerNames = 40, enabledExtensionCount = 48, ppEnabledExtensionNames = 56, pEnabledFeatures = 64 } },
    VkBufferCreateInfo = { 56, { sType = 0, pNext = 8, flags = 16, size = 24, usage = 32, sharingMode = 36, queueFamilyIndexCount = 40, pQueueFamilyIndices = 48 } },
    VkMemoryAllocateInfo = { 32, { sType = 0, pNext = 8, allocationSize = 16, memoryTypeIndex = 24 } },
    VkShaderModuleCreateInfo = { 40, { sType = 0, pNext = 8, flags = 16, codeSize = 24, pCode = 32 } },
    VkDescriptorSetLayoutBinding = { 24, { binding = 0, descriptorType = 4, descriptorCount = 8, stageFlags = 12, pImmutableSamplers = 16 } },
    VkDescriptorSetLayoutCreateInfo = { 32, { sType = 0, pNext = 8, flags = 16, bindingCount = 20, pBindings = 24 } },
    VkPushConstantRange = { 12, { stageFlags = 0, offset = 4, size = 8 } },
    VkPipelineLayoutCreateInfo = { 48, { sType = 0, pNext = 8, flags = 16, setLayoutCount = 20, pSetLayouts = 24, pushConstantRangeCount = 32, pPushConstantRanges = 40 } },
    VkComputePipelineCreateInfo = { 96, { sType = 0, pNext = 8, flags = 16, stage = 24, layout = 72, basePipelineHandle = 80, basePipelineIndex = 88 } },
    VkDescriptorPoolSize = { 8, { type = 0, descriptorCount = 4 } },
    VkDescriptorPoolCreateInfo = { 40, { sType = 0, pNext = 8, flags = 16, maxSets = 20, poolSizeCount = 24, pPoolSizes = 32 } },
    VkDescriptorSetAllocateInfo = { 40, { sType = 0, pNext = 8, descriptorPool = 16, descriptorSetCount = 24, pSetLayouts = 32 } },
    VkDescriptorBufferInfo = { 24, { buffer = 0, offset = 8, range = 16 } },
    VkWriteDescriptorSet = { 64, { sType = 0, pNext = 8, dstSet = 16, dstBinding = 24, dstArrayElement = 28, descriptorCount = 32, descriptorType = 36, pImageInfo = 40, pBufferInfo = 48, pTexelBufferView = 56 } },
    VkCommandPoolCreateInfo = { 24, { sType = 0, pNext = 8, flags = 16, queueFamilyIndex = 20 } },
    VkCommandBufferAllocateInfo = { 32, { sType = 0, pNext = 8, commandPool = 16, level = 24, commandBufferCount = 28 } },
    VkCommandBufferBeginInfo = { 32, { sType = 0, pNext = 8, flags = 16, pInheritanceInfo = 24 } },
    VkSubmitInfo = { 72, { sType = 0, pNext = 8, waitSemaphoreCount = 16, pWaitSemaphores = 24, pWaitDstStageMask = 32, commandBufferCount = 40, pCommandBuffers = 48, signalSemaphoreCount = 56, pSignalSemaphores = 64 } },
    VkFenceCreateInfo = { 24, { sType = 0, pNext = 8, flags = 16 } },
    VkBufferMemoryBarrier = { 56, { sType = 0, pNext = 8, srcAccessMask = 16, dstAccessMask = 20, srcQueueFamilyIndex = 24, dstQueueFamilyIndex = 28, buffer = 32, offset = 40, size = 48 } },
    VkPipelineShaderStageCreateInfo = { 48, { sType = 0, pNext = 8, flags = 16, stage = 20, module = 24, pName = 32, pSpecializationInfo = 40 } },
    VkSpecializationMapEntry = { 16, { constantID = 0, offset = 4, size = 8 } },
    VkSpecializationInfo = { 32, { mapEntryCount = 0, pMapEntries = 8, dataSize = 16, pData = 24 } },
}
_M.ABI_GOLDEN = ABI_GOLDEN

-- abi_check() -> true | nil, err(ABI_MISMATCH). Pure FFI struct definitions:
-- valid and assertable without a loader. On LP64 compares sizeof/offsetof to
-- the golden table; on 32-bit asserts non-zero sizes and monotonic field order.
function _M.abi_check()
    if not ffi then
        return nil, mkerr(_M.ERRORS.ABI_MISMATCH, 'ffi unavailable; cannot validate ABI')
    end
    define_types()
    local is64 = ffi.abi('64bit')
    for name, golden in pairs(ABI_GOLDEN) do
        local t = ffi.typeof(name)
        local sz = ffi.sizeof(t)
        if sz == 0 then
            return nil, mkerr(_M.ERRORS.ABI_MISMATCH, ('%s has zero size'):format(name))
        end
        if is64 then
            local expected_sz, fields = golden[1], golden[2]
            if sz ~= expected_sz then
                return nil, mkerr(_M.ERRORS.ABI_MISMATCH,
                    ('%s sizeof = %d, expected %d'):format(name, sz, expected_sz))
            end
            -- Compare every declared field against the golden table (when listed).
            local ctype = ffi.typeof(name)
            for field, expected in pairs(fields) do
                local got = ffi.offsetof(ctype, field)
                if got ~= expected then
                    return nil, mkerr(_M.ERRORS.ABI_MISMATCH,
                        ('%s.%s offsetof = %d, expected %d'):format(name, field, got, expected))
                end
            end
        else
            -- 32-bit: verify field offsets are monotonic (declaration order) and sane.
            local ctype = ffi.typeof(name)
            local _, fields = golden[1], golden[2]
            local prev = -1
            local ordered = {}
            for field in pairs(fields) do ordered[#ordered + 1] = field end
            table.sort(ordered, function(a, b) return fields[a] < fields[b] end)
            for _, field in ipairs(ordered) do
                local got = ffi.offsetof(ctype, field)
                if got < prev then
                    return nil, mkerr(_M.ERRORS.ABI_MISMATCH,
                        ('%s.%s offsetof = %d regressed vs %d'):format(name, field, got, prev))
                end
                prev = got
            end
        end
    end
    return true
end
return _M
