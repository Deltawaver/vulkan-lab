#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>

#include <imgui.h>

#include "cylinder.hpp"
#include "mat4.hpp"

namespace application {

using graphics::internal::context;

namespace {

// --- Параметры цилиндра -----------------------------------------------------
constexpr uint32_t cylinder_segments = 50;  // вершин в каждом основании
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

// --- Параметры камеры -------------------------------------------------------
constexpr float camera_distance = 5.0f;
constexpr float fov_y = 45.0f * std::numbers::pi_v<float> / 180.0f;
constexpr float z_near = 0.1f;
constexpr float z_far = 100.0f;

// --- Состояние приложения ---------------------------------------------------
bool auto_rotate = true;
float angle_x = 0.4f;
float angle_y = 0.0f;
double previous_time = -1.0;

// --- Ресурсы Vulkan ---------------------------------------------------------
VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;
uint32_t index_count = 0;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

// Данные, которые уходят в вершинный шейдер через push constants.
struct PushConstants {
	float mvp[16];
};

bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	// Память, доступная CPU: для маленькой статичной модели staging-буфер не нужен.
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &buffer, &allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	if (vmaCopyMemoryToAllocation(context.allocator, data, allocation, 0, size) != VK_SUCCESS) {
		std::cerr << "Failed to upload data to Vulkan buffer\n";
		return false;
	}

	return true;
}

bool loadShader(const char* path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader '" << path << "'. "
		          << "Is the working directory the project root? "
		          << "Is compile_shader(...) added in CMakeLists.txt?\n";
		return false;
	}

	const size_t size = size_t(file.tellg());
	std::vector<uint32_t> code((size + sizeof(uint32_t) - 1) / sizeof(uint32_t));

	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from '" << path << "'\n";
		return false;
	}

	return true;
}

bool createPipeline() {
	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;

	if (!loadShader("shaders/cylinder.vert.spv", vert) ||
	    !loadShader("shaders/cylinder.frag.spv", frag)) {
		if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vert, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(cylinder::Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(cylinder::Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(cylinder::Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	// Размер окна меняется, поэтому viewport и scissor задаём при записи команд.
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Отсечение граней выключено, чтобы не зависеть от порядка обхода вершин.
	// Корректность всё равно обеспечивает буфер глубины.
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};

	bool ok = true;

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		ok = false;
	}

	if (ok) {
		const VkGraphicsPipelineCreateInfo pipeline_info = {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = sizeof(stages) / sizeof(stages[0]),
			.pStages = stages,
			.pVertexInputState = &vertex_input,
			.pInputAssemblyState = &input_assembly,
			.pViewportState = &viewport,
			.pRasterizationState = &rasterization,
			.pMultisampleState = &multisample,
			.pDepthStencilState = &depth_stencil,
			.pColorBlendState = &color_blend,
			.pDynamicState = &dynamic,
			.layout = pipeline_layout,
			.renderPass = context.render_pass,
			.subpass = 0,
		};

		if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
		                              nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan graphics pipeline\n";
			ok = false;
		}
	}

	// Шейдерные модули нужны только на время создания пайплайна.
	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	return ok;
}

} // namespace

bool initialize() {
	const cylinder::Mesh mesh =
		cylinder::make(cylinder_segments, cylinder_radius, cylinder_height);

	index_count = uint32_t(mesh.indices.size());

	if (!createBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(cylinder::Vertex),
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer, vertex_allocation)) {
		return false;
	}

	if (!createBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
	                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer, index_allocation)) {
		return false;
	}

	return createPipeline();
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
}

void update(double time) {
	const float dt = (previous_time < 0.0) ? 0.0f : float(time - previous_time);
	previous_time = time;

	constexpr float two_pi = 2.0f * std::numbers::pi_v<float>;

	if (auto_rotate) {
		angle_y = std::fmod(angle_y + 0.8f * dt, two_pi);
		angle_x = std::fmod(angle_x + 0.3f * dt, two_pi);
	}

	ImGui::Begin("Cylinder");
	ImGui::Checkbox("Auto rotate", &auto_rotate);
	ImGui::SliderAngle("Rotation X", &angle_x, -360.0f, 360.0f);
	ImGui::SliderAngle("Rotation Y", &angle_y, -360.0f, 360.0f);
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	// prepare() вернул пустой кадр (например, не удалось получить изображение).
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkCommandBuffer cmd = fd.command_buffer;

	// --- Матрицы: clip = Projection * View * Model * position ---
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	const math::Mat4 model = math::rotateY(angle_y) * math::rotateX(angle_x);
	const math::Mat4 view = math::translate(0.0f, 0.0f, -camera_distance);
	const math::Mat4 projection = math::perspective(fov_y, aspect, z_near, z_far);

	const math::Mat4 mvp = projection * view * model;

	// --- Запись команд ---
	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2] = {};
	clear_values[0].color = {{0.05f, 0.05f, 0.08f, 1.0f}};
	clear_values[1].depthStencil = {1.0f, 0};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(cmd, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	PushConstants push = {};
	for (int i = 0; i < 16; ++i) {
		push.mvp[i] = mvp.m[i];
	}
	vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
	                   sizeof(push), &push);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

} // namespace application