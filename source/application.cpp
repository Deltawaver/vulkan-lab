#include "application.hpp"

#include <array>
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

constexpr float pi = std::numbers::pi_v<float>;

// --- Параметры цилиндра -----------------------------------------------------
constexpr uint32_t cylinder_segments = 50;  // вершин в каждом основании
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

constexpr float z_near = 0.1f;
constexpr float z_far = 100.0f;

// --- Объекты сцены ----------------------------------------------------------
constexpr uint32_t object_count = 3;

// Всё, что уходит в uniform-буфер объекта (раскладка std140: mat4, затем vec4).
struct ObjectUniforms {
	float mvp[16];
	float color[4];
};

struct Object {
	const char* name;

	// Базовое преобразование, которое задаётся из интерфейса.
	float position[3] = {0.0f, 0.0f, 0.0f};
	float rotation[3] = {0.0f, 0.0f, 0.0f};  // углы вокруг X, Y, Z в радианах
	float scale[3] = {1.0f, 1.0f, 1.0f};

	float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

	// Анимация: волнистая окружность в плоскости XZ + кувыркание.
	bool animate = true;
	float radius = 0.0f;      // радиус окружности
	float height = 0.0f;      // амплитуда колебаний по Y
	float waves = 3.0f;       // сколько волн на один оборот
	float phase = 0.0f;       // сдвиг по траектории
	float spin_speed = 1.0f;  // скорость кувыркания вокруг своих осей

	// Ресурсы GPU: свой буфер и свой набор дескрипторов на каждый объект.
	VkBuffer uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation uniform_allocation = VK_NULL_HANDLE;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

std::array<Object, object_count> objects = {{
	Object{
		.name = "Center",
		.spin_speed = 0.8f,
	},
	Object{
		.name = "Satellite A",
		.scale = {0.4f, 0.4f, 0.4f},
		.radius = 3.0f,
		.height = 0.8f,
		.waves = 3.0f,
		.phase = 0.0f,
		.spin_speed = 2.0f,
	},
	Object{
		.name = "Satellite B",
		.scale = {0.4f, 0.4f, 0.4f},
		.color = {1.0f, 0.8f, 0.5f, 1.0f},
		.radius = 2.0f,
		.height = 0.5f,
		.waves = 5.0f,
		.phase = pi,
		.spin_speed = -2.5f,
	},
}};

// --- Камера и проекция ------------------------------------------------------
enum class Projection { Perspective, Orthographic };

Projection projection_mode = Projection::Perspective;
float fov_y = 45.0f * pi / 180.0f;
float ortho_size = 4.0f;       // половина высоты видимой области
float camera_distance = 9.0f;
float camera_pitch = 0.45f;    // наклон камеры вниз, радианы

// --- Анимация ---------------------------------------------------------------
bool playing = true;
float animation_speed = 1.0f;
double animation_time = 0.0;
double previous_time = -1.0;

// --- Ресурсы Vulkan ---------------------------------------------------------
VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;
VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;
uint32_t index_count = 0;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	// Память, доступная CPU: для маленьких буферов staging не нужен.
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

// Макет набора дескрипторов (binding 0 - uniform-буфер), пул и по одному набору
// дескрипторов с собственным uniform-буфером на каждый объект.
bool createDescriptors() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = object_count,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = object_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	std::array<VkDescriptorSetLayout, object_count> layouts;
	layouts.fill(descriptor_set_layout);

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = layouts.data(),
	};

	std::array<VkDescriptorSet, object_count> sets;

	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets\n";
		return false;
	}

	std::array<VkDescriptorBufferInfo, object_count> buffer_infos;
	std::array<VkWriteDescriptorSet, object_count> writes;

	for (uint32_t i = 0; i < object_count; ++i) {
		Object& object = objects[i];
		object.descriptor_set = sets[i];

		const ObjectUniforms initial = {};
		if (!createBuffer(&initial, sizeof(initial), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  object.uniform_buffer, object.uniform_allocation)) {
			return false;
		}

		buffer_infos[i] = {
			.buffer = object.uniform_buffer,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};

		writes[i] = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = object.descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[i],
		};
	}

	vkUpdateDescriptorSets(context.device, object_count, writes.data(), 0, nullptr);

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
			.offset = offsetof(cylinder::Vertex, color),
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

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
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

// Матрица модели: T(позиция + смещение по траектории) * кувыркание * R(z, y, x) * S.
math::Mat4 modelMatrix(const Object& o) {
	float offset[3] = {0.0f, 0.0f, 0.0f};
	float spin = 0.0f;

	if (o.animate) {
		// Траектория - окружность в плоскости XZ, по которой бежит синусоида по Y.
		const float t = float(animation_time) + o.phase;
		offset[0] = o.radius * std::cos(t);
		offset[1] = o.height * std::sin(o.waves * t);
		offset[2] = o.radius * std::sin(t);

		spin = o.spin_speed * float(animation_time);
	}

	const math::Mat4 translation = math::translate(o.position[0] + offset[0],
	                                               o.position[1] + offset[1],
	                                               o.position[2] + offset[2]);
	const math::Mat4 tumble = math::rotateY(spin) * math::rotateX(spin * 0.5f);
	const math::Mat4 rotation = math::rotateZ(o.rotation[2]) *
	                            math::rotateY(o.rotation[1]) *
	                            math::rotateX(o.rotation[0]);
	const math::Mat4 scaling = math::scale(o.scale[0], o.scale[1], o.scale[2]);

	return translation * tumble * rotation * scaling;
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

	return createDescriptors() && createPipeline();
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	// Наборы дескрипторов освобождаются вместе с пулом.
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);

	for (Object& object : objects) {
		vmaDestroyBuffer(context.allocator, object.uniform_buffer, object.uniform_allocation);
	}

	vmaDestroyBuffer(context.allocator, index_buffer, index_allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
}

void update(double time) {
	float dt = (previous_time < 0.0) ? 0.0f : float(time - previous_time);
	previous_time = time;

	// При перетаскивании окна кадры могут надолго зависать: не даём анимации "прыгать".
	if (dt > 0.1f) {
		dt = 0.1f;
	}

	if (playing) {
		animation_time += double(dt) * double(animation_speed);
	}

	ImGui::SetNextWindowSize(ImVec2(380.0f, 640.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Scene");

	if (ImGui::CollapsingHeader("Camera and projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::RadioButton("Perspective", projection_mode == Projection::Perspective)) {
			projection_mode = Projection::Perspective;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Orthographic", projection_mode == Projection::Orthographic)) {
			projection_mode = Projection::Orthographic;
		}

		if (projection_mode == Projection::Perspective) {
			ImGui::SliderAngle("Field of view", &fov_y, 20.0f, 120.0f);
		} else {
			ImGui::SliderFloat("Ortho size", &ortho_size, 0.5f, 12.0f);
		}

		ImGui::SliderFloat("Camera distance", &camera_distance, 2.0f, 40.0f);
		ImGui::SliderAngle("Camera pitch", &camera_pitch, -89.0f, 89.0f);
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(playing ? "Pause" : "Play")) {
			playing = !playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			animation_time = 0.0;
		}
		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
	}

	for (uint32_t i = 0; i < object_count; ++i) {
		Object& o = objects[i];

		ImGui::PushID(int(i));

		if (ImGui::CollapsingHeader(o.name, i == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
			ImGui::DragFloat3("Position", o.position, 0.05f);
			ImGui::SliderAngle("Rotation X", &o.rotation[0], -180.0f, 180.0f);
			ImGui::SliderAngle("Rotation Y", &o.rotation[1], -180.0f, 180.0f);
			ImGui::SliderAngle("Rotation Z", &o.rotation[2], -180.0f, 180.0f);
			ImGui::DragFloat3("Scale", o.scale, 0.02f, 0.05f, 10.0f);

			ImGui::ColorEdit3("Color", o.color);

			ImGui::Checkbox("Animate", &o.animate);
			ImGui::SliderFloat("Radius", &o.radius, 0.0f, 8.0f);
			ImGui::SliderFloat("Wave height", &o.height, 0.0f, 3.0f);
			ImGui::SliderFloat("Waves per turn", &o.waves, 0.0f, 12.0f);
			ImGui::SliderAngle("Phase", &o.phase, 0.0f, 360.0f);
			ImGui::SliderFloat("Spin speed", &o.spin_speed, -6.0f, 6.0f);
		}

		ImGui::PopID();
	}

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

	math::Mat4 projection;
	if (projection_mode == Projection::Perspective) {
		projection = math::perspective(fov_y, aspect, z_near, z_far);
	} else {
		const float half_height = ortho_size;
		const float half_width = ortho_size * aspect;
		projection = math::orthographic(-half_width, half_width, -half_height, half_height,
		                                z_near, z_far);
	}

	const math::Mat4 view = math::translate(0.0f, 0.0f, -camera_distance) *
	                        math::rotateX(camera_pitch);

	// Обновляем uniform-буфер каждого объекта. Здесь это безопасно: prepare() уже
	// дождался окончания предыдущего кадра на GPU (fence), буферы никем не читаются.
	for (Object& object : objects) {
		const math::Mat4 mvp = projection * view * modelMatrix(object);

		ObjectUniforms uniforms = {};
		for (int i = 0; i < 16; ++i) {
			uniforms.mvp[i] = mvp.m[i];
		}
		for (int i = 0; i < 4; ++i) {
			uniforms.color[i] = object.color[i];
		}

		vmaCopyMemoryToAllocation(context.allocator, &uniforms, object.uniform_allocation,
		                          0, sizeof(uniforms));
	}

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

	// Меш у всех объектов общий, различается только набор дескрипторов.
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);

	for (const Object& object : objects) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &object.descriptor_set, 0, nullptr);
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

} // namespace application