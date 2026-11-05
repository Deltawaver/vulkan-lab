#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

constexpr uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 1.0f;
constexpr float cylinder_height = 2.0f;

constexpr float z_near = 0.1f;
constexpr float z_far = 100.0f;

constexpr uint32_t max_lights = 32;           // емкость каждого storage-буфера
constexpr uint32_t object_count = 6;
constexpr uint32_t max_draws = object_count + 3 * max_lights;  // объекты + маркеры + конусы

constexpr float cone_length = 2.5f;           // длина каркаса конуса прожектора
constexpr float marker_size = 0.2f;

// --- Раскладки данных для GPU (std140 / std430) -----------------------------
struct SceneUniforms {
	float view_proj[16];
	float camera_pos[4];
	float ambient[4];
	float dir_direction[4];
	float dir_color[4];
	int32_t counts[4];
};

struct PointLightGpu {
	float position[4];
	float color[4];
};

struct SpotLightGpu {
	float position[4];
	float direction[4];
	float color[4];
	float cone[4];
};

struct ModelUniforms {
	float model[16];
	float albedo[4];
	float specular[4];
	float flags[4];
};

// --- Параметры сцены (меняются из интерфейса) -------------------------------
struct Object {
	const char* name;

	float position[3] = {0.0f, 0.0f, 0.0f};
	float rotation[3] = {0.0f, 0.0f, 0.0f};
	float scale[3] = {1.0f, 1.0f, 1.0f};

	// Материал Блинн-Фонга.
	float albedo[3] = {0.8f, 0.8f, 0.8f};
	float specular[3] = {0.5f, 0.5f, 0.5f};
	float shininess = 64.0f;

	bool animate = true;
	float radius = 0.0f;
	float height = 0.0f;
	float waves = 3.0f;
	float phase = 0.0f;
	float spin_speed = 1.0f;
};

std::array<Object, object_count> objects = {{
	Object{
		.name = "Center",
		.albedo = {0.8f, 0.8f, 0.85f},
		.spin_speed = 0.8f,
	},
	Object{
		.name = "Satellite A",
		.scale = {0.4f, 0.4f, 0.4f},
		.albedo = {0.9f, 0.3f, 0.3f},
		.radius = 3.0f,
		.height = 0.8f,
		.waves = 3.0f,
		.phase = 0.0f,
		.spin_speed = 2.0f,
	},
	Object{
		.name = "Satellite B",
		.scale = {0.4f, 0.4f, 0.4f},
		.albedo = {0.3f, 0.5f, 0.9f},
		.radius = 2.0f,
		.height = 0.5f,
		.waves = 5.0f,
		.phase = pi,
		.spin_speed = -2.5f,
	},
	Object{
		.name = "Floor",
		.position = {0.0f, -1.6f, 0.0f},
		.scale = {8.0f, 0.1f, 8.0f},
		.albedo = {0.5f, 0.5f, 0.5f},
		.specular = {0.2f, 0.2f, 0.2f},
		.shininess = 16.0f,
		.animate = false,
	},
	// Две колонны для красивых теней... точнее бликов и пятен света.
	Object{
		.name = "Pillar Left",
		.position = {-5.0f, -0.4f, -3.0f},
		.scale = {0.6f, 1.2f, 0.6f},
		.albedo = {0.85f, 0.8f, 0.7f},
		.animate = false,
	},
	Object{
		.name = "Pillar Right",
		.position = {5.0f, -0.4f, -3.0f},
		.scale = {0.6f, 1.2f, 0.6f},
		.albedo = {0.85f, 0.8f, 0.7f},
		.animate = false,
	},
}};

struct DirectionalLight {
	float direction[3] = {-0.3f, -1.0f, -0.4f};
	float color[3] = {1.0f, 1.0f, 1.0f};
	float intensity = 0.5f;
} directional;

struct PointLight {
	float position[3] = {0.0f, 1.5f, 0.0f};
	float color[3] = {1.0f, 0.9f, 0.7f};
	float intensity = 12.0f;
	int attached = -1;  // индекс объекта, к которому прикреплён источник (-1 - нет)
};

struct SpotLight {
	float position[3] = {4.0f, 4.0f, 0.0f};
	float direction[3] = {-1.0f, -1.0f, 0.0f};
	float color[3] = {0.6f, 0.8f, 1.0f};
	float intensity = 40.0f;
	float inner_angle = 15.0f * pi / 180.0f;
	float outer_angle = 28.0f * pi / 180.0f;
	float sweep_speed = 0.0f;  // вращение луча вокруг оси Y (рад/с), "прожектор-маяк"
};

// Два огонька летают вместе со спутниками; два прожектора слева и справа качают лучом.
std::vector<PointLight> point_lights = {
	PointLight{.position = {0.0f, 0.6f, 0.0f}, .color = {1.0f, 0.4f, 0.3f}, .intensity = 10.0f, .attached = 1},
	PointLight{.position = {0.0f, 0.6f, 0.0f}, .color = {0.3f, 0.5f, 1.0f}, .intensity = 8.0f, .attached = 2},
};
std::vector<SpotLight> spot_lights = {
	SpotLight{.sweep_speed = 0.8f},
	SpotLight{.position = {-4.0f, 4.0f, 0.0f}, .direction = {1.0f, -1.0f, 0.0f},
	          .color = {1.0f, 0.6f, 0.9f}, .sweep_speed = -0.6f},
};

float ambient_color[3] = {0.08f, 0.08f, 0.1f};
bool show_light_markers = true;

// --- Камера -----------------------------------------------------------------
enum class Projection { Perspective, Orthographic };

Projection projection_mode = Projection::Perspective;
float fov_y = 45.0f * pi / 180.0f;
float ortho_size = 4.0f;
float camera_position[3] = {0.0f, 3.5f, 9.0f};
float camera_yaw = 0.0f;     // поворот вокруг Y (положительный - влево)
float camera_pitch = -0.3f;  // положительный - вверх

// --- Анимация ---------------------------------------------------------------
bool playing = true;
float animation_speed = 1.0f;
double animation_time = 0.0;
double previous_time = -1.0;

// --- Ресурсы Vulkan ---------------------------------------------------------
struct GpuMesh {
	VkBuffer vertex_buffer = VK_NULL_HANDLE;
	VmaAllocation vertex_allocation = VK_NULL_HANDLE;
	VkBuffer index_buffer = VK_NULL_HANDLE;
	VmaAllocation index_allocation = VK_NULL_HANDLE;
	uint32_t index_count = 0;
};

GpuMesh mesh_cylinder, mesh_cube, mesh_cone;

struct GpuBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
};

GpuBuffer scene_buffer, point_buffer, spot_buffer, model_buffer;
VkDeviceSize model_stride = sizeof(ModelUniforms);

VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline_triangles = VK_NULL_HANDLE;
VkPipeline pipeline_lines = VK_NULL_HANDLE;

bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, GpuBuffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	if (vmaCopyMemoryToAllocation(context.allocator, data, out.allocation, 0, size) != VK_SUCCESS) {
		std::cerr << "Failed to upload data to Vulkan buffer\n";
		return false;
	}

	return true;
}

bool createZeroBuffer(VkDeviceSize size, VkBufferUsageFlags usage, GpuBuffer& out) {
	const std::vector<uint8_t> zeros(size);
	return createBuffer(zeros.data(), size, usage, out);
}

bool createMesh(const cylinder::Mesh& mesh, GpuMesh& out) {
	out.index_count = uint32_t(mesh.indices.size());

	GpuBuffer v, i;
	if (!createBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(cylinder::Vertex),
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, v) ||
	    !createBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
	                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, i)) {
		return false;
	}

	out.vertex_buffer = v.buffer;
	out.vertex_allocation = v.allocation;
	out.index_buffer = i.buffer;
	out.index_allocation = i.allocation;
	return true;
}

void destroyMesh(GpuMesh& m) {
	if (m.vertex_buffer) vmaDestroyBuffer(context.allocator, m.vertex_buffer, m.vertex_allocation);
	if (m.index_buffer) vmaDestroyBuffer(context.allocator, m.index_buffer, m.index_allocation);
}

void destroyBuffer(GpuBuffer& b) {
	if (b.buffer) vmaDestroyBuffer(context.allocator, b.buffer, b.allocation);
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

// Набор 0 - сцена (uniform + 2 storage-буфера), набор 1 - модель (динамический uniform).
bool createDescriptors() {
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &scene_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &model_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor set layouts\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 3,
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = {scene_layout, model_layout};
	VkDescriptorSet sets[2];

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 2,
		.pSetLayouts = layouts,
	};

	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan descriptor sets\n";
		return false;
	}

	scene_set = sets[0];
	model_set = sets[1];

	// Буферы. Выравнивание динамических смещений зависит от устройства.
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(context.physical_device, &props);
	const VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
	model_stride = (sizeof(ModelUniforms) + align - 1) / align * align;

	if (!createZeroBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) ||
	    !createZeroBuffer(max_lights * sizeof(PointLightGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_buffer) ||
	    !createZeroBuffer(max_lights * sizeof(SpotLightGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_buffer) ||
	    !createZeroBuffer(model_stride * max_draws, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer)) {
		return false;
	}

	const VkDescriptorBufferInfo infos[] = {
		{ scene_buffer.buffer, 0, sizeof(SceneUniforms) },
		{ point_buffer.buffer, 0, max_lights * sizeof(PointLightGpu) },
		{ spot_buffer.buffer, 0, max_lights * sizeof(SpotLightGpu) },
		{ model_buffer.buffer, 0, sizeof(ModelUniforms) },
	};

	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set, .dstBinding = 0, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &infos[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set, .dstBinding = 1, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set, .dstBinding = 2, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[2],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = model_set, .dstBinding = 0, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &infos[3],
		},
	};

	vkUpdateDescriptorSets(context.device, 4, writes, 0, nullptr);

	return true;
}

bool createPipeline(VkShaderModule vert, VkShaderModule frag, VkPrimitiveTopology topology,
                    VkPipeline& out) {
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
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(cylinder::Vertex, position) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(cylinder::Vertex, normal) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

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

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
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
	                              nullptr, &out) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}
	return true;
}

// Один и тот же шейдер для треугольников и для линий (каркас конуса рисуется
// с флагом "без освещения", поэтому отдельные шейдеры не нужны).
bool createPipelines() {
	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;

	if (!loadShader("shaders/cylinder.vert.spv", vert) ||
	    !loadShader("shaders/cylinder.frag.spv", frag)) {
		if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vert, nullptr);
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = {scene_layout, model_layout};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};

	bool ok = vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) == VK_SUCCESS;
	if (!ok) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
	}

	ok = ok && createPipeline(vert, frag, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, pipeline_triangles) &&
	     createPipeline(vert, frag, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, pipeline_lines);

	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	return ok;
}

math::Mat4 modelMatrix(const Object& o) {
	float offset[3] = {0.0f, 0.0f, 0.0f};
	float spin = 0.0f;

	if (o.animate) {
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

// Положение точечного источника: смещение + (если прикреплён) положение объекта.
void pointLightPosition(const PointLight& l, float out[3]) {
	for (int k = 0; k < 3; ++k) out[k] = l.position[k];
	if (l.attached >= 0 && l.attached < int(object_count)) {
		const math::Mat4 m = modelMatrix(objects[l.attached]);
		for (int k = 0; k < 3; ++k) out[k] += m.at(k, 3);
	}
}

// Направление прожектора с учётом качания луча вокруг оси Y.
void spotLightDirection(const SpotLight& l, float out[3]) {
	const float a = l.sweep_speed * float(animation_time);
	const float c = std::cos(a), s = std::sin(a);
	out[0] = c * l.direction[0] + s * l.direction[2];
	out[1] = l.direction[1];
	out[2] = -s * l.direction[0] + c * l.direction[2];
}

void normalize3(const float in[3], float out[3]) {
	const float len = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
	if (len < 1e-6f) {
		out[0] = 0.0f; out[1] = -1.0f; out[2] = 0.0f;
		return;
	}
	for (int i = 0; i < 3; ++i) out[i] = in[i] / len;
}

// Матрица, переводящая локальную ось -Z в направление dir (для каркаса конуса).
math::Mat4 orientAlongMinusZ(const float dir[3]) {
	float d[3];
	normalize3(dir, d);
	const float z[3] = {-d[0], -d[1], -d[2]};
	const float up[3] = {std::fabs(z[1]) > 0.99f ? 1.0f : 0.0f, std::fabs(z[1]) > 0.99f ? 0.0f : 1.0f, 0.0f};

	float x[3] = {up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]};
	normalize3(x, x);
	const float y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]};

	math::Mat4 r = math::identity();
	for (int i = 0; i < 3; ++i) {
		r.at(i, 0) = x[i];
		r.at(i, 1) = y[i];
		r.at(i, 2) = z[i];
	}
	return r;
}

void cameraVectors(float forward[3], float right[3]) {
	const float cy = std::cos(camera_yaw), sy = std::sin(camera_yaw);
	const float cp = std::cos(camera_pitch), sp = std::sin(camera_pitch);
	forward[0] = -sy * cp; forward[1] = sp; forward[2] = -cy * cp;
	right[0] = cy; right[1] = 0.0f; right[2] = -sy;
}

} // namespace

bool initialize() {
	return createMesh(cylinder::make(cylinder_segments, cylinder_radius, cylinder_height), mesh_cylinder) &&
	       createMesh(cylinder::makeCube(), mesh_cube) &&
	       createMesh(cylinder::makeConeLines(), mesh_cone) &&
	       createDescriptors() && createPipelines();
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline_triangles, nullptr);
	vkDestroyPipeline(context.device, pipeline_lines, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr);

	destroyBuffer(scene_buffer);
	destroyBuffer(point_buffer);
	destroyBuffer(spot_buffer);
	destroyBuffer(model_buffer);

	destroyMesh(mesh_cylinder);
	destroyMesh(mesh_cube);
	destroyMesh(mesh_cone);
}

void update(double time) {
	float dt = (previous_time < 0.0) ? 0.0f : float(time - previous_time);
	previous_time = time;
	if (dt > 0.1f) dt = 0.1f;

	if (playing) {
		animation_time += double(dt) * double(animation_speed);
	}

	// Управление камерой: ПКМ + движение мыши - поворот, WASD - перемещение, Q/E - вниз/вверх.
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera_yaw -= io.MouseDelta.x * 0.005f;
		camera_pitch = std::clamp(camera_pitch - io.MouseDelta.y * 0.005f, -1.55f, 1.55f);
	}
	if (!io.WantCaptureKeyboard) {
		float forward[3], right[3];
		cameraVectors(forward, right);
		const float speed = (io.KeyShift ? 15.0f : 5.0f) * dt;
		const float move_f = float(ImGui::IsKeyDown(ImGuiKey_W)) - float(ImGui::IsKeyDown(ImGuiKey_S));
		const float move_r = float(ImGui::IsKeyDown(ImGuiKey_D)) - float(ImGui::IsKeyDown(ImGuiKey_A));
		const float move_u = float(ImGui::IsKeyDown(ImGuiKey_E)) - float(ImGui::IsKeyDown(ImGuiKey_Q));
		for (int i = 0; i < 3; ++i) {
			camera_position[i] += (forward[i] * move_f + right[i] * move_r) * speed;
		}
		camera_position[1] += move_u * speed;
	}

	ImGui::SetNextWindowSize(ImVec2(400.0f, 700.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Scene");

	if (ImGui::CollapsingHeader("Camera and projection", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextDisabled("RMB drag: look, WASD/QE: move, Shift: fast");
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

		ImGui::DragFloat3("Camera position", camera_position, 0.05f);
		ImGui::SliderAngle("Camera yaw", &camera_yaw, -180.0f, 180.0f);
		ImGui::SliderAngle("Camera pitch", &camera_pitch, -89.0f, 89.0f);
	}

	if (ImGui::CollapsingHeader("Animation")) {
		if (ImGui::Button(playing ? "Pause" : "Play")) playing = !playing;
		ImGui::SameLine();
		if (ImGui::Button("Restart")) animation_time = 0.0;
		ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient", ambient_color);
		ImGui::Checkbox("Show light markers", &show_light_markers);

		if (ImGui::TreeNodeEx("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Direction", directional.direction, 0.01f);
			ImGui::ColorEdit3("Color", directional.color);
			ImGui::SliderFloat("Intensity", &directional.intensity, 0.0f, 5.0f);
			ImGui::TreePop();
		}

		if (ImGui::TreeNodeEx("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
			int remove = -1;
			for (int i = 0; i < int(point_lights.size()); ++i) {
				PointLight& l = point_lights[i];
				ImGui::PushID(i);
				if (ImGui::TreeNode("Point", "Point %d", i)) {
					ImGui::DragFloat3("Position", l.position, 0.05f);
					ImGui::ColorEdit3("Color", l.color);
					ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 100.0f);
					ImGui::SliderInt("Attach to object (-1 = none)", &l.attached, -1, int(object_count) - 1);
					if (ImGui::Button("Remove")) remove = i;
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			if (remove >= 0) point_lights.erase(point_lights.begin() + remove);
			if (ImGui::Button("Add point light") && point_lights.size() < max_lights) {
				point_lights.push_back(PointLight{});
			}
			ImGui::TreePop();
		}

		if (ImGui::TreeNodeEx("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
			int remove = -1;
			for (int i = 0; i < int(spot_lights.size()); ++i) {
				SpotLight& l = spot_lights[i];
				ImGui::PushID(i);
				if (ImGui::TreeNode("Spot", "Spot %d", i)) {
					ImGui::DragFloat3("Position", l.position, 0.05f);
					ImGui::DragFloat3("Direction", l.direction, 0.01f);
					ImGui::ColorEdit3("Color", l.color);
					ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 300.0f);
					ImGui::SliderAngle("Inner angle", &l.inner_angle, 0.0f, 89.0f);
					ImGui::SliderAngle("Outer angle", &l.outer_angle, 0.0f, 89.0f);
					ImGui::SliderFloat("Sweep speed", &l.sweep_speed, -3.0f, 3.0f);
					if (ImGui::Button("Remove")) remove = i;
					ImGui::TreePop();
				}
				ImGui::PopID();
			}
			if (remove >= 0) spot_lights.erase(spot_lights.begin() + remove);
			if (ImGui::Button("Add spot light") && spot_lights.size() < max_lights) {
				spot_lights.push_back(SpotLight{});
			}
			ImGui::TreePop();
		}
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

			ImGui::ColorEdit3("Albedo", o.albedo);
			ImGui::ColorEdit3("Specular color", o.specular);
			ImGui::SliderFloat("Shininess", &o.shininess, 1.0f, 256.0f);

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
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkCommandBuffer cmd = fd.command_buffer;

	// --- Матрицы ---
	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	math::Mat4 projection;
	if (projection_mode == Projection::Perspective) {
		projection = math::perspective(fov_y, aspect, z_near, z_far);
	} else {
		projection = math::orthographic(-ortho_size * aspect, ortho_size * aspect,
		                                -ortho_size, ortho_size, z_near, z_far);
	}

	// Мировая матрица камеры = T(pos) * Ry(yaw) * Rx(pitch); вид - обратная к ней.
	const math::Mat4 view = math::rotateX(-camera_pitch) * math::rotateY(-camera_yaw) *
	                        math::translate(-camera_position[0], -camera_position[1], -camera_position[2]);
	const math::Mat4 view_proj = projection * view;

	// --- Данные сцены ---
	const uint32_t point_count = uint32_t(std::min<size_t>(point_lights.size(), max_lights));
	const uint32_t spot_count = uint32_t(std::min<size_t>(spot_lights.size(), max_lights));

	SceneUniforms scene = {};
	std::memcpy(scene.view_proj, view_proj.m, sizeof(scene.view_proj));
	for (int i = 0; i < 3; ++i) {
		scene.camera_pos[i] = camera_position[i];
		scene.ambient[i] = ambient_color[i];
		scene.dir_color[i] = directional.color[i] * directional.intensity;
	}
	normalize3(directional.direction, scene.dir_direction);
	scene.counts[0] = int32_t(point_count);
	scene.counts[1] = int32_t(spot_count);
	vmaCopyMemoryToAllocation(context.allocator, &scene, scene_buffer.allocation, 0, sizeof(scene));

	// Массивы источников света - storage-буферы (размер в шейдере не фиксирован).
	std::vector<PointLightGpu> point_gpu(point_count);
	for (uint32_t i = 0; i < point_count; ++i) {
		float pos[3];
		pointLightPosition(point_lights[i], pos);
		for (int k = 0; k < 3; ++k) {
			point_gpu[i].position[k] = pos[k];
			point_gpu[i].color[k] = point_lights[i].color[k] * point_lights[i].intensity;
		}
	}
	if (point_count > 0) {
		vmaCopyMemoryToAllocation(context.allocator, point_gpu.data(), point_buffer.allocation,
		                          0, point_count * sizeof(PointLightGpu));
	}

	std::vector<SpotLightGpu> spot_gpu(spot_count);
	for (uint32_t i = 0; i < spot_count; ++i) {
		const SpotLight& l = spot_lights[i];
		float dir[3];
		spotLightDirection(l, dir);
		normalize3(dir, spot_gpu[i].direction);
		for (int k = 0; k < 3; ++k) {
			spot_gpu[i].position[k] = l.position[k];
			spot_gpu[i].color[k] = l.color[k] * l.intensity;
		}
		spot_gpu[i].cone[0] = std::cos(l.inner_angle);
		spot_gpu[i].cone[1] = std::cos(l.outer_angle);
	}
	if (spot_count > 0) {
		vmaCopyMemoryToAllocation(context.allocator, spot_gpu.data(), spot_buffer.allocation,
		                          0, spot_count * sizeof(SpotLightGpu));
	}

	// --- Список отрисовки: данные моделей лежат в одном буфере с динамическими смещениями ---
	struct Draw {
		const GpuMesh* mesh;
		VkPipeline pipeline;
	};
	std::vector<Draw> draws;
	std::vector<uint8_t> model_data(model_stride * max_draws);

	auto addDraw = [&](const GpuMesh& mesh, VkPipeline pipeline, const math::Mat4& model,
	                   const float albedo[3], const float specular[3], float shininess, bool unlit) {
		if (draws.size() >= max_draws) return;
		ModelUniforms u = {};
		std::memcpy(u.model, model.m, sizeof(u.model));
		for (int k = 0; k < 3; ++k) {
			u.albedo[k] = albedo[k];
			u.specular[k] = specular[k];
		}
		u.albedo[3] = 1.0f;
		u.specular[3] = shininess;
		u.flags[0] = unlit ? 1.0f : 0.0f;
		std::memcpy(model_data.data() + draws.size() * model_stride, &u, sizeof(u));
		draws.push_back({&mesh, pipeline});
	};

	const float no_specular[3] = {0.0f, 0.0f, 0.0f};
	const float white[3] = {1.0f, 1.0f, 1.0f};

	for (const Object& o : objects) {
		addDraw(mesh_cylinder, pipeline_triangles, modelMatrix(o), o.albedo, o.specular, o.shininess, false);
	}

	// Визуализация источников: постоянный цвет, без освещения.
	if (show_light_markers) {
		for (uint32_t i = 0; i < point_count; ++i) {
			const PointLight& l = point_lights[i];
			float pos[3];
			pointLightPosition(l, pos);
			addDraw(mesh_cube, pipeline_triangles,
			        math::translate(pos[0], pos[1], pos[2]) *
			        math::scale(marker_size, marker_size, marker_size),
			        l.color, no_specular, 1.0f, true);
		}
		for (uint32_t i = 0; i < spot_count; ++i) {
			const SpotLight& l = spot_lights[i];
			const math::Mat4 position = math::translate(l.position[0], l.position[1], l.position[2]);
			addDraw(mesh_cube, pipeline_triangles,
			        position * math::scale(marker_size, marker_size, marker_size),
			        l.color, no_specular, 1.0f, true);

			// Конус: высота cone_length, радиус основания = высота * tan(внешний угол).
			const float r = cone_length * std::tan(std::min(l.outer_angle, 1.5f));
			addDraw(mesh_cone, pipeline_lines,
			        position * orientAlongMinusZ(spot_gpu[i].direction) * math::scale(r, r, cone_length),
			        white, no_specular, 1.0f, true);
		}
	}

	vmaCopyMemoryToAllocation(context.allocator, model_data.data(), model_buffer.allocation,
	                          0, model_data.size());

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

	// Набор 0 (сцена) привязывается один раз на кадр.
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_set, 0, nullptr);

	VkPipeline bound = VK_NULL_HANDLE;
	const VkDeviceSize offset = 0;

	for (size_t i = 0; i < draws.size(); ++i) {
		if (draws[i].pipeline != bound) {
			bound = draws[i].pipeline;
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bound);
		}

		const uint32_t dynamic_offset = uint32_t(i * model_stride);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &model_set, 1, &dynamic_offset);

		vkCmdBindVertexBuffers(cmd, 0, 1, &draws[i].mesh->vertex_buffer, &offset);
		vkCmdBindIndexBuffer(cmd, draws[i].mesh->index_buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(cmd, draws[i].mesh->index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);

	vkEndCommandBuffer(cmd);
}

} // namespace application