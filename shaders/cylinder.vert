#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// Набор дескрипторов 0: сцена.
layout(set = 0, binding = 0) uniform Scene {
	mat4 view_proj;
	vec4 camera_pos;
	vec4 ambient;
	vec4 dir_direction; 
	vec4 dir_color;      // цвет * интенсивность
	ivec4 counts;        // x - точечных, y - прожекторных
} scene;

// Набор дескрипторов 1: модель (динамический uniform-буфер).
layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	vec4 albedo;
	vec4 specular;  // rgb - цвет блика, a - коэффициент блика
	vec4 flags;     // x > 0.5 - без освещения
} model;

layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec2 out_uv;

void main() {
	const vec4 world = model.model * vec4(in_position, 1.0);
	gl_Position = scene.view_proj * world;
	out_world_pos = world.xyz;
	out_world_normal = transpose(inverse(mat3(model.model))) * in_normal;
	out_uv = in_uv;
}
