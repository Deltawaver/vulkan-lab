#version 450

layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_proj;
	vec4 camera_pos;
	vec4 ambient;
	vec4 dir_direction; 
	vec4 dir_color;      // цвет * интенсивность
	ivec4 counts;        // x - точечных, y - прожекторных
	mat4 light_view_proj[2];
} scene;
layout(set = 1, binding = 0) uniform Model {
	mat4 model;
} object;
layout(push_constant) uniform ShadowPass {
	layout(offset = 12) int light;
} shadow;

void main() {
	gl_Position = scene.light_view_proj[shadow.light] * object.model * vec4(in_position, 1.0);
}
