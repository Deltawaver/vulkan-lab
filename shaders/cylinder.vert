#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(push_constant) uniform PushConstants {
	mat4 mvp;
} pc;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = pc.mvp * vec4(in_position, 1.0);

	// Временная раскраска по нормали, чтобы объём было видно без освещения.
	out_color = in_normal * 0.5 + 0.5;
}
