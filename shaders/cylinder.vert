#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// Набор дескрипторов объекта: у каждого объекта свой uniform-буфер.
layout(set = 0, binding = 0) uniform Object {
	mat4 mvp;
	vec4 color;
} object;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = object.mvp * vec4(in_position, 1.0);

	// Цвет вершины (по её локальной позиции) умножается на выбранный в интерфейсе цвет.
	out_color = in_color * object.color.rgb;
}
