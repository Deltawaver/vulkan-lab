#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform Scene {
	mat4 view_proj;
	vec4 camera_pos;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	ivec4 counts;
} scene;

struct PointLight {
	vec4 position;
	vec4 color;  // rgb * интенсивность
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;  // rgb * интенсивность
	vec4 cone;   // x - cos(внутреннего угла), y - cos(внешнего угла)
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights { PointLight point_lights[]; };
layout(std430, set = 0, binding = 2) readonly buffer SpotLights { SpotLight spot_lights[]; };

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	vec4 albedo;
	vec4 specular;
	vec4 flags;
} model;

// Модель Блинн-Фонга для одного источника. L - от точки к источнику.
vec3 blinnPhong(vec3 L, vec3 radiance, vec3 N, vec3 V) {
	const float ndl = max(dot(N, L), 0.0);
	const vec3 H = normalize(L + V);
	const float spec = ndl > 0.0 ? pow(max(dot(N, H), 0.0), model.specular.a) : 0.0;
	return (model.albedo.rgb * ndl + model.specular.rgb * spec) * radiance;
}

void main() {
	if (model.flags.x > 0.5) {
		out_color = vec4(model.albedo.rgb, 1.0);
		return;
	}

	const vec3 N = normalize(in_world_normal);
	const vec3 V = normalize(scene.camera_pos.xyz - in_world_pos);

	vec3 color = scene.ambient.rgb * model.albedo.rgb;

	// Направленный источник.
	color += blinnPhong(normalize(-scene.dir_direction.xyz), scene.dir_color.rgb, N, V);

	// Точечные источники: затухание 1 / d^2.
	for (int i = 0; i < scene.counts.x; ++i) {
		const vec3 to_light = point_lights[i].position.xyz - in_world_pos;
		const float d2 = max(dot(to_light, to_light), 1e-4);
		color += blinnPhong(to_light * inversesqrt(d2), point_lights[i].color.rgb / d2, N, V);
	}

	// Прожекторные источники: 1 / d^2 и мягкий край между внутренним и внешним углом.
	for (int i = 0; i < scene.counts.y; ++i) {
		const vec3 to_light = spot_lights[i].position.xyz - in_world_pos;
		const float d2 = max(dot(to_light, to_light), 1e-4);
		const vec3 L = to_light * inversesqrt(d2);

		const float cos_theta = dot(-L, normalize(spot_lights[i].direction.xyz));
		const float cos_inner = spot_lights[i].cone.x;
		const float cos_outer = spot_lights[i].cone.y;
		float t = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 1e-4), 0.0, 1.0);
		t = t * t * (3.0 - 2.0 * t);

		color += blinnPhong(L, spot_lights[i].color.rgb * (t / d2), N, V);
	}

	out_color = vec4(color, 1.0);
}
