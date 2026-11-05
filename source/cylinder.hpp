#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace cylinder {

struct Vertex {
	float position[3];
	float normal[3];
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

// Цилиндр с осью вдоль Y, центр в начале координат.
// Нормали генерируются процедурно: у боковой стенки - радиальные (плавные),
// у оснований - ровно вверх/вниз. Поэтому вершины стенки и оснований раздельные.
inline Mesh make(uint32_t segments, float radius, float height) {
	Mesh mesh;

	const float half = height * 0.5f;
	const float step = 2.0f * std::numbers::pi_v<float> / float(segments);

	// Боковая стенка: (segments + 1) пар вершин (шов дублируется).
	for (uint32_t i = 0; i <= segments; ++i) {
		const float c = std::cos(step * float(i)), s = std::sin(step * float(i));
		mesh.vertices.push_back({{radius * c, +half, radius * s}, {c, 0.0f, s}});
		mesh.vertices.push_back({{radius * c, -half, radius * s}, {c, 0.0f, s}});
	}
	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t t0 = 2 * i, b0 = 2 * i + 1, t1 = 2 * i + 2, b1 = 2 * i + 3;
		mesh.indices.insert(mesh.indices.end(), {t0, b0, t1, t1, b0, b1});
	}

	// Основания: центр + кольцо с нормалью вдоль оси.
	for (int cap = 0; cap < 2; ++cap) {
		const float y = cap == 0 ? +half : -half;
		const float ny = cap == 0 ? 1.0f : -1.0f;
		const uint32_t center = uint32_t(mesh.vertices.size());
		mesh.vertices.push_back({{0.0f, y, 0.0f}, {0.0f, ny, 0.0f}});
		for (uint32_t i = 0; i < segments; ++i) {
			const float a = step * float(i);
			mesh.vertices.push_back({{radius * std::cos(a), y, radius * std::sin(a)}, {0.0f, ny, 0.0f}});
		}
		for (uint32_t i = 0; i < segments; ++i) {
			const uint32_t a = center + 1 + i, b = center + 1 + (i + 1) % segments;
			mesh.indices.insert(mesh.indices.end(), {center, a, b});
		}
	}

	return mesh;
}

// Единичный куб [-0.5, 0.5]^3 с нормалями граней (для маркеров источников света).
inline Mesh makeCube() {
	Mesh mesh;
	const float f[6][9] = {  // нормаль, касательная u, касательная v
		{ 1, 0, 0,  0, 1, 0,  0, 0, 1}, {-1, 0, 0,  0, 1, 0,  0, 0, 1},
		{ 0, 1, 0,  1, 0, 0,  0, 0, 1}, { 0,-1, 0,  1, 0, 0,  0, 0, 1},
		{ 0, 0, 1,  1, 0, 0,  0, 1, 0}, { 0, 0,-1,  1, 0, 0,  0, 1, 0},
	};
	const float su[4] = {-1, 1, 1, -1}, sv[4] = {-1, -1, 1, 1};
	for (const auto& d : f) {
		const uint32_t base = uint32_t(mesh.vertices.size());
		for (int k = 0; k < 4; ++k) {
			Vertex v{};
			for (int j = 0; j < 3; ++j) {
				v.position[j] = 0.5f * (d[j] + su[k] * d[3 + j] + sv[k] * d[6 + j]);
				v.normal[j] = d[j];
			}
			mesh.vertices.push_back(v);
		}
		mesh.indices.insert(mesh.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
	}
	return mesh;
}

// Каркас конуса (линии): вершина в начале координат, основание радиуса 1 на z = -1.
// Масштабируется и поворачивается матрицей модели. Топология - LINE_LIST.
inline Mesh makeConeLines(uint32_t segments = 32) {
	Mesh mesh;
	const float step = 2.0f * std::numbers::pi_v<float> / float(segments);
	mesh.vertices.push_back({{0, 0, 0}, {0, 0, 1}});
	for (uint32_t i = 0; i < segments; ++i) {
		mesh.vertices.push_back({{std::cos(step * float(i)), std::sin(step * float(i)), -1.0f}, {0, 0, 1}});
	}
	for (uint32_t i = 0; i < segments; ++i) {
		mesh.indices.insert(mesh.indices.end(), {1 + i, 1 + (i + 1) % segments});
		if (i % (segments / 4) == 0) {
			mesh.indices.insert(mesh.indices.end(), {0, 1 + i});
		}
	}
	return mesh;
}

} // namespace cylinder