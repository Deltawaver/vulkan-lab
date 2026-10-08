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
// В каждом основании `segments` вершин по окружности (+ центр для веера).
// Вершины основания и боковой поверхности дублируются: у них разные нормали.
inline Mesh make(uint32_t segments, float radius, float height) {
	Mesh mesh;

	const float half = height * 0.5f;
	const float step = 2.0f * std::numbers::pi_v<float> / float(segments);

	// --- Боковая поверхность: верхнее кольцо [0, n), нижнее кольцо [n, 2n) ---
	for (uint32_t i = 0; i < segments; ++i) {
		const float c = std::cos(step * float(i)), s = std::sin(step * float(i));
		mesh.vertices.push_back(Vertex{{radius * c, +half, radius * s}, {c, 0.0f, s}});
	}
	for (uint32_t i = 0; i < segments; ++i) {
		const float c = std::cos(step * float(i)), s = std::sin(step * float(i));
		mesh.vertices.push_back(Vertex{{radius * c, -half, radius * s}, {c, 0.0f, s}});
	}
	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t j = (i + 1) % segments;
		const uint32_t top_i = i, top_j = j;
		const uint32_t bottom_i = segments + i, bottom_j = segments + j;

		mesh.indices.insert(mesh.indices.end(), {top_i, bottom_i, top_j});
		mesh.indices.insert(mesh.indices.end(), {top_j, bottom_i, bottom_j});
	}

	// --- Основания: центр + кольцо, треугольники веером ---
	for (int cap = 0; cap < 2; ++cap) {
		const float y = (cap == 0) ? +half : -half;
		const float ny = (cap == 0) ? +1.0f : -1.0f;

		const uint32_t center = uint32_t(mesh.vertices.size());
		mesh.vertices.push_back(Vertex{{0.0f, y, 0.0f}, {0.0f, ny, 0.0f}});

		const uint32_t ring = uint32_t(mesh.vertices.size());
		for (uint32_t i = 0; i < segments; ++i) {
			const float c = std::cos(step * float(i)), s = std::sin(step * float(i));
			mesh.vertices.push_back(Vertex{{radius * c, y, radius * s}, {0.0f, ny, 0.0f}});
		}

		for (uint32_t i = 0; i < segments; ++i) {
			const uint32_t j = (i + 1) % segments;
			mesh.indices.insert(mesh.indices.end(), {center, ring + j, ring + i});
		}
	}

	return mesh;
}

} // namespace cylinder