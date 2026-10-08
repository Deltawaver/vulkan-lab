#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace cylinder {

struct Vertex {
	float position[3];
	float color[3];
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

// Цилиндр с осью вдоль Y, центр в начале координат.
//
// Вершины (всего 2 * segments + 2):
//   [0, n)      - верхнее основание (окружность)
//   [n, 2n)     - нижнее основание (окружность)
//   2n          - центр верхнего основания
//   2n + 1      - центр нижнего основания
// Боковая стенка и основания используют одни и те же вершины, потому что
// цвет зависит только от положения вершины, а нормали нам не нужны.
//
// Цвет вершины - её локальная позиция, отображённая из [-размер, +размер]
// в [0, 1]: (x, y, z) -> (R, G, B).
inline Mesh make(uint32_t segments, float radius, float height) {
	Mesh mesh;

	const float half = height * 0.5f;
	const float step = 2.0f * std::numbers::pi_v<float> / float(segments);

	auto makeVertex = [&](float x, float y, float z) {
		return Vertex{
			{x, y, z},
			{x / radius * 0.5f + 0.5f, y / half * 0.5f + 0.5f, z / radius * 0.5f + 0.5f},
		};
	};

	for (int ring = 0; ring < 2; ++ring) {
		const float y = (ring == 0) ? +half : -half;
		for (uint32_t i = 0; i < segments; ++i) {
			const float a = step * float(i);
			mesh.vertices.push_back(makeVertex(radius * std::cos(a), y, radius * std::sin(a)));
		}
	}

	const uint32_t top_center = 2 * segments;
	const uint32_t bottom_center = 2 * segments + 1;
	mesh.vertices.push_back(makeVertex(0.0f, +half, 0.0f));
	mesh.vertices.push_back(makeVertex(0.0f, -half, 0.0f));

	for (uint32_t i = 0; i < segments; ++i) {
		const uint32_t j = (i + 1) % segments;
		const uint32_t top_i = i, top_j = j;
		const uint32_t bottom_i = segments + i, bottom_j = segments + j;

		// Боковая стенка: два треугольника на сегмент.
		mesh.indices.insert(mesh.indices.end(), {top_i, bottom_i, top_j});
		mesh.indices.insert(mesh.indices.end(), {top_j, bottom_i, bottom_j});

		// Основания: треугольник от центра к двум соседним вершинам окружности.
		mesh.indices.insert(mesh.indices.end(), {top_center, top_j, top_i});
		mesh.indices.insert(mesh.indices.end(), {bottom_center, bottom_i, bottom_j});
	}

	return mesh;
}

} // namespace cylinder