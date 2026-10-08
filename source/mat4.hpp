#pragma once

#include <cmath>

namespace math {

// Матрица 4x4, хранится по СТОЛБЦАМ (как в GLSL):
// элемент (строка r, столбец c) лежит в m[c * 4 + r].
// Векторы - столбцы, поэтому преобразование записывается как M * v,
// а A * B означает "сначала B, потом A".
struct Mat4 {
	float m[16] = {};

	float& at(int r, int c) { return m[c * 4 + r]; }
	float at(int r, int c) const { return m[c * 4 + r]; }
};

inline Mat4 identity() {
	Mat4 r;
	r.at(0, 0) = r.at(1, 1) = r.at(2, 2) = r.at(3, 3) = 1.0f;
	return r;
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (int c = 0; c < 4; ++c) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.at(row, k) * b.at(k, c);
			}
			r.at(row, c) = sum;
		}
	}
	return r;
}

inline Mat4 translate(float x, float y, float z) {
	Mat4 r = identity();
	r.at(0, 3) = x;
	r.at(1, 3) = y;
	r.at(2, 3) = z;
	return r;
}

inline Mat4 scale(float x, float y, float z) {
	Mat4 r = identity();
	r.at(0, 0) = x;
	r.at(1, 1) = y;
	r.at(2, 2) = z;
	return r;
}

inline Mat4 rotateX(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.at(1, 1) = c;  r.at(1, 2) = -s;
	r.at(2, 1) = s;  r.at(2, 2) = c;
	return r;
}

inline Mat4 rotateY(float angle) {
	const float c = std::cos(angle), s = std::sin(angle);
	Mat4 r = identity();
	r.at(0, 0) = c;   r.at(0, 2) = s;
	r.at(2, 0) = -s;  r.at(2, 2) = c;
	return r;
}

// Перспективная проекция под Vulkan:
//  - камера смотрит вдоль -Z (правосторонняя система),
//  - глубина после деления на w попадает в [0, 1] (в OpenGL было бы [-1, 1]),
//  - ось Y в Vulkan направлена ВНИЗ, поэтому у элемента (1,1) стоит минус.
inline Mat4 perspective(float fov_y, float aspect, float z_near, float z_far) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);
	Mat4 r;
	r.at(0, 0) = f / aspect;
	r.at(1, 1) = -f;
	r.at(2, 2) = z_far / (z_near - z_far);
	r.at(2, 3) = (z_near * z_far) / (z_near - z_far);
	r.at(3, 2) = -1.0f;
	return r;
}

} // namespace math