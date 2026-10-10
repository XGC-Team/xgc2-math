#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace xgc2_math::scene::detail {

constexpr double kPi = 3.14159265358979323846;
constexpr double kInf = std::numeric_limits<double>::infinity();

// Point `i` of `n` on the unit sphere along a Fibonacci lattice.
inline Eigen::Vector3d fibonacciPoint(int i, int n) {
    const double golden = kPi * (3.0 - std::sqrt(5.0));
    const double z = 1.0 - 2.0 * (i + 0.5) / n;
    const double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
    const double phi = golden * i;
    return Eigen::Vector3d(rho * std::cos(phi), rho * std::sin(phi), z);
}

// Number of lattice points that cover a sphere of `radius` at about `spacing` metres.
inline int sphereSampleCount(double radius, double spacing) {
    return std::max(12, static_cast<int>(std::ceil(4.0 * kPi * radius * radius / (spacing * spacing))));
}

inline void appendRing(double radius, double z, double spacing, std::vector<Eigen::Vector3d>* out) {
    const int n = std::max(8, static_cast<int>(std::ceil(2.0 * kPi * radius / spacing)));
    for (int k = 0; k < n; ++k) {
        const double a = 2.0 * kPi * k / n;
        out->emplace_back(radius * std::cos(a), radius * std::sin(a), z);
    }
}

inline void appendDisc(double radius, double z, double spacing, std::vector<Eigen::Vector3d>* out) {
    out->emplace_back(0.0, 0.0, z);
    const int rings = std::max(1, static_cast<int>(std::ceil(radius / spacing)));
    for (int k = 1; k <= rings; ++k) {
        appendRing(radius * k / rings, z, spacing, out);
    }
}

// Samples a planar convex polygon (`m` ordered vertices, outward `normal`) at `spacing`: the boundary at edge
// subdivisions plus interior cell centres. `cell` is scratch space for the polygon in plane coordinates.
inline void appendPolygon(const Eigen::Vector3d* poly, std::size_t m, const Eigen::Vector3d& normal, double spacing,
                          std::vector<Eigen::Vector2d>* cell, std::vector<Eigen::Vector3d>* out) {
    for (std::size_t i = 0; i < m; ++i) {
        const Eigen::Vector3d& a = poly[i];
        const Eigen::Vector3d& b = poly[(i + 1) % m];
        const int k = std::max(1, static_cast<int>(std::ceil((b - a).norm() / spacing)));
        for (int j = 0; j < k; ++j) {
            out->push_back(a + (b - a) * (static_cast<double>(j) / k));
        }
    }
    const Eigen::Vector3d u = (poly[1] - poly[0]).normalized();
    const Eigen::Vector3d w = normal.cross(u);
    cell->resize(m);
    std::vector<Eigen::Vector2d>& p2 = *cell;
    Eigen::Vector2d lo(kInf, kInf), hi(-kInf, -kInf);
    for (std::size_t i = 0; i < m; ++i) {
        const Eigen::Vector3d r = poly[i] - poly[0];
        p2[i] = Eigen::Vector2d(r.dot(u), r.dot(w));
        lo = lo.cwiseMin(p2[i]);
        hi = hi.cwiseMax(p2[i]);
    }
    double area = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const Eigen::Vector2d& a = p2[i];
        const Eigen::Vector2d& b = p2[(i + 1) % m];
        area += a.x() * b.y() - a.y() * b.x();
    }
    const double sign = area >= 0.0 ? 1.0 : -1.0;
    const int nu = std::max(1, static_cast<int>(std::ceil((hi.x() - lo.x()) / spacing)));
    const int nw = std::max(1, static_cast<int>(std::ceil((hi.y() - lo.y()) / spacing)));
    const double du = (hi.x() - lo.x()) / nu;
    const double dw = (hi.y() - lo.y()) / nw;
    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < nw; ++j) {
            const Eigen::Vector2d q(lo.x() + (i + 0.5) * du, lo.y() + (j + 0.5) * dw);
            bool inside = true;
            for (std::size_t e = 0; e < m && inside; ++e) {
                const Eigen::Vector2d& a = p2[e];
                const Eigen::Vector2d& b = p2[(e + 1) % m];
                const double cross = (b.x() - a.x()) * (q.y() - a.y()) - (b.y() - a.y()) * (q.x() - a.x());
                inside = sign * cross >= -1e-12;
            }
            if (inside) {
                out->push_back(poly[0] + q.x() * u + q.y() * w);
            }
        }
    }
}

} // namespace xgc2_math::scene::detail
