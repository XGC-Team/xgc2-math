#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace xgc2_math::scene {
struct MeshTriangle {
    Eigen::Vector3d a, b, c;
};
inline void validateTriangleMesh(const std::vector<Eigen::Vector3d>& vertices,
                                 const std::vector<std::array<uint32_t, 3>>& triangles) {
    if (vertices.size() < 4 || triangles.empty())
        throw std::invalid_argument("mesh requires vertices and triangles");
    for (const auto& v : vertices)
        if (!v.allFinite())
            throw std::invalid_argument("nonfinite mesh vertex");
    std::map<std::pair<uint32_t, uint32_t>, std::pair<unsigned, int>> edges;
    for (const auto& t : triangles) {
        for (auto i : t)
            if (i >= vertices.size())
                throw std::invalid_argument("mesh index out of range");
        if ((vertices[t[1]] - vertices[t[0]]).cross(vertices[t[2]] - vertices[t[0]]).squaredNorm() == 0)
            throw std::invalid_argument("degenerate mesh triangle");
        for (int j = 0; j < 3; ++j) {
            const auto a = t[j], b = t[(j + 1) % 3];
            auto& e = edges[{std::min(a, b), std::max(a, b)}];
            ++e.first;
            e.second += a < b ? 1 : -1;
        }
    }
    for (const auto& e : edges)
        if (e.second.first != 2 || e.second.second != 0)
            throw std::invalid_argument("mesh must have closed consistently oriented manifold edges");
}
inline double triangleDistance(const Eigen::Vector3d& p, const MeshTriangle& t) {
    const Eigen::Vector3d ab = t.b - t.a, ac = t.c - t.a, ap = p - t.a;
    const double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0)
        return ap.norm();
    const Eigen::Vector3d bp = p - t.b;
    const double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3)
        return bp.norm();
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
        return (p - (t.a + ab * (d1 / (d1 - d3)))).norm();
    const Eigen::Vector3d cp = p - t.c;
    const double d5 = ab.dot(cp), d6 = ac.dot(cp);
    if (d6 >= 0 && d5 <= d6)
        return cp.norm();
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
        return (p - (t.a + ac * (d2 / (d2 - d6)))).norm();
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
        return (p - (t.b + (t.c - t.b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).norm();
    return std::abs(ap.dot(ab.cross(ac).normalized()));
}
inline double meshResidual(const MeshTriangle* triangles, std::size_t count, const Eigen::Vector3d& p) {
    double distance = std::numeric_limits<double>::infinity();
    long double winding = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& t = triangles[i];
        distance = std::min(distance, triangleDistance(p, t));
        const Eigen::Vector3d a = t.a - p, b = t.b - p, c = t.c - p;
        const double an = a.norm(), bn = b.norm(), cn = c.norm();
        winding +=
            2 * std::atan2(static_cast<long double>(a.dot(b.cross(c))),
                           static_cast<long double>(an) * bn * cn + static_cast<long double>(a.dot(b)) * cn +
                               static_cast<long double>(b.dot(c)) * an + static_cast<long double>(c.dot(a)) * bn);
    }
    return std::abs(winding) > 2 * std::acos(-1.L) ? -distance : distance;
}
inline double meshRayClosest(const MeshTriangle* triangles, std::size_t count, const Eigen::Vector3d& o,
                             const Eigen::Vector3d& d, double maxRange) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        const auto& t = triangles[i];
        const Eigen::Vector3d e1 = t.b - t.a, e2 = t.c - t.a, h = d.cross(e2);
        const double det = e1.dot(h);
        if (std::abs(det) <= 1e-14 * e1.norm() * e2.norm())
            continue;
        const Eigen::Vector3d s = o - t.a;
        const double u = s.dot(h) / det;
        if (u < -1e-12 || u > 1 + 1e-12)
            continue;
        const Eigen::Vector3d q = s.cross(e1);
        const double v = d.dot(q) / det;
        if (v < -1e-12 || u + v > 1 + 1e-12)
            continue;
        const double hit = e2.dot(q) / det;
        if (hit >= 0 && hit <= maxRange)
            best = std::min(best, hit);
    }
    return best;
}
} // namespace xgc2_math::scene
