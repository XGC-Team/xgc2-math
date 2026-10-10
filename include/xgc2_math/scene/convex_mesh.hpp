#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "xgc2_math/geometry/quickhull.hpp"

namespace xgc2_math::scene {

// A closed triangle mesh: vertex positions and index triplets (counter-clockwise seen from outside).
struct ConvexMesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<std::uint32_t> triangles;
};

namespace detail {

// Python-order dot product: ((x0 y0 + x1 y1) + x2 y2).
inline double dot3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

inline Eigen::Vector3d cross3(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return Eigen::Vector3d(a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(), a.x() * b.y() - a.y() * b.x());
}

} // namespace detail

// The surface of the convex hull of `points` as a closed triangle mesh, with outward counter-clockwise triangles, as
// a scene document (xgc2.scene.v1) stores a convex part: only the hull vertices are kept, in input order, and the
// triangles come from the hull computation with `epsilon` as its tolerance (relative to the extent of the points).
// Throws std::invalid_argument for fewer than four points or points that do not span a volume.
inline ConvexMesh convexMeshOf(const std::vector<Eigen::Vector3d>& points, double epsilon = 1e-7) {
    if (points.size() < 4) {
        throw std::invalid_argument("convex hull needs >= 4 points");
    }
    if (std::any_of(points.begin(), points.end(), [](const Eigen::Vector3d& p) {
            return !p.allFinite();
        })) {
        throw std::invalid_argument("non-finite convex vertex");
    }
    quickhull::QuickHull<double> quickhull;
    const auto hull = quickhull.getConvexHull(points.front().data(), points.size(), false, true, epsilon);
    const std::vector<std::size_t>& indices = hull.getIndexBuffer();
    if (quickhull.getDiagnostics().m_failedHorizonEdges != 0) {
        throw std::invalid_argument("convex hull could not be computed");
    }
    std::vector<char> on_hull(points.size(), 0);
    for (std::size_t index : indices) {
        on_hull[index] = 1;
    }
    ConvexMesh mesh;
    std::vector<std::uint32_t> compact(points.size(), 0);
    Eigen::Vector3d inside = Eigen::Vector3d::Zero();
    Eigen::Vector3d lo = points.front(), hi = points.front();
    for (std::size_t i = 0; i < points.size(); ++i) {
        lo = lo.cwiseMin(points[i]);
        hi = hi.cwiseMax(points[i]);
        if (on_hull[i]) {
            compact[i] = static_cast<std::uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(points[i]);
            inside += points[i];
        }
    }
    if (mesh.vertices.size() < 4) {
        throw std::invalid_argument("convex hull is flat");
    }
    inside /= static_cast<double>(mesh.vertices.size());
    const double extent = (hi - lo).norm();
    const double degenerate = 1e-10 * extent * extent;
    double volume = 0.0;
    for (std::size_t t = 0; t + 2 < indices.size(); t += 3) {
        std::array<std::uint32_t, 3> triangle{{compact[indices[t]], compact[indices[t + 1]], compact[indices[t + 2]]}};
        const Eigen::Vector3d& a = mesh.vertices[triangle[0]];
        const Eigen::Vector3d normal = (mesh.vertices[triangle[1]] - a).cross(mesh.vertices[triangle[2]] - a);
        if (normal.norm() < degenerate) {
            continue;
        }
        const double height = normal.dot(inside - a);
        if (height > 0.0) {
            std::swap(triangle[1], triangle[2]);
        }
        volume += std::abs(height) / 6.0;
        mesh.triangles.insert(mesh.triangles.end(), triangle.begin(), triangle.end());
    }
    // Points on one plane come back as a two-sided sheet of triangles without volume.
    if (mesh.triangles.size() < 12 || volume <= 1e-9 * extent * extent * extent) {
        throw std::invalid_argument("convex hull is flat");
    }
    return mesh;
}

// Checks the surface of a convex part of a scene document (xgc2.scene.v1): `vertices` and `indices` (triangle index
// triplets) must describe a closed, strictly convex surface of non-zero volume, and returns the triangles oriented
// counter-clockwise seen from outside. Limits: 4 to 4096 vertices with finite coordinates of at most 1e6 in magnitude,
// no duplicates, 4 to 8192 triangles, every vertex on the surface. Throws std::invalid_argument with the message of the
// scene runtime's validator.
inline std::vector<std::uint32_t> validateConvexMesh(const std::vector<Eigen::Vector3d>& vertices,
                                                     const std::vector<std::int64_t>& indices) {
    if (vertices.size() < 4 || vertices.size() > 4096) {
        throw std::invalid_argument("Convex geometry requires 4 to 4096 vertices");
    }
    for (const auto& p : vertices) {
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(p[k])) {
                throw std::invalid_argument("vertex must be finite");
            }
            if (!(-1e6 <= p[k] && p[k] <= 1e6)) {
                throw std::invalid_argument("vertex is outside the supported range");
            }
        }
    }
    {
        std::vector<std::array<double, 3>> sorted(vertices.size());
        // -0.0 and 0.0 are the same coordinate.
        std::transform(vertices.begin(), vertices.end(), sorted.begin(), [](const Eigen::Vector3d& p) {
            return std::array<double, 3>{{p.x() + 0.0, p.y() + 0.0, p.z() + 0.0}};
        });
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("Convex geometry has duplicate vertices");
        }
    }
    if (indices.size() < 12 || indices.size() > 24576 || indices.size() % 3 != 0) {
        throw std::invalid_argument("Convex geometry requires triangle index triplets");
    }
    std::vector<char> used(vertices.size(), 0);
    for (std::int64_t i : indices) {
        if (i < 0 || i >= static_cast<std::int64_t>(vertices.size())) {
            throw std::invalid_argument("Convex triangle index is invalid");
        }
        used[static_cast<std::size_t>(i)] = 1;
    }
    if (std::find(used.begin(), used.end(), 0) != used.end()) {
        throw std::invalid_argument("Every convex vertex must belong to its surface");
    }
    // Summed in vertex order, like the scene runtime does.
    Eigen::Vector3d center = std::accumulate(vertices.begin(), vertices.end(), Eigen::Vector3d(Eigen::Vector3d::Zero()),
                                             [](const Eigen::Vector3d& sum, const Eigen::Vector3d& p) {
                                                 return Eigen::Vector3d(sum + p);
                                             });
    center /= static_cast<double>(vertices.size());
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> edges;
    std::vector<std::uint32_t> triangles;
    triangles.reserve(indices.size());
    double volume = 0.0;
    for (std::size_t offset = 0; offset < indices.size(); offset += 3) {
        std::uint32_t a = static_cast<std::uint32_t>(indices[offset]);
        std::uint32_t b = static_cast<std::uint32_t>(indices[offset + 1]);
        std::uint32_t c = static_cast<std::uint32_t>(indices[offset + 2]);
        Eigen::Vector3d normal = detail::cross3(vertices[b] - vertices[a], vertices[c] - vertices[a]);
        const double magnitude = std::sqrt(detail::dot3(normal, normal));
        if (magnitude < 1e-12) {
            throw std::invalid_argument("Convex geometry has a degenerate face");
        }
        if (detail::dot3(normal, center - vertices[a]) > 0) {
            std::swap(b, c);
            normal = -normal;
        }
        if (std::any_of(vertices.begin(), vertices.end(), [&](const Eigen::Vector3d& p) {
                return detail::dot3(normal, p - vertices[a]) > 1e-8 * magnitude;
            })) {
            throw std::invalid_argument("Geometry is not convex; represent it as separate convex parts");
        }
        volume += std::abs(detail::dot3(normal, center - vertices[a])) / 6.0;
        triangles.push_back(a);
        triangles.push_back(b);
        triangles.push_back(c);
        ++edges[{a, b}];
        ++edges[{b, c}];
        ++edges[{c, a}];
    }
    const auto open = [&edges](const std::pair<const std::pair<std::uint32_t, std::uint32_t>, int>& edge) {
        const auto reverse = edges.find({edge.first.second, edge.first.first});
        return edge.second != 1 || reverse == edges.end() || reverse->second != 1;
    };
    if (volume < 1e-12 || std::any_of(edges.begin(), edges.end(), open)) {
        throw std::invalid_argument("Convex surface must be a closed nonzero-volume manifold");
    }
    return triangles;
}

} // namespace xgc2_math::scene
