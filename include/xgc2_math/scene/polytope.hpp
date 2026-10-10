#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "xgc2_math/geometry/quickhull.hpp"

namespace xgc2_math::scene {

// Supporting plane of a convex solid with outward unit normal `n`: the solid lies in n . x <= d.
struct Plane {
    Eigen::Vector3d n = Eigen::Vector3d::UnitZ();
    double d = 0.0;
};

// Boundary representation of the convex hull of a point set.
struct Polytope {
    std::vector<Eigen::Vector3d> vertices;      // hull vertices, in input order
    std::vector<Plane> planes;                  // one plane per face
    std::vector<std::size_t> face_first;        // face f is face_vertices[face_first[f], face_first[f + 1])
    std::vector<Eigen::Vector3d> face_vertices; // face polygons, counter-clockwise seen from outside
};

namespace detail {

// A face of the polytope while it is assembled: its plane, the hull vertices on it and its sort key.
struct PolytopeFace {
    Plane plane;
    std::array<std::size_t, 3> key{{0, 0, 0}};
    std::vector<std::size_t> members; // ascending vertex indices
};

inline bool samePlane(const Plane& a, const Plane& b, double eps) {
    return a.n.dot(b.n) > 1.0 - 1e-9 && std::abs(a.d - b.d) < eps;
}

} // namespace detail

// The convex hull of `input` as planes and face polygons. Duplicate points (closer than 1e-9) are merged and points
// that are not hull vertices are ignored. Faces are ordered by the smallest vertex triple that spans them, which
// is the order the first face of each plane is met when all vertex triples are enumerated; the vertices of a face run
// counter-clockwise from the angle -pi, measured from the direction of its lowest input index. Throws
// std::invalid_argument for non-finite input, fewer than four distinct points or a flat point set (all points within
// about 1e-7 of the extent from one plane).
inline Polytope buildPolytope(const std::vector<Eigen::Vector3d>& input) {
    std::vector<Eigen::Vector3d> v;
    v.reserve(input.size());
    for (const auto& p : input) {
        if (!p.allFinite()) {
            throw std::invalid_argument("non-finite convex vertex");
        }
        const bool duplicate = std::any_of(v.begin(), v.end(), [&p](const Eigen::Vector3d& q) {
            return (p - q).squaredNorm() < 1e-18;
        });
        if (!duplicate) {
            v.push_back(p);
        }
    }
    if (v.size() < 4) {
        throw std::invalid_argument("convex obstacle needs >= 4 distinct vertices");
    }
    Eigen::Vector3d lo = v[0], hi = v[0];
    for (const auto& p : v) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    const double extent = std::max((hi - lo).norm(), 1e-9);
    const double eps = 1e-7 * extent;
    const double degenerate = 1e-10 * extent * extent;

    // Triangles of the hull. The cloud is centred so that the hull tolerance does not depend on the offset.
    const Eigen::Vector3d center = 0.5 * (lo + hi);
    std::vector<Eigen::Vector3d> centered(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        centered[i] = v[i] - center;
    }
    quickhull::QuickHull<double> quickhull;
    const auto hull = quickhull.getConvexHull(centered.front().data(), centered.size(), false, true, 1e-7);
    if (quickhull.getDiagnostics().m_failedHorizonEdges != 0) {
        throw std::invalid_argument("convex hull could not be computed");
    }
    const std::vector<std::size_t>& triangles = hull.getIndexBuffer();

    std::vector<char> on_hull(v.size(), 0);
    std::vector<std::array<std::size_t, 3>> facets;
    for (std::size_t t = 0; t + 2 < triangles.size(); t += 3) {
        const std::array<std::size_t, 3> f{{triangles[t], triangles[t + 1], triangles[t + 2]}};
        if ((v[f[1]] - v[f[0]]).cross(v[f[2]] - v[f[0]]).norm() < degenerate) {
            continue;
        }
        facets.push_back(f);
        for (std::size_t index : f) {
            on_hull[index] = 1;
        }
    }
    Eigen::Vector3d inside = Eigen::Vector3d::Zero();
    std::size_t hull_count = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (on_hull[i]) {
            inside += v[i];
            ++hull_count;
        }
    }
    if (hull_count < 4) {
        throw std::invalid_argument("convex obstacle is flat");
    }
    inside /= static_cast<double>(hull_count);

    // Plane through three vertices, oriented away from the interior; false for a (nearly) collinear triple.
    const auto planeOf = [&](std::size_t i, std::size_t j, std::size_t k, Plane* plane) {
        Eigen::Vector3d nn = (v[j] - v[i]).cross(v[k] - v[i]);
        const double len = nn.norm();
        if (len < degenerate) {
            return false;
        }
        nn /= len;
        plane->n = nn;
        plane->d = nn.dot(v[i]);
        if (nn.dot(inside) - plane->d > 0.0) {
            plane->n = -nn;
            plane->d = -plane->d;
        }
        return true;
    };

    // One face per distinct facet plane.
    std::vector<Plane> facet_planes;
    for (const auto& f : facets) {
        Plane plane;
        if (!planeOf(f[0], f[1], f[2], &plane)) {
            continue;
        }
        if (std::none_of(facet_planes.begin(), facet_planes.end(), [&](const Plane& q) {
                return detail::samePlane(q, plane, eps);
            })) {
            facet_planes.push_back(plane);
        }
    }
    if (facet_planes.size() < 4) {
        throw std::invalid_argument("convex obstacle is flat");
    }
    std::vector<detail::PolytopeFace> faces;
    faces.reserve(facet_planes.size());
    for (const Plane& facet_plane : facet_planes) {
        detail::PolytopeFace face;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (on_hull[i] && std::abs(facet_plane.n.dot(v[i]) - facet_plane.d) <= eps) {
                face.members.push_back(i);
            }
        }
        // The plane is that of the first non-collinear vertex triple of the face, as in an enumeration of all triples.
        face.plane = facet_plane;
        bool found = false;
        for (std::size_t a = 0; a < face.members.size() && !found; ++a) {
            for (std::size_t b = a + 1; b < face.members.size() && !found; ++b) {
                for (std::size_t c = b + 1; c < face.members.size() && !found; ++c) {
                    Plane plane;
                    if (planeOf(face.members[a], face.members[b], face.members[c], &plane)) {
                        face.plane = plane;
                        face.key = {{face.members[a], face.members[b], face.members[c]}};
                        found = true;
                    }
                }
            }
        }
        face.members.clear();
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (on_hull[i] && std::abs(face.plane.n.dot(v[i]) - face.plane.d) <= eps) {
                face.members.push_back(i);
            }
        }
        faces.push_back(std::move(face));
    }
    std::stable_sort(faces.begin(), faces.end(), [](const detail::PolytopeFace& a, const detail::PolytopeFace& b) {
        return a.key < b.key;
    });

    Polytope polytope;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (on_hull[i]) {
            polytope.vertices.push_back(v[i]);
        }
    }
    polytope.face_first.push_back(0);
    std::vector<std::pair<double, std::size_t>> angles;
    for (const auto& face : faces) {
        Eigen::Vector3d c =
            std::accumulate(face.members.begin(), face.members.end(), Eigen::Vector3d(Eigen::Vector3d::Zero()),
                            [&v](const Eigen::Vector3d& sum, std::size_t index) {
                                return Eigen::Vector3d(sum + v[index]);
                            });
        c /= static_cast<double>(face.members.size());
        const Eigen::Vector3d u = (v[face.members.front()] - c).normalized();
        const Eigen::Vector3d w = face.plane.n.cross(u);
        angles.clear();
        for (std::size_t index : face.members) {
            const Eigen::Vector3d r = v[index] - c;
            angles.emplace_back(std::atan2(r.dot(w), r.dot(u)), index);
        }
        std::sort(angles.begin(), angles.end());
        for (const auto& entry : angles) {
            polytope.face_vertices.push_back(v[entry.second]);
        }
        polytope.face_first.push_back(polytope.face_vertices.size());
        polytope.planes.push_back(face.plane);
    }
    return polytope;
}

} // namespace xgc2_math::scene
