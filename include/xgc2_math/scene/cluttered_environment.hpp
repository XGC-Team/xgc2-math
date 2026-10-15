#pragma once

// Obstacles of a cluttered-environment scenario: the construction math of the cluttered_environment generator, without
// ROS. An obstacle is a primitive (sphere, cylinder, capsule, cube) or a convex polytope given by half-spaces
// (H-polytope) or by vertices (V-polytope), placed by a pose and a per-axis scale in its local frame:
//
//   sphere     radius scale.x
//   cylinder   radius scale.x, full height scale.z along local z
//   capsule    radius scale.x, full length scale.z along local z, caps included
//   cube       full side lengths scale
//   polytope   unit-scale vertices multiplied component-wise by scale
//
// A polytope given by half-spaces h0 x + h1 y + h2 z + h3 <= 0 (one row per plane) has the vertices
// geo_utils::enumerateVs() finds; its hull triangles are those of the quickhull of those vertices.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgc2_math/geometry/geo_utils.hpp"
#include "xgc2_math/geometry/occupied_sets/box_set.h"
#include "xgc2_math/geometry/occupied_sets/capsule_set.h"
#include "xgc2_math/geometry/occupied_sets/cylinder_set.h"
#include "xgc2_math/geometry/occupied_sets/point_set.h"
#include "xgc2_math/geometry/occupied_sets/sphere_set.h"
#include "xgc2_math/scene/convex_mesh.hpp"
#include "xgc2_math/scene/obstacle.hpp"
#include "xgc2_math/scene/surface_sampling.hpp"

namespace xgc2_math::scene {

struct ClutteredObstacle {
    enum Kind { kSphere, kCylinder, kCapsule, kCube, kHPolytope, kVPolytope };

    Kind kind = kSphere;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d scale = Eigen::Vector3d::Ones();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::MatrixX4d h_matrix;             // kHPolytope
    std::vector<Eigen::Vector3d> vertices; // kVPolytope, unit scale, local frame
    double epsilon = 1e-6;                 // polytope tolerance
};

namespace detail {

constexpr std::size_t kMinHullVertices = 4;

} // namespace detail

// The vertices of a polytope obstacle at unit scale. Throws std::invalid_argument for a non-polytope, an H-polytope
// with fewer than four rows or one whose vertices cannot be enumerated, or fewer than four vertices.
inline std::vector<Eigen::Vector3d> polytopeVertices(const ClutteredObstacle& obstacle) {
    std::vector<Eigen::Vector3d> vertices;
    if (obstacle.kind == ClutteredObstacle::kHPolytope) {
        if (obstacle.h_matrix.cols() != 4 ||
            obstacle.h_matrix.rows() < static_cast<Eigen::Index>(detail::kMinHullVertices)) {
            throw std::invalid_argument("H-polytope: need at least 4 valid H-rows, got " +
                                        std::to_string(obstacle.h_matrix.rows()));
        }
        Eigen::Matrix3Xd enumerated;
        if (!geo_utils::enumerateVs(obstacle.h_matrix, enumerated, obstacle.epsilon)) {
            throw std::invalid_argument("Failed to enumerate vertices from H-representation");
        }
        vertices.reserve(static_cast<std::size_t>(enumerated.cols()));
        for (Eigen::Index i = 0; i < enumerated.cols(); ++i) {
            vertices.push_back(enumerated.col(i));
        }
    } else if (obstacle.kind == ClutteredObstacle::kVPolytope) {
        vertices = obstacle.vertices;
    } else {
        throw std::invalid_argument("shape is not a polytope");
    }
    if (vertices.size() < detail::kMinHullVertices) {
        throw std::invalid_argument("polytope: need at least 4 vertices for 3D hull, got " +
                                    std::to_string(vertices.size()));
    }
    return vertices;
}

// A polytope obstacle in vertex form: an H-polytope becomes the V-polytope of its enumerated vertices, so that repeated
// calls of toObstacle() or polytopeMesh() (a moving obstacle needs one per update) do not enumerate
// again. Other obstacles are returned unchanged.
inline ClutteredObstacle vertexForm(const ClutteredObstacle& obstacle) {
    ClutteredObstacle converted = obstacle;
    if (obstacle.kind == ClutteredObstacle::kHPolytope) {
        converted.vertices = polytopeVertices(obstacle);
        converted.kind = ClutteredObstacle::kVPolytope;
        converted.h_matrix.resize(0, 4);
    }
    return converted;
}

// The obstacle as a solid for ray casting. A capsule's length is that of the whole capsule, a scene capsule's height
// that of its straight part.
inline Obstacle toObstacle(const ClutteredObstacle& obstacle) {
    const Eigen::Quaterniond q = obstacle.orientation.normalized();
    const Eigen::Vector3d& s = obstacle.scale;
    switch (obstacle.kind) {
    case ClutteredObstacle::kSphere:
        return Obstacle::sphere(obstacle.position, s.x());
    case ClutteredObstacle::kCylinder:
        return Obstacle::cylinder(obstacle.position, s.x(), s.z(), q);
    case ClutteredObstacle::kCapsule:
        return Obstacle::capsule(obstacle.position, s.x(), std::max(0.0, s.z() - 2.0 * s.x()), q);
    case ClutteredObstacle::kCube:
        return Obstacle::box(obstacle.position, s, q);
    case ClutteredObstacle::kHPolytope:
    case ClutteredObstacle::kVPolytope: {
        std::vector<Eigen::Vector3d> local = polytopeVertices(obstacle);
        for (auto& v : local) {
            v = v.cwiseProduct(s);
        }
        return Obstacle::convexHull(local, obstacle.position, q);
    }
    }
    throw std::invalid_argument("unknown obstacle kind");
}

// Support points of the obstacle's entry in a geometry library, at unit scale: the corners of the unit cube,
// `resolution` angles of two rings of the unit cylinder, the vertices of a polytope; none for the analytic sphere and
// capsule (and for a cylinder at resolution 0).
inline std::vector<Eigen::Vector3d> templateSupportPoints(const ClutteredObstacle& obstacle, int resolution) {
    std::vector<Eigen::Vector3d> points;
    switch (obstacle.kind) {
    case ClutteredObstacle::kSphere:
    case ClutteredObstacle::kCapsule:
        break;
    case ClutteredObstacle::kCylinder:
        for (int i = 0; i < std::max(0, resolution); ++i) {
            const double angle = 2.0 * detail::kPi * i / resolution;
            points.emplace_back(std::cos(angle), std::sin(angle), -0.5);
            points.emplace_back(std::cos(angle), std::sin(angle), 0.5);
        }
        break;
    case ClutteredObstacle::kCube:
        points = {Eigen::Vector3d(-0.5, -0.5, -0.5), Eigen::Vector3d(0.5, -0.5, -0.5), Eigen::Vector3d(0.5, 0.5, -0.5),
                  Eigen::Vector3d(-0.5, 0.5, -0.5),  Eigen::Vector3d(-0.5, -0.5, 0.5), Eigen::Vector3d(0.5, -0.5, 0.5),
                  Eigen::Vector3d(0.5, 0.5, 0.5),    Eigen::Vector3d(-0.5, 0.5, 0.5)};
        break;
    case ClutteredObstacle::kHPolytope:
    case ClutteredObstacle::kVPolytope:
        points = polytopeVertices(obstacle);
        break;
    }
    return points;
}

// The boundary of a polytope obstacle in its local frame (scaled), as an indexed closed triangle mesh with outward
// counter-clockwise triangles, for display or for a scene document.
inline ConvexMesh polytopeMesh(const ClutteredObstacle& obstacle) {
    std::vector<Eigen::Vector3d> vertices = polytopeVertices(obstacle);
    ConvexMesh mesh = convexMeshOf(vertices, obstacle.epsilon);
    for (auto& v : mesh.vertices) {
        v = v.cwiseProduct(obstacle.scale);
    }
    return mesh;
}

// A frame of four cubes (two posts, a top and a bottom edge) in the plane through `center` turned by `yaw` about +z,
// `width` wide between the post axes and `height` tall between the edge axes, all `rod_thickness` thick.
struct CubeFrame {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    double yaw = 0.0;
    double width = 2.0;
    double height = 2.0;
    double rod_thickness = 0.25;
};

// The cubes of a frame: left_side, right_side, top_edge and bottom_edge. Throws std::invalid_argument for a negative
// size that is not positive.
inline std::vector<ClutteredObstacle> expandCubeFrame(const CubeFrame& frame) {
    if (!(frame.width > 0.0 && frame.height > 0.0 && frame.rod_thickness > 0.0)) {
        throw std::invalid_argument("cube_frame width, height, and rod_thickness must be positive");
    }
    const Eigen::Quaterniond orientation(std::cos(0.5 * frame.yaw), 0.0, 0.0, std::sin(0.5 * frame.yaw));
    const double half_width = 0.5 * frame.width;
    const double half_height = 0.5 * frame.height;
    const Eigen::Vector3d lateral(std::cos(frame.yaw), std::sin(frame.yaw), 0.0);
    const auto cube = [&](const Eigen::Vector3d& position, const Eigen::Vector3d& scale) {
        ClutteredObstacle c;
        c.kind = ClutteredObstacle::kCube;
        c.position = position;
        c.orientation = orientation;
        c.scale = scale;
        return c;
    };
    const double t = frame.rod_thickness;
    return {cube(frame.center - half_width * lateral, Eigen::Vector3d(t, t, frame.height)),
            cube(frame.center + half_width * lateral, Eigen::Vector3d(t, t, frame.height)),
            cube(frame.center + Eigen::Vector3d(0.0, 0.0, half_height), Eigen::Vector3d(frame.width + t, t, t)),
            cube(frame.center - Eigen::Vector3d(0.0, 0.0, half_height), Eigen::Vector3d(frame.width + t, t, t))};
}

} // namespace xgc2_math::scene
