#pragma once

// Pure numerical shape + rigid pose conversion. IDs, sessions, template/asset resolution and
// source error annotation belong to the caller, before/after this boundary.
#include "xgc2_math/geometry/geometry_payload.hpp"
#include "xgc2_math/geometry/prepared_geometry.hpp"
#include "xgc2_math/geometry/se3.hpp"
#include "xgc2_math/scene/obstacle.hpp"
#include "xgc2_math/scene/shape_table.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace xgc2_math::scene {
struct ScenePartDescription {
    geometry::GeometryPayload geometry;
    Pose3 pose;
};
struct SceneObstacleDescription {
    Pose3 pose;
    std::vector<ScenePartDescription> parts;
};
namespace detail {
inline Eigen::Quaterniond unitOrientation(const Eigen::Quaterniond& q) {
    if (!q.coeffs().allFinite() || q.norm() < 1e-9)
        throw std::invalid_argument("invalid orientation");
    return normalizedQuaternion(q);
}
} // namespace detail
inline Obstacle obstacleFromGeometry(const geometry::GeometryPayload& payload,
                                     const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
                                     const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
    using geometry::GeometryType;
    const auto g = geometry::decodeGeometry(payload);
    const auto q = detail::unitOrientation(orientation);
    if (!position.allFinite())
        throw std::invalid_argument("invalid position");
    Obstacle o;
    switch (g.type) {
    case GeometryType::Box:
        o = Obstacle::box(position, {g.size[0], g.size[1], g.size[2]}, q);
        break;
    case GeometryType::Sphere:
        o = Obstacle::sphere(position, g.radius);
        break;
    case GeometryType::Cylinder:
        o = Obstacle::cylinder(position, g.radius, g.height, q);
        break;
    case GeometryType::Capsule:
        o = Obstacle::capsule(position, g.radius, g.height, q);
        break;
    case GeometryType::Ring:
        o = Obstacle::ring(position, {g.radii[0], g.radii[1]}, g.tubeRadius, q);
        break;
    case GeometryType::Convex:
        o = Obstacle::convexHull(geometry::preparation_detail::eigenVertices(g), position, q);
        break;
    case GeometryType::Halfspaces: {
        const auto parts = geometry::prepareConvexParts(payload);
        o = Obstacle::convexHull(geometry::preparation_detail::eigenVertices(parts.front().geometry), position, q);
        break;
    }
    case GeometryType::Ellipsoid:
        o = Obstacle::ellipsoid(position, {g.semiaxes[0], g.semiaxes[1], g.semiaxes[2]}, q);
        break;
    case GeometryType::Mesh:
        o = Obstacle::mesh(geometry::preparation_detail::eigenVertices(g), g.triangles, position, q);
        break;
    }
    validateObstacle(o);
    return o;
}
inline std::vector<Obstacle> toObstacles(const std::vector<SceneObstacleDescription>& shapes) {
    std::vector<Obstacle> out;
    for (const auto& shape : shapes)
        for (const auto& part : shape.parts) {
            const auto parent = detail::unitOrientation(shape.pose.orientation);
            out.push_back(obstacleFromGeometry(part.geometry, shape.pose.position + parent * part.pose.position,
                                               parent * detail::unitOrientation(part.pose.orientation)));
        }
    return out;
}
} // namespace xgc2_math::scene
