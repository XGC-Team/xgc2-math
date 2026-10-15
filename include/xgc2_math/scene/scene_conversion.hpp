#pragma once

// Scene descriptions -> Obstacle, without ROS.
//
// A scene document (xgc2.scene.v1) lists obstacles that each hold parts; the Gazebo simulator publishes the obstacle
// truth as a geometry library plus convex body instances. A ROS node or a service copies each message field into the
// description structs below; the rules here are the whole conversion and are unit-tested without ROS.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xgc2_math/geometry/se3.hpp"
#include "xgc2_math/scene/obstacle.hpp"
#include "xgc2_math/scene/shape_table.hpp"

namespace xgc2_math::scene {

// One part of a scene document obstacle. A part's world pose is obstacle_pose * part_pose; box `size` is the full side
// length; cylinder `height` is the full height along local z; capsule `height` is the straight part; convex `vertices`
// are part-frame points (triangles are not needed, the solid is their convex hull).
struct ScenePartDescription {
    std::string type; // box | sphere | cylinder | capsule | convex | ring
    Pose3 pose;       // part frame in the obstacle frame
    Eigen::Vector3d size = Eigen::Vector3d::Zero();
    double radius = 0.0;
    double height = 0.0;
    Eigen::Vector2d radii = Eigen::Vector2d::Zero();
    double tubeRadius = 0.0;
    std::vector<Eigen::Vector3d> vertices;
};

struct SceneObstacleDescription {
    std::string id;
    Pose3 pose; // obstacle frame in the world (already the current state pose for a mover)
    std::vector<ScenePartDescription> parts;
};

namespace detail {

inline Eigen::Quaterniond unitOrientation(const Eigen::Quaterniond& q) {
    const double n = q.norm();
    if (!std::isfinite(n) || n < 1e-9) {
        throw std::invalid_argument("invalid orientation");
    }
    return Eigen::Quaterniond(q.coeffs() / n);
}

inline Obstacle partObstacle(const ScenePartDescription& part, const Eigen::Vector3d& position,
                             const Eigen::Quaterniond& orientation) {
    if (part.type == "box") {
        return Obstacle::box(position, part.size, orientation);
    }
    if (part.type == "sphere") {
        return Obstacle::sphere(position, part.radius);
    }
    if (part.type == "cylinder") {
        return Obstacle::cylinder(position, part.radius, part.height, orientation);
    }
    if (part.type == "capsule") {
        return Obstacle::capsule(position, part.radius, part.height, orientation);
    }
    if (part.type == "ring") {
        return Obstacle::ring(position, part.radii, part.tubeRadius, orientation);
    }
    if (part.type == "convex") {
        return Obstacle::convexHull(part.vertices, position, orientation);
    }
    throw std::invalid_argument("unknown geometry type '" + part.type + "'");
}

} // namespace detail

// One Obstacle per part. Throws std::invalid_argument naming the obstacle and part on an unknown type or invalid
// geometry.
inline std::vector<Obstacle> toObstacles(const std::vector<SceneObstacleDescription>& scene) {
    std::vector<Obstacle> out;
    for (const auto& obstacle : scene) {
        for (std::size_t i = 0; i < obstacle.parts.size(); ++i) {
            const auto& part = obstacle.parts[i];
            try {
                const Eigen::Quaterniond parent = detail::unitOrientation(obstacle.pose.orientation);
                const Eigen::Vector3d position = obstacle.pose.position + parent * part.pose.position;
                const Eigen::Quaterniond orientation = parent * detail::unitOrientation(part.pose.orientation);
                Obstacle o = detail::partObstacle(part, position, orientation);
                validateObstacle(o);
                out.push_back(std::move(o));
            } catch (const std::invalid_argument& e) {
                throw std::invalid_argument("obstacle '" + obstacle.id + "' part " + std::to_string(i) + ": " +
                                            e.what());
            }
        }
    }
    return out;
}

// The Gazebo simulator's obstacle truth (/xgc2/simulation/obstacles/geometry_library and .../instances, published by
// the xgc2_gazebo_scene system plugin from the actual collision geometry of every managed obstacle). Each instance is
// one convex part in the world frame:
//
//   geometry_type "cube"      box, `scale` = full side lengths
//   geometry_type "sphere"    sphere, radius = scale.x
//   geometry_type "cylinder"  cylinder along local z, radius = scale.x, full height = scale.z
//   geometry_type "capsule"   capsule along local z, radius = scale.x, full length (caps included) = scale.z
//   any other type            the library template of that type; its support points, multiplied component-wise by
//                             `scale`, are local-frame vertices whose convex hull is the solid (collision meshes arrive
//                             this way, already reduced to their convex vertices)
//
// `pose` places the part: local -> world.
struct GeometryTemplateDescription {
    std::string type;
    std::vector<Eigen::Vector3d> support_points;
};

struct ConvexBodyDescription {
    std::string name;
    std::string geometry_type;
    Pose3 pose; // world
    Eigen::Vector3d scale = Eigen::Vector3d::Ones();
};

// Throws std::invalid_argument naming the instance on a missing template or invalid geometry.
inline std::vector<Obstacle> fromConvexBodies(const std::vector<GeometryTemplateDescription>& library,
                                              const std::vector<ConvexBodyDescription>& bodies) {
    std::map<std::string, const GeometryTemplateDescription*> templates;
    for (const auto& t : library) {
        templates[t.type] = &t;
    }
    std::vector<Obstacle> out;
    out.reserve(bodies.size());
    for (const auto& body : bodies) {
        try {
            if (!body.pose.position.allFinite()) {
                throw std::invalid_argument("invalid position");
            }
            const Eigen::Quaterniond q = detail::unitOrientation(body.pose.orientation);
            const Eigen::Vector3d& p = body.pose.position;
            const Eigen::Vector3d& scale = body.scale;
            if (!(scale.allFinite() && scale.x() > 0.0 && scale.y() > 0.0 && scale.z() > 0.0)) {
                throw std::invalid_argument("scale must be positive and finite");
            }
            Obstacle o;
            if (body.geometry_type == "cube") {
                o = Obstacle::box(p, scale, q);
            } else if (body.geometry_type == "sphere") {
                o = Obstacle::sphere(p, scale.x());
            } else if (body.geometry_type == "cylinder") {
                o = Obstacle::cylinder(p, scale.x(), scale.z(), q);
            } else if (body.geometry_type == "capsule") {
                o = Obstacle::capsule(p, scale.x(), std::max(0.0, scale.z() - 2.0 * scale.x()), q);
            } else {
                const auto found = templates.find(body.geometry_type);
                if (found == templates.end()) {
                    throw std::invalid_argument("no geometry template '" + body.geometry_type + "'");
                }
                const std::vector<Eigen::Vector3d>& support = found->second->support_points;
                std::vector<Eigen::Vector3d> local;
                local.reserve(support.size());
                std::transform(support.begin(), support.end(), std::back_inserter(local),
                               [&scale](const Eigen::Vector3d& s) {
                                   return Eigen::Vector3d(s.cwiseProduct(scale));
                               });
                o = Obstacle::convexHull(local, p, q);
            }
            validateObstacle(o);
            out.push_back(std::move(o));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("convex body '" + body.name + "': " + e.what());
        }
    }
    return out;
}

} // namespace xgc2_math::scene
