#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <vector>

namespace xgc2_math::scene {

// A closed convex solid. Every shape is defined in its local frame and placed by (position, orientation), local ->
// world. Conventions follow the scene geometry messages: box `size` is the full side length, cylinder `height`
// is the full height along local z, capsule `height` is the straight cylindrical part along local z, convex `vertices`
// are local-frame points whose convex hull is the solid.
struct Obstacle {
    enum Type { kBox, kSphere, kCylinder, kCapsule, kConvex };

    Type type = kBox;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d size = Eigen::Vector3d::Zero();
    double radius = 0.0;
    double height = 0.0;
    std::vector<Eigen::Vector3d> vertices;

    static Obstacle box(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
                        const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
        Obstacle o;
        o.type = kBox;
        o.position = center;
        o.orientation = orientation;
        o.size = size;
        return o;
    }

    static Obstacle alignedBox(const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner) {
        return box(0.5 * (min_corner + max_corner), max_corner - min_corner);
    }

    static Obstacle sphere(const Eigen::Vector3d& center, double radius) {
        Obstacle o;
        o.type = kSphere;
        o.position = center;
        o.radius = radius;
        return o;
    }

    static Obstacle cylinder(const Eigen::Vector3d& center, double radius, double height,
                             const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
        Obstacle o;
        o.type = kCylinder;
        o.position = center;
        o.orientation = orientation;
        o.radius = radius;
        o.height = height;
        return o;
    }

    static Obstacle capsule(const Eigen::Vector3d& center, double radius, double height,
                            const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
        Obstacle o = cylinder(center, radius, height, orientation);
        o.type = kCapsule;
        return o;
    }

    // Local-frame vertices placed by a pose; the default pose treats them as world-frame vertices.
    static Obstacle convexHull(const std::vector<Eigen::Vector3d>& local_vertices,
                               const Eigen::Vector3d& position = Eigen::Vector3d::Zero(),
                               const Eigen::Quaterniond& orientation = Eigen::Quaterniond::Identity()) {
        Obstacle o;
        o.type = kConvex;
        o.position = position;
        o.orientation = orientation;
        o.vertices = local_vertices;
        return o;
    }
};

} // namespace xgc2_math::scene
