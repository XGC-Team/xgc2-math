#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "xgc2_math/geometry/occupied_sets/box_set.h"
#include "xgc2_math/geometry/occupied_sets/capsule_set.h"
#include "xgc2_math/geometry/occupied_sets/convex_set_base.h"
#include "xgc2_math/geometry/occupied_sets/cylinder_set.h"
#include "xgc2_math/geometry/occupied_sets/point_set.h"
#include "xgc2_math/geometry/occupied_sets/sphere_set.h"
#include "xgc2_math/scene/obstacle.hpp"
#include "xgc2_math/scene/polytope.hpp"
#include "xgc2_math/scene/shape_table.hpp"

namespace xgc2_math::scene {

// The obstacle in the world frame as a convex set with a support function, for the collision queries of the geometry
// component (GJK distance and separation, warm-started tracking). A convex obstacle becomes the set of the vertices of
// its hull. Throws std::invalid_argument for an obstacle that cannot be compiled.
inline std::shared_ptr<const ConvexSet3D> occupiedSet(const Obstacle& obstacle) {
    validateObstacle(obstacle);
    const Eigen::Quaterniond q = obstacle.orientation.normalized();
    switch (obstacle.type) {
    case Obstacle::kBox:
        return std::make_shared<BoxSet>(obstacle.position, obstacle.size, q);
    case Obstacle::kSphere:
        return std::make_shared<SphereSet>(obstacle.position, obstacle.radius);
    case Obstacle::kCylinder:
        return std::make_shared<CylinderSet>(obstacle.position, obstacle.radius, obstacle.height, q);
    case Obstacle::kCapsule:
        // The set takes the length of the whole capsule, caps included.
        return std::make_shared<CapsuleSet>(obstacle.position, obstacle.radius, obstacle.height + 2.0 * obstacle.radius,
                                            q);
    case Obstacle::kConvex: {
        std::vector<Eigen::Vector3d> offsets = buildPolytope(obstacle.vertices).vertices;
        std::transform(offsets.begin(), offsets.end(), offsets.begin(), [&q](const Eigen::Vector3d& vertex) {
            return Eigen::Vector3d(q * vertex);
        });
        return std::make_shared<SupportPointSet>(obstacle.position, std::move(offsets));
    }
    }
    throw std::invalid_argument("unknown obstacle type");
}

} // namespace xgc2_math::scene
