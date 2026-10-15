#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xgc2_math/scene/ellipse_ring.hpp"
#include "xgc2_math/scene/obstacle.hpp"
#include "xgc2_math/scene/polytope.hpp"
#include "xgc2_math/scene/surface_sampling.hpp"

namespace xgc2_math::scene {
namespace detail {

// Axis-aligned box.
struct Aabb {
    Eigen::Vector3d lo = Eigen::Vector3d::Constant(kInf);
    Eigen::Vector3d hi = Eigen::Vector3d::Constant(-kInf);

    void grow(const Eigen::Vector3d& p) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    void grow(const Aabb& b) {
        lo = lo.cwiseMin(b.lo);
        hi = hi.cwiseMax(b.hi);
    }
    bool overlaps(const Aabb& b) const {
        return (lo.array() <= b.hi.array()).all() && (b.lo.array() <= hi.array()).all();
    }
    Eigen::Vector3d center() const { return 0.5 * (lo + hi); }
};

// Unit rotation matrix of a quaternion; throws std::invalid_argument for a (nearly) zero or non-finite quaternion.
inline Eigen::Matrix3d rotationOf(const Eigen::Quaterniond& q) {
    const double n = q.norm();
    if (!std::isfinite(n) || n < 1e-9) {
        throw std::invalid_argument("invalid orientation quaternion");
    }
    return Eigen::Quaterniond(q.coeffs() / n).toRotationMatrix();
}

// Interval of the line o + t d (t real) inside the slab |x| <= h along one axis, intersected with [*t0, *t1].
// Returns false when the line misses the slab.
inline bool slab(double o, double d, double h, double* t0, double* t1) {
    if (std::abs(d) < 1e-15) {
        if (std::abs(o) > h) {
            return false;
        }
        return true;
    }
    double a = (-h - o) / d;
    double b = (h - o) / d;
    if (a > b) {
        std::swap(a, b);
    }
    *t0 = std::max(*t0, a);
    *t1 = std::min(*t1, b);
    return *t0 <= *t1;
}

// Line o + t d against the ball |x - c| <= r, d unit.
inline bool sphereInterval(const Eigen::Vector3d& o, const Eigen::Vector3d& d, const Eigen::Vector3d& c, double r,
                           double* t0, double* t1) {
    const Eigen::Vector3d oc = o - c;
    const double b = oc.dot(d);
    const double disc = b * b - (oc.squaredNorm() - r * r);
    if (disc < 0.0) {
        return false;
    }
    const double s = std::sqrt(disc);
    *t0 = -b - s;
    *t1 = -b + s;
    return true;
}

// Line against the infinite z-cylinder x^2 + y^2 <= r^2 cut by the slab |z| <= hh, d unit.
inline bool cylinderInterval(const Eigen::Vector3d& o, const Eigen::Vector3d& d, double r, double hh, double* t0,
                             double* t1) {
    double lo = -kInf, hi = kInf;
    const double a = d.x() * d.x() + d.y() * d.y();
    const double c = o.x() * o.x() + o.y() * o.y() - r * r;
    if (a < 1e-15) {
        if (c > 0.0) {
            return false;
        }
    } else {
        const double b = o.x() * d.x() + o.y() * d.y();
        const double disc = b * b - a * c;
        if (disc < 0.0) {
            return false;
        }
        const double s = std::sqrt(disc);
        lo = (-b - s) / a;
        hi = (-b + s) / a;
    }
    if (!slab(o.z(), d.z(), hh, &lo, &hi)) {
        return false;
    }
    *t0 = lo;
    *t1 = hi;
    return true;
}

// Per-shape record that ray casting and containment tests read: one contiguous element per shape.
struct ShapeCore {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity(); // local -> world
    Eigen::Vector3d position = Eigen::Vector3d::Zero();     // world
    // box: half extents; sphere: radius in x; cylinder/capsule: radius and half height; ring: a,b,tube radius.
    Eigen::Vector3d params = Eigen::Vector3d::Zero();
    Obstacle::Type type = Obstacle::kBox;
    std::uint32_t plane_first = 0; // convex: planes [plane_first, plane_first + plane_count) of the table
    std::uint32_t plane_count = 0;
};

// The compiled shapes of a scene. Hot data (ShapeCore, planes, bounds) is kept in flat arrays indexed by shape;
// the polygon vertices of convex faces are only needed to sample the surface and are kept apart.
class ShapeTable {
  public:
    std::size_t size() const { return core_.size(); }
    const Aabb& bounds(std::size_t i) const { return bounds_[i]; }

    void reserve(std::size_t n) {
        core_.reserve(n);
        bounds_.reserve(n);
    }

    // Compiles `o` and appends it; throws std::invalid_argument for a degenerate or non-finite obstacle.
    void append(const Obstacle& o) {
        if (!o.position.allFinite()) {
            throw std::invalid_argument("non-finite obstacle position");
        }
        ShapeCore s;
        s.type = o.type;
        s.rotation = rotationOf(o.orientation);
        s.position = o.position;
        const auto positive = [](double x, const char* what) {
            if (!std::isfinite(x) || x <= 0.0) {
                throw std::invalid_argument(std::string("invalid ") + what);
            }
        };
        Aabb box;
        switch (o.type) {
        case Obstacle::kBox: {
            positive(o.size.x(), "box size");
            positive(o.size.y(), "box size");
            positive(o.size.z(), "box size");
            s.params = 0.5 * o.size;
            for (int corner = 0; corner < 8; ++corner) {
                box.grow(s.position + s.rotation * boxCorner(s.params, corner));
            }
            break;
        }
        case Obstacle::kSphere:
            positive(o.radius, "sphere radius");
            s.params = Eigen::Vector3d(o.radius, 0.0, 0.0);
            box.lo = s.position - Eigen::Vector3d::Constant(o.radius);
            box.hi = s.position + Eigen::Vector3d::Constant(o.radius);
            break;
        case Obstacle::kCylinder:
            positive(o.radius, "cylinder radius");
            positive(o.height, "cylinder height");
            s.params = Eigen::Vector3d(o.radius, 0.5 * o.height, 0.0);
            roundBounds(s, o.type, &box);
            break;
        case Obstacle::kCapsule:
            positive(o.radius, "capsule radius");
            if (!std::isfinite(o.height) || o.height < 0.0) {
                throw std::invalid_argument("invalid capsule height");
            }
            s.params = Eigen::Vector3d(o.radius, 0.5 * o.height, 0.0);
            roundBounds(s, o.type, &box);
            break;
        case Obstacle::kRing: {
            validateEllipticRing(o.radii, o.tubeRadius);
            s.params = Eigen::Vector3d(o.radii.x(), o.radii.y(), o.tubeRadius);
            Eigen::Vector3d extent;
            for (int k = 0; k < 3; ++k)
                extent[k] = std::hypot(o.radii.x() * s.rotation(k, 0), o.radii.y() * s.rotation(k, 1)) + o.tubeRadius;
            box.lo = s.position - extent;
            box.hi = s.position + extent;
            break;
        }
        case Obstacle::kConvex: {
            const Polytope hull = buildPolytope(o.vertices);
            s.plane_first = static_cast<std::uint32_t>(planes_.size());
            s.plane_count = static_cast<std::uint32_t>(hull.planes.size());
            for (std::size_t f = 0; f < hull.planes.size(); ++f) {
                planes_.push_back(hull.planes[f]);
                face_first_.push_back(face_vertices_.size());
                const auto first = hull.face_vertices.begin() + static_cast<std::ptrdiff_t>(hull.face_first[f]);
                const auto last = hull.face_vertices.begin() + static_cast<std::ptrdiff_t>(hull.face_first[f + 1]);
                face_vertices_.insert(face_vertices_.end(), first, last);
            }
            for (const auto& p : hull.vertices) {
                box.grow(s.position + s.rotation * p);
            }
            break;
        }
        default:
            throw std::invalid_argument("unknown obstacle type");
        }
        core_.push_back(s);
        bounds_.push_back(box);
    }

    // Appends shape `i` of `other` unchanged.
    void appendFrom(const ShapeTable& other, std::size_t i) {
        ShapeCore s = other.core_[i];
        if (s.type == Obstacle::kConvex) {
            const std::size_t first = s.plane_first;
            s.plane_first = static_cast<std::uint32_t>(planes_.size());
            for (std::uint32_t f = 0; f < s.plane_count; ++f) {
                planes_.push_back(other.planes_[first + f]);
                face_first_.push_back(face_vertices_.size());
                const auto begin =
                    other.face_vertices_.begin() + static_cast<std::ptrdiff_t>(other.face_first_[first + f]);
                const auto end = other.face_vertices_.begin() + static_cast<std::ptrdiff_t>(other.faceEnd(first + f));
                face_vertices_.insert(face_vertices_.end(), begin, end);
            }
        }
        core_.push_back(s);
        bounds_.push_back(other.bounds_[i]);
    }

    // Interval [*t0, *t1] of the world line o + t d (d unit) inside shape `i`; false when the line misses it.
    bool intersect(std::size_t i, const Eigen::Vector3d& o, const Eigen::Vector3d& d, double* t0, double* t1) const {
        const ShapeCore& s = core_[i];
        const Eigen::Vector3d ol = s.rotation.transpose() * (o - s.position);
        const Eigen::Vector3d dl = s.rotation.transpose() * d;
        switch (s.type) {
        case Obstacle::kRing:
            throw std::invalid_argument("ring is nonconvex; use closestSurface, not a single line interval");
        case Obstacle::kSphere:
            return sphereInterval(ol, dl, Eigen::Vector3d::Zero(), s.params.x(), t0, t1);
        case Obstacle::kCylinder:
            return cylinderInterval(ol, dl, s.params.x(), s.params.y(), t0, t1);
        case Obstacle::kCapsule: {
            // Convex union: the line meets it in one interval, the hull of the pieces' intervals.
            double lo = kInf, hi = -kInf, a, b;
            const double hh = s.params.y();
            if (cylinderInterval(ol, dl, s.params.x(), hh, &a, &b)) {
                lo = std::min(lo, a), hi = std::max(hi, b);
            }
            if (sphereInterval(ol, dl, Eigen::Vector3d(0, 0, hh), s.params.x(), &a, &b)) {
                lo = std::min(lo, a), hi = std::max(hi, b);
            }
            if (sphereInterval(ol, dl, Eigen::Vector3d(0, 0, -hh), s.params.x(), &a, &b)) {
                lo = std::min(lo, a), hi = std::max(hi, b);
            }
            if (lo > hi) {
                return false;
            }
            *t0 = lo;
            *t1 = hi;
            return true;
        }
        case Obstacle::kBox: {
            double lo = -kInf, hi = kInf;
            for (int k = 0; k < 3; ++k) {
                if (!slab(ol[k], dl[k], s.params[k], &lo, &hi)) {
                    return false;
                }
            }
            *t0 = lo;
            *t1 = hi;
            return true;
        }
        case Obstacle::kConvex: {
            double lo = -kInf, hi = kInf;
            const Plane* plane = planes_.data() + s.plane_first;
            for (std::uint32_t k = 0; k < s.plane_count; ++k, ++plane) {
                const double den = plane->n.dot(dl);
                const double num = plane->d - plane->n.dot(ol);
                if (std::abs(den) < 1e-15) {
                    if (num < 0.0) {
                        return false;
                    }
                    continue;
                }
                const double t = num / den;
                if (den < 0.0) {
                    lo = std::max(lo, t);
                } else {
                    hi = std::min(hi, t);
                }
                if (lo > hi) {
                    return false;
                }
            }
            *t0 = lo;
            *t1 = hi;
            return true;
        }
        }
        return false;
    }

    // Nearest forward boundary, including an inside origin's exit. Rings may
    // intersect a line in disjoint intervals and cannot use intersect().
    double closestSurface(std::size_t i, const Eigen::Vector3d& o, const Eigen::Vector3d& d, double maxRange) const {
        const auto& s = core_[i];
        if (s.type == Obstacle::kRing)
            return ellipticRingRayClosest(s.rotation.transpose() * (o - s.position), s.rotation.transpose() * d,
                                          s.params.head<2>(), s.params.z(), maxRange);
        double lo, hi;
        if (!intersect(i, o, d, &lo, &hi))
            return kInf;
        const double t = lo >= 0 ? lo : hi;
        return t >= 0 && t <= maxRange ? t : kInf;
    }

    // 0 on the surface of shape `i`, < 0 inside and > 0 outside; |value| is at most the distance to the surface.
    double residual(std::size_t i, const Eigen::Vector3d& world) const {
        const ShapeCore& s = core_[i];
        const Eigen::Vector3d q = s.rotation.transpose() * (world - s.position);
        switch (s.type) {
        case Obstacle::kRing:
            return ellipticRingResidual(q, s.params.head<2>(), s.params.z());
        case Obstacle::kSphere:
            return q.norm() - s.params.x();
        case Obstacle::kCylinder:
            return std::max(std::hypot(q.x(), q.y()) - s.params.x(), std::abs(q.z()) - s.params.y());
        case Obstacle::kCapsule: {
            const double z = std::clamp(q.z(), -s.params.y(), s.params.y());
            return (q - Eigen::Vector3d(0, 0, z)).norm() - s.params.x();
        }
        case Obstacle::kBox:
            return (q.cwiseAbs() - s.params).maxCoeff();
        case Obstacle::kConvex: {
            double r = -kInf;
            const Plane* plane = planes_.data() + s.plane_first;
            for (std::uint32_t k = 0; k < s.plane_count; ++k, ++plane) {
                r = std::max(r, plane->n.dot(q) - plane->d);
            }
            return r;
        }
        }
        return kInf;
    }

    // Appends the world-frame surface samples of shape `i` at about `spacing` metres (faces, edges and caps
    // included). `cell` is scratch space.
    void appendSamples(std::size_t i, double spacing, std::vector<Eigen::Vector3d>* out,
                       std::vector<Eigen::Vector2d>* cell) const {
        if (!std::isfinite(spacing) || spacing <= 0.0) {
            throw std::invalid_argument("invalid surface spacing");
        }
        const ShapeCore& s = core_[i];
        const std::size_t begin = out->size();
        switch (s.type) {
        case Obstacle::kRing: {
            const auto points = sampleEllipticRingSurface(s.params.head<2>(), s.params.z(), spacing);
            out->insert(out->end(), points.begin(), points.end());
            break;
        }
        case Obstacle::kBox:
            for (int axis = 0; axis < 3; ++axis) {
                for (int sign = -1; sign <= 1; sign += 2) {
                    std::array<Eigen::Vector3d, 4> face;
                    boxFace(s.params, axis, sign, &face);
                    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                    normal[axis] = sign;
                    appendPolygon(face.data(), face.size(), normal, spacing, cell, out);
                }
            }
            break;
        case Obstacle::kConvex:
            for (std::uint32_t f = 0; f < s.plane_count; ++f) {
                const std::size_t plane = s.plane_first + f;
                appendPolygon(face_vertices_.data() + face_first_[plane], faceEnd(plane) - face_first_[plane],
                              planes_[plane].n, spacing, cell, out);
            }
            break;
        case Obstacle::kSphere: {
            const int n = sphereSampleCount(s.params.x(), spacing);
            for (int k = 0; k < n; ++k) {
                out->push_back(s.params.x() * fibonacciPoint(k, n));
            }
            break;
        }
        case Obstacle::kCylinder:
        case Obstacle::kCapsule: {
            const double hh = s.params.y();
            const double radius = s.params.x();
            const int rings = std::max(1, static_cast<int>(std::ceil(2.0 * hh / spacing)));
            if (hh > 0.0) {
                for (int k = 0; k <= rings; ++k) {
                    appendRing(radius, -hh + 2.0 * hh * k / rings, spacing, out);
                }
            }
            if (s.type == Obstacle::kCylinder) {
                appendDisc(radius, hh, spacing, out);
                appendDisc(radius, -hh, spacing, out);
            } else {
                const int n = sphereSampleCount(radius, spacing);
                for (int k = 0; k < n; ++k) {
                    const Eigen::Vector3d p = fibonacciPoint(k, n);
                    out->push_back(radius * p + Eigen::Vector3d(0, 0, p.z() >= 0.0 ? hh : -hh));
                }
            }
            break;
        }
        }
        for (std::size_t k = begin; k < out->size(); ++k) {
            (*out)[k] = s.position + s.rotation * (*out)[k];
        }
    }

  private:
    // Corner `corner` (bit 0: x, bit 1: y, bit 2: z; set means positive) of a box with half extents `h`.
    static Eigen::Vector3d boxCorner(const Eigen::Vector3d& h, int corner) {
        return Eigen::Vector3d((corner & 1) ? h.x() : -h.x(), (corner & 2) ? h.y() : -h.y(),
                               (corner & 4) ? h.z() : -h.z());
    }

    // The face of a box on `axis` at side `sign` (-1, +1), corners in the order that samples are generated.
    static void boxFace(const Eigen::Vector3d& h, int axis, int sign, std::array<Eigen::Vector3d, 4>* face) {
        const int a = (axis + 1) % 3, b = (axis + 2) % 3;
        const std::array<std::array<int, 2>, 4> corners{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};
        for (std::size_t k = 0; k < corners.size(); ++k) {
            Eigen::Vector3d p;
            p[axis] = sign * h[axis];
            p[a] = corners[k][0] * h[a];
            p[b] = corners[k][1] * h[b];
            (*face)[k] = p;
        }
    }

    // World bounds of a cylinder or capsule from the extent of its axis.
    static void roundBounds(const ShapeCore& s, Obstacle::Type type, Aabb* box) {
        const Eigen::Vector3d axis = s.rotation.col(2);
        Eigen::Vector3d ext;
        for (int k = 0; k < 3; ++k) {
            const double a = std::abs(axis[k]);
            ext[k] = s.params.y() * a +
                     (type == Obstacle::kCapsule ? s.params.x() : s.params.x() * std::sqrt(std::max(0.0, 1.0 - a * a)));
        }
        box->lo = s.position - ext;
        box->hi = s.position + ext;
    }

    // One past the last face vertex of face `plane` (global plane index).
    std::size_t faceEnd(std::size_t plane) const {
        return plane + 1 < face_first_.size() ? face_first_[plane + 1] : face_vertices_.size();
    }

    std::vector<ShapeCore> core_;
    std::vector<Plane> planes_;                  // convex faces of every shape, shape by shape
    std::vector<std::size_t> face_first_;        // face of planes_[k]: face_vertices_[face_first_[k], faceEnd(k))
    std::vector<Eigen::Vector3d> face_vertices_; // face polygons for sampling
    std::vector<Aabb> bounds_;
};

} // namespace detail

// Checks that an obstacle can be compiled: finite pose and a valid size, radius and height (and, for a convex
// solid, at least four distinct, non-flat vertices). Throws std::invalid_argument naming the problem.
inline void validateObstacle(const Obstacle& obstacle) {
    detail::ShapeTable table;
    table.append(obstacle);
}

// Surface samples of one obstacle in the world frame, at roughly `spacing` metres (faces, edges and caps included).
// Throws std::invalid_argument on a degenerate obstacle or a spacing that is not finite and positive.
inline std::vector<Eigen::Vector3d> sampleSurface(const Obstacle& obstacle, double spacing) {
    detail::ShapeTable table;
    table.append(obstacle);
    std::vector<Eigen::Vector3d> out;
    std::vector<Eigen::Vector2d> cell;
    table.appendSamples(0, spacing, &out, &cell);
    return out;
}

} // namespace xgc2_math::scene
