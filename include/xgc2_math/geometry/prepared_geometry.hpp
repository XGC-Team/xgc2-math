#pragma once

#include "xgc2_math/geometry/geo_utils.hpp"
#include "xgc2_math/geometry/geometry_payload.hpp"
#include "xgc2_math/geometry/occupied_sets/convex_set_base.h"
#include "xgc2_math/geometry/se3.hpp"
#include "xgc2_math/scene/bvh.hpp"
#include "xgc2_math/scene/ellipse_ring.hpp"
#include "xgc2_math/scene/polytope.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace xgc2_math::geometry {
struct PreparationOptions {
    std::size_t ringSegments = 128; // numerical resolution, never an author/wire field
    std::size_t maxHalfspaceRows = 128;
    std::size_t maxMeshPlanes = 128;
    std::size_t maxPieces = 65536;
};
struct ConvexPiece {
    Geometry geometry;
    Pose3 pose;                    // piece frame -> original geometry frame
    double approximationError = 0; // outward Euclidean distance bound, zero for exact pieces
};
namespace preparation_detail {
inline Eigen::Vector3d eigen(const GeometryVertex& v) {
    return {v[0], v[1], v[2]};
}
inline GeometryVertex array(const Eigen::Vector3d& v) {
    return {v.x(), v.y(), v.z()};
}
inline Geometry convexOf(const std::vector<Eigen::Vector3d>& vertices) {
    std::vector<GeometryVertex> out;
    out.reserve(vertices.size());
    std::transform(vertices.begin(), vertices.end(), std::back_inserter(out), array);
    return Geometry::convex(std::move(out));
}
inline std::vector<Eigen::Vector3d> eigenVertices(const Geometry& g) {
    std::vector<Eigen::Vector3d> out;
    out.reserve(g.vertices.size());
    std::transform(g.vertices.begin(), g.vertices.end(), std::back_inserter(out), eigen);
    return out;
}
inline Geometry finiteHalfspaces(const Geometry& g, const PreparationOptions& options) {
    if (g.planes.size() > options.maxHalfspaceRows)
        throw std::invalid_argument("halfspace preparation row limit exceeded");
    Eigen::MatrixX4d h(g.planes.size(), 4);
    for (std::size_t i = 0; i < g.planes.size(); ++i) {
        const auto& p = g.planes[i];
        const double scale = std::max({std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
        const double n = eigen({p[0] / scale, p[1] / scale, p[2] / scale}).norm();
        for (int j = 0; j < 4; ++j)
            h(static_cast<Eigen::Index>(i), j) = (p[static_cast<std::size_t>(j)] / scale) / n;
        if (!h.row(static_cast<Eigen::Index>(i)).allFinite())
            throw std::invalid_argument("halfspace range is not representable");
    }
    const Eigen::MatrixX3d a = h.leftCols<3>();
    const Eigen::VectorXd b = -h.rightCols<1>();
    for (int axis = 0; axis < 3; ++axis)
        for (double sign : {-1., 1.}) {
            Eigen::Vector3d c = Eigen::Vector3d::Zero(), x;
            c[axis] = sign;
            if (!std::isfinite(sdlp::linprog<3>(c, a, b, x)))
                throw std::invalid_argument("halfspace solid is empty or unbounded");
        }
    Eigen::Matrix3Xd vertices;
    if (!geo_utils::enumerateVs(h, vertices))
        throw std::invalid_argument("halfspace solid has no full-dimensional interior");
    std::vector<Eigen::Vector3d> v;
    v.reserve(static_cast<std::size_t>(vertices.cols()));
    for (Eigen::Index i = 0; i < vertices.cols(); ++i)
        v.push_back(vertices.col(i));
    return convexOf(scene::buildPolytope(v).vertices);
}
// Oriented manifold mesh: the winding integral classifies arrangement cells, including cavities.
inline bool insideMesh(const Geometry& g, const Eigen::Vector3d& p) {
    long double sum = 0;
    for (const auto& t : g.triangles) {
        const Eigen::Vector3d a = eigen(g.vertices[t[0]]) - p, b = eigen(g.vertices[t[1]]) - p,
                              c = eigen(g.vertices[t[2]]) - p;
        const double an = a.norm(), bn = b.norm(), cn = c.norm();
        sum += 2 * std::atan2(static_cast<long double>(a.dot(b.cross(c))),
                              static_cast<long double>(an) * bn * cn + static_cast<long double>(a.dot(b)) * cn +
                                  static_cast<long double>(b.dot(c)) * an + static_cast<long double>(c.dot(a)) * bn);
    }
    return std::abs(sum) > 2 * std::acos(-1.L);
}
inline std::vector<ConvexPiece> meshParts(const Geometry& g, const PreparationOptions& options) {
    std::map<std::pair<uint32_t, uint32_t>, std::pair<unsigned, int>> edges;
    std::vector<scene::Plane> planes;
    Eigen::Vector3d lo = eigen(g.vertices.front()), hi = lo;
    for (const auto& v : g.vertices) {
        lo = lo.cwiseMin(eigen(v));
        hi = hi.cwiseMax(eigen(v));
    }
    const double scale = (hi - lo).norm(), eps = 1e-9 * scale;
    if (!(scale > 0))
        throw std::invalid_argument("degenerate mesh");
    for (const auto& t : g.triangles) {
        for (std::size_t j = 0; j < 3; ++j) {
            const auto a = t[j], b = t[(j + 1) % 3];
            auto& edge = edges[{std::min(a, b), std::max(a, b)}];
            ++edge.first;
            edge.second += a < b ? 1 : -1;
        }
        Eigen::Vector3d n = (eigen(g.vertices[t[1]]) - eigen(g.vertices[t[0]]))
                                .cross(eigen(g.vertices[t[2]]) - eigen(g.vertices[t[0]]));
        if (n.norm() <= 1e-14 * scale * scale)
            throw std::invalid_argument("degenerate mesh triangle");
        n.normalize();
        for (int j = 0; j < 3; ++j)
            if (std::abs(n[j]) > 1e-12) {
                if (n[j] < 0)
                    n = -n;
                break;
            }
        scene::Plane p{n, n.dot(eigen(g.vertices[t[0]]))};
        if (std::none_of(planes.begin(), planes.end(), [&](const scene::Plane& q) {
                return n.dot(q.n) > 1 - 1e-12 && std::abs(p.d - q.d) <= eps;
            }))
            planes.push_back(p);
    }
    if (std::any_of(edges.begin(), edges.end(), [](const auto& e) {
            return e.second.first != 2 || e.second.second != 0;
        }))
        throw std::invalid_argument("mesh must be closed with consistently oriented manifold edges");
    if (planes.size() > options.maxMeshPlanes)
        throw std::invalid_argument("mesh decomposition plane limit exceeded");
    std::vector<Eigen::Vector3d> corners;
    for (unsigned i = 0; i < 8; ++i)
        corners.emplace_back((i & 1) ? hi.x() : lo.x(), (i & 2) ? hi.y() : lo.y(), (i & 4) ? hi.z() : lo.z());
    std::vector<std::vector<Eigen::Vector3d>> cells{std::move(corners)};
    // A boundary triangle lies in one of these planes, hence the solid indicator is constant
    // in every open arrangement cell. Clip convex cells and classify once, never hull the union.
    for (const auto& p : planes) {
        std::vector<std::vector<Eigen::Vector3d>> next;
        for (auto& cell : cells) {
            bool pos = false, neg = false;
            for (const auto& v : cell) {
                const double d = p.n.dot(v) - p.d;
                pos |= d > eps;
                neg |= d < -eps;
            }
            if (!(pos && neg)) {
                next.push_back(std::move(cell));
                continue;
            }
            std::vector<Eigen::Vector3d> a, b;
            for (const auto& v : cell) {
                const double d = p.n.dot(v) - p.d;
                if (d <= eps)
                    a.push_back(v);
                if (d >= -eps)
                    b.push_back(v);
            }
            for (std::size_t i = 0; i < cell.size(); ++i)
                for (std::size_t j = 0; j < i; ++j) {
                    const double di = p.n.dot(cell[i]) - p.d, dj = p.n.dot(cell[j]) - p.d;
                    if ((di > eps && dj < -eps) || (di < -eps && dj > eps)) {
                        const Eigen::Vector3d v = cell[i] + (cell[j] - cell[i]) * (di / (di - dj));
                        a.push_back(v);
                        b.push_back(v);
                    }
                }
            next.push_back(scene::buildPolytope(a).vertices);
            next.push_back(scene::buildPolytope(b).vertices);
            if (next.size() > options.maxPieces)
                throw std::invalid_argument("mesh decomposition cell limit exceeded");
        }
        cells = std::move(next);
    }
    std::vector<ConvexPiece> result;
    for (auto& cell : cells) {
        Eigen::Vector3d center = std::accumulate(cell.begin(), cell.end(), Eigen::Vector3d(Eigen::Vector3d::Zero()),
                                                 [](Eigen::Vector3d sum, const Eigen::Vector3d& v) {
                                                     sum += v;
                                                     return sum;
                                                 });
        center /= double(cell.size());
        if (insideMesh(g, center))
            result.push_back({convexOf(cell), {}, 0});
    }
    if (result.empty())
        throw std::invalid_argument("mesh has no occupied volume");
    return result;
}
} // namespace preparation_detail

inline std::vector<ConvexPiece> prepareConvexParts(const GeometryPayload& payload,
                                                   const PreparationOptions& options = {}) {
    Geometry g = decodeGeometry(payload);
    if (options.maxPieces == 0)
        throw std::invalid_argument("zero convex piece limit");
    if (g.type == GeometryType::Halfspaces)
        return {{preparation_detail::finiteHalfspaces(g, options), {}, 0}};
    if (g.type == GeometryType::Convex) {
        g = preparation_detail::convexOf(scene::buildPolytope(preparation_detail::eigenVertices(g)).vertices);
        return {{std::move(g), {}, 0}};
    }
    if (g.type == GeometryType::Mesh)
        return preparation_detail::meshParts(g, options);
    if (g.type != GeometryType::Ring)
        return {{std::move(g), {}, 0}};
    std::size_t n = options.ringSegments;
    if (n < 8 || n > options.maxPieces)
        throw std::invalid_argument("ring segments outside preparation bounds");
    const double pi = std::acos(-1.);
    const double hole = std::min(g.radii[0], g.radii[1]) - g.tubeRadius;
    const double required = pi * std::sqrt(std::max(g.radii[0], g.radii[1]) / hole);
    if (required >= double(n)) {
        if (!std::isfinite(required) || required >= double(options.maxPieces))
            throw std::invalid_argument("ring hole requires more parts than preparation limit");
        n = static_cast<std::size_t>(required) + 1;
    }
    const double h = 2 * pi / double(n);
    // Interpolation remainder ||C-C_chord|| <= max(a,b)*h^2/8 = delta.
    // Inflate each chord by r+delta: covers C+B_r and lies within its 2*delta offset.
    const double delta = std::max(g.radii[0], g.radii[1]) * h * h / 8;
    if (g.tubeRadius + 2 * delta >= std::min(g.radii[0], g.radii[1]))
        throw std::invalid_argument("ring resolution insufficient to preserve hole");
    std::vector<ConvexPiece> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double u = h * double(i), v = h * double(i + 1);
        const Eigen::Vector3d a(g.radii[0] * std::cos(u), g.radii[1] * std::sin(u), 0),
            b(g.radii[0] * std::cos(v), g.radii[1] * std::sin(v), 0);
        ConvexPiece p;
        p.geometry = Geometry::capsule(g.tubeRadius + delta, (b - a).norm());
        p.pose.position = (a + b) * .5;
        p.pose.orientation = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), b - a);
        p.approximationError = 2 * delta;
        out.push_back(std::move(p));
    }
    return out;
}

// Concrete local kernels selected once outside a support/GJK loop. All storage is immutable.
inline Eigen::Vector3d unitSupportDirection(const Eigen::Vector3d& d) {
    const double n2 = d.squaredNorm();
    if (n2 >= std::numeric_limits<double>::min() && std::isfinite(n2))
        return d / std::sqrt(n2);
    const double scale = d.cwiseAbs().maxCoeff();
    if (scale == 0)
        return Eigen::Vector3d::Zero();
    const Eigen::Vector3d scaled = d / scale;
    return scaled / scaled.norm();
}
struct LocalBox {
    Eigen::Vector3d half;
    Eigen::Vector3d center() const { return Eigen::Vector3d::Zero(); }
    Eigen::Vector3d supportPoint(const Eigen::Vector3d& d) const {
        return {(d.x() < 0 ? -1 : 1) * half.x(), (d.y() < 0 ? -1 : 1) * half.y(), (d.z() < 0 ? -1 : 1) * half.z()};
    }
};
struct LocalSphere {
    double radius;
    Eigen::Vector3d center() const { return Eigen::Vector3d::Zero(); }
    Eigen::Vector3d supportPoint(const Eigen::Vector3d& d) const { return radius * unitSupportDirection(d); }
};
struct LocalCylinder {
    double radius, halfHeight;
    Eigen::Vector3d center() const { return Eigen::Vector3d::Zero(); }
    Eigen::Vector3d supportPoint(const Eigen::Vector3d& d) const {
        const double n = std::hypot(d.x(), d.y());
        return {n > 0 ? radius * d.x() / n : radius, n > 0 ? radius * d.y() / n : 0, (d.z() < 0 ? -1 : 1) * halfHeight};
    }
};
struct LocalCapsule {
    double radius, halfHeight;
    Eigen::Vector3d center() const { return Eigen::Vector3d::Zero(); }
    Eigen::Vector3d supportPoint(const Eigen::Vector3d& d) const {
        Eigen::Vector3d p = LocalSphere{radius}.supportPoint(d);
        p.z() += (d.z() < 0 ? -1 : 1) * halfHeight;
        return p;
    }
};
struct LocalEllipsoid {
    Eigen::Vector3d axes;
    Eigen::Vector3d center() const { return Eigen::Vector3d::Zero(); }
    Eigen::Vector3d supportPoint(const Eigen::Vector3d& d) const {
        return axes.cwiseProduct(unitSupportDirection(axes.cwiseProduct(d)));
    }
};
struct LocalHull {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<double> x, y, z; // large hull: cold SoA, four independent exact dot-product lanes
    Eigen::Vector3d center() const { return Eigen::Vector3d::Zero(); } // same local anchor as PointSet
    Eigen::Vector3d supportPoint(const Eigen::Vector3d& d) const {
        if (x.empty()) {
            double best = -std::numeric_limits<double>::infinity();
            Eigen::Vector3d point = vertices.front();
            for (const auto& v : vertices) {
                const double value = v.dot(d);
                if (value > best) {
                    best = value;
                    point = v;
                }
            }
            return point;
        }
        Eigen::Array4d best = Eigen::Array4d::Constant(-std::numeric_limits<double>::infinity());
        Eigen::Array4i indices = Eigen::Array4i::Zero();
        std::size_t i = 0;
        for (; i + 4 <= vertices.size(); i += 4) {
            const Eigen::Array4d value = (Eigen::Map<const Eigen::Array4d>(x.data() + i) * d.x() +
                                          Eigen::Map<const Eigen::Array4d>(y.data() + i) * d.y()) +
                                         Eigen::Map<const Eigen::Array4d>(z.data() + i) * d.z();
            const auto greater = (value > best).eval();
            indices = greater.select(Eigen::Array4i(int(i), int(i + 1), int(i + 2), int(i + 3)), indices);
            best = greater.select(value, best);
        }
        int lane = 0;
        for (int k = 1; k < 4; ++k)
            if (best[k] > best[lane] || (best[k] == best[lane] && indices[k] < indices[lane]))
                lane = k;
        std::size_t winner = static_cast<std::size_t>(indices[lane]);
        double value = best[lane];
        for (; i < vertices.size(); ++i) {
            const double v = vertices[i].dot(d);
            if (v > value) {
                value = v;
                winner = i;
            }
        }
        return vertices[winner];
    }
};
using LocalConvex = std::variant<LocalBox, LocalSphere, LocalCylinder, LocalCapsule, LocalEllipsoid, LocalHull>;
struct CompiledConvexPiece {
    LocalConvex local;
    Pose3 pose;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    double approximationError = 0;
};
inline CompiledConvexPiece compileConvexPiece(const ConvexPiece& piece) {
    const auto& g = piece.geometry;
    validateGeometry(g);
    if (!isFinite(piece.pose) || piece.pose.orientation.norm() < 1e-12)
        throw std::invalid_argument("invalid convex piece pose");
    CompiledConvexPiece p;
    p.pose = piece.pose;
    p.rotation = normalizedQuaternion(piece.pose.orientation).toRotationMatrix();
    p.approximationError = piece.approximationError;
    switch (g.type) {
    case GeometryType::Box:
        p.local = LocalBox{preparation_detail::eigen(g.size) * .5};
        break;
    case GeometryType::Sphere:
        p.local = LocalSphere{g.radius};
        break;
    case GeometryType::Cylinder:
        p.local = LocalCylinder{g.radius, g.height * .5};
        break;
    case GeometryType::Capsule:
        p.local = LocalCapsule{g.radius, g.height * .5};
        break;
    case GeometryType::Ellipsoid:
        p.local = LocalEllipsoid{preparation_detail::eigen(g.semiaxes)};
        break;
    case GeometryType::Convex: {
        LocalHull h;
        h.vertices = preparation_detail::eigenVertices(g);
        if (h.vertices.size() >= 32) {
            h.x.reserve(h.vertices.size());
            h.y.reserve(h.vertices.size());
            h.z.reserve(h.vertices.size());
            for (const auto& v : h.vertices) {
                h.x.push_back(v.x());
                h.y.push_back(v.y());
                h.z.push_back(v.z());
            }
        }
        p.local = std::move(h);
        break;
    }
    default:
        throw std::invalid_argument("compileConvexPiece requires a prepared convex part");
    }
    return p;
}
template <class Local> class RigidConvexView final : public ConvexSet3D {
    const Local* local_;
    Eigen::Matrix3d rotation_;
    Eigen::Vector3d translation_;

  public:
    RigidConvexView(const Local& local, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation)
        : local_(&local), rotation_(rotation), translation_(translation) {}
    Eigen::Vector3d center() const override { return translation_ + rotation_ * local_->center(); }
    SupportQueryResult support(const Eigen::Vector3d& d) const override {
        if constexpr (std::is_same_v<Local, LocalSphere>)
            return makeSupportQueryResult(translation_ + local_->supportPoint(d), d);
        else
            return makeSupportQueryResult(translation_ + rotation_ * local_->supportPoint(rotation_.transpose() * d),
                                          d);
    }
    const Local& local() const { return *local_; }
};
// Cache this numerical transform at a state/prediction update, then reuse it across query batches.
struct ConvexInstanceTransform {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    ConvexInstanceTransform() = default;
    explicit ConvexInstanceTransform(const Pose3& pose) {
        if (!isFinite(pose) || pose.orientation.norm() < 1e-12)
            throw std::invalid_argument("invalid geometry instance pose");
        rotation = normalizedQuaternion(pose.orientation).toRotationMatrix();
        translation = pose.position;
    }
};
// Prepared at a numerical state update. The owner keeps the immutable piece alive.
struct ConvexPieceTransform {
    const CompiledConvexPiece* piece = nullptr;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};
inline ConvexPieceTransform prepareConvexPieceTransform(const CompiledConvexPiece& piece,
                                                        const ConvexInstanceTransform& instance) {
    return {&piece, instance.rotation * piece.rotation, instance.translation + instance.rotation * piece.pose.position};
}
// Hot callback receives a concrete set type; call existing DistanceGjkQuery::query here.
// This overload only reads cached R/T: no pose validation/normalization/composition.
template <class Fn> decltype(auto) visitConvexPiece(const ConvexPieceTransform& transformed, Fn&& fn) {
    return std::visit(
        [&](const auto& local) -> decltype(auto) {
            return fn(
                RigidConvexView<std::decay_t<decltype(local)>>(local, transformed.rotation, transformed.translation));
        },
        transformed.piece->local);
}
// Convenience at the update boundary; repeated queries should keep ConvexPieceTransform instead.
template <class Fn>
decltype(auto) visitConvexPiece(const CompiledConvexPiece& piece, const ConvexInstanceTransform& instance, Fn&& fn) {
    return visitConvexPiece(prepareConvexPieceTransform(piece, instance), std::forward<Fn>(fn));
}
template <class Fn> decltype(auto) visitConvexPiece(const CompiledConvexPiece& piece, const Pose3& instance, Fn&& fn) {
    return visitConvexPiece(piece, ConvexInstanceTransform(instance), std::forward<Fn>(fn));
}

struct PreparedGeometry {
    GeometryPayload original;
    Geometry geometry;
    std::vector<ConvexPiece> parts;
    std::vector<CompiledConvexPiece> compiledParts;
    scene::detail::ShapeTable partBounds;
    scene::detail::Bvh partIndex;
};
inline const std::vector<ConvexPiece>& convexPieces(const PreparedGeometry& g) {
    return g.parts;
}
inline PreparedGeometry prepareGeometry(const GeometryPayload& payload, const PreparationOptions& options = {}) {
    PreparedGeometry out;
    out.original = payload;
    out.geometry = decodeGeometry(payload);
    out.parts = prepareConvexParts(payload, options);
    out.compiledParts.reserve(out.parts.size());
    for (const auto& p : out.parts) {
        out.compiledParts.push_back(compileConvexPiece(p));
        const auto& g = p.geometry;
        scene::Obstacle o;
        switch (g.type) {
        case GeometryType::Box:
            o = scene::Obstacle::box(p.pose.position, preparation_detail::eigen(g.size), p.pose.orientation);
            break;
        case GeometryType::Sphere:
            o = scene::Obstacle::sphere(p.pose.position, g.radius);
            break;
        case GeometryType::Cylinder:
            o = scene::Obstacle::cylinder(p.pose.position, g.radius, g.height, p.pose.orientation);
            break;
        case GeometryType::Capsule:
            o = scene::Obstacle::capsule(p.pose.position, g.radius, g.height, p.pose.orientation);
            break;
        case GeometryType::Ellipsoid:
            o = scene::Obstacle::ellipsoid(p.pose.position, preparation_detail::eigen(g.semiaxes), p.pose.orientation);
            break;
        case GeometryType::Convex:
            o = scene::Obstacle::convexHull(preparation_detail::eigenVertices(g), p.pose.position, p.pose.orientation);
            break;
        default:
            throw std::invalid_argument("unexpected nonconvex prepared part");
        }
        out.partBounds.append(o);
    }
    out.partIndex.build(out.partBounds);
    return out;
}
// Existing scene BVH in the original local frame. Motion only transforms the query;
// this prunes pieces without rebuilding a tree or transforming/copying their vertices.
template <class Fn>
void forEachConvexPieceNear(const PreparedGeometry& g, const ConvexInstanceTransform& instance,
                            const Eigen::Vector3d& worldCenter, double radius, Fn&& fn) {
    if (!worldCenter.allFinite() || !std::isfinite(radius) || radius < 0)
        throw std::invalid_argument("invalid convex piece neighborhood");
    const Eigen::Vector3d local = instance.rotation.transpose() * (worldCenter - instance.translation);
    g.partIndex.forEachInSphere(g.partBounds, local, radius * radius, [&](std::size_t i) {
        fn(i, g.compiledParts[i]);
    });
}
template <class Fn>
void forEachConvexPieceNear(const PreparedGeometry& g, const Pose3& instance, const Eigen::Vector3d& worldCenter,
                            double radius, Fn&& fn) {
    return forEachConvexPieceNear(g, ConvexInstanceTransform(instance), worldCenter, radius, std::forward<Fn>(fn));
}
// Exact original ring queries: the decomposition is never substituted for these.
inline double ringPointResidual(const PreparedGeometry& g, const Eigen::Vector3d& point) {
    if (g.geometry.type != GeometryType::Ring)
        throw std::invalid_argument("ring query requires ring geometry");
    return scene::ellipticRingResidual(point, {g.geometry.radii[0], g.geometry.radii[1]}, g.geometry.tubeRadius);
}
// Owner/thread-local cold cache. Full payload equality defeats hash collisions; resolution is part of its key.
class PreparedGeometryCache {
    struct Entry {
        PreparationOptions options;
        std::shared_ptr<const PreparedGeometry> value;
    };
    std::vector<Entry> entries_;
    std::size_t capacity_;

  public:
    explicit PreparedGeometryCache(std::size_t capacity = 128) : capacity_(capacity) {
        if (!capacity)
            throw std::invalid_argument("zero geometry cache capacity");
    }
    std::shared_ptr<const PreparedGeometry> prepare(const GeometryPayload& p, const PreparationOptions& o = {}) {
        const auto entry = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) {
            return e.value->original == p && e.options.ringSegments == o.ringSegments &&
                   e.options.maxHalfspaceRows == o.maxHalfspaceRows && e.options.maxMeshPlanes == o.maxMeshPlanes &&
                   e.options.maxPieces == o.maxPieces;
        });
        if (entry != entries_.end())
            return entry->value;
        auto value = std::make_shared<const PreparedGeometry>(prepareGeometry(p, o));
        if (entries_.size() == capacity_)
            entries_.erase(entries_.begin());
        entries_.push_back({o, value});
        return value;
    }
    std::size_t size() const { return entries_.size(); }
    void clear() { entries_.clear(); }
};
} // namespace xgc2_math::geometry
