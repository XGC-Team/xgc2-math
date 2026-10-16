#include "xgc2_math/geometry/collision/distance_gjk_query.h"
#include "xgc2_math/geometry/occupied_sets/sphere_set.h"
#include "xgc2_math/geometry/prepared_geometry.hpp"
#include "xgc2_math/scene/lidar_scene.hpp"
#include "xgc2_math/scene/scene_conversion.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <new>
#include <random>

static std::atomic<std::size_t> allocations{0};
static bool counting = false;
void* operator new(std::size_t n) {
    if (counting)
        ++allocations;
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    return ::operator new(n);
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete[](void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}
using namespace xgc2_math;
using namespace xgc2_math::geometry;
static void check(bool x, const char* text) {
    if (!x)
        throw std::runtime_error(text);
}
static void near(double a, double b, double eps, const char* text) {
    check(std::abs(a - b) <= eps, text);
}
template <class F> static void rejects(F f) {
    bool rejected = false;
    try {
        f();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "expected explicit refusal");
}
static Geometry squareFrame() {
    std::vector<GeometryVertex> v;
    const std::array<std::array<double, 2>, 4> corners{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};
    for (double z : {-.5, .5})
        for (double r : {2., 1.})
            std::transform(corners.begin(), corners.end(), std::back_inserter(v), [&](const auto& c) {
                return GeometryVertex{c[0] * r, c[1] * r, z};
            });
    std::vector<GeometryTriangle> t;
    const auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        t.push_back({a, b, c});
        t.push_back({a, c, d});
    };
    for (uint32_t i = 0; i < 4; ++i) {
        const auto j = (i + 1) % 4;
        quad(i, j, j + 8, i + 8);
        quad(i + 4, i + 12, j + 12, j + 4);
        quad(i + 8, j + 8, j + 12, i + 12);
        quad(i + 4, j + 4, j, i);
    }
    return Geometry::mesh(v, t);
}
static void supportAndGjk() {
    std::vector<Geometry> shapes{
        Geometry::box({2, 3, 4}),
        Geometry::sphere(1),
        Geometry::cylinder(1, 3),
        Geometry::capsule(1, 3),
        Geometry::ellipsoid({1, 2, 3}),
        Geometry::convex(
            {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}})};
    for (int n : {64, 256}) {
        std::vector<GeometryVertex> vertices;
        for (int i = 0; i < n; ++i) {
            const double z = 1 - 2 * (i + .5) / n, u = 2.39996322972865332 * i, r = std::sqrt(1 - z * z);
            vertices.push_back({r * std::cos(u), r * std::sin(u), z});
        }
        shapes.push_back(Geometry::convex(std::move(vertices)));
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u(-1, 1);
    for (const auto& shape : shapes) {
        auto prepared = prepareGeometry(encodeGeometry(shape));
        Pose3 pose;
        pose.position = {1e5, -3, 4};
        pose.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(.7, Eigen::Vector3d(1, 2, 3).normalized()));
        visitConvexPiece(prepared.compiledParts[0], pose, [&](const auto& set) {
            const auto* borrowed = &set.local();
            for (int i = 0; i < 1000; ++i) {
                const Eigen::Vector3d d(u(rng), u(rng), u(rng));
                const Eigen::Vector3d dl = pose.orientation.conjugate() * d;
                double expected = 0;
                switch (shape.type) {
                case GeometryType::Box:
                    expected = .5 * preparation_detail::eigen(shape.size).dot(dl.cwiseAbs());
                    break;
                case GeometryType::Sphere:
                    expected = shape.radius * dl.norm();
                    break;
                case GeometryType::Cylinder:
                    expected = shape.radius * std::hypot(dl.x(), dl.y()) + .5 * shape.height * std::abs(dl.z());
                    break;
                case GeometryType::Capsule:
                    expected = shape.radius * dl.norm() + .5 * shape.height * std::abs(dl.z());
                    break;
                case GeometryType::Ellipsoid:
                    expected = preparation_detail::eigen(shape.semiaxes).cwiseProduct(dl).norm();
                    break;
                default:
                    expected = -std::numeric_limits<double>::infinity();
                    for (auto v : shape.vertices)
                        expected = std::max(expected, preparation_detail::eigen(v).dot(dl));
                    break;
                }
                near(set.support(d).support_value, expected + pose.position.dot(d), 5e-10, "support oracle mismatch");
            }
            for (const Eigen::Vector3d& d :
                 std::array<Eigen::Vector3d, 3>{Eigen::Vector3d::Zero(), Eigen::Vector3d(1e-200, -2e-200, 3e-200),
                                                Eigen::Vector3d(1e150, -2e150, 3e150)}) {
                const auto result = set.support(d);
                check(result.support_point.allFinite() && std::isfinite(result.support_value),
                      "extreme/zero direction support");
                if (d.isZero(0))
                    near(result.support_value, 0, 0, "zero support value");
            }
            SphereSet b(pose.position + Eigen::Vector3d(10, 0, 0), .3);
            allocations = 0;
            counting = true;
            Eigen::internal::set_is_malloc_allowed(false);
            double sum = 0;
            for (int i = 0; i < 1000; ++i) {
                const auto r = gjk::DistanceGjkQuery::query(set, b, nullptr, .01, 128, 1e-10);
                sum += r.separator.margin;
            }
            Eigen::internal::set_is_malloc_allowed(true);
            counting = false;
            check(allocations == 0, "hot GJK allocated");
            check(&set.local() == borrowed, "shape copied");
            check(sum > 0, "GJK sum");
        });
    }
    auto compiled = prepareGeometry(encodeGeometry(Geometry::box({1, 2, 3})));
    ConvexPieceTransform view;
    SphereSet other({5, 0, 0}, .2);
    allocations = 0;
    counting = true;
    for (int i = 0; i < 1000; ++i) {
        Pose3 pose;
        pose.position = {i * .0001, 0, 0};
        view = prepareConvexPieceTransform(compiled.compiledParts[0], ConvexInstanceTransform(pose));
        visitConvexPiece(view, [&](const auto& set) {
            auto r = gjk::DistanceGjkQuery::query(set, other, nullptr, .01, 128, 1e-10);
            check(r.status == gjk::Result::Status::kSuccess, "moving cached query");
        });
    }
    counting = false;
    check(allocations == 0, "instance update/query allocated");
    auto flat = encodeGeometry(Geometry::convex({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}));
    rejects([&] {
        prepareGeometry(flat);
    });
    auto h = encodeGeometry(Geometry::halfspaces(
        {{1, 0, 0, -1}, {-1, 0, 0, -1}, {0, 1, 0, -1}, {0, -1, 0, -1}, {0, 0, 1, -1}, {0, 0, -1, -1}}));
    auto p = prepareGeometry(h);
    check(p.parts[0].geometry.vertices.size() == 8, "H-to-V cube");
    const auto unbounded = encodeGeometry(Geometry::halfspaces({{1, 0, 0, -1}}));
    rejects([&] {
        prepareGeometry(unbounded);
    });
    PreparedGeometryCache cache(2);
    auto a = cache.prepare(h);
    check(cache.prepare(h) == a, "same numeric geometry cache miss");
    auto b = cache.prepare(encodeGeometry(Geometry::sphere(1)));
    auto c = cache.prepare(encodeGeometry(Geometry::sphere(2)));
    check(b != c && cache.size() == 2, "bounded content cache");
    check(a->parts.size() == 1, "evicted live geometry invalidated");
}
static void ring() {
    auto prepared = prepareGeometry(encodeGeometry(Geometry::ring({3, 2}, .25)));
    check(prepared.original == encodeGeometry(Geometry::ring({3, 2}, .25)), "original ring changed");
    auto eccentric = prepareConvexParts(encodeGeometry(Geometry::ring({1000, .1}, .01)));
    check(eccentric.size() > 128 && eccentric[0].geometry.radius + eccentric[0].approximationError * .5 < .1,
          "adaptive resolution must preserve eccentric ring hole");
    const double error = prepared.parts[0].approximationError;
    std::size_t candidates = 0;
    forEachConvexPieceNear(prepared, ConvexInstanceTransform{}, Eigen::Vector3d(3, 0, 0), .4,
                           [&](std::size_t, const auto&) {
                               ++candidates;
                           });
    check(candidates > 0 && candidates < prepared.parts.size() / 4, "ring BVH failed to prune parts");
    double worst = 0, coverage = -1;
    for (int i = 0; i < 512; ++i) {
        const double u = 2 * std::acos(-1.) * i / 512;
        const Eigen::Vector3d center(3 * std::cos(u), 2 * std::sin(u), 0),
            normal = (Eigen::Vector3d(std::cos(u) / 3, std::sin(u) / 2, 0)).normalized();
        for (int j = 0; j < 16; ++j) {
            const double v = 2 * std::acos(-1.) * j / 16;
            const Eigen::Vector3d point =
                center + .25 * (normal * std::cos(v) + Eigen::Vector3d::UnitZ() * std::sin(v));
            double best = 1e9;
            for (const auto& p : prepared.parts) {
                const Eigen::Vector3d q = p.pose.orientation.conjugate() * (point - p.pose.position);
                const double z = std::clamp(q.z(), -.5 * p.geometry.height, .5 * p.geometry.height);
                best = std::min(best, (q - Eigen::Vector3d(0, 0, z)).norm() - p.geometry.radius);
            }
            coverage = std::max(coverage, best);
            check(best < 1e-12, "ring cover has a gap");
        }
    }
    for (const auto& p : prepared.parts)
        for (int j = 0; j < 16; ++j) {
            const double v = 2 * std::acos(-1.) * j / 16;
            const Eigen::Vector3d point =
                p.pose.position + p.pose.orientation * Eigen::Vector3d(p.geometry.radius * std::cos(v),
                                                                       p.geometry.radius * std::sin(v), 0);
            worst = std::max(worst, ringPointResidual(prepared, point));
        }
    check(worst <= error + 1e-10, "ring cover exceeds bound");
    near(ringPointResidual(prepared, {0, 0, 0}), 1.75, 1e-12, "ring hole filled");
    scene::LidarScene scene({scene::obstacleFromGeometry(prepared.original)}, .2, true);
    check(std::isinf(scene.closest({0, 0, 0}, Eigen::Vector3d::UnitZ(), 10)), "exact ring axial hole");
    near(scene.closest({0, 0, 0}, Eigen::Vector3d::UnitX(), 10), 2.75, 1e-10, "exact ring radial hit");
    std::cout << "ring pieces=" << prepared.parts.size() << " bound=" << error << " measured outward=" << worst
              << " maximum cover residual=" << coverage << "\n";
}
static void mesh() {
    const auto g = squareFrame();
    const auto payload = encodeGeometry(g);
    auto p = prepareGeometry(payload);
    check(p.parts.size() == 8, "square-frame exact arrangement should have eight cells");
    for (const auto& part : p.parts)
        check(part.approximationError == 0, "polyhedral decomposition approximated");
    for (int i = -19; i <= 19; ++i)
        for (int j = -19; j <= 19; ++j) {
            const Eigen::Vector3d point(i * .1, j * .1, .1);
            const bool expected = std::abs(point.x()) > 1 + 1e-8 || std::abs(point.y()) > 1 + 1e-8;
            if (std::abs(std::abs(point.x()) - 1) < 1e-8 || std::abs(std::abs(point.y()) - 1) < 1e-8)
                continue;
            bool inside = false;
            for (const auto& part : p.parts) {
                const auto hull = scene::buildPolytope(preparation_detail::eigenVertices(part.geometry));
                const bool yes = std::all_of(hull.planes.begin(), hull.planes.end(), [&](const auto& plane) {
                    return plane.n.dot(point) <= plane.d + 1e-9;
                });
                inside |= yes;
            }
            check(inside == expected, "polyhedral hole classification");
        }
    const auto o = scene::obstacleFromGeometry(payload);
    check(o.type == scene::Obstacle::kMesh && o.meshData->triangleIndices() == g.triangles, "mesh topology lost");
    scene::LidarScene scene({o}, .2, true);
    check(std::isinf(scene.closest({0, 0, -2}, Eigen::Vector3d::UnitZ(), 10)), "mesh hole filled");
    near(scene.closest({1.5, 0, -2}, Eigen::Vector3d::UnitZ(), 10), 1.5, 1e-10, "mesh top/bottom ray");
    near(scene.closest({0, 0, 0}, Eigen::Vector3d::UnitX(), 10), 1, 1e-10, "mesh inner wall ray");
    auto ellipsoid = scene::obstacleFromGeometry(encodeGeometry(Geometry::ellipsoid({1, 2, 3})));
    scene::LidarScene e({ellipsoid}, .2, true);
    near(e.closest({0, 0, 0}, Eigen::Vector3d::UnitY(), 10), 2, 1e-12, "ellipsoid analytic ray");
    std::cout << "mesh exact pieces=" << p.parts.size() << " hole samples and original topology queries pass\n";
}
int main() {
    supportAndGjk();
    ring();
    mesh();
    std::cout << "prepared geometry: support, original GJK, H bounds, cache, zero hot allocations pass\n";
}
