// Manual same-build comparison with existing occupied-set kernels. No host perf policy changes.
#include "xgc2_math/geometry/collision/distance_gjk_query.h"
#include "xgc2_math/geometry/occupied_sets/box_set.h"
#include "xgc2_math/geometry/occupied_sets/capsule_set.h"
#include "xgc2_math/geometry/occupied_sets/cylinder_set.h"
#include "xgc2_math/geometry/occupied_sets/ellipsoid_set.h"
#include "xgc2_math/geometry/occupied_sets/point_set.h"
#include "xgc2_math/geometry/occupied_sets/sphere_set.h"
#include "xgc2_math/geometry/prepared_geometry.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <memory>
#include <new>
#include <vector>

static std::size_t heapCalls = 0, heapBytes = 0, liveBytes = 0;
extern "C" void* __real_malloc(std::size_t);
extern "C" void __real_free(void*);
extern "C" void* __wrap_malloc(std::size_t n) {
    void* p = __real_malloc(n);
    if (p) {
        ++heapCalls;
        heapBytes += malloc_usable_size(p);
        liveBytes += malloc_usable_size(p);
    }
    return p;
}
extern "C" void __wrap_free(void* p) {
    if (p)
        liveBytes -= malloc_usable_size(p);
    __real_free(p);
}
void* operator new(std::size_t n) {
    if (void* p = __wrap_malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    return ::operator new(n);
}
void operator delete(void* p) noexcept {
    __wrap_free(p);
}
void operator delete[](void* p) noexcept {
    __wrap_free(p);
}
void operator delete(void* p, std::size_t) noexcept {
    __wrap_free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    __wrap_free(p);
}
using namespace xgc2_math;
using namespace xgc2_math::geometry;
using Clock = std::chrono::steady_clock;
static volatile double sink = 0;
static double ns(Clock::time_point t) {
    return std::chrono::duration<double, std::nano>(Clock::now() - t).count();
}
static std::shared_ptr<const ConvexSet3D> baseline(const Geometry& g, const Pose3& p) {
    switch (g.type) {
    case GeometryType::Box:
        return std::make_shared<BoxSet>(p.position, preparation_detail::eigen(g.size), p.orientation);
    case GeometryType::Sphere:
        return std::make_shared<SphereSet>(p.position, g.radius);
    case GeometryType::Cylinder:
        return std::make_shared<CylinderSet>(p.position, g.radius, g.height, p.orientation);
    case GeometryType::Capsule:
        return std::make_shared<CapsuleSet>(p.position, g.radius, g.height + 2 * g.radius, p.orientation);
    case GeometryType::Ellipsoid:
        return std::make_shared<EllipsoidSet>(p.position, preparation_detail::eigen(g.semiaxes), p.orientation);
    default: {
        auto v = preparation_detail::eigenVertices(g);
        std::transform(v.begin(), v.end(), v.begin(), [&](const auto& x) {
            return Eigen::Vector3d(p.orientation * x);
        });
        return std::make_shared<SupportPointSet>(p.position, std::move(v));
    }
    }
}
static double quantile(std::vector<double>& v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::size_t(q * static_cast<double>(v.size() - 1))];
}
template <class Set> struct Counted {
    const Set& set;
    std::size_t& calls;
    Eigen::Vector3d center() const { return set.center(); }
    SupportQueryResult support(const Eigen::Vector3d& d) const {
        ++calls;
        return set.support(d);
    }
};
struct Timing {
    double p50 = 0, p95 = 0, p99 = 0;
    std::size_t allocations = 0, bytes = 0, calls = 0, iterations = 0;
};
template <class Run> static Timing measure(std::size_t n, Run run) {
    std::vector<double> times(n);
    Timing out;
    const auto callsBefore = heapCalls, bytesBefore = heapBytes;
    for (std::size_t i = 0; i < n; ++i) {
        auto t = Clock::now();
        run(i, out);
        times[i] = ns(t);
    }
    out.allocations = heapCalls - callsBefore;
    out.bytes = heapBytes - bytesBefore;
    out.p50 = quantile(times, .50);
    out.p95 = quantile(times, .95);
    out.p99 = quantile(times, .99);
    return out;
}
static void row(const std::string& shape, const std::string& phase, const std::string& impl, const std::string& state,
                const std::string& temperature, std::size_t n, const Timing& t) {
    std::cout << shape << ',' << phase << ',' << impl << ',' << state << ',' << temperature << ',' << n << ',' << t.p50
              << ',' << t.p95 << ',' << t.p99 << ',' << t.calls << ',' << t.iterations << ',' << t.allocations << ','
              << t.bytes << '\n';
}
static Geometry sampledHull(std::size_t n) {
    std::vector<GeometryVertex> v;
    for (std::size_t i = 0; i < n; ++i) {
        const double z = 1 - 2 * (static_cast<double>(i) + .5) / static_cast<double>(n),
                     u = 2.39996322972865332 * static_cast<double>(i), r = std::sqrt(1 - z * z);
        v.push_back({r * std::cos(u), r * std::sin(u), z});
    }
    return Geometry::convex(v);
}
static void bench(const std::string& name, const Geometry& shape) {
    const auto payload = encodeGeometry(shape);
    const auto beforeCalls = heapCalls, beforeBytes = heapBytes, beforeLive = liveBytes;
    auto t = Clock::now();
    const auto prepared = prepareGeometry(payload);
    const auto prepareNs = ns(t);
    const auto prepCalls = heapCalls - beforeCalls, prepBytes = heapBytes - beforeBytes,
               retained = liveBytes - beforeLive;
    std::cout << "# prepare," << name << ",ns=" << prepareNs << ",allocations=" << prepCalls
              << ",allocated_bytes=" << prepBytes << ",retained_heap_bytes=" << retained
              << ",wire_bytes=" << payload.parameters.size() << ",parts=" << prepared.parts.size() << '\n';
    PreparedGeometryCache cache;
    const auto resident = cache.prepare(payload);
    auto hit = measure(1000, [&](std::size_t, Timing&) {
        auto g = cache.prepare(payload);
        sink += double(g->compiledParts.size());
    });
    row(name, "prepare_cache", "compiled", "static", "hot", 1000, hit);
    constexpr std::size_t n = 1500;
    std::vector<Pose3> poses(n);
    std::vector<ConvexPieceTransform> transforms(n);
    std::vector<std::shared_ptr<const ConvexSet3D>> old(n);
    std::array<Eigen::Vector3d, 64> dirs;
    for (std::size_t i = 0; i < dirs.size(); ++i)
        dirs[i] = Eigen::Vector3d(std::cos(.13 * static_cast<double>(i)), std::sin(.17 * static_cast<double>(i)),
                                  .2 + std::sin(.31 * static_cast<double>(i)));
    for (bool moving : {false, true}) {
        for (std::size_t i = 0; i < n; ++i) {
            const double u = moving ? double(i) / n : 0;
            poses[i].position = {u * .3, -u * .1, u * .05};
            poses[i].orientation =
                Eigen::Quaterniond(Eigen::AngleAxisd(.4 + u * .2, Eigen::Vector3d(1, 2, 3).normalized()));
        }
        auto update = measure(n, [&](std::size_t i, Timing&) {
            old[i] = baseline(prepared.parts[0].geometry, poses[i]);
        });
        row(name, "pose_update", "existing", moving ? "dynamic" : "static", "cold", n, update);
        auto viewUpdate = measure(n, [&](std::size_t i, Timing&) {
            transforms[i] = prepareConvexPieceTransform(prepared.compiledParts[0], ConvexInstanceTransform(poses[i]));
            sink += transforms[i].translation.x();
        });
        row(name, "pose_update", "compiled", moving ? "dynamic" : "static", "hot", n, viewUpdate);
        auto oldSupport = measure(n, [&](std::size_t i, Timing& stat) {
            for (const auto& d : dirs) {
                sink += old[i]->support(d).support_value;
                ++stat.calls;
            }
        });
        const auto supportBatch = [&](const auto& set, Timing& stat) {
            for (const auto& d : dirs) {
                sink += set.support(d).support_value;
                ++stat.calls;
            }
        };
        Timing newSupport;
        if (moving)
            newSupport = measure(n, [&](std::size_t i, Timing& stat) {
                visitConvexPiece(transforms[i], [&](const auto& set) {
                    supportBatch(set, stat);
                });
            });
        else
            newSupport = visitConvexPiece(transforms[0], [&](const auto& set) {
                return measure(n, [&](std::size_t, Timing& stat) {
                    supportBatch(set, stat);
                });
            });
        // Batch measurements divided by 64 yield per-support quantiles with amortized clock cost.
        for (auto* s : {&oldSupport, &newSupport}) {
            s->p50 /= static_cast<double>(dirs.size());
            s->p95 /= static_cast<double>(dirs.size());
            s->p99 /= static_cast<double>(dirs.size());
        }
        row(name, "support", "existing", moving ? "dynamic" : "static", "hot", n * dirs.size(), oldSupport);
        row(name, "support", "compiled", moving ? "dynamic" : "static", "hot", n * dirs.size(), newSupport);
        for (bool warm : {false, true}) {
            gjk::WarmStart wo, wn;
            SphereSet other({5, 1, .3}, .4);
            auto runOld = measure(n, [&](std::size_t i, Timing& stat) {
                Counted<ConvexSet3D> a{*old[i], stat.calls};
                Counted<SphereSet> b{other, stat.calls};
                const auto r = gjk::DistanceGjkQuery::query(a, b, warm ? &wo : nullptr, .01, 128, 1e-10);
                stat.iterations += static_cast<std::size_t>(r.iterations);
                sink += r.separator.margin;
                wo = {r.separator, r.status == gjk::Result::Status::kSuccess};
            });
            const auto queryNew = [&](const auto& set, Timing& stat) {
                Counted<std::decay_t<decltype(set)>> a{set, stat.calls};
                Counted<SphereSet> b{other, stat.calls};
                const auto r = gjk::DistanceGjkQuery::query(a, b, warm ? &wn : nullptr, .01, 128, 1e-10);
                stat.iterations += static_cast<std::size_t>(r.iterations);
                sink += r.separator.margin;
                wn = {r.separator, r.status == gjk::Result::Status::kSuccess};
            };
            Timing runNew;
            if (moving)
                runNew = measure(n, [&](std::size_t i, Timing& stat) {
                    visitConvexPiece(transforms[i], [&](const auto& set) {
                        queryNew(set, stat);
                    });
                });
            else
                runNew = visitConvexPiece(transforms[0], [&](const auto& set) {
                    return measure(n, [&](std::size_t, Timing& stat) {
                        queryNew(set, stat);
                    });
                });
            row(name, "gjk", "existing", moving ? "dynamic" : "static", warm ? "warm" : "cold", n, runOld);
            row(name, "gjk", "compiled", moving ? "dynamic" : "static", warm ? "warm" : "cold", n, runNew);
        }
    }
}
static void ringGjkBatch(const PreparedGeometry& ring) {
    constexpr std::size_t count = 300;
    const std::size_t parts = ring.compiledParts.size();
    std::vector<ConvexInstanceTransform> transforms(count);
    std::vector<ConvexPieceTransform> views(count * parts);
    std::vector<std::shared_ptr<const ConvexSet3D>> previous(count * parts);
    std::vector<SphereSet> targets;
    targets.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        Pose3 pose;
        pose.position = {double(i) * .001, 0, 0};
        pose.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(double(i) * .001, Eigen::Vector3d::UnitY()));
        transforms[i] = ConvexInstanceTransform(pose);
        const double u = double(i) * .013;
        const Eigen::Vector3d local(3 * std::cos(u), 2 * std::sin(u), 0),
            normal = Eigen::Vector3d(std::cos(u) / 3, std::sin(u) / 2, 0).normalized();
        targets.emplace_back(pose.position + pose.orientation * (local + .8 * normal), .2);
    }
    const auto update = measure(count, [&](std::size_t i, Timing&) {
        for (std::size_t k = 0; k < parts; ++k)
            views[i * parts + k] = prepareConvexPieceTransform(ring.compiledParts[k], transforms[i]);
    });
    row("ring128", "pose_update_128_parts", "compiled", "dynamic", "hot", count, update);
    const auto updateOld = measure(count, [&](std::size_t i, Timing&) {
        for (std::size_t k = 0; k < parts; ++k) {
            Pose3 pose;
            pose.position = views[i * parts + k].translation;
            pose.orientation = Eigen::Quaterniond(views[i * parts + k].rotation);
            previous[i * parts + k] = baseline(ring.parts[k].geometry, pose);
        }
    });
    row("ring128", "pose_update_128_parts", "existing", "dynamic", "cold", count, updateOld);
    for (bool warm : {false, true}) {
        std::vector<gjk::WarmStart> oldWarm(parts), newWarm(parts);
        const auto oldQuery = measure(count, [&](std::size_t i, Timing& stat) {
            forEachConvexPieceNear(ring, transforms[i], targets[i].center(), 1.1, [&](std::size_t k, const auto&) {
                Counted<ConvexSet3D> a{*previous[i * parts + k], stat.calls};
                Counted<SphereSet> b{targets[i], stat.calls};
                const auto r = gjk::DistanceGjkQuery::query(a, b, warm ? &oldWarm[k] : nullptr, .01, 128, 1e-10);
                stat.iterations += static_cast<std::size_t>(r.iterations);
                sink += r.separator.margin;
                oldWarm[k] = {r.separator, r.status == gjk::Result::Status::kSuccess};
            });
        });
        const auto newQuery = measure(count, [&](std::size_t i, Timing& stat) {
            forEachConvexPieceNear(ring, transforms[i], targets[i].center(), 1.1, [&](std::size_t k, const auto&) {
                visitConvexPiece(views[i * parts + k], [&](const auto& set) {
                    Counted<std::decay_t<decltype(set)>> a{set, stat.calls};
                    Counted<SphereSet> b{targets[i], stat.calls};
                    const auto r = gjk::DistanceGjkQuery::query(a, b, warm ? &newWarm[k] : nullptr, .01, 128, 1e-10);
                    stat.iterations += static_cast<std::size_t>(r.iterations);
                    sink += r.separator.margin;
                    newWarm[k] = {r.separator, r.status == gjk::Result::Status::kSuccess};
                });
            });
        });
        row("ring128", "gjk_candidate_batch", "existing", "dynamic", warm ? "warm" : "cold", count, oldQuery);
        row("ring128", "gjk_candidate_batch", "compiled", "dynamic", warm ? "warm" : "cold", count, newQuery);
    }
}
int main() {
    std::cout << std::fixed << std::setprecision(2)
              << "shape,phase,implementation,state,temperature,queries,p50_ns,p95_ns,p99_ns,support_calls,iterations,"
                 "heap_allocations,allocated_bytes\n";
    bench("box", Geometry::box({2, 3, 4}));
    bench("sphere", Geometry::sphere(1));
    bench("cylinder", Geometry::cylinder(1, 3));
    bench("capsule", Geometry::capsule(1, 3));
    bench("ellipsoid", Geometry::ellipsoid({1, 2, 3}));
    for (std::size_t n : std::array<std::size_t, 3>{8, 64, 256})
        bench("hull" + std::to_string(n), sampledHull(n));
    bench("halfspaces6",
          Geometry::halfspaces(
              {{1, 0, 0, -1}, {-1, 0, 0, -1}, {0, 1, 0, -1}, {0, -1, 0, -1}, {0, 0, 1, -1}, {0, 0, -1, -1}}));
    const auto start = Clock::now();
    const auto ring = prepareGeometry(encodeGeometry(Geometry::ring({3, 2}, .25)));
    const double prep = ns(start);
    ringGjkBatch(ring);
    std::size_t candidates = 0;
    const auto scan = measure(10000, [&](std::size_t i, Timing& stat) {
        const Eigen::Vector3d p(3 * std::cos(static_cast<double>(i) * .001),
                                2 * std::sin(static_cast<double>(i) * .001), 0);
        forEachConvexPieceNear(ring, ConvexInstanceTransform{}, p, .4, [&](std::size_t, const auto&) {
            ++candidates;
            ++stat.calls;
        });
    });
    row("ring128", "bvh_candidates", "compiled", "dynamic", "hot", 10000, scan);
    std::cout << "# ring,prepare_ns=" << prep << ",pieces=" << ring.parts.size()
              << ",outward_bound_m=" << std::setprecision(9) << ring.parts[0].approximationError
              << ",mean_candidates=" << double(candidates) / 10000 << ",sink=" << sink << '\n';
}
