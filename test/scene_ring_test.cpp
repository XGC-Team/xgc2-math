#include "xgc2_math/scene.hpp"
#include "xgc2_math/scene/ellipse_ring.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace xgc2_math::scene;
namespace {
constexpr double pi = 3.14159265358979323846;
void check(bool ok, const char* what) {
    if (!ok)
        throw std::runtime_error(what);
}
void near(double a, double b, double eps, const char* what) {
    if (!std::isfinite(a) || std::abs(a - b) > eps) {
        std::cerr << what << " actual " << a << " expected " << b << '\n';
        throw std::runtime_error(what);
    }
}
// Independent parameter minimization: find every sampled local minimum, then
// golden-section refine in angle. No Lagrange multiplier or production helper.
double oracleDistance(const Eigen::Vector3d& p, const Eigen::Vector2d& ab) {
    const auto f = [&](double u) {
        return (p - Eigen::Vector3d(ab.x() * std::cos(u), ab.y() * std::sin(u), 0)).squaredNorm();
    };
    constexpr int n = 4096;
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const double u = 2 * pi * i / n, step = 2 * pi / n;
        if (f(u) > f(u - step) || f(u) > f(u + step))
            continue;
        double lo = u - step, hi = u + step;
        for (int k = 0; k < 75; ++k) {
            const double x = lo + (hi - lo) * 0.3819660112501051, y = lo + (hi - lo) * 0.6180339887498949;
            if (f(x) < f(y))
                hi = y;
            else
                lo = x;
        }
        best = std::min(best, f((lo + hi) / 2));
    }
    return std::sqrt(best);
}
void distanceTests() {
    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> unit(-1, 1);
    for (const Eigen::Vector2d& ab : {Eigen::Vector2d(1, 1), {1, 1.2}, {2, 1}, {1, 8}, {100, .01}}) {
        for (int i = 0; i < 80; ++i) {
            Eigen::Vector3d p(unit(rng) * ab.x() * 2, unit(rng) * ab.y() * 2, unit(rng));
            near(ellipseCenterlineDistance(p, ab), oracleDistance(p, ab), 2e-9 * ab.maxCoeff(), "distance oracle");
        }
        for (const Eigen::Vector3d& p : {Eigen::Vector3d::Zero().eval(),
                                         Eigen::Vector3d(.5 * ab.x(), 0, 0),
                                         {0, .5 * ab.y(), 0},
                                         {1e-14, 1e-14, 0}})
            near(ellipseCenterlineDistance(p, ab), oracleDistance(p, ab), 2e-9 * ab.maxCoeff(), "axis distance");
    }
}
void rayTests() {
    const Eigen::Vector2d circle(1, 1);
    const auto ray = [&](Eigen::Vector3d o, Eigen::Vector3d d, double range = 10) {
        return ellipticRingRayClosest(o, d, circle, .1, range);
    };
    near(ray({0, 0, 0}, {1, 0, 0}), .9, 1e-11, "hole nearest entry");
    near(ray({-2, 0, 0}, {1, 0, 0}), .9, 1e-11, "outer nearest entry");
    near(ray({1, 0, 0}, {1, 0, 0}), .1, 1e-11, "inside exit");
    near(ray({1, 0, 0}, {-1, 0, 0}), .1, 1e-11, "inside inward exit before hole");
    near(ray({.9, 0, 0}, {1, 0, 0}), 0, 1e-11, "boundary origin");
    check(std::isinf(ray({0, 0, 0}, {0, 0, 1})), "hole axial miss");
    check(std::isinf(ray({0, 0, 0}, {1, 0, 0}, .89)), "range excludes entry");
    near(ray({0, 0, 0}, {1, 0, 0}, .9), .9, 1e-11, "range includes entry");
    near(ray({0, 0, 0}, {1, 0, 0}, std::numeric_limits<double>::infinity()), .9, 1e-11, "unbounded range");
    near(ray({-2, 1.1, 0}, {1, 0, 0}), 2, 2e-7, "outer tangent");
    near(ray({1, 0, -2}, {0, 0, 1}), 1.9, 1e-11, "normal tube hit");
    // General ellipses: a dense union of actual balls is an independent inner
    // approximation. It cannot invent a hit before the continuum union.
    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> unit(-1, 1);
    for (const Eigen::Vector2d& ab : {Eigen::Vector2d(1, 1.2), {2, 1}})
        for (int i = 0; i < 120; ++i) {
            const double radius = ab.x() == 2 ? .75 : .05;
            const Eigen::Vector3d o(3 * unit(rng), 3 * unit(rng), unit(rng));
            const Eigen::Vector3d d = Eigen::Vector3d(unit(rng), unit(rng), unit(rng)).normalized();
            const double actual = ellipticRingRayClosest(o, d, ab, radius, 10);
            // Oracle scans the signed distance along the finite ray and bisects
            // every sign change, independently of the algebraic ray solver.
            double previous = oracleDistance(o, ab) - radius, expected = INFINITY;
            if (std::abs(previous) < 1e-10)
                expected = 0;
            for (int j = 1; j <= 600 && !std::isfinite(expected); ++j) {
                const double t = j * .01, now = ellipseCenterlineDistance(o + t * d, ab) - radius;
                if ((now < 0) != (previous < 0)) {
                    double lo = t - .01, hi = t;
                    const bool sign = previous < 0;
                    for (int k = 0; k < 45; ++k) {
                        double mid = (lo + hi) / 2;
                        if ((oracleDistance(o + mid * d, ab) - radius < 0) == sign)
                            lo = mid;
                        else
                            hi = mid;
                    }
                    expected = (lo + hi) / 2;
                }
                previous = now;
            }
            if (std::isfinite(expected))
                near(actual, expected, 2e-8, "ray distance-crossing oracle");
            else if (std::isfinite(actual) && actual <= 6)
                near(oracleDistance(o + actual * d, ab), radius, 2e-8, "ray residual");
        }
}
void samplingTests() {
    double worst = 0;
    for (const Eigen::Vector2d& ab : {Eigen::Vector2d(1, 1.2), {2, 1}, {1, 2}}) {
        const double r = ab.maxCoeff() == 2 ? .75 : .05;
        const auto points = sampleEllipticRingSurface(ab, r, .15);
        for (size_t i = 0; i < points.size(); i += std::max(size_t(1), points.size() / 200)) {
            worst = std::max(worst, std::abs(oracleDistance(points[i], ab) - r));
            near(oracleDistance(points[i], ab), r, 1e-9, "true exposed surface including r >= reach");
        }
        check(!points.empty(), "samples nonempty");
    }
    std::cout << "sample independent residual max=" << worst << '\n';
}
void sceneTests() {
    const Eigen::Quaterniond rotation =
        Eigen::Quaterniond(Eigen::AngleAxisd(.7, Eigen::Vector3d(1, 2, 3).normalized()));
    const Eigen::Vector3d center(3, 4, 5);
    auto ring = Obstacle::ring(center, {1, 1.2}, .05, rotation);
    LidarScene scene({ring}, .2, true);
    near(scene.closest(center, rotation * Eigen::Vector3d::UnitX(), 10), .95, 1e-10, "rotated hole ray");
    check(std::isinf(scene.closest(center, rotation * Eigen::Vector3d::UnitZ(), 10)), "rotated hole axial miss");
    for (int field = 0; field < 3; ++field) {
        auto changed = ring;
        if (field < 2)
            changed.radii[field] += .1;
        else
            changed.tubeRadius += .02;
        LidarScene reuse({changed}, .2, true, scene), fresh({changed}, .2, true);
        check(reuse.samples().size() == fresh.samples().size(), "cache sample count");
        for (size_t i = 0; i < fresh.samples().size(); ++i)
            check(reuse.samples()[i] == fresh.samples()[i], "cache parameter identity");
        near(reuse.closest(center, rotation * Eigen::Vector3d::UnitX(), 10),
             fresh.closest(center, rotation * Eigen::Vector3d::UnitX(), 10), 1e-12, "cache query");
    }
    bool rejected = false;
    try {
        occupiedSet(ring);
    } catch (const std::invalid_argument& e) {
        rejected = std::string(e.what()).find("nonconvex") != std::string::npos;
    }
    check(rejected, "convex occupied set explicitly rejects ring");
    SceneObstacleDescription o;
    o.id = "r";
    o.pose.position = center;
    o.pose.orientation = rotation;
    ScenePartDescription p;
    p.type = "ring";
    p.radii = {1, 1.2};
    p.tubeRadius = .05;
    p.pose.position = {.3, .2, .1};
    p.pose.orientation = rotation;
    o.parts.push_back(p);
    const auto result = toObstacles({o});
    check(result.size() == 1 && result[0].type == Obstacle::kRing, "one typed part");
    near((result[0].position - (center + rotation * p.pose.position)).norm(), 0, 1e-12, "composed position");
    for (const auto& invalid :
         {Obstacle::ring(Eigen::Vector3d::Zero(), {1, 2}, 1), Obstacle::ring(Eigen::Vector3d::Zero(), {0, 2}, .1)}) {
        bool failed = false;
        try {
            validateObstacle(invalid);
        } catch (const std::invalid_argument&) {
            failed = true;
        }
        check(failed, "invalid ring rejected");
    }
}
} // namespace
int main() {
    try {
        distanceTests();
        rayTests();
        samplingTests();
        sceneTests();
        std::cout << "PASS ring distance/ray/sampling/cache/typed pose/convex rejection\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
