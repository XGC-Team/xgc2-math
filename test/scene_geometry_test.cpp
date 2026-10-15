// Geometry of xgc2_math::scene: obstacle validation, convex hulls, surface sampling, ray casting, the bounding-volume
// hierarchy (build, refit), scene rebuilds from a previous scene and the sampled-map index. Every case is checked
// against an independent computation (analytic distances, brute force over single shapes, an enumeration of all vertex
// triples, a distance filter) and runs with fixed seeds.

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "xgc2_math/scene.hpp"
#include "xgc2_math/scene/bvh.hpp"

namespace {

using namespace xgc2_math::scene;

constexpr std::uint64_t kSeed = 0x58474332534345ULL;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.141592653589793238462643383279502884;

// Use the engine bits directly so cases do not depend on the standard library's choice of distribution
// implementation.
struct Cases {
    std::mt19937_64 engine{kSeed};
    double unit() { return static_cast<double>(engine() >> 11) / 9007199254740992.0; }
    double scalar() { return 2.0 * unit() - 1.0; }
    double between(double lo, double hi) { return lo + (hi - lo) * unit(); }
    double gaussian() {
        const double u1 = 1.0 - unit(), u2 = unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
    Eigen::Vector3d vector3(double half_extent = 1.0) {
        const double x = scalar(), y = scalar(), z = scalar();
        return half_extent * Eigen::Vector3d(x, y, z);
    }
    Eigen::Vector3d direction() {
        for (;;) {
            const Eigen::Vector3d v = vector3();
            if (v.norm() > 0.1 && v.norm() < 1.0) {
                return v.normalized();
            }
        }
    }
    Eigen::Quaterniond rotation() {
        const double w = gaussian(), x = gaussian(), y = gaussian(), z = gaussian();
        return Eigen::Quaterniond(w, x, y, z).normalized();
    }
    // irregular_rock_1 of the cluttered-environment V-polytope library, scaled by `s`
    std::vector<Eigen::Vector3d> rock(double s) {
        std::vector<Eigen::Vector3d> v{{-0.45, -0.40, -0.50}, {0.50, -0.35, -0.50}, {0.45, 0.48, -0.50},
                                       {-0.38, 0.42, -0.50},  {-0.25, -0.15, 0.80}, {0.30, -0.20, 0.75},
                                       {0.28, 0.25, 0.85},    {-0.22, 0.30, 0.70}};
        std::transform(v.begin(), v.end(), v.begin(), [s](const Eigen::Vector3d& p) {
            return Eigen::Vector3d(s * p);
        });
        return v;
    }
    // A mixed scene: every obstacle type, rotated.
    std::vector<Obstacle> scene(int count, double half_extent) {
        std::vector<Obstacle> out;
        for (int i = 0; i < count; ++i) {
            const double x = between(-half_extent, half_extent), y = between(-half_extent, half_extent);
            const Eigen::Vector3d c(x, y, between(-2.0, 2.0));
            const auto size = [this] {
                return between(0.3, 2.0);
            };
            switch (i % 5) {
            case 0: {
                const double a = size(), b = size(), d = size();
                out.push_back(Obstacle::box(c, Eigen::Vector3d(a, b, d), rotation()));
                break;
            }
            case 1:
                out.push_back(Obstacle::sphere(c, 0.5 * size()));
                break;
            case 2: {
                const double r = 0.5 * size(), h = 2.0 * size();
                out.push_back(Obstacle::cylinder(c, r, h, rotation()));
                break;
            }
            case 3: {
                const double r = 0.4 * size(), h = size();
                out.push_back(Obstacle::capsule(c, r, h, rotation()));
                break;
            }
            default:
                out.push_back(Obstacle::convexHull(rock(size()), c, rotation()));
                break;
            }
        }
        return out;
    }
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance, const std::string& context) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << std::setprecision(17) << context << ": actual=" << actual << ", expected=" << expected
            << ", tolerance=" << tolerance;
        throw std::runtime_error(out.str());
    }
}

template <class F> bool refuses(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

bool sameBytes(const std::vector<Eigen::Vector3d>& a, const std::vector<Eigen::Vector3d>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(Eigen::Vector3d)) == 0);
}

// ---- obstacles ---------------------------------------------------------------------------------------------------

void obstacleValidation() {
    const Eigen::Vector3d one = Eigen::Vector3d::Ones();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Valid definitions, including a capsule that is a sphere.
    validateObstacle(Obstacle::box({0, 0, 0}, one));
    validateObstacle(Obstacle::alignedBox({-1, -1, -1}, {1, 2, 3}));
    validateObstacle(Obstacle::sphere({1, 2, 3}, 0.5));
    validateObstacle(Obstacle::cylinder({0, 0, 0}, 0.5, 2.0));
    validateObstacle(Obstacle::capsule({0, 0, 0}, 0.5, 0.0));
    validateObstacle(Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}));
    // Refusals name the problem.
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::box({0, 0, 0}, {1, 0, 1}));
                },
                "box size"),
            "zero box side");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::box({0, 0, 0}, {1, -1, 1}));
                },
                "box size"),
            "negative box side");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::box({0, 0, 0}, {1, nan, 1}));
                },
                "box size"),
            "NaN box side");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::sphere({0, 0, 0}, 0.0));
                },
                "sphere radius"),
            "zero radius");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::sphere({0, 0, nan}, 1.0));
                },
                "non-finite obstacle position"),
            "NaN position");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::cylinder({0, 0, 0}, 1.0, 0.0));
                },
                "cylinder height"),
            "zero height cylinder");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::cylinder({0, 0, 0}, -1.0, 1.0));
                },
                "cylinder radius"),
            "negative radius cylinder");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::capsule({0, 0, 0}, 1.0, -0.1));
                },
                "capsule height"),
            "negative capsule height");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::capsule({0, 0, 0}, 0.0, 1.0));
                },
                "capsule radius"),
            "zero capsule radius");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::box({0, 0, 0}, one, Eigen::Quaterniond(0, 0, 0, 0)));
                },
                "orientation"),
            "zero quaternion");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::box({0, 0, 0}, one, Eigen::Quaterniond(nan, 0, 0, 0)));
                },
                "orientation"),
            "NaN quaternion");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}));
                },
                "needs >= 4 distinct vertices"),
            "three vertices");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::convexHull({{0, 0, 0}, {0, 0, 0}, {1, 0, 0}, {1, 0, 1e-10}}));
                },
                "needs >= 4 distinct vertices"),
            "duplicates merge");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}));
                },
                "flat"),
            "coplanar vertices");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0}}));
                },
                "flat"),
            "collinear vertices");
    require(refuses(
                [&] {
                    validateObstacle(Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, nan}}));
                },
                "non-finite convex vertex"),
            "NaN vertex");
    Obstacle unknown;
    unknown.type = static_cast<Obstacle::Type>(7); // the enumeration holds values up to 7
    require(refuses(
                [&] {
                    validateObstacle(unknown);
                },
                "unknown obstacle type"),
            "unknown type");
    require(refuses(
                [&] {
                    sampleSurface(Obstacle::sphere({0, 0, 0}, 1.0), 0.0);
                },
                "spacing"),
            "zero spacing");
    require(refuses(
                [&] {
                    sampleSurface(Obstacle::sphere({0, 0, 0}, 1.0), nan);
                },
                "spacing"),
            "NaN spacing");
    require(refuses(
                [&] {
                    LidarScene(std::vector<Obstacle>{}, -1.0);
                },
                "spacing"),
            "scene spacing");
    // A scene refuses a degenerate obstacle anywhere in the list.
    require(refuses(
                [&] {
                    LidarScene({Obstacle::sphere({0, 0, 0}, 1.0), Obstacle::sphere({0, 0, 0}, 0.0)});
                },
                "sphere radius"),
            "scene with a degenerate obstacle");
}

// ---- convex hulls ------------------------------------------------------------------------------------------------

// The supporting planes of all vertex triples, in enumeration order: what a hull of tens of vertices used to cost
// O(n^4) for. An independent oracle for the order and the planes of buildPolytope().
std::vector<Plane> enumeratedPlanes(const std::vector<Eigen::Vector3d>& input) {
    std::vector<Eigen::Vector3d> v;
    for (const auto& p : input) {
        if (std::none_of(v.begin(), v.end(), [&p](const Eigen::Vector3d& q) {
                return (p - q).norm() < 1e-9;
            })) {
            // cppcheck-suppress useStlAlgorithm
            v.push_back(p);
        }
    }
    Eigen::Vector3d lo = v[0], hi = v[0];
    for (const auto& p : v) {
        lo = lo.cwiseMin(p);
        hi = hi.cwiseMax(p);
    }
    const double extent = std::max((hi - lo).norm(), 1e-9);
    const double eps = 1e-7 * extent;
    std::vector<Plane> planes;
    for (std::size_t i = 0; i < v.size(); ++i) {
        for (std::size_t j = i + 1; j < v.size(); ++j) {
            for (std::size_t k = j + 1; k < v.size(); ++k) {
                Eigen::Vector3d nn = (v[j] - v[i]).cross(v[k] - v[i]);
                const double len = nn.norm();
                if (len < 1e-10 * extent * extent) {
                    continue;
                }
                nn /= len;
                const double d = nn.dot(v[i]);
                double smax = -kInf, smin = kInf;
                for (const auto& p : v) {
                    smax = std::max(smax, nn.dot(p) - d);
                    smin = std::min(smin, nn.dot(p) - d);
                }
                Plane plane;
                if (smax <= eps) {
                    plane.n = nn;
                    plane.d = d;
                } else if (smin >= -eps) {
                    plane.n = -nn;
                    plane.d = -d;
                } else {
                    continue;
                }
                const bool duplicate = std::any_of(planes.begin(), planes.end(), [&](const Plane& q) {
                    return q.n.dot(plane.n) > 1.0 - 1e-9 && std::abs(q.d - plane.d) < eps;
                });
                if (!duplicate) {
                    planes.push_back(plane);
                }
            }
        }
    }
    return planes;
}

void checkPolytope(const Polytope& hull, const std::vector<Eigen::Vector3d>& points, const std::string& context) {
    require(hull.face_first.size() == hull.planes.size() + 1, context + ": face index size");
    std::set<std::pair<std::size_t, std::size_t>> edges;
    std::size_t edge_uses = 0;
    for (std::size_t f = 0; f < hull.planes.size(); ++f) {
        const Plane& plane = hull.planes[f];
        near(plane.n.norm(), 1.0, 1e-12, context + ": unit normal");
        for (const auto& p : points) {
            require(plane.n.dot(p) - plane.d <= 1e-6, context + ": a point lies outside a plane");
        }
        const std::size_t first = hull.face_first[f], last = hull.face_first[f + 1];
        require(last - first >= 3, context + ": face with fewer than three vertices");
        for (std::size_t k = first; k < last; ++k) {
            near(plane.n.dot(hull.face_vertices[k]) - plane.d, 0.0, 1e-6, context + ": face vertex off its plane");
        }
        // Counter-clockwise seen from outside, convex.
        for (std::size_t k = first; k < last; ++k) {
            const Eigen::Vector3d& a = hull.face_vertices[k];
            const Eigen::Vector3d& b = hull.face_vertices[first + (k - first + 1) % (last - first)];
            const Eigen::Vector3d& c = hull.face_vertices[first + (k - first + 2) % (last - first)];
            require((b - a).cross(c - b).dot(plane.n) > 0.0, context + ": face not counter-clockwise and convex");
            // vertex identity by position
            const auto index = [&](const Eigen::Vector3d& p) {
                for (std::size_t i = 0; i < hull.vertices.size(); ++i) {
                    if ((hull.vertices[i] - p).norm() < 1e-12) {
                        return i;
                    }
                }
                throw std::runtime_error(context + ": face vertex is not a hull vertex");
            };
            const std::size_t ia = index(a), ib = index(b);
            edges.insert({std::min(ia, ib), std::max(ia, ib)});
            ++edge_uses;
        }
    }
    require(edge_uses == 2 * edges.size(), context + ": every edge borders exactly two faces");
    const long euler = static_cast<long>(hull.vertices.size()) - static_cast<long>(edges.size()) +
                       static_cast<long>(hull.planes.size());
    require(euler == 2, context + ": Euler characteristic " + std::to_string(euler));
}

void polytopeHull() {
    // The unit cube with interior, edge-middle and face-centre points, which are not hull vertices.
    const std::vector<Eigen::Vector3d> corners{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                               {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    std::vector<Eigen::Vector3d> cube = corners;
    cube.emplace_back(0.5, 0.5, 0.5);
    cube.emplace_back(0.5, 0.0, 0.0);
    cube.emplace_back(0.5, 0.5, 0.0);
    cube.emplace_back(0.0, 0.5, 1.0);
    const Polytope hull = buildPolytope(cube);
    require(hull.vertices.size() == 8, "cube has eight hull vertices");
    require(hull.planes.size() == 6, "cube has six faces");
    for (const Plane& plane : hull.planes) {
        require(std::abs(plane.n.cwiseAbs().maxCoeff() - 1.0) < 1e-12, "cube normals are axes");
        require(std::abs(plane.d - 0.0) < 1e-12 || std::abs(plane.d - 1.0) < 1e-12, "cube plane offsets");
    }
    for (std::size_t f = 0; f < 6; ++f) {
        require(hull.face_first[f + 1] - hull.face_first[f] == 4, "cube faces are quadrilaterals");
    }
    checkPolytope(hull, cube, "cube");

    // A tetrahedron is its own hull.
    const std::vector<Eigen::Vector3d> tetra{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const Polytope t = buildPolytope(tetra);
    require(t.planes.size() == 4 && t.vertices.size() == 4, "tetrahedron");
    checkPolytope(t, tetra, "tetrahedron");

    // Duplicates and a far offset do not change the faces; the order of the input does not change the planes.
    Cases cases;
    const std::vector<Eigen::Vector3d> rock = cases.rock(1.3);
    const Polytope base = buildPolytope(rock);
    checkPolytope(base, rock, "rock");
    std::vector<Eigen::Vector3d> shifted;
    shifted.reserve(2 * rock.size());
    for (const auto& p : rock) {
        shifted.emplace_back(p + Eigen::Vector3d(1000.0, -2000.0, 300.0));
        shifted.emplace_back(shifted.back() + Eigen::Vector3d(1e-12, 0, 0)); // a duplicate within 1e-9
    }
    const Polytope far = buildPolytope(shifted);
    require(far.planes.size() == base.planes.size(), "faces survive an offset of 2000 m and duplicates");
    for (std::size_t f = 0; f < base.planes.size(); ++f) {
        near((far.planes[f].n - base.planes[f].n).norm(), 0.0, 1e-6,
             "offset keeps the normal of face " + std::to_string(f));
    }

    // Against the enumeration of all vertex triples: the same planes in the same order for tens of vertices.
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<Eigen::Vector3d> cloud;
        const int count = 6 + 3 * trial;
        cloud.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            cloud.emplace_back(Eigen::Vector3d(cases.gaussian(), cases.gaussian(), cases.gaussian()).normalized() *
                               cases.between(0.5, 1.0));
        }
        const Polytope hull_n = buildPolytope(cloud);
        const std::vector<Plane> oracle = enumeratedPlanes(cloud);
        const std::string where = "cloud " + std::to_string(trial);
        require(hull_n.planes.size() == oracle.size(), where + ": " + std::to_string(hull_n.planes.size()) +
                                                           " planes vs " + std::to_string(oracle.size()) +
                                                           " enumerated");
        for (std::size_t f = 0; f < oracle.size(); ++f) {
            near((hull_n.planes[f].n - oracle[f].n).norm(), 0.0, 1e-12,
                 where + ": normal of face " + std::to_string(f));
            near(hull_n.planes[f].d, oracle[f].d, 1e-12, where + ": offset of face " + std::to_string(f));
        }
        checkPolytope(hull_n, cloud, where);
    }

    // A large cloud (the enumeration is O(n^4), the hull is not).
    std::vector<Eigen::Vector3d> sphere;
    sphere.reserve(1500);
    for (int i = 0; i < 1500; ++i) {
        sphere.emplace_back(Eigen::Vector3d(cases.gaussian(), cases.gaussian(), cases.gaussian()).normalized());
    }
    const Polytope big = buildPolytope(sphere);
    require(big.vertices.size() > 1000, "points on a sphere are all hull vertices");
    checkPolytope(big, sphere, "sphere cloud");
}

// ---- surface sampling --------------------------------------------------------------------------------------------

void surfaceSampling() {
    Cases cases;
    const std::vector<Obstacle> obstacles = cases.scene(40, 6.0);
    for (const double spacing : {0.1, 0.25, 0.8}) {
        for (std::size_t i = 0; i < obstacles.size(); ++i) {
            const std::string where = "obstacle " + std::to_string(i) + " spacing " + std::to_string(spacing);
            const std::vector<Eigen::Vector3d> samples = sampleSurface(obstacles[i], spacing);
            require(!samples.empty(), where + ": no samples");
            // The scene of this obstacle alone holds the same samples (nothing buries them) and measures them.
            const LidarScene alone({obstacles[i]}, spacing, true);
            require(sameBytes(alone.samples(), samples), where + ": scene samples differ from sampleSurface");
            for (const auto& p : samples) {
                near(alone.surfaceResidual(p), 0.0, 1e-9, where + ": sample off the surface");
            }
        }
    }
    // The count follows the area: halving the spacing quadruples it.
    for (const Obstacle& o : {Obstacle::sphere({0, 0, 0}, 2.0), Obstacle::box({0, 0, 0}, {3, 4, 5}),
                              Obstacle::cylinder({0, 0, 0}, 1.5, 4.0), Obstacle::capsule({0, 0, 0}, 1.0, 3.0)}) {
        const double coarse = static_cast<double>(sampleSurface(o, 0.2).size());
        const double fine = static_cast<double>(sampleSurface(o, 0.1).size());
        require(fine / coarse > 3.2 && fine / coarse < 4.8,
                "sample count ratio " + std::to_string(fine / coarse) + " for type " + std::to_string(o.type));
    }
    // A box is sampled at its corners and edges.
    const Obstacle box = Obstacle::box({1, 2, 3}, {2, 4, 6});
    const std::vector<Eigen::Vector3d> samples = sampleSurface(box, 0.5);
    for (int corner = 0; corner < 8; ++corner) {
        const Eigen::Vector3d c(1 + ((corner & 1) ? 1.0 : -1.0), 2 + ((corner & 2) ? 2.0 : -2.0),
                                3 + ((corner & 4) ? 3.0 : -3.0));
        require(std::any_of(samples.begin(), samples.end(),
                            [&c](const Eigen::Vector3d& p) {
                                return (p - c).norm() < 1e-12;
                            }),
                "box corner " + std::to_string(corner) + " is sampled");
    }
    // No two samples of a sphere are closer than a fraction of the spacing: the lattice does not clump.
    const std::vector<Eigen::Vector3d> sphere = sampleSurface(Obstacle::sphere({0, 0, 0}, 1.0), 0.1);
    double closest = kInf;
    for (std::size_t i = 0; i < sphere.size(); ++i) {
        for (std::size_t j = i + 1; j < sphere.size(); ++j) {
            closest = std::min(closest, (sphere[i] - sphere[j]).norm());
        }
    }
    require(closest > 0.05, "sphere samples keep a distance of " + std::to_string(closest));
}

// ---- ray casting -------------------------------------------------------------------------------------------------

double castOne(const Obstacle& o, const Eigen::Vector3d& origin, const Eigen::Vector3d& direction, double max_range) {
    const LidarScene scene({o}, 0.5, true);
    return scene.closest(origin, direction.normalized(), max_range);
}

void rayAnalytic() {
    const Eigen::Vector3d x = Eigen::Vector3d::UnitX(), z = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    // Sphere of radius 1 at x = 5: enters at 4, a ray from inside leaves at 1, a ray past it misses, tmax cuts.
    const Obstacle ball = Obstacle::sphere({5, 0, 0}, 1.0);
    near(castOne(ball, origin, x, 20), 4.0, 1e-12, "sphere front");
    near(castOne(ball, {5, 0, 0}, x, 20), 1.0, 1e-12, "sphere from the centre");
    near(castOne(ball, {5.5, 0, 0}, -x, 20), 1.5, 1e-12, "sphere exit behind the origin");
    require(std::isinf(castOne(ball, {0, 1.01, 0}, x, 20)), "sphere missed by 1 cm");
    near(castOne(ball, {0, 0.999, 0}, x, 20), 5.0 - std::sqrt(1 - 0.999 * 0.999), 1e-9, "sphere grazed");
    require(std::isinf(castOne(ball, origin, x, 3.99)), "sphere beyond max range");
    near(castOne(ball, origin, x, 4.0), 4.0, 1e-12, "sphere at max range");
    require(std::isinf(castOne(ball, origin, -x, 20)), "sphere behind the ray");
    // Axis-aligned box [4, 6] x [-1, 1]^2: faces are met at the near side, parallel rays outside miss.
    const Obstacle wall = Obstacle::alignedBox({4, -1, -1}, {6, 1, 1});
    near(castOne(wall, origin, x, 20), 4.0, 1e-12, "box front");
    near(castOne(wall, {5, 0, 0}, x, 20), 1.0, 1e-12, "box from inside");
    near(castOne(wall, {0, 0, 0.5}, Eigen::Vector3d(4.5, 0, -0.5), 20), 4.0 * std::sqrt(4.5 * 4.5 + 0.25) / 4.5, 1e-12,
         "box oblique");
    require(std::isinf(castOne(wall, {0, 1.5, 0}, x, 20)), "box missed along a parallel ray");
    require(std::isinf(castOne(wall, {0, 0, 1.5}, x, 20)), "box missed above");
    near(castOne(wall, {5, 0, 5}, -z, 20), 4.0, 1e-12, "box top");
    // A box turned by 45 degrees about z shows an edge whose corner is a half diagonal, sqrt(2), from the centre; a ray
    // beside the corner meets the face that slopes away at 45 degrees.
    const Obstacle diamond = Obstacle::box({5, 0, 0}, {2, 2, 2}, Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 4, z)));
    near(castOne(diamond, {0, 0.1, 0}, x, 20), 5.0 - std::sqrt(2.0) + 0.1, 1e-12, "turned box face");
    near(castOne(diamond, {0, -0.1, 0}, x, 20), 5.0 - std::sqrt(2.0) + 0.1, 1e-12, "turned box other face");
    // Cylinder of radius 1, height 2 about the z axis at x = 5.
    const Obstacle pillar = Obstacle::cylinder({5, 0, 0}, 1.0, 2.0);
    near(castOne(pillar, origin, x, 20), 4.0, 1e-12, "cylinder side");
    near(castOne(pillar, {5, 0, 5}, -z, 20), 4.0, 1e-12, "cylinder top cap");
    near(castOne(pillar, {5, 0.5, 5}, -z, 20), 4.0, 1e-12, "cylinder top cap off axis");
    require(std::isinf(castOne(pillar, {0, 0, 1.01}, x, 20)), "cylinder missed above its top");
    near(castOne(pillar, {5, 0, 0}, z, 20), 1.0, 1e-12, "cylinder from inside, up");
    // Capsule: radius 0.5, straight part 2, so 3 tall with round caps.
    const Obstacle pill = Obstacle::capsule({5, 0, 0}, 0.5, 2.0);
    near(castOne(pill, origin, x, 20), 4.5, 1e-12, "capsule side");
    near(castOne(pill, {5, 0, 5}, -z, 20), 3.5, 1e-12, "capsule top");
    // 0.2 m above the centre of the upper cap sphere (centre z = 1, radius 0.5).
    near(castOne(pill, {0, 0, 1.2}, x, 20), 5.0 - std::sqrt(0.25 - 0.2 * 0.2), 1e-12, "capsule cap");
    require(std::isinf(castOne(pill, {0, 0, 1.6}, x, 20)), "capsule missed above its cap");
    // A capsule of height zero is a sphere.
    near(castOne(Obstacle::capsule({5, 0, 0}, 1.0, 0.0), origin, x, 20), 4.0, 1e-12, "degenerate capsule");
    // Tetrahedron: the ray along x at y = z = 0.2 meets the slanted face x + y + z = 1 at x = 0.6.
    const Obstacle tetra = Obstacle::convexHull({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    near(castOne(tetra, {-2, 0.2, 0.2}, x, 20), 2.0, 1e-12, "tetrahedron back face");
    near(castOne(tetra, {-2, 0.6, 0.1}, x, 20), 2.0, 1e-12, "tetrahedron second ray");
    near(castOne(tetra, {0.2, 0.2, 0.2}, x, 20), 0.6 - 0.2, 1e-12, "tetrahedron from inside");
    require(std::isinf(castOne(tetra, {-2, 0.8, 0.8}, x, 20)), "tetrahedron missed");
    // A rotated, moved convex obstacle: the distance follows the rigid motion.
    Cases cases;
    for (int i = 0; i < 20; ++i) {
        const Eigen::Quaterniond q = cases.rotation();
        const Eigen::Vector3d p = cases.vector3(5.0);
        const Obstacle a =
            Obstacle::convexHull(cases.rock(1.0), Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
        const Obstacle b = Obstacle::convexHull(a.vertices, p, q);
        const Eigen::Vector3d o = cases.vector3(3.0), d = cases.direction();
        const double ta = castOne(a, o, d, 40), tb = castOne(b, p + q * o, q * d, 40);
        require(std::isinf(ta) == std::isinf(tb), "rigid motion keeps hits");
        if (std::isfinite(ta)) {
            near(ta, tb, 1e-9, "rigid motion keeps the distance");
        }
    }
}

void rayAgainstBruteForce() {
    // Many obstacles: the closest hit of the hierarchy equals the closest of the single-obstacle scenes.
    Cases cases;
    const std::vector<Obstacle> scene = cases.scene(100, 15.0);
    const LidarScene all(scene, 0.5, true);
    std::vector<std::unique_ptr<LidarScene>> singles;
    std::transform(scene.begin(), scene.end(), std::back_inserter(singles), [](const Obstacle& o) {
        return std::make_unique<LidarScene>(std::vector<Obstacle>{o}, 0.5, true);
    });
    int hits = 0;
    for (int i = 0; i < 3000; ++i) {
        const Eigen::Vector3d o(cases.between(-15, 15), cases.between(-15, 15), cases.between(-3, 3));
        const Eigen::Vector3d d = cases.direction();
        double brute = kInf;
        for (const auto& single : singles) {
            brute = std::min(brute, single->closest(o, d, 25.0));
        }
        const double t = all.closest(o, d, 25.0);
        hits += std::isfinite(t) ? 1 : 0;
        require(t == brute, "hierarchy " + std::to_string(t) + " vs brute force " + std::to_string(brute) +
                                " for ray " + std::to_string(i));
    }
    require(hits > 100, "enough rays hit");
    // The empty scene and a scene of one obstacle.
    require(std::isinf(LidarScene(std::vector<Obstacle>{}).closest({0, 0, 0}, {1, 0, 0}, 10)), "empty scene");
    near(LidarScene({Obstacle::sphere({3, 0, 0}, 1)}).closest({0, 0, 0}, {1, 0, 0}, 10), 2.0, 1e-12, "one obstacle");
}

// ---- bounding-volume hierarchy -----------------------------------------------------------------------------------

detail::ShapeTable compileAll(const std::vector<Obstacle>& obstacles) {
    detail::ShapeTable table;
    for (const Obstacle& o : obstacles) {
        table.append(o);
    }
    return table;
}

double bruteClosest(const detail::ShapeTable& table, const Eigen::Vector3d& o, const Eigen::Vector3d& d, double tmax) {
    double best = kInf;
    for (std::size_t i = 0; i < table.size(); ++i) {
        double t0, t1;
        if (!table.intersect(i, o, d, &t0, &t1)) {
            continue;
        }
        const double t = t0 >= 0.0 ? t0 : t1;
        if (t >= 0.0 && t <= tmax && t < best) {
            best = t;
        }
    }
    return best;
}

void checkRays(const detail::Bvh& bvh, const detail::ShapeTable& table, Cases* cases, int rays, double extent,
               const std::string& context) {
    for (int i = 0; i < rays; ++i) {
        const Eigen::Vector3d o = cases->vector3(extent), d = cases->direction();
        const double tmax = cases->between(1.0, 2.0 * extent);
        require(bvh.closest(table, o, d, tmax) == bruteClosest(table, o, d, tmax),
                context + ": ray " + std::to_string(i));
    }
}

void bvhBuildAndRefit() {
    Cases cases;
    // Scenes of every small size, then a few hundred shapes.
    for (const int count : {0, 1, 2, 3, 4, 5, 7, 8, 9, 33, 250, 1000}) {
        const std::vector<Obstacle> scene = cases.scene(count, 10.0);
        const detail::ShapeTable table = compileAll(scene);
        detail::Bvh bvh;
        bvh.build(table);
        require(bvh.empty() == (count == 0), "empty hierarchy exactly for no shapes");
        checkRays(bvh, table, &cases, 300, 14.0, "build of " + std::to_string(count));
        // Every overlap query returns exactly the shapes whose bounds overlap.
        for (int q = 0; q < 20 && count > 0; ++q) {
            detail::Aabb box;
            const Eigen::Vector3d c = cases.vector3(10.0);
            box.grow(c - Eigen::Vector3d::Constant(cases.between(0.1, 3.0)));
            box.grow(c + Eigen::Vector3d::Constant(cases.between(0.1, 3.0)));
            const std::size_t skip = static_cast<std::size_t>(q) % static_cast<std::size_t>(count);
            std::vector<std::uint32_t> found;
            bvh.overlapping(table, box, skip, &found);
            std::sort(found.begin(), found.end());
            std::vector<std::uint32_t> expected;
            for (std::size_t i = 0; i < table.size(); ++i) {
                if (i != skip && box.overlaps(table.bounds(i))) {
                    expected.push_back(static_cast<std::uint32_t>(i));
                }
            }
            require(found == expected, "overlap query of a " + std::to_string(count) + " shape hierarchy");
        }
    }

    // Refit: the shapes move a little, the topology stays, every ray still agrees with brute force.
    std::vector<Obstacle> scene = cases.scene(400, 12.0);
    detail::ShapeTable table = compileAll(scene);
    detail::Bvh bvh;
    bvh.build(table);
    for (int step = 0; step < 6; ++step) {
        for (Obstacle& o : scene) {
            o.position += cases.vector3(0.2);
        }
        table = compileAll(scene);
        const bool fit = bvh.refit(table);
        require(fit, "small motions keep the refit tree good (step " + std::to_string(step) + ")");
        checkRays(bvh, table, &cases, 300, 14.0, "refit step " + std::to_string(step));
    }
    // Moving everything far apart ruins the tree: the refit says a rebuild is due, and is still correct.
    for (Obstacle& o : scene) {
        o.position = cases.vector3(60.0);
    }
    table = compileAll(scene);
    require(!bvh.refit(table), "scattering the shapes asks for a rebuild");
    checkRays(bvh, table, &cases, 300, 60.0, "refit after scattering");
    bvh.build(table);
    checkRays(bvh, table, &cases, 300, 60.0, "rebuild after scattering");
    require(refuses(
                [&] {
                    detail::ShapeTable fewer = compileAll(std::vector<Obstacle>(scene.begin(), scene.begin() + 10));
                    bvh.refit(fewer);
                },
                "shape count"),
            "refit refuses another shape count");
}

void bvhDegenerateScenes() {
    Cases cases;
    // Thousands of identical shapes: no split separates them, the depth stays bounded.
    std::vector<Obstacle> same(3000, Obstacle::sphere({1, 2, 3}, 0.5));
    detail::ShapeTable table = compileAll(same);
    detail::Bvh bvh;
    bvh.build(table);
    near(bvh.closest(table, {-5, 2, 3}, {1, 0, 0}, 20), 5.5, 1e-12, "identical spheres");
    // Shapes in a geometric progression along a line: the most unbalanced input for a surface-area split.
    std::vector<Obstacle> line;
    double x = 0.0, step = 0.01;
    for (int i = 0; i < 3000; ++i) {
        line.push_back(Obstacle::sphere({x, 0, 0}, 0.001));
        x += step;
        step *= 1.004;
        if (!std::isfinite(x) || x > 1e12) {
            break;
        }
    }
    table = compileAll(line);
    bvh.build(table);
    for (int i = 0; i < 400; ++i) {
        const double target =
            line[static_cast<std::size_t>(cases.between(0, static_cast<double>(line.size())))].position.x();
        const Eigen::Vector3d o(target - 1.0, 0.0, 0.0);
        const double t = bvh.closest(table, o, Eigen::Vector3d::UnitX(), 1e13);
        require(t == bruteClosest(table, o, Eigen::Vector3d::UnitX(), 1e13),
                "geometric progression ray " + std::to_string(i));
    }
    // A flat sheet of coplanar shapes and a long thin scene (all centroids on one axis).
    std::vector<Obstacle> sheet;
    for (int i = 0; i < 40; ++i) {
        for (int j = 0; j < 40; ++j) {
            sheet.push_back(Obstacle::box({0.1 * i, 0.1 * j, 0.0}, {0.09, 0.09, 0.0 + 0.001}));
        }
    }
    table = compileAll(sheet);
    bvh.build(table);
    checkRays(bvh, table, &cases, 400, 3.0, "coplanar sheet");
    // Rays with a zero direction component and rays starting on a box face.
    const detail::ShapeTable wall = compileAll({Obstacle::alignedBox({4, -1, -1}, {6, 1, 1})});
    detail::Bvh one;
    one.build(wall);
    near(one.closest(wall, {0, 0, 0}, {1, 0, 0}, 10), 4.0, 1e-12, "axis-parallel ray");
    require(std::isinf(one.closest(wall, {0, 1.5, 0}, {1, 0, 0}, 10)), "axis-parallel ray beside the box");
    // A ray that starts on a surface meets it at distance zero, whichever way it points.
    near(one.closest(wall, {4, 0, 0}, {1, 0, 0}, 10), 0.0, 0.0, "ray starting on the near face");
    near(one.closest(wall, {6, 0, 0}, {1, 0, 0}, 10), 0.0, 0.0, "ray starting on the far face");
}

// ---- scene rebuilds and the sampled-map index ----------------------------------------------------------------------

std::vector<std::uint64_t> markedBits(const LidarScene& scene, const Eigen::Vector3d& center, double radius,
                                      std::size_t* count) {
    std::vector<std::uint64_t> bits((scene.sampleCount() + 63) / 64, 0);
    *count = scene.mark(center, radius * radius, bits.data());
    return bits;
}

void checkIndex(const LidarScene& scene, Cases* cases, const std::string& context) {
    for (int q = 0; q < 12; ++q) {
        const Eigen::Vector3d c = cases->vector3(9.0);
        const double r = cases->between(0.5, 7.0);
        std::size_t count = 0;
        const std::vector<std::uint64_t> bits = markedBits(scene, c, r, &count);
        std::size_t expected = 0;
        for (std::size_t k = 0; k < scene.sampleCount(); ++k) {
            const bool inside = (scene.samples()[k] - c).squaredNorm() <= r * r;
            const bool marked = (bits[k >> 6] >> (k & 63)) & 1u;
            require(inside == marked, context + ": sample " + std::to_string(k) + " of query " + std::to_string(q));
            expected += inside ? 1 : 0;
        }
        require(count == expected, context + ": marked count");
    }
}

void sampleIndex() {
    Cases cases;
    for (const bool keep : {true, false}) {
        const LidarScene scene(cases.scene(60, 8.0), 0.2, keep);
        checkIndex(scene, &cases, keep ? "kept samples" : "dropped samples");
    }
    checkIndex(LidarScene(std::vector<Obstacle>{}, 0.2, true), &cases, "empty scene");
    // Moving a few obstacles re-blocks only their samples, and queries still see every sample once.
    std::vector<Obstacle> obstacles = cases.scene(80, 8.0);
    auto previous = std::make_shared<const LidarScene>(obstacles, 0.25, true);
    for (int revision = 0; revision < 6; ++revision) {
        obstacles[3].position.x() += 0.25;
        obstacles[4].position.y() -= 0.25;
        if (revision == 3) {
            obstacles[10].position += Eigen::Vector3d(0.3, 0.1, 0.0);
        }
        previous = std::make_shared<const LidarScene>(obstacles, 0.25, true, *previous);
        checkIndex(*previous, &cases, "revision " + std::to_string(revision));
    }
}

void sceneRebuild() {
    // A scene rebuilt from the previous revision equals a scene built from scratch: the same samples in the same
    // order, the same ray casts. Revisions move a few obstacles, many, none, push one into another (burial changes),
    // change a size, change the obstacle count, then keep moving the same two (the dynamic case).
    Cases cases;
    for (const bool keep : {true, false}) {
        for (const double spacing : {0.1, 0.25}) {
            std::vector<Obstacle> obstacles = cases.scene(80, 7.0);
            auto previous = std::make_shared<const LidarScene>(obstacles, spacing, keep);
            for (int rev = 0; rev < 14; ++rev) {
                const int moves[] = {1, 3, 0, 1, 10, 50, 3, 1};
                if (rev < 8) {
                    for (int k = 0; k < moves[rev]; ++k) {
                        const std::size_t which = rev == 3 ? 7 : static_cast<std::size_t>(cases.between(0, 80));
                        obstacles[which].position += cases.vector3(0.6);
                    }
                } else if (rev == 8) {
                    obstacles[5].position = obstacles[6].position; // overlap: burial changes
                } else if (rev == 9) {
                    obstacles[11].radius *= 1.3;
                    obstacles[12].orientation = cases.rotation();
                } else if (rev == 10) {
                    obstacles.push_back(Obstacle::sphere(Eigen::Vector3d(1, 1, 1), 0.7));
                } else {
                    obstacles[3].position.x() += 0.25;
                    obstacles[4].position.y() -= 0.25;
                }
                const auto rebuilt = std::make_shared<const LidarScene>(obstacles, spacing, keep, *previous);
                const LidarScene full(obstacles, spacing, keep);
                const std::string where = std::string(keep ? "kept" : "dropped") + " spacing " +
                                          std::to_string(spacing) + " revision " + std::to_string(rev);
                require(rebuilt->sampleCount() == full.sampleCount(), where + ": sample count");
                require(sameBytes(rebuilt->samples(), full.samples()), where + ": samples differ from a full build");
                for (int k = 0; k < 200; ++k) {
                    const Eigen::Vector3d o = cases.vector3(9.0), d = cases.direction();
                    require(rebuilt->closest(o, d, 20.0) == full.closest(o, d, 20.0), where + ": ray cast");
                }
                checkIndex(*rebuilt, &cases, where);
                previous = rebuilt;
            }
        }
    }
    // The previous scene is ignored when the policy or the list length differs, and an unchanged list shares the
    // previous index (the same samples, a copy of nothing).
    const std::vector<Obstacle> obstacles = cases.scene(20, 5.0);
    const LidarScene a(obstacles, 0.2, true);
    const LidarScene b(obstacles, 0.3, true, a);
    require(sameBytes(b.samples(), LidarScene(obstacles, 0.3, true).samples()), "another spacing builds from scratch");
    const LidarScene c(obstacles, 0.2, false, a);
    require(sameBytes(c.samples(), LidarScene(obstacles, 0.2, false).samples()), "another policy builds from scratch");
    std::vector<Obstacle> fewer(obstacles.begin(), obstacles.end() - 1);
    const LidarScene d(fewer, 0.2, true, a);
    require(sameBytes(d.samples(), LidarScene(fewer, 0.2, true).samples()), "another length builds from scratch");
    const LidarScene e(obstacles, 0.2, true, a);
    require(sameBytes(e.samples(), a.samples()), "an unchanged list keeps the samples");
    // A degenerate obstacle in a revision is refused and leaves the previous scene untouched.
    std::vector<Obstacle> broken = obstacles;
    broken[4].radius = -1.0;
    broken[4].type = Obstacle::kSphere;
    require(refuses(
                [&] {
                    LidarScene(broken, 0.2, true, a);
                },
                "sphere radius"),
            "a bad revision is refused");
    require(sameBytes(a.samples(), LidarScene(obstacles, 0.2, true).samples()), "the previous scene is unchanged");
}

void sceneMapLayout() {
    // The sampled map of a scene is the concatenation of the samples of its obstacles in list order.
    Cases cases;
    const std::vector<Obstacle> obstacles = cases.scene(30, 5.0);
    const LidarScene scene(obstacles, 0.3, true);
    std::vector<Eigen::Vector3d> expected;
    for (const Obstacle& o : obstacles) {
        const auto samples = sampleSurface(o, 0.3);
        expected.insert(expected.end(), samples.begin(), samples.end());
    }
    require(sameBytes(scene.samples(), expected), "scene map is the obstacle samples in list order");
    require(scene.obstacleCount() == obstacles.size(), "obstacle count");
    require(scene.spacing() == 0.3 && scene.keepBuried(), "sampling policy");
    require(sameBytes(scene.resampled(0.3, true), expected), "resampling at the same policy");
    require(scene.resampled(0.3, false).size() < expected.size(), "dropping buried samples drops some");
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 10> tests{{
        {"obstacle_validation", obstacleValidation},
        {"polytope_hull", polytopeHull},
        {"surface_sampling", surfaceSampling},
        {"ray_analytic", rayAnalytic},
        {"ray_against_brute_force", rayAgainstBruteForce},
        {"bvh_build_and_refit", bvhBuildAndRefit},
        {"bvh_degenerate_scenes", bvhDegenerateScenes},
        {"sample_index", sampleIndex},
        {"scene_rebuild", sceneRebuild},
        {"scene_map_layout", sceneMapLayout},
    }};
    if (argc > 2) {
        std::cerr << "Usage: scene_geometry_test [case_name|--list]\n";
        return 2;
    }
    const std::string requested = argc == 2 ? argv[1] : "";
    if (requested == "--list") {
        for (const auto& test : tests) {
            std::cout << test.name << '\n';
        }
        return 0;
    }
    int failures = 0;
    int executed = 0;
    for (const auto& test : tests) {
        if (!requested.empty() && requested != test.name) {
            continue;
        }
        ++executed;
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << " (seed=" << kSeed << "): " << error.what() << '\n';
        }
    }
    if (executed == 0) {
        std::cerr << "Unknown case: " << requested << " (use --list)\n";
        return 2;
    }
    // Explicit checks remain active in Release/RelWithDebInfo, including NDEBUG.
    return failures == 0 ? 0 : 1;
}
