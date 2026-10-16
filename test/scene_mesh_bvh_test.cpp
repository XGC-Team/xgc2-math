#include "xgc2_math/scene/lidar_scene.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <random>

namespace {
bool counting = false;
std::size_t allocations = 0, allocatedBytes = 0;
// Keep the allocation-counter boundary intact under an optimizing test build.
[[gnu::noinline]] void release(void* p) {
    std::free(p);
}
} // namespace

void* operator new(std::size_t n) {
    if (counting) {
        ++allocations;
        allocatedBytes += n;
    }
    if (void* p = std::malloc(n ? n : 1))
        return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    return ::operator new(n);
}
void operator delete(void* p) noexcept {
    release(p);
}
void operator delete[](void* p) noexcept {
    release(p);
}
void operator delete(void* p, std::size_t) noexcept {
    release(p);
}
void operator delete[](void* p, std::size_t) noexcept {
    release(p);
}

using namespace xgc2_math::scene;
using Vec = Eigen::Vector3d;
using Index = std::array<uint32_t, 3>;

static void require(bool good, const char* message) {
    if (!good)
        throw std::runtime_error(message);
}

static void equalHit(double actual, double expected) {
    require((std::isinf(actual) && std::isinf(expected)) ||
                std::abs(actual - expected) <= 1e-12 * std::max(1., std::abs(expected)),
            "BVH differs from unchanged linear Moller-Trumbore oracle");
}

static TriangleMeshPtr torus(uint32_t major = 128, uint32_t minor = 16) {
    std::vector<Vec> vertices;
    std::vector<Index> indices;
    const double pi = std::acos(-1.);
    for (uint32_t i = 0; i < major; ++i) {
        const double u = 2 * pi * i / major;
        for (uint32_t j = 0; j < minor; ++j) {
            const double v = 2 * pi * j / minor, radius = 1.5 + .35 * std::cos(v);
            vertices.push_back({radius * std::cos(u), radius * std::sin(u), .35 * std::sin(v)});
        }
    }
    for (uint32_t i = 0; i < major; ++i)
        for (uint32_t j = 0; j < minor; ++j) {
            const auto a = i * minor + j, b = ((i + 1) % major) * minor + j;
            const auto c = ((i + 1) % major) * minor + (j + 1) % minor, d = i * minor + (j + 1) % minor;
            indices.push_back({a, b, c});
            indices.push_back({a, c, d});
        }
    const auto* vertexAddress = vertices.data();
    const auto* indexAddress = indices.data();
    auto result = prepareTriangleMesh(std::move(vertices), std::move(indices));
    require(result->vertices().data() == vertexAddress && result->triangleIndices().data() == indexAddress,
            "cold source buffers were copied instead of moved");
    return result;
}

static double linear(const TriangleMeshData& mesh, const Vec& o, const Vec& d, double range) {
    return meshRayClosest(mesh.triangles().data(), mesh.triangles().size(), o, d, range);
}

static void correctness(const TriangleMeshPtr& mesh) {
    std::mt19937_64 rng(18321);
    std::uniform_real_distribution<double> coordinate(-4, 4), range(0, 9);
    for (unsigned i = 0; i < 6000; ++i) {
        const Vec o(coordinate(rng), coordinate(rng), coordinate(rng));
        const Vec d = Vec(coordinate(rng), coordinate(rng), coordinate(rng)).normalized();
        const double limit = i % 7 == 0 ? std::numeric_limits<double>::infinity() : range(rng);
        equalHit(mesh->rayClosest(o, d, limit), linear(*mesh, o, d, limit));
    }
    require(std::isinf(mesh->rayClosest({0, 0, -2}, Vec::UnitZ(), 5)), "mesh hole was filled");
    for (const auto& o :
         std::array<Vec, 6>{{{1.5, 0, 0}, {0, 0, 0}, {1.85, 0, 0}, {1.85, 0, -2}, {1.5, 0, -.35}, {1.5, 0, 2}}})
        for (const auto& d : std::array<Vec, 4>{{Vec::UnitX(), Vec::UnitY(), Vec::UnitZ(), -Vec::UnitZ()}}) {
            const double hit = linear(*mesh, o, d, 8);
            equalHit(mesh->rayClosest(o, d, 8), hit);
            if (std::isfinite(hit)) {
                equalHit(mesh->rayClosest(o, d, hit), hit);
                if (hit > 1e-10)
                    equalHit(mesh->rayClosest(o, d, hit - 1e-10), linear(*mesh, o, d, hit - 1e-10));
            }
        }
    // The old leaf deliberately accepts just outside a triangle's exact barycentric bounds.
    auto tetra =
        prepareTriangleMesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}});
    for (const Vec& o :
         std::array<Vec, 4>{{{-.5e-12, .5, -1}, {.5, -.5e-12, -1}, {-.5e-12, -.5e-12, -1}, {1 + .5e-12, 0, -1}}}) {
        const double expected = linear(*tetra, o, Vec::UnitZ(), 5);
        require(std::isfinite(expected), "tolerance fixture did not exercise an accepted outside-edge hit");
        equalHit(tetra->rayClosest(o, Vec::UnitZ(), 5), expected);
        detail::ShapeTable table;
        table.append(Obstacle::mesh(tetra));
        detail::Bvh outer;
        outer.build(table);
        equalHit(outer.closest(table, o, Vec::UnitZ(), 5), expected);
    }
    const Eigen::Quaterniond q(Eigen::AngleAxisd(.83, Vec(1, 2, 3).normalized()));
    const Vec position(4, -3, 7);
    detail::ShapeTable table;
    table.append(Obstacle::mesh(mesh, position, q));
    detail::Bvh outer;
    outer.build(table);
    for (unsigned i = 0; i < 1000; ++i) {
        const Vec o(coordinate(rng), coordinate(rng), coordinate(rng));
        const Vec d = Vec(coordinate(rng), coordinate(rng), coordinate(rng)).normalized();
        const Vec world = position + q * o, direction = q * d;
        // Compare using exactly the production world-to-local transform, including its rounding.
        const Eigen::Matrix3d rotation = detail::rotationOf(q);
        const double expected =
            linear(*mesh, rotation.transpose() * (world - position), rotation.transpose() * direction, 9);
        equalHit(table.closestSurface(0, world, direction, 9), expected);
        equalHit(outer.closest(table, world, direction, 9), expected);
    }
    for (std::size_t i = 0; i < mesh->triangles().size(); ++i) {
        const auto& t = mesh->triangleIndices()[i];
        const auto& triangle = mesh->triangles()[i];
        require(triangle.a == mesh->vertices()[t[0]] && triangle.b == mesh->vertices()[t[1]] &&
                    triangle.c == mesh->vertices()[t[2]],
                "BVH changed original triangle/sampling order");
    }
    // Old world AABBs already lost this exact-range hit at a large translation, although the local leaf hit.
    // Keep that distinct from the new local BVH, and exercise the mesh-only outward transform bounds.
    std::vector<Vec> gridVertices;
    std::vector<Index> gridIndices;
    for (uint32_t i = 0; i < 128; ++i) {
        const Vec base(static_cast<double>(i % 8) * 3, static_cast<double>(i / 8) * 3, 0);
        const auto first = static_cast<uint32_t>(gridVertices.size());
        for (const Vec& v : std::array<Vec, 4>{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}})
            gridVertices.push_back(base + v);
        for (auto index : std::array<Index, 4>{{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}}) {
            for (auto& component : index)
                component += first;
            gridIndices.push_back(index);
        }
    }
    auto grid = prepareTriangleMesh(std::move(gridVertices), std::move(gridIndices));
    const Vec largePosition(52473.747595763794, 28796.266830740879, -97002.019966757522);
    const Eigen::Quaterniond largeRotation(.74585019034536926, .34962592841470586, -.30055695307437358,
                                           .48076472593211339);
    const Vec origin(52473.781514187358, 28806.210380605433, -97003.08046929483);
    const Vec direction(-.0033918423563922273, -.99435498645557396, .1060502537307903);
    const auto rotation = detail::rotationOf(largeRotation);
    const double exactRange =
        linear(*grid, rotation.transpose() * (origin - largePosition), rotation.transpose() * direction, 100);
    require(std::isfinite(exactRange), "large-coordinate exact-range fixture missed");
    detail::ShapeTable large;
    large.append(Obstacle::mesh(grid, largePosition, largeRotation));
    detail::Bvh largeWorld;
    largeWorld.build(large);
    equalHit(largeWorld.closest(large, origin, direction, exactRange), exactRange);
    std::cout << "oracle random=6000 rotated=1000 hole/inside/exit/tangent/range/tolerant-edge PASS\n";
}

static void sharingAndRebuild(const TriangleMeshPtr& mesh) {
    std::vector<Obstacle> instances;
    instances.reserve(100);
    for (unsigned i = 0; i < 100; ++i)
        instances.push_back(Obstacle::mesh(mesh, {static_cast<double>(i) * 100, 0, 0}));
    const auto before = mesh.use_count();
    allocations = allocatedBytes = 0;
    counting = true;
    detail::ShapeTable table;
    table.reserve(instances.size());
    for (const auto& obstacle : instances)
        table.append(obstacle);
    counting = false;
    const auto instanceBytes = allocatedBytes;
    require(mesh.use_count() == before + 1, "table duplicated a mesh handle for every instance");
    require(instanceBytes < mesh->triangles().size() * sizeof(MeshTriangle), "100 instances copied triangle storage");
    detail::ShapeTable copied;
    const auto priorCopy = mesh.use_count();
    for (std::size_t i = 0; i < table.size(); ++i)
        copied.appendFrom(table, i);
    require(mesh.use_count() == priorCopy + 1, "appendFrom duplicated mesh storage/handles");
    std::cout << "100-instance table allocated_bytes=" << instanceBytes
              << " unique_mesh_handles=1 triangle_storage_bytes=" << mesh->triangles().size() * sizeof(MeshTriangle)
              << " previous_linear_triangle_bytes=" << 100 * mesh->triangles().size() * sizeof(MeshTriangle) << "\n";

    auto small = torus(16, 8);
    std::vector<Obstacle> two{Obstacle::mesh(small), Obstacle::box({20, 0, 0}, {1, 1, 1})};
    LidarScene first(two, .6, true);
    auto moved = two;
    moved[0].position = {1, 2, 3};
    LidarScene incremental(moved, .6, true, first);
    LidarScene fresh(moved, .6, true);
    require(incremental.samples() == fresh.samples(), "moved instance incremental samples differ");
    require(moved[0].meshData.get() == two[0].meshData.get(), "pose update rebuilt cold mesh");
    auto scaledVertices = small->vertices();
    for (auto& v : scaledVertices)
        v.x() *= 1.5;
    moved[0].meshData = prepareTriangleMesh(std::move(scaledVertices), small->triangleIndices());
    LidarScene changed(moved, .6, true, incremental);
    LidarScene changedFresh(moved, .6, true);
    require(changed.samples() == changedFresh.samples(), "changed mesh cache was not invalidated");
    const auto expectedSamples = changed.samples();
    LidarScene survivesTemporary({Obstacle::mesh(prepareTriangleMesh(small->vertices(), small->triangleIndices()))}, .6,
                                 true);
    require(std::isfinite(survivesTemporary.closest({0, 0, 0}, Vec::UnitX(), 5)),
            "mesh lifetime ended with owner temporary");
    require(!expectedSamples.empty(), "sampling fixture empty");
}

struct Ray {
    Vec origin, direction;
};

static void performance(const TriangleMeshPtr& mesh) {
    std::vector<Ray> rays;
    std::mt19937_64 rng(4312);
    std::uniform_real_distribution<double> u(-1, 1);
    for (unsigned i = 0; i < 256; ++i) {
        const Vec origin(3 * u(rng), 3 * u(rng), 2 * u(rng));
        rays.push_back({origin, (Vec(1.5 * u(rng), 1.5 * u(rng), .2 * u(rng)) - origin).normalized()});
    }
    std::size_t visits = 0;
    for (const auto& ray : rays) {
        const double hit = mesh->rayClosest(
            ray.origin, ray.direction, 6, [&](const MeshTriangle& triangle, const Vec& o, const Vec& d, double limit) {
                ++visits;
                return meshRayClosest(&triangle, 1, o, d, limit);
            });
        equalHit(hit, linear(*mesh, ray.origin, ray.direction, 6));
    }
    require(visits < rays.size() * mesh->triangles().size() / 20, "BVH still visits a linear fraction of triangles");
    allocations = 0;
    counting = true;
    double checksum = 0;
    for (unsigned repeat = 0; repeat < 20; ++repeat)
        for (const auto& ray : rays) {
            const double hit = mesh->rayClosest(ray.origin, ray.direction, 6);
            if (std::isfinite(hit))
                checksum += hit;
        }
    counting = false;
    require(allocations == 0, "hot mesh ray allocated");
    std::cout << "triangles_per_ray=" << static_cast<double>(visits) / static_cast<double>(rays.size())
              << " linear=" << mesh->triangles().size() << " hot_allocations=" << allocations << "\n";
    for (const unsigned count : {20u, 100u}) {
        detail::ShapeTable table;
        table.reserve(count);
        for (unsigned i = 0; i < count; ++i)
            table.append(Obstacle::mesh(mesh, {static_cast<double>(i) * 100, 0, 0}));
        detail::Bvh outer;
        outer.build(table);
        const auto measure = [&](bool accelerated) {
            const auto begin = std::chrono::steady_clock::now();
            double sum = 0;
            for (unsigned i = 0; i < count; ++i)
                for (const auto& ray : rays) {
                    const Vec offset(static_cast<double>(i) * 100, 0, 0);
                    const double hit = accelerated ? outer.closest(table, ray.origin + offset, ray.direction, 6)
                                                   : linear(*mesh, (ray.origin + offset) - offset, ray.direction, 6);
                    if (std::isfinite(hit))
                        sum += hit;
                }
            return std::make_pair(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count(), sum);
        };
        const auto baseline = measure(false), accelerated = measure(true);
        require(std::abs(baseline.second - accelerated.second) <= 1e-8 * baseline.second, "fleet checksum mismatch");
        std::cout << "instances=" << count << " rays=" << count * rays.size() << " linear_seconds=" << baseline.first
                  << " bvh_seconds=" << accelerated.first << " speedup=" << baseline.first / accelerated.first << "\n";
        allocations = 0;
        counting = true;
        for (const auto& ray : rays) {
            const double hit = outer.closest(table, ray.origin, ray.direction, 6);
            if (std::isfinite(hit))
                checksum += hit;
        }
        counting = false;
        require(allocations == 0, "top-level/mesh hot ray allocated");
    }
    require(checksum > 0, "benchmark checksum empty");
}

int main() {
    auto mesh = torus();
    correctness(mesh);
    sharingAndRebuild(mesh);
    performance(mesh);
}
