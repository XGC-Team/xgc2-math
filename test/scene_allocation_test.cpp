// Steady-state allocation of xgc2_math::scene scans. A sensor that has scanned a scene once (and a caller that reuses
// its output buffers) must not allocate again: not per ray, not per point, not per scan, range noise and vehicle bodies
// included. The test replaces the global allocation functions with counting ones and repeats scans from the same
// poses, so no buffer needs to grow.

#include <Eigen/Geometry>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "xgc2_math/scene.hpp"

namespace {

std::atomic<std::size_t> g_allocations{0};
std::atomic<bool> g_counting{false};

void* allocate(std::size_t size, std::size_t alignment) {
    if (g_counting.load(std::memory_order_relaxed)) {
        g_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = nullptr;
    if (alignment <= alignof(std::max_align_t)) {
        p = std::malloc(size ? size : 1);
    } else if (posix_memalign(&p, alignment, size ? size : 1) != 0) {
        p = nullptr;
    }
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    return p;
}

} // namespace

void* operator new(std::size_t size) {
    return allocate(size, alignof(std::max_align_t));
}
void* operator new[](std::size_t size) {
    return allocate(size, alignof(std::max_align_t));
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return allocate(size, alignof(std::max_align_t));
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    try {
        return allocate(size, alignof(std::max_align_t));
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    std::free(p);
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
void operator delete(void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}

namespace {

using namespace xgc2_math::scene;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Runs `work` and returns how many allocations it made.
template <class F> std::size_t allocationsOf(F&& work) {
    g_allocations.store(0);
    g_counting.store(true);
    work();
    g_counting.store(false);
    return g_allocations.load();
}

std::vector<Obstacle> clutter() {
    std::mt19937_64 engine(5);
    const auto unit = [&engine] {
        return static_cast<double>(engine() >> 11) / 9007199254740992.0;
    };
    std::vector<Obstacle> out;
    for (int i = 0; i < 60; ++i) {
        const Eigen::Vector3d c(20.0 * unit() - 10.0, 20.0 * unit() - 10.0, 4.0 * unit() - 2.0);
        const Eigen::Quaterniond q =
            Eigen::Quaterniond(unit() - 0.5, unit() - 0.5, unit() - 0.5, unit() + 0.5).normalized();
        switch (i % 5) {
        case 0:
            out.push_back(Obstacle::box(c, Eigen::Vector3d(1.0 + unit(), 1.0 + unit(), 1.0 + unit()), q));
            break;
        case 1:
            out.push_back(Obstacle::sphere(c, 0.4 + 0.5 * unit()));
            break;
        case 2:
            out.push_back(Obstacle::cylinder(c, 0.3 + 0.3 * unit(), 1.0 + unit(), q));
            break;
        case 3:
            out.push_back(Obstacle::capsule(c, 0.2 + 0.2 * unit(), 1.0 + unit(), q));
            break;
        default:
            out.push_back(Obstacle::convexHull({{-0.4, -0.4, -0.5},
                                                {0.5, -0.3, -0.5},
                                                {0.4, 0.5, -0.5},
                                                {-0.4, 0.4, -0.5},
                                                {-0.2, -0.1, 0.8},
                                                {0.3, -0.2, 0.7},
                                                {0.3, 0.2, 0.8},
                                                {-0.2, 0.3, 0.7}},
                                               c, q));
            break;
        }
    }
    return out;
}

void counterSeesAllocations() {
    // The counting is only worth something if it sees what the library might do.
    std::size_t sink = 0;
    const std::size_t vector_allocations = allocationsOf([&] {
        std::vector<Eigen::Vector3d> points(100);
        sink += points.size();
    });
    const std::size_t seed_allocations = allocationsOf([&] {
        std::seed_seq seq{1u, 2u, 3u};
        sink += seq.size();
    });
    require(vector_allocations >= 1, "the counter sees a vector of points");
    require(seed_allocations >= 1, "the counter sees the allocation of a std::seed_seq, which scans avoid");
    require(sink == 103, "the probes ran");
}

void steadyStateScans() {
    const std::vector<Obstacle> obstacles = clutter();
    const std::vector<VehicleBody> no_bodies;
    const std::vector<VehicleBody> bodies{{3, {1.5, 0.5, 0.0}, 0.4}, {7, {-3.0, 2.0, 0.5}, 0.3}};
    const std::array<Eigen::Vector3d, 4> poses{{{0, 0, 3}, {2, -1, 3}, {-3, 4, 3.5}, {5, 5, 2}}};
    const Eigen::Quaterniond attitude =
        Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, 3).normalized()));
    const SensorConfig::Mode modes[] = {SensorConfig::kRaycast, SensorConfig::kPenetrating,
                                        SensorConfig::kDepthFrustum};
    for (const SensorConfig::Mode mode : modes) {
        for (const double noise : {0.0, 0.02}) {
            for (const bool with_bodies : {false, true}) {
                SensorConfig c;
                c.mode = mode;
                c.range = 9.0;
                c.noise_std = noise;
                c.seed = 12;
                c.v_fov_deg = mode == SensorConfig::kPenetrating ? 180.0 : 40.0;
                c.h_fov_deg = mode == SensorConfig::kDepthFrustum ? 90.0 : 360.0;
                c.width = 64;
                c.height = 48;
                c.penetrating_heading_crop = mode == SensorConfig::kPenetrating && noise > 0.0;
                WorldLidar lidar(c);
                lidar.setScene(obstacles);
                const std::vector<VehicleBody>& others = with_bodies ? bodies : no_bodies;
                std::vector<std::uint8_t> packed;
                std::vector<TaggedPoint> tagged;
                BeamScan beam_scan;
                const std::string where = "mode " + std::to_string(mode) + " noise " + std::to_string(noise) +
                                          (with_bodies ? " with bodies" : " without bodies");
                const auto sweep = [&] {
                    for (const Eigen::Vector3d& position : poses) {
                        lidar.scanInto(position, attitude, others, true, &packed);
                        lidar.scanTaggedInto(position, attitude, others, &tagged);
                        if (mode != SensorConfig::kPenetrating) {
                            lidar.scanBeamsInto(position, attitude, others, &beam_scan);
                        }
                    }
                };
                sweep(); // warm-up: buffers grow to the largest scan of the sweep
                sweep();
                const std::size_t allocations = allocationsOf([&] {
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        sweep();
                    }
                });
                require(allocations == 0, where + ": " + std::to_string(allocations) + " allocations in steady state");
                require(!packed.empty() || !tagged.empty() || !beam_scan.range.empty(), where + ": scans are empty");
            }
        }
    }
}

void steadyStateQueries() {
    const std::vector<Obstacle> obstacles = clutter();
    SensorConfig c;
    c.mode = SensorConfig::kRaycast;
    WorldLidar lidar(c);
    lidar.setScene(obstacles);
    const auto scene = lidar.scene();
    double sink = 0.0;
    const std::size_t allocations = allocationsOf([&] {
        for (int i = 0; i < 2000; ++i) {
            const Eigen::Vector3d origin(0.01 * i - 10.0, 0.5, 1.0);
            const Eigen::Vector3d direction = Eigen::Vector3d(1.0, 0.3, -0.05 * (i % 7)).normalized();
            sink += lidar.castRay(origin, direction, 30.0);
            sink += scene->closest(origin, direction, 30.0);
            sink += lidar.surfaceResidual(origin);
        }
        RangeNoise noise;
        for (std::uint64_t index = 0; index < 200; ++index) {
            noise.reseed(7, index, 0.05);
            sink += noise() + noise();
        }
    });
    require(allocations == 0, std::to_string(allocations) + " allocations by queries and noise");
    require(std::isfinite(sink) || std::isinf(sink), "the queries ran");
}

void scanScratchIsBounded() {
    // The scratch space of a scan follows the scene and the bodies, not the number of scans: a thousand scans of a
    // sensor leave the capacity of the output buffers where the first one put it.
    const std::vector<Obstacle> obstacles = clutter();
    SensorConfig c;
    c.mode = SensorConfig::kPenetrating;
    c.range = 8.0;
    c.noise_std = 0.01;
    WorldLidar lidar(c);
    lidar.setScene(obstacles);
    std::vector<std::uint8_t> packed;
    lidar.scanInto({0, 0, 3}, Eigen::Quaterniond::Identity(), {}, false, &packed);
    const std::size_t after_first = packed.capacity();
    for (int i = 0; i < 1000; ++i) {
        lidar.scanInto({0, 0, 3}, Eigen::Quaterniond::Identity(), {}, false, &packed);
    }
    require(packed.capacity() == after_first, "the output capacity stays at " + std::to_string(after_first) + ", is " +
                                                  std::to_string(packed.capacity()));
}

} // namespace

int main(int argc, char** argv) {
    struct Test {
        const char* name;
        void (*run)();
    };
    const std::array<Test, 4> tests{{
        {"counter_sees_allocations", counterSeesAllocations},
        {"steady_state_scans", steadyStateScans},
        {"steady_state_queries", steadyStateQueries},
        {"scan_scratch_is_bounded", scanScratchIsBounded},
    }};
    if (argc > 2) {
        std::cerr << "Usage: scene_allocation_test [case_name|--list]\n";
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
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    if (executed == 0) {
        std::cerr << "Unknown case: " << requested << " (use --list)\n";
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
