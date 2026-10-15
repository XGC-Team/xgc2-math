#include "xgc2_math/geometry/geometry_payload.hpp"
#include <functional>
#include <iostream>
using namespace xgc2_math::geometry;
static void require(bool value) {
    if (!value)
        throw std::runtime_error("codec test failed");
}
static void rejects(const std::function<void()>& f) {
    bool caught = false;
    try {
        f();
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    require(caught);
}
int main() {
    const std::vector<GeometryVertex> v{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const std::vector<GeometryTriangle> t{{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}};
    const std::vector<Geometry> shapes{
        Geometry::box({2, 3, 4}),
        Geometry::sphere(1),
        Geometry::cylinder(2, 4),
        Geometry::capsule(1, 0),
        Geometry::convex(v),
        Geometry::ring({3, 2}, .2),
        Geometry::halfspaces({{1, 0, 0, -1}, {0, 1, 0, -1}, {0, 0, 1, -1}, {-1, -1, -1, -1}}),
        Geometry::ellipsoid({1, 2, 3}),
        Geometry::mesh(v, t)};
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        auto p = encodeGeometry(shapes[i]);
        require(p.type == i + 1);
        require(encodeGeometry(decodeGeometry(p)) == p);
        auto truncated = p;
        truncated.parameters.pop_back();
        rejects([&] {
            decodeGeometry(truncated);
        });
        p.parameters.push_back(0);
        rejects([&] {
            decodeGeometry(p);
        });
    }
    require(encodeGeometry(Geometry::sphere(1)).parameters == std::vector<uint8_t>({0, 0, 0, 0, 0, 0, 240, 63}));
    rejects([] {
        decodeGeometry({42, {}});
    });
    rejects([] {
        decodeGeometry({5, {255, 255, 255, 255, 255, 255, 255, 255}});
    });
    rejects([] {
        encodeGeometry(Geometry::ring({2, 3}, 2));
    });
    rejects([] {
        encodeGeometry(Geometry::box({1, 0, 2}));
    });
    rejects([] {
        encodeGeometry(Geometry::sphere(std::numeric_limits<double>::infinity()));
    });
    auto bad = Geometry::mesh(v, t);
    bad.triangles[0][0] = 42;
    rejects([&] {
        encodeGeometry(bad);
    });
    std::cout << "geometry codec: 9 layouts, endian golden, roundtrip and malformed inputs pass\n";
}
