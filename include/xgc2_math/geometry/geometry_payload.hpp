#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xgc2_math::geometry {

// Wire identity: no pose, source identity, version or approximation lives in this payload.
struct GeometryPayload {
    uint32_t type = 0;
    std::vector<uint8_t> parameters;
    bool operator==(const GeometryPayload& other) const { return type == other.type && parameters == other.parameters; }
    bool operator!=(const GeometryPayload& other) const { return !(*this == other); }
};

enum class GeometryType : uint32_t {
#define XGC2_GEOMETRY(cpp, name, number, fields) cpp = number,
#include "xgc2_math/geometry/geometry_registry.def"
#undef XGC2_GEOMETRY
};
struct GeometryTypeInfo {
    GeometryType type;
    std::string_view name;
    std::string_view fields;
};
inline constexpr GeometryTypeInfo geometryRegistry[] = {
#define XGC2_GEOMETRY(cpp, name, number, fields) {GeometryType::cpp, #name, fields},
#include "xgc2_math/geometry/geometry_registry.def"
#undef XGC2_GEOMETRY
};
inline const GeometryTypeInfo& geometryTypeInfo(uint32_t type) {
    for (const auto& info : geometryRegistry)
        if (static_cast<uint32_t>(info.type) == type)
            return info;
    throw std::invalid_argument("unknown geometry type: " + std::to_string(type));
}
inline GeometryType geometryTypeFromName(std::string_view name) {
    const auto info = std::find_if(std::begin(geometryRegistry), std::end(geometryRegistry), [&](const auto& entry) {
        return entry.name == name;
    });
    if (info != std::end(geometryRegistry))
        return info->type;
    throw std::invalid_argument("unknown geometry name: " + std::string(name));
}
inline std::string_view geometryTypeName(GeometryType type) {
    return geometryTypeInfo(static_cast<uint32_t>(type)).name;
}

using GeometryVertex = std::array<double, 3>;
using GeometryTriangle = std::array<uint32_t, 3>;
using GeometryHalfspace = std::array<double, 4>; // n dot x + d <= 0

// Author/decoded representation, local XYZ, metres. Only fields belonging to `type` are used.
struct Geometry {
    GeometryType type = GeometryType::Box;
    std::array<double, 3> size{}; // box: full lengths
    double radius = 0;
    double height = 0;             // cylinder: full height; capsule: straight height
    std::array<double, 2> radii{}; // ring ellipse semiaxes in local XY
    double tubeRadius = 0;         // Euclidean circular tube
    std::array<double, 3> semiaxes{};
    std::vector<GeometryVertex> vertices;
    std::vector<GeometryTriangle> triangles;
    std::vector<GeometryHalfspace> planes;

    static Geometry box(std::array<double, 3> size) {
        Geometry g;
        g.size = size;
        return g;
    }
    static Geometry sphere(double radius) {
        Geometry g;
        g.type = GeometryType::Sphere;
        g.radius = radius;
        return g;
    }
    static Geometry cylinder(double radius, double height) {
        auto g = sphere(radius);
        g.type = GeometryType::Cylinder;
        g.height = height;
        return g;
    }
    static Geometry capsule(double radius, double straightHeight) {
        auto g = cylinder(radius, straightHeight);
        g.type = GeometryType::Capsule;
        return g;
    }
    static Geometry convex(std::vector<GeometryVertex> vertices, std::vector<GeometryTriangle> triangles = {}) {
        Geometry g;
        g.type = GeometryType::Convex;
        g.vertices = std::move(vertices);
        g.triangles = std::move(triangles);
        return g;
    }
    static Geometry ring(std::array<double, 2> radii, double tubeRadius) {
        Geometry g;
        g.type = GeometryType::Ring;
        g.radii = radii;
        g.tubeRadius = tubeRadius;
        return g;
    }
    static Geometry halfspaces(std::vector<GeometryHalfspace> planes) {
        Geometry g;
        g.type = GeometryType::Halfspaces;
        g.planes = std::move(planes);
        return g;
    }
    static Geometry ellipsoid(std::array<double, 3> semiaxes) {
        Geometry g;
        g.type = GeometryType::Ellipsoid;
        g.semiaxes = semiaxes;
        return g;
    }
    static Geometry mesh(std::vector<GeometryVertex> vertices, std::vector<GeometryTriangle> triangles) {
        auto g = convex(std::move(vertices), std::move(triangles));
        g.type = GeometryType::Mesh;
        return g;
    }
};
using DecodedGeometry = Geometry;

inline void validateGeometry(const Geometry& g) {
    (void)geometryTypeInfo(static_cast<uint32_t>(g.type));
    const auto positive = [](double x) {
        if (!std::isfinite(x) || x <= 0)
            throw std::invalid_argument("geometry dimension must be positive and finite");
    };
    switch (g.type) {
    case GeometryType::Box:
        for (double x : g.size)
            positive(x);
        break;
    case GeometryType::Ellipsoid:
        for (double x : g.semiaxes)
            positive(x);
        break;
    case GeometryType::Sphere:
        positive(g.radius);
        break;
    case GeometryType::Cylinder:
        positive(g.radius);
        positive(g.height);
        break;
    case GeometryType::Capsule:
        positive(g.radius);
        if (!std::isfinite(g.height) || g.height < 0)
            throw std::invalid_argument("capsule straight height must be nonnegative and finite");
        break;
    case GeometryType::Ring:
        for (double x : g.radii)
            positive(x);
        positive(g.tubeRadius);
        if (g.tubeRadius >= std::min(g.radii[0], g.radii[1]))
            throw std::invalid_argument("ring tubeRadius must preserve hole");
        break;
    case GeometryType::Convex:
    case GeometryType::Mesh:
        if (g.vertices.size() < 4 || g.vertices.size() > UINT32_MAX || g.triangles.size() > UINT32_MAX)
            throw std::invalid_argument("geometry vertex/triangle count out of range");
        if (g.type == GeometryType::Mesh && g.triangles.empty())
            throw std::invalid_argument("mesh requires explicit triangles");
        for (const auto& v : g.vertices)
            for (double x : v)
                if (!std::isfinite(x))
                    throw std::invalid_argument("nonfinite vertex");
        for (const auto& t : g.triangles) {
            if (std::any_of(t.begin(), t.end(), [&](auto i) {
                    return i >= g.vertices.size();
                }))
                throw std::invalid_argument("triangle index out of range");
            if (t[0] == t[1] || t[0] == t[2] || t[1] == t[2])
                throw std::invalid_argument("repeated triangle index");
        }
        break;
    case GeometryType::Halfspaces:
        if (g.planes.empty() || g.planes.size() > UINT32_MAX)
            throw std::invalid_argument("halfspaces require nonempty uint32 row count");
        for (const auto& p : g.planes) {
            for (double x : p)
                if (!std::isfinite(x))
                    throw std::invalid_argument("nonfinite halfspace");
            if (std::hypot(p[0], std::hypot(p[1], p[2])) == 0)
                throw std::invalid_argument("zero halfspace normal");
        }
        break;
    }
}

namespace payload_detail {
inline void u32(std::vector<uint8_t>& out, uint32_t x) {
    for (unsigned i = 0; i < 4; ++i)
        out.push_back(static_cast<uint8_t>(x >> (i * 8)));
}
inline void f64(std::vector<uint8_t>& out, double x) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "binary64 required");
    uint64_t bits;
    std::memcpy(&bits, &x, 8);
    for (unsigned i = 0; i < 8; ++i)
        out.push_back(static_cast<uint8_t>(bits >> (i * 8)));
}
struct Reader {
    const std::vector<uint8_t>& data;
    std::size_t offset = 0;
    void require(std::size_t n) const {
        if (n > data.size() - offset)
            throw std::invalid_argument("truncated geometry parameters");
    }
    uint32_t u32() {
        require(4);
        uint32_t x = 0;
        for (unsigned i = 0; i < 4; ++i)
            x |= uint32_t(data[offset++]) << (i * 8);
        return x;
    }
    double f64() {
        require(8);
        uint64_t x = 0;
        for (unsigned i = 0; i < 8; ++i)
            x |= uint64_t(data[offset++]) << (i * 8);
        double d;
        std::memcpy(&d, &x, 8);
        return d;
    }
};
} // namespace payload_detail

inline GeometryPayload encodeGeometry(const Geometry& g) {
    validateGeometry(g);
    GeometryPayload out{static_cast<uint32_t>(g.type), {}};
    auto& bytes = out.parameters;
    const auto scalar = [&](double x) {
        payload_detail::f64(bytes, x);
    };
    switch (g.type) {
    case GeometryType::Box:
        for (double x : g.size)
            scalar(x);
        break;
    case GeometryType::Sphere:
        scalar(g.radius);
        break;
    case GeometryType::Cylinder:
    case GeometryType::Capsule:
        scalar(g.radius);
        scalar(g.height);
        break;
    case GeometryType::Ring:
        for (double x : g.radii)
            scalar(x);
        scalar(g.tubeRadius);
        break;
    case GeometryType::Ellipsoid:
        for (double x : g.semiaxes)
            scalar(x);
        break;
    case GeometryType::Convex:
    case GeometryType::Mesh:
        payload_detail::u32(bytes, static_cast<uint32_t>(g.vertices.size()));
        payload_detail::u32(bytes, static_cast<uint32_t>(g.triangles.size()));
        for (const auto& v : g.vertices)
            for (double x : v)
                scalar(x);
        for (const auto& t : g.triangles)
            for (auto x : t)
                payload_detail::u32(bytes, x);
        break;
    case GeometryType::Halfspaces:
        payload_detail::u32(bytes, static_cast<uint32_t>(g.planes.size()));
        for (const auto& p : g.planes)
            for (double x : p)
                scalar(x);
        break;
    }
    return out;
}
inline Geometry decodeGeometry(const GeometryPayload& payload) {
    Geometry g;
    g.type = geometryTypeInfo(payload.type).type;
    payload_detail::Reader r{payload.parameters};
    switch (g.type) {
    case GeometryType::Box:
        std::generate(g.size.begin(), g.size.end(), [&] {
            return r.f64();
        });
        break;
    case GeometryType::Sphere:
        g.radius = r.f64();
        break;
    case GeometryType::Cylinder:
    case GeometryType::Capsule:
        g.radius = r.f64();
        g.height = r.f64();
        break;
    case GeometryType::Ring:
        std::generate(g.radii.begin(), g.radii.end(), [&] {
            return r.f64();
        });
        g.tubeRadius = r.f64();
        break;
    case GeometryType::Ellipsoid:
        std::generate(g.semiaxes.begin(), g.semiaxes.end(), [&] {
            return r.f64();
        });
        break;
    case GeometryType::Convex:
    case GeometryType::Mesh: {
        const auto nv = r.u32(), nt = r.u32();
        // Validate wire counts against remaining bytes before any allocation.
        if (uint64_t(nv) * 24 + uint64_t(nt) * 12 != payload.parameters.size() - r.offset)
            throw std::invalid_argument("geometry array counts do not match payload length");
        g.vertices.resize(nv);
        g.triangles.resize(nt);
        for (auto& v : g.vertices)
            std::generate(v.begin(), v.end(), [&] {
                return r.f64();
            });
        for (auto& t : g.triangles)
            std::generate(t.begin(), t.end(), [&] {
                return r.u32();
            });
        break;
    }
    case GeometryType::Halfspaces: {
        const auto n = r.u32();
        if (uint64_t(n) * 32 != payload.parameters.size() - r.offset)
            throw std::invalid_argument("halfspace count does not match payload length");
        g.planes.resize(n);
        for (auto& p : g.planes)
            std::generate(p.begin(), p.end(), [&] {
                return r.f64();
            });
        break;
    }
    }
    if (r.offset != payload.parameters.size())
        throw std::invalid_argument("trailing geometry parameters");
    validateGeometry(g);
    return g;
}
} // namespace xgc2_math::geometry
