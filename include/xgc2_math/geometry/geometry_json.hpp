#pragma once
#include "xgc2_math/geometry/geometry_payload.hpp"

namespace xgc2_math::geometry {
namespace json_detail {
template <class Json> double scalar(const Json& value) {
    if (!value.is_number())
        throw std::invalid_argument("geometry parameter must be a number, not boolean");
    const double number = value.template get<double>();
    if (!std::isfinite(number))
        throw std::invalid_argument("geometry parameter must be finite");
    return number;
}
template <std::size_t N, class Json> std::array<double, N> vector(const Json& value) {
    if (!value.is_array() || value.size() != N)
        throw std::invalid_argument("geometry array dimension mismatch");
    std::array<double, N> out{};
    for (std::size_t i = 0; i < N; ++i)
        out[i] = scalar(value.at(i));
    return out;
}
} // namespace json_detail
// Json is a nlohmann-compatible JSON value. No JSON dependency enters the math component.
// Human-readable author representation <-> canonical wire payload; cold path only.
template <class Json> GeometryPayload geometryFromJson(const Json& json) {
    Geometry g;
    g.type = geometryTypeFromName(json.at("type").template get<std::string>());
    switch (g.type) {
    case GeometryType::Box:
        g.size = json_detail::vector<3>(json.at("size"));
        break;
    case GeometryType::Sphere:
        g.radius = json_detail::scalar(json.at("radius"));
        break;
    case GeometryType::Cylinder:
    case GeometryType::Capsule:
        g.radius = json_detail::scalar(json.at("radius"));
        g.height = json_detail::scalar(json.at("height"));
        break;
    case GeometryType::Ring:
        g.radii = json_detail::vector<2>(json.at("radii"));
        g.tubeRadius = json_detail::scalar(json.at("tubeRadius"));
        break;
    case GeometryType::Ellipsoid:
        g.semiaxes = json_detail::vector<3>(json.at("semiaxes"));
        break;
    case GeometryType::Convex:
    case GeometryType::Mesh:
        if (!json.at("vertices").is_array())
            throw std::invalid_argument("vertices must be an array");
        for (const auto& vertex : json.at("vertices"))
            g.vertices.push_back(json_detail::vector<3>(vertex));
        if (json.contains("triangles")) {
            if (!json.at("triangles").is_array())
                throw std::invalid_argument("triangles must be a flat array");
            std::vector<uint32_t> indices;
            for (const auto& index : json.at("triangles")) {
                if (!index.is_number_integer() || index.template get<double>() < 0 ||
                    index.template get<double>() > UINT32_MAX)
                    throw std::invalid_argument("triangle indices must be uint32 integers");
                indices.push_back(index.template get<uint32_t>());
            }
            if (indices.size() % 3 != 0)
                throw std::invalid_argument("triangles must contain index triples");
            g.triangles.reserve(indices.size() / 3);
            for (std::size_t i = 0; i < indices.size(); i += 3)
                g.triangles.push_back({indices[i], indices[i + 1], indices[i + 2]});
        }
        break;
    case GeometryType::Halfspaces:
        if (!json.at("halfspaces").is_array())
            throw std::invalid_argument("halfspaces must be an array");
        for (const auto& plane : json.at("halfspaces"))
            g.planes.push_back(json_detail::vector<4>(plane));
        break;
    }
    return encodeGeometry(g);
}
template <class Json> Json geometryToJson(const Geometry& g) {
    validateGeometry(g);
    Json out;
    out["type"] = std::string(geometryTypeName(g.type));
    switch (g.type) {
    case GeometryType::Box:
        out["size"] = g.size;
        break;
    case GeometryType::Sphere:
        out["radius"] = g.radius;
        break;
    case GeometryType::Cylinder:
    case GeometryType::Capsule:
        out["radius"] = g.radius;
        out["height"] = g.height;
        break;
    case GeometryType::Ring:
        out["radii"] = g.radii;
        out["tubeRadius"] = g.tubeRadius;
        break;
    case GeometryType::Ellipsoid:
        out["semiaxes"] = g.semiaxes;
        break;
    case GeometryType::Convex:
    case GeometryType::Mesh:
        out["vertices"] = g.vertices;
        if (!g.triangles.empty() || g.type == GeometryType::Mesh) {
            std::vector<uint32_t> indices;
            indices.reserve(g.triangles.size() * 3);
            for (const auto& t : g.triangles)
                indices.insert(indices.end(), t.begin(), t.end());
            out["triangles"] = std::move(indices);
        }
        break;
    case GeometryType::Halfspaces:
        out["halfspaces"] = g.planes;
        break;
    }
    return out;
}
template <class Json> Json geometryToJson(const GeometryPayload& payload) {
    return geometryToJson<Json>(decodeGeometry(payload));
}
} // namespace xgc2_math::geometry
