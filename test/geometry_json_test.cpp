// Optional nlohmann JSON bridge and cross-language corpus export; no JSON dependency in Math runtime.
#include "xgc2_math/geometry/geometry_json.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
using nlohmann::json;
using namespace xgc2_math::geometry;
int main() {
    const auto corpus = json::parse(R"([
        {"type":"box","size":[1,2,3]}, {"type":"sphere","radius":1},
        {"type":"cylinder","radius":0.3,"height":2}, {"type":"capsule","radius":0.5,"height":0},
        {"type":"convex","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],"triangles":[0,2,1,0,1,3,0,3,2,1,2,3]},
        {"type":"ring","radii":[3,2],"tubeRadius":0.25}, {"type":"halfspaces","halfspaces":[[1,0,0,-1]]},
        {"type":"ellipsoid","semiaxes":[1,2,3]},
        {"type":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],"triangles":[0,2,1,0,1,3,0,3,2,1,2,3]}
    ])");
    json output = json::array();
    for (const auto& g : corpus) {
        const auto p = geometryFromJson(g);
        if (geometryToJson<json>(p) != g)
            throw std::runtime_error("named JSON roundtrip changed");
        output.push_back({{"geometry", g}, {"type", p.type}, {"parameters", p.parameters}});
    }
    auto bad = corpus[4];
    bad["triangles"] = {0, 1};
    bool rejected = false;
    try {
        geometryFromJson(bad);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("incomplete triangle accepted");
    const auto invalid = json::parse(R"([
      {"type":"ring","radii":[true,1.2],"tubeRadius":0.1},
      {"type":"ring","radii":[1],"tubeRadius":0.1},
      {"type":"ring","radii":[1,1.2,1.3],"tubeRadius":0.1},
      {"type":"ring","radii":[0,1.2],"tubeRadius":0.1},
      {"type":"box","size":[1,2,3,4]}, {"type":"sphere","radius":true},
      {"type":"ellipsoid","semiaxes":[1,2]},
      {"type":"convex","vertices":[[0,0,0,1],[1,0,0],[0,1,0],[0,0,1]]},
      {"type":"convex","vertices":[[true,0,0],[1,0,0],[0,1,0],[0,0,1]]},
      {"type":"halfspaces","halfspaces":[[1,0,0,-1,9]]},
      {"type":"halfspaces","halfspaces":[[1,0,false,-1]]},
      {"type":"convex","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],"triangles":[0,1.5,2]}
    ])");
    for (const auto& g : invalid) {
        bool failed = false;
        try {
            geometryFromJson(g);
        } catch (const std::exception&) {
            failed = true;
        }
        if (!failed)
            throw std::runtime_error("invalid JSON accepted: " + g.dump());
    }
    std::cout << output.dump(2) << '\n';
}
