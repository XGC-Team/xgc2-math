#pragma once

#include "xgc2_math/scene/triangle_mesh.hpp"

#include <numeric>
#include <utility>

namespace xgc2_math::scene {

// One immutable local mesh, prepared by its numerical owner and shared by rigid instances.
// Vertices/indices are the source; triangles, tolerant bounds and the hierarchy are derived once.
class TriangleMeshData {
  public:
    TriangleMeshData(const TriangleMeshData&) = delete;
    TriangleMeshData& operator=(const TriangleMeshData&) = delete;

    TriangleMeshData(std::vector<Eigen::Vector3d> vertices, std::vector<std::array<uint32_t, 3>> indices)
        : vertices_(std::move(vertices)), indices_(std::move(indices)) {
        validateTriangleMesh(vertices_, indices_);
        if (indices_.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("too many mesh triangles");
        triangles_.reserve(indices_.size());
        std::vector<Bounds> bounds;
        bounds.reserve(indices_.size());
        for (const auto& t : indices_) {
            triangles_.push_back({vertices_[t[0]], vertices_[t[1]], vertices_[t[2]]});
            const auto& triangle = triangles_.back();
            const Eigen::Vector3d e1 = triangle.b - triangle.a, e2 = triangle.c - triangle.a;
            // The unchanged leaf accepts u,v >= -1e-12 and u+v <= 1+1e-12.
            // Its accepted barycentric polygon extends past the exact triangle bounds.
            const Eigen::Vector3d magnitude =
                triangle.a.cwiseAbs().cwiseMax(triangle.b.cwiseAbs()).cwiseMax(triangle.c.cwiseAbs());
            const Eigen::Vector3d padding =
                2e-12 * (e1.cwiseAbs() + e2.cwiseAbs()) + 32 * std::numeric_limits<double>::epsilon() * magnitude;
            Bounds box{triangle.a.cwiseMin(triangle.b).cwiseMin(triangle.c) - padding,
                       triangle.a.cwiseMax(triangle.b).cwiseMax(triangle.c) + padding};
            for (int axis = 0; axis < 3; ++axis) {
                box.lo[axis] = std::nextafter(box.lo[axis], -std::numeric_limits<double>::infinity());
                box.hi[axis] = std::nextafter(box.hi[axis], std::numeric_limits<double>::infinity());
            }
            bounds.push_back(box);
        }
        order_.resize(triangles_.size());
        std::iota(order_.begin(), order_.end(), std::uint32_t{0});
        nodes_.reserve(triangles_.size());
        build(bounds, 0, static_cast<std::uint32_t>(order_.size()));
    }

    const std::vector<Eigen::Vector3d>& vertices() const { return vertices_; }
    const std::vector<std::array<uint32_t, 3>>& triangleIndices() const { return indices_; }
    const std::vector<MeshTriangle>& triangles() const { return triangles_; }
    const Eigen::Vector3d& minCorner() const { return nodes_.front().bounds.lo; }
    const Eigen::Vector3d& maxCorner() const { return nodes_.front().bounds.hi; }

    double rayClosest(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction, double maxRange) const {
        return rayClosest(
            origin, direction, maxRange,
            [](const MeshTriangle& triangle, const Eigen::Vector3d& o, const Eigen::Vector3d& d, double limit) {
                return meshRayClosest(&triangle, 1, o, d, limit);
            });
    }

    // The visitor is the triangle intersection kernel; traversal itself owns no scratch allocation or lock.
    template <class Intersection>
    double rayClosest(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction, double maxRange,
                      Intersection&& intersection) const {
        struct Entry {
            std::uint32_t node;
            double entry;
        };
        // Median splits and uint32 triangle indices bound depth by 32; one far sibling per level.
        std::array<Entry, 64> stack;
        std::size_t top = 0;
        double best = std::numeric_limits<double>::infinity();
        std::uint32_t bestIndex = std::numeric_limits<std::uint32_t>::max();
        const double root = enter(nodes_.front().bounds, origin, direction, maxRange);
        if (root < best)
            stack[top++] = {0, root};
        while (top != 0) {
            const Entry entry = stack[--top];
            if (entry.entry > best)
                continue;
            const auto& node = nodes_[entry.node];
            if (node.count != 0) {
                for (std::uint32_t k = node.first; k < node.first + node.count; ++k) {
                    const auto index = order_[k];
                    const double hit = intersection(triangles_[index], origin, direction, std::min(best, maxRange));
                    // Match the linear oracle's original-index choice even for equal signed-zero hits.
                    if (hit < best || (hit == best && index < bestIndex)) {
                        best = hit;
                        bestIndex = index;
                    }
                }
                continue;
            }
            const double limit = std::min(best, maxRange);
            const double left = enter(nodes_[node.first].bounds, origin, direction, limit);
            const double right = enter(nodes_[node.right].bounds, origin, direction, limit);
            if (left <= right) {
                if (right < std::numeric_limits<double>::infinity())
                    stack[top++] = {node.right, right};
                if (left < std::numeric_limits<double>::infinity())
                    stack[top++] = {node.first, left};
            } else {
                if (left < std::numeric_limits<double>::infinity())
                    stack[top++] = {node.first, left};
                if (right < std::numeric_limits<double>::infinity())
                    stack[top++] = {node.right, right};
            }
        }
        return best;
    }

  private:
    struct Bounds {
        Eigen::Vector3d lo, hi;
    };
    struct Node {
        Bounds bounds;
        std::uint32_t first = 0, count = 0, right = 0;
    };

    static double enter(const Bounds& box, const Eigen::Vector3d& o, const Eigen::Vector3d& d, double limit) {
        double lo = 0, hi = limit;
        for (int axis = 0; axis < 3; ++axis) {
            if (d[axis] == 0) {
                if (o[axis] < box.lo[axis] || o[axis] > box.hi[axis])
                    return std::numeric_limits<double>::infinity();
                continue;
            }
            double a = (box.lo[axis] - o[axis]) / d[axis], b = (box.hi[axis] - o[axis]) / d[axis];
            if (a > b)
                std::swap(a, b);
            lo = std::max(lo, a);
            hi = std::min(hi, b);
            if (lo > hi)
                return std::numeric_limits<double>::infinity();
        }
        return lo <= hi ? lo : std::numeric_limits<double>::infinity();
    }

    std::uint32_t build(const std::vector<Bounds>& bounds, std::uint32_t first, std::uint32_t count) {
        const auto index = static_cast<std::uint32_t>(nodes_.size());
        Node node;
        node.bounds = bounds[order_[first]];
        for (std::uint32_t k = first + 1; k < first + count; ++k) {
            node.bounds.lo = node.bounds.lo.cwiseMin(bounds[order_[k]].lo);
            node.bounds.hi = node.bounds.hi.cwiseMax(bounds[order_[k]].hi);
        }
        nodes_.push_back(node);
        if (count <= 4) {
            nodes_[index].first = first;
            nodes_[index].count = count;
            return index;
        }
        Eigen::Index axis;
        (node.bounds.hi - node.bounds.lo).maxCoeff(&axis);
        const auto half = count / 2;
        std::nth_element(order_.begin() + first, order_.begin() + first + half, order_.begin() + first + count,
                         [&](auto a, auto b) {
                             const double ca = .5 * bounds[a].lo[axis] + .5 * bounds[a].hi[axis];
                             const double cb = .5 * bounds[b].lo[axis] + .5 * bounds[b].hi[axis];
                             return ca < cb || (ca == cb && a < b);
                         });
        const auto left = build(bounds, first, half);
        const auto right = build(bounds, first + half, count - half);
        nodes_[index].first = left;
        nodes_[index].right = right;
        return index;
    }

    std::vector<Eigen::Vector3d> vertices_;
    std::vector<std::array<uint32_t, 3>> indices_;
    std::vector<MeshTriangle> triangles_; // Original order, also used for residuals and sampling.
    std::vector<Node> nodes_;
    std::vector<std::uint32_t> order_; // BVH partitioning never changes the source index/triangle order.
};

// Passing vectors by value lets a cold owner move its source arrays into the one immutable mesh.
inline TriangleMeshPtr prepareTriangleMesh(std::vector<Eigen::Vector3d> vertices,
                                           std::vector<std::array<uint32_t, 3>> triangles) {
    return std::make_shared<const TriangleMeshData>(std::move(vertices), std::move(triangles));
}

} // namespace xgc2_math::scene
