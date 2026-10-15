#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "xgc2_math/scene/shape_table.hpp"

namespace xgc2_math::scene::detail {

// Bounding-volume hierarchy over the bounds of the shapes of a ShapeTable, built top-down with a binned surface-area
// heuristic and at most kLeafSize shapes per leaf.
//
// A node holds the boxes of both its children next to their codes, laid out axis by axis with the two children side
// by side, so a traversal step reads one contiguous record and tests both child boxes with two-lane vector
// operations without touching the children. A child code is either the index of an inner node or, with the high bit
// set, a leaf: (first << 3 | count) into the leaf order. Nodes are stored in pre-order, so children always come after
// their parent.
class Bvh {
  public:
    static constexpr std::uint32_t kLeafSize = 2;
    // A refit tree is rebuilt when its summed box area exceeds this multiple of the area it had when it was built.
    static constexpr double kRefitCostLimit = 1.5;

    bool empty() const { return root_ == kEmpty; }

    // Builds the hierarchy over all shapes of the table.
    void build(const ShapeTable& shapes) {
        if (shapes.size() >= (std::size_t{1} << 28)) {
            throw std::invalid_argument("too many shapes");
        }
        nodes_.clear();
        order_.resize(shapes.size());
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            order_[i] = static_cast<std::uint32_t>(i);
        }
        root_ = kEmpty;
        root_box_ = Aabb();
        if (shapes.size() == 0) {
            built_cost_ = cost_ = 0.0;
            return;
        }
        std::vector<Eigen::Vector3d> centers(shapes.size());
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            centers[i] = shapes.bounds(i).center();
        }
        root_ = buildRange(shapes, centers, 0, static_cast<std::uint32_t>(shapes.size()), 0, &root_box_);
        cost_ = built_cost_ = totalArea();
    }

    // Recomputes every box from the current bounds of the shapes, keeping the topology. The table must hold the same
    // number of shapes as at build time. Returns false when the boxes have grown so much that a new build is due.
    bool refit(const ShapeTable& shapes) {
        if (shapes.size() != order_.size()) {
            throw std::invalid_argument("refit needs the shape count of the build");
        }
        if (empty()) {
            return true;
        }
        for (std::size_t k = nodes_.size(); k-- > 0;) {
            Node& node = nodes_[k];
            for (int c = 0; c < 2; ++c) {
                const Aabb box = boxOf(shapes, node.child[c]);
                setBox(&node, c, box);
            }
        }
        root_box_ = boxOf(shapes, root_);
        cost_ = totalArea();
        return cost_ <= kRefitCostLimit * built_cost_;
    }

    // Distance along the unit direction `d` from `o` to the first surface within [0, tmax], or +infinity. A ray that
    // starts inside a solid reports the surface it leaves through.
    double closest(const ShapeTable& shapes, const Eigen::Vector3d& o, const Eigen::Vector3d& d, double tmax) const {
        if (empty()) {
            return kInf;
        }
        double inv[3];
        int near_side[3]; // box side that a ray enters through along each axis: 0 low, 1 high
        for (int k = 0; k < 3; ++k) {
            inv[k] = 1.0 / (std::abs(d[k]) < 1e-300 ? 1e-300 : d[k]);
            near_side[k] = inv[k] < 0.0 ? 1 : 0;
        }
        // Distance at which the ray enters the box [lo, hi] within [0, limit], or +infinity.
        const auto enter = [&](const double* lo, const double* hi, double limit) {
            const double* side[2] = {lo, hi};
            double t0 = 0.0, t1 = limit;
            for (int k = 0; k < 3; ++k) {
                const double a = (side[near_side[k]][k] - o[k]) * inv[k];
                const double b = (side[1 - near_side[k]][k] - o[k]) * inv[k];
                t0 = std::max(t0, a);
                t1 = std::min(t1, b);
            }
            return t0 <= t1 ? t0 : kInf;
        };
        // An entry keeps the distance at which its box is entered. A node is pushed only when its box is entered
        // within min(best, tmax) at that time, so comparing with a smaller `best` at pop time rejects it exactly when
        // that entry distance exceeds `best`.
        struct Entry {
            std::uint32_t code;
            double t;
        };
        Entry stack[kStackSize];
        int top = 0;
        double best = kInf;
        const double t_root = enter(root_box_.lo.data(), root_box_.hi.data(), tmax);
        if (t_root < kInf) {
            stack[top++] = {root_, t_root};
        }
        using Lane = Eigen::Array<double, 2, 1>; // the two children of a node
        const Lane origin_lane[3] = {Lane::Constant(o[0]), Lane::Constant(o[1]), Lane::Constant(o[2])};
        const Lane inverse_lane[3] = {Lane::Constant(inv[0]), Lane::Constant(inv[1]), Lane::Constant(inv[2])};
        while (top > 0) {
            const Entry entry = stack[--top];
            if (entry.t > best) {
                continue;
            }
            if (entry.code & kLeafFlag) {
                const std::uint32_t first = (entry.code & ~kLeafFlag) >> 3, count = entry.code & 7u;
                for (std::uint32_t i = first; i < first + count; ++i) {
                    const std::uint32_t shape = order_[i];
                    // The leaf box encloses its shapes loosely; most rays that enter it miss the shape itself, which
                    // a test against the shape's own box finds out more cheaply than the shape test.
                    const Aabb& box = shapes.bounds(shape);
                    if (enter(box.lo.data(), box.hi.data(), std::min(best, tmax)) == kInf) {
                        continue;
                    }
                    const double t = shapes.closestSurface(shape, o, d, std::min(tmax, best));
                    if (t >= 0.0 && t <= tmax && t < best) {
                        best = t;
                    }
                }
                continue;
            }
            const Node& node = nodes_[entry.code];
            Lane t0 = Lane::Zero();
            Lane t1 = Lane::Constant(std::min(best, tmax));
            for (int k = 0; k < 3; ++k) {
                const Lane near_bound = Eigen::Map<const Lane>(node.bound[near_side[k]][k]);
                const Lane far_bound = Eigen::Map<const Lane>(node.bound[1 - near_side[k]][k]);
                t0 = t0.max((near_bound - origin_lane[k]) * inverse_lane[k]);
                t1 = t1.min((far_bound - origin_lane[k]) * inverse_lane[k]);
            }
            const double tl = t0[0] <= t1[0] ? t0[0] : kInf;
            const double tr = t0[1] <= t1[1] ? t0[1] : kInf;
            // Push the far child first so the near one is visited first.
            if (tl <= tr) {
                if (tr < kInf) {
                    stack[top++] = {node.child[1], tr};
                }
                if (tl < kInf) {
                    stack[top++] = {node.child[0], tl};
                }
            } else {
                if (tl < kInf) {
                    stack[top++] = {node.child[0], tl};
                }
                if (tr < kInf) {
                    stack[top++] = {node.child[1], tr};
                }
            }
        }
        return best;
    }

    // Appends every shape other than `skip` whose bounds overlap `box` (closed boxes). Each node box encloses the
    // bounds of its shapes, so pruning never drops an overlapping shape.
    void overlapping(const ShapeTable& shapes, const Aabb& box, std::size_t skip,
                     std::vector<std::uint32_t>* out) const {
        if (empty() || !box.overlaps(root_box_)) {
            return;
        }
        std::uint32_t stack[kStackSize];
        int top = 0;
        stack[top++] = root_;
        while (top > 0) {
            const std::uint32_t code = stack[--top];
            if (code & kLeafFlag) {
                const std::uint32_t first = (code & ~kLeafFlag) >> 3, count = code & 7u;
                for (std::uint32_t i = first; i < first + count; ++i) {
                    const std::uint32_t j = order_[i];
                    if (j != skip && box.overlaps(shapes.bounds(j))) {
                        out->push_back(j);
                    }
                }
                continue;
            }
            const Node& node = nodes_[code];
            for (int c = 1; c >= 0; --c) {
                if (box.overlaps(boxOf(node, c))) {
                    stack[top++] = node.child[c];
                }
            }
        }
    }

    // Calls visit(shape) for every shape whose bounds come within sqrt(r2) of `center`, in no particular order.
    template <class Visitor>
    void forEachInSphere(const ShapeTable& shapes, const Eigen::Vector3d& center, double r2, Visitor&& visit) const {
        if (empty() || distance2(root_box_, center) > r2) {
            return;
        }
        std::uint32_t stack[kStackSize];
        int top = 0;
        stack[top++] = root_;
        while (top > 0) {
            const std::uint32_t code = stack[--top];
            if (code & kLeafFlag) {
                const std::uint32_t first = (code & ~kLeafFlag) >> 3, count = code & 7u;
                for (std::uint32_t i = first; i < first + count; ++i) {
                    const std::uint32_t shape = order_[i];
                    if (distance2(shapes.bounds(shape), center) <= r2) {
                        visit(shape);
                    }
                }
                continue;
            }
            const Node& node = nodes_[code];
            for (int c = 1; c >= 0; --c) {
                if (distance2(boxOf(node, c), center) <= r2) {
                    stack[top++] = node.child[c];
                }
            }
        }
    }

  private:
    static constexpr std::uint32_t kLeafFlag = 0x80000000u;
    static constexpr std::uint32_t kEmpty = 0xffffffffu;
    // Splits by area stop paying off in clutter that cannot be told apart; below this depth the tree is split at the
    // median, which keeps it shallower than kMaxSahDepth + log2(shapes) <= 24 + 28 for any scene.
    static constexpr std::uint32_t kMaxSahDepth = 24;
    static constexpr int kStackSize = 64;
    static constexpr int kBins = 16;

    struct Node {
        double bound[2][3][2]; // [low, high][axis][child]
        std::uint32_t child[2];
    };

    static void setBox(Node* node, int child, const Aabb& box) {
        for (int axis = 0; axis < 3; ++axis) {
            node->bound[0][axis][child] = box.lo[axis];
            node->bound[1][axis][child] = box.hi[axis];
        }
    }

    static Aabb boxOf(const Node& node, int child) {
        Aabb box;
        box.lo = Eigen::Vector3d(node.bound[0][0][child], node.bound[0][1][child], node.bound[0][2][child]);
        box.hi = Eigen::Vector3d(node.bound[1][0][child], node.bound[1][1][child], node.bound[1][2][child]);
        return box;
    }

    // Squared distance from `point` to the box (zero inside).
    static double distance2(const Aabb& box, const Eigen::Vector3d& point) {
        const Eigen::Vector3d outside = (box.lo - point).cwiseMax(point - box.hi).cwiseMax(0.0);
        return outside.squaredNorm();
    }

    static double area(const Aabb& b) {
        const Eigen::Vector3d e = (b.hi - b.lo).cwiseMax(0.0);
        return 2.0 * (e.x() * e.y() + e.y() * e.z() + e.z() * e.x());
    }

    double totalArea() const {
        return std::accumulate(nodes_.begin(), nodes_.end(), area(root_box_), [](double sum, const Node& node) {
            return sum + (area(boxOf(node, 0)) + area(boxOf(node, 1)));
        });
    }

    // Box enclosing the shapes below a child code, from the current shape bounds and the stored child boxes.
    Aabb boxOf(const ShapeTable& shapes, std::uint32_t code) const {
        Aabb box;
        if (code & kLeafFlag) {
            const std::uint32_t first = (code & ~kLeafFlag) >> 3, count = code & 7u;
            for (std::uint32_t i = first; i < first + count; ++i) {
                box.grow(shapes.bounds(order_[i]));
            }
            return box;
        }
        const Node& node = nodes_[code];
        box.grow(boxOf(node, 0));
        box.grow(boxOf(node, 1));
        return box;
    }

    // Reorders order_[first, first + count) around the cheapest split by the binned surface-area heuristic along
    // `axis` and returns the first index of the right part, or `first` when no bin boundary separates the shapes.
    std::uint32_t splitBySurfaceArea(const ShapeTable& shapes, const std::vector<Eigen::Vector3d>& centers,
                                     const Aabb& spread, int axis, std::uint32_t first, std::uint32_t count) {
        const double extent = spread.hi[axis] - spread.lo[axis];
        if (!(extent > 0.0)) {
            return first;
        }
        const double scale = kBins / extent;
        const auto binOf = [&](std::uint32_t shape) {
            return std::min(kBins - 1, static_cast<int>((centers[shape][axis] - spread.lo[axis]) * scale));
        };
        Aabb bin_box[kBins];
        std::uint32_t bin_count[kBins] = {};
        for (std::uint32_t i = first; i < first + count; ++i) {
            const int b = binOf(order_[i]);
            ++bin_count[b];
            bin_box[b].grow(shapes.bounds(order_[i]));
        }
        double right_cost[kBins]; // area * count of the shapes in bins b .. kBins - 1
        Aabb right;
        std::uint32_t right_count = 0;
        for (int b = kBins - 1; b > 0; --b) {
            right.grow(bin_box[b]);
            right_count += bin_count[b];
            right_cost[b] = right_count ? area(right) * right_count : 0.0;
        }
        Aabb left;
        std::uint32_t left_count = 0;
        double best_cost = kInf;
        int best_bin = -1;
        for (int b = 0; b < kBins - 1; ++b) {
            left.grow(bin_box[b]);
            left_count += bin_count[b];
            if (left_count == 0 || left_count == count) {
                continue;
            }
            const double cost = area(left) * left_count + right_cost[b + 1];
            if (cost < best_cost) {
                best_cost = cost;
                best_bin = b;
            }
        }
        if (best_bin < 0) {
            return first;
        }
        const auto middle =
            std::partition(order_.begin() + first, order_.begin() + first + count, [&](std::uint32_t shape) {
                return binOf(shape) <= best_bin;
            });
        return static_cast<std::uint32_t>(middle - order_.begin());
    }

    // Builds the subtree over order_[first, first + count) and returns its code; *box receives the enclosing box.
    std::uint32_t buildRange(const ShapeTable& shapes, const std::vector<Eigen::Vector3d>& centers, std::uint32_t first,
                             std::uint32_t count, std::uint32_t depth, Aabb* box) {
        Aabb spread;
        for (std::uint32_t i = first; i < first + count; ++i) {
            box->grow(shapes.bounds(order_[i]));
            spread.grow(centers[order_[i]]);
        }
        if (count <= kLeafSize) {
            return kLeafFlag | (first << 3) | count;
        }
        int axis = 0;
        (spread.hi - spread.lo).maxCoeff(&axis);
        std::uint32_t mid =
            depth < kMaxSahDepth ? splitBySurfaceArea(shapes, centers, spread, axis, first, count) : first;
        if (mid <= first || mid >= first + count) {
            mid = first + count / 2;
            std::nth_element(order_.begin() + first, order_.begin() + mid, order_.begin() + first + count,
                             [&](std::uint32_t a, std::uint32_t b) {
                                 return centers[a][axis] < centers[b][axis];
                             });
        }
        const std::uint32_t index = static_cast<std::uint32_t>(nodes_.size());
        nodes_.emplace_back();
        Aabb left_box, right_box;
        const std::uint32_t left = buildRange(shapes, centers, first, mid - first, depth + 1, &left_box);
        const std::uint32_t right = buildRange(shapes, centers, mid, first + count - mid, depth + 1, &right_box);
        Node& node = nodes_[index];
        node.child[0] = left;
        node.child[1] = right;
        setBox(&node, 0, left_box);
        setBox(&node, 1, right_box);
        return index;
    }

    std::vector<Node> nodes_;
    std::vector<std::uint32_t> order_; // shape indices in leaf order
    Aabb root_box_;
    std::uint32_t root_ = kEmpty;
    double built_cost_ = 0.0;
    double cost_ = 0.0;
};

} // namespace xgc2_math::scene::detail
