#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "xgc2_math/scene/bvh.hpp"
#include "xgc2_math/scene/obstacle.hpp"
#include "xgc2_math/scene/shape_table.hpp"

namespace xgc2_math::scene {

// Immutable scene geometry and sampled map, shared by a fleet: the compiled solids, the hierarchy that casts rays
// against them and the surface samples of every solid with the blocks that find the samples around a sensor. Build it
// once when scene contents change (from the previous scene when only some obstacles moved); scans never resample the
// map and never write to the scene, so any number of sensors on any number of threads can share it.
class LidarScene {
  public:
    // Throws std::invalid_argument for a degenerate obstacle or a spacing that is not finite and positive.
    explicit LidarScene(const std::vector<Obstacle>& obstacles, double spacing = 0.1, bool keep_buried = false) {
        build(obstacles, spacing, keep_buried, nullptr);
    }

    // The scene LidarScene(obstacles, spacing, keep_buried) builds (the same samples in the same order), built from
    // `previous`: the shapes, samples and sample blocks of obstacles that are bitwise unchanged at the same position in
    // the list are reused, so after some obstacles moved only they (and, without buried samples, the obstacles they
    // touch) are compiled and sampled again, and the hierarchy is refit instead of rebuilt. Builds from scratch when
    // the spacing, the buried-sample policy or the number of obstacles differ.
    LidarScene(const std::vector<Obstacle>& obstacles, double spacing, bool keep_buried, const LidarScene& previous) {
        build(obstacles, spacing, keep_buried, &previous);
    }

    LidarScene(const LidarScene&) = delete;
    LidarScene& operator=(const LidarScene&) = delete;

    std::size_t obstacleCount() const { return shapes_.size(); }
    std::size_t sampleCount() const { return points_.size(); }
    double spacing() const { return spacing_; }
    bool keepBuried() const { return keep_buried_; }
    // The obstacles this scene was built from.
    const std::vector<Obstacle>& obstacles() const { return obstacles_; }
    // The sampled map: surface samples of every obstacle in obstacle order, buried samples dropped unless kept.
    const std::vector<Eigen::Vector3d>& samples() const { return points_; }

    // Distance along the unit direction to the first surface within [0, max_range], or +infinity. A ray that starts
    // inside a solid reports the surface it leaves through.
    double closest(const Eigen::Vector3d& origin, const Eigen::Vector3d& unit_direction, double max_range) const {
        return bvh_.closest(shapes_, origin, unit_direction, max_range);
    }

    // Scene residual: min over obstacles of a per-obstacle function that is 0 on the surface, < 0 inside and > 0
    // outside (|value| <= distance to that surface); +infinity for an empty scene.
    double surfaceResidual(const Eigen::Vector3d& point) const {
        double r = detail::kInf;
        for (std::size_t i = 0; i < shapes_.size(); ++i) {
            r = std::min(r, shapes_.residual(i, point));
        }
        return r;
    }

    // Sets bit k of `bits` (ceil(sampleCount() / 64) zeroed words) for every sample k with
    // (p_k - center).squaredNorm() <= r2 and returns how many bits it set. Obstacles whose bounds miss the sphere are
    // skipped; of the rest, blocks of consecutive samples that lie wholly inside are marked whole, blocks that lie
    // wholly outside are skipped and only the others are tested sample by sample.
    std::size_t mark(const Eigen::Vector3d& center, double r2, std::uint64_t* bits) const {
        // A box counts as clearly inside or outside only beyond a margin far larger than rounding, so that every sample
        // that is not clearly either gets the plain distance test.
        const double r = std::sqrt(r2);
        const double outer = (r + kMargin) * (r + kMargin);
        const double inner = r > kMargin ? (r - kMargin) * (r - kMargin) : -1.0;
        std::size_t marked = 0;
        bvh_.forEachInSphere(shapes_, center, outer, [&](std::uint32_t shape) {
            const std::size_t base = first_[shape];
            for (std::size_t b = block_first_[shape]; b < block_first_[shape + 1]; ++b) {
                const Block& block = blocks_[b];
                const Eigen::Vector3d nearest = (block.box.lo - center).cwiseMax(center - block.box.hi).cwiseMax(0.0);
                if (nearest.squaredNorm() > outer) {
                    continue;
                }
                const std::size_t begin = base + block.offset, end = begin + block.count;
                const Eigen::Vector3d farthest =
                    (block.box.lo - center).cwiseAbs().cwiseMax((block.box.hi - center).cwiseAbs());
                if (farthest.squaredNorm() <= inner) {
                    setBits(bits, begin, end);
                    marked += block.count;
                    continue;
                }
                for (std::size_t k = begin; k < end; ++k) {
                    if ((points_[k] - center).squaredNorm() <= r2) {
                        bits[k >> 6] |= std::uint64_t{1} << (k & 63u);
                        ++marked;
                    }
                }
            }
        });
        return marked;
    }

    // The sampled map of this geometry at another spacing or buried-sample policy.
    std::vector<Eigen::Vector3d> resampled(double spacing, bool keep_buried) const {
        checkSpacing(spacing);
        std::vector<Eigen::Vector3d> out, scratch;
        std::vector<Eigen::Vector2d> cell;
        std::vector<std::uint32_t> candidates;
        for (std::size_t i = 0; i < shapes_.size(); ++i) {
            appendShapeSamples(i, spacing, keep_buried, &scratch, &cell, &candidates, &out);
        }
        return out;
    }

  private:
    // Consecutive samples of one obstacle form blocks of up to kBlockSize; a block keeps the box of its samples and
    // where it starts within the obstacle's samples, so blocks stay valid when other obstacles gain or lose samples.
    static constexpr std::size_t kBlockSize = 64;
    static constexpr double kMargin = 1e-9;

    struct Block {
        detail::Aabb box;
        std::uint32_t offset = 0;
        std::uint32_t count = 0;
    };

    static void checkSpacing(double spacing) {
        if (!std::isfinite(spacing) || spacing <= 0.0) {
            throw std::invalid_argument("scene sample spacing must be finite and > 0");
        }
    }

    // Sets bits [first, last).
    static void setBits(std::uint64_t* bits, std::size_t first, std::size_t last) {
        if (first >= last) {
            return;
        }
        const std::size_t first_word = first >> 6, last_word = (last - 1) >> 6;
        const std::uint64_t head = ~std::uint64_t{0} << (first & 63u);
        const std::uint64_t tail = ~std::uint64_t{0} >> (63u - ((last - 1) & 63u));
        if (first_word == last_word) {
            bits[first_word] |= head & tail;
            return;
        }
        bits[first_word] |= head;
        for (std::size_t w = first_word + 1; w < last_word; ++w) {
            bits[w] = ~std::uint64_t{0};
        }
        bits[last_word] |= tail;
    }

    static std::uint64_t bitsOf(double v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        return bits;
    }

    static bool sameBits(const double* a, const double* b, std::size_t n) {
        for (std::size_t k = 0; k < n; ++k) {
            if (bitsOf(a[k]) != bitsOf(b[k])) {
                return false;
            }
        }
        return true;
    }

    // Bitwise equal definitions compile and sample to bitwise equal shapes.
    static bool sameObstacle(const Obstacle& a, const Obstacle& b) {
        if (a.type != b.type || a.vertices.size() != b.vertices.size() || a.triangles != b.triangles ||
            !sameBits(a.position.data(), b.position.data(), 3) ||
            !sameBits(a.orientation.coeffs().data(), b.orientation.coeffs().data(), 4) ||
            !sameBits(a.size.data(), b.size.data(), 3) || !sameBits(&a.radius, &b.radius, 1) ||
            !sameBits(&a.height, &b.height, 1) || !sameBits(a.radii.data(), b.radii.data(), 2) ||
            !sameBits(&a.tubeRadius, &b.tubeRadius, 1)) {
            return false;
        }
        for (std::size_t i = 0; i < a.vertices.size(); ++i) {
            if (!sameBits(a.vertices[i].data(), b.vertices[i].data(), 3)) {
                return false;
            }
        }
        return true;
    }

    // Appends the samples of shape `i` at `spacing` to `out`, without those strictly inside another shape unless
    // `keep_buried`. `scratch`, `cell` and `candidates` are reusable work space.
    void appendShapeSamples(std::size_t i, double spacing, bool keep_buried, std::vector<Eigen::Vector3d>* scratch,
                            std::vector<Eigen::Vector2d>* cell, std::vector<std::uint32_t>* candidates,
                            std::vector<Eigen::Vector3d>* out) const {
        scratch->clear();
        shapes_.appendSamples(i, spacing, scratch, cell);
        if (keep_buried) {
            out->insert(out->end(), scratch->begin(), scratch->end());
            return;
        }
        candidates->clear();
        bvh_.overlapping(shapes_, shapes_.bounds(i), i, candidates);
        for (const Eigen::Vector3d& p : *scratch) {
            // A point strictly inside a shape lies inside its box, which is much cheaper to test than the shape.
            const bool buried = std::any_of(candidates->begin(), candidates->end(), [&](std::uint32_t other) {
                const detail::Aabb& box = shapes_.bounds(other);
                return (box.lo.array() <= p.array()).all() && (p.array() <= box.hi.array()).all() &&
                       shapes_.residual(other, p) < -1e-9;
            });
            if (!buried) {
                out->push_back(p);
            }
        }
    }

    // Appends the blocks of the samples points_[begin, end) of one shape.
    void appendBlocks(std::size_t begin, std::size_t end) {
        if (end - begin > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("too many samples for one obstacle");
        }
        for (std::size_t start = begin; start < end; start += kBlockSize) {
            const std::size_t stop = std::min(end, start + kBlockSize);
            Block block;
            for (std::size_t k = start; k < stop; ++k) {
                block.box.grow(points_[k]);
            }
            block.offset = static_cast<std::uint32_t>(start - begin);
            block.count = static_cast<std::uint32_t>(stop - start);
            blocks_.push_back(block);
        }
    }

    // Builds the scene for `obstacles`; with `prev` (the scene of the previous revision) the shapes, samples and blocks
    // of unchanged obstacles are reused. The result equals a build without `prev`: the same samples in the same order.
    void build(const std::vector<Obstacle>& obstacles, double spacing, bool keep_buried, const LidarScene* prev) {
        checkSpacing(spacing);
        spacing_ = spacing;
        keep_buried_ = keep_buried;
        if (prev && (prev->spacing_ != spacing || prev->keep_buried_ != keep_buried ||
                     prev->obstacles_.size() != obstacles.size())) {
            prev = nullptr;
        }
        const std::size_t n = obstacles.size();
        std::vector<char> changed(n, 1);
        shapes_.reserve(n);
        bool any_changed = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (prev && sameObstacle(obstacles[i], prev->obstacles_[i])) {
                changed[i] = 0;
                shapes_.appendFrom(prev->shapes_, i);
            } else {
                shapes_.append(obstacles[i]);
                any_changed = true;
            }
        }
        obstacles_ = obstacles;
        if (prev) {
            bvh_ = prev->bvh_;
            if (any_changed && !bvh_.refit(shapes_)) {
                bvh_.build(shapes_);
            }
        } else {
            bvh_.build(shapes_);
        }
        // Shapes sampled again: the changed ones and, when buried samples are dropped, every shape that a changed one
        // overlapped before or overlaps now.
        std::vector<char> resample = changed;
        std::vector<std::uint32_t> candidates;
        if (prev && !keep_buried) {
            for (std::size_t j = 0; j < n; ++j) {
                if (!changed[j]) {
                    continue;
                }
                candidates.clear();
                bvh_.overlapping(shapes_, prev->shapes_.bounds(j), n, &candidates);
                bvh_.overlapping(shapes_, shapes_.bounds(j), n, &candidates);
                for (std::uint32_t k : candidates) {
                    resample[k] = 1;
                }
            }
        }
        std::vector<Eigen::Vector3d> scratch;
        std::vector<Eigen::Vector2d> cell;
        first_.assign(1, 0);
        block_first_.assign(1, 0);
        if (prev) {
            points_.reserve(prev->points_.size());
            blocks_.reserve(prev->blocks_.size());
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (prev && !resample[i]) {
                const auto first = prev->points_.begin() + static_cast<std::ptrdiff_t>(prev->first_[i]);
                const auto last = prev->points_.begin() + static_cast<std::ptrdiff_t>(prev->first_[i + 1]);
                points_.insert(points_.end(), first, last);
                const auto first_block = prev->blocks_.begin() + static_cast<std::ptrdiff_t>(prev->block_first_[i]);
                const auto last_block = prev->blocks_.begin() + static_cast<std::ptrdiff_t>(prev->block_first_[i + 1]);
                blocks_.insert(blocks_.end(), first_block, last_block);
            } else {
                const std::size_t begin = points_.size();
                appendShapeSamples(i, spacing, keep_buried, &scratch, &cell, &candidates, &points_);
                appendBlocks(begin, points_.size());
            }
            first_.push_back(points_.size());
            block_first_.push_back(blocks_.size());
        }
    }

    double spacing_ = 0.1;
    bool keep_buried_ = false;
    std::vector<Obstacle> obstacles_; // definitions, to find what a rebuild changes
    detail::ShapeTable shapes_;
    detail::Bvh bvh_;
    std::vector<Eigen::Vector3d> points_;
    std::vector<std::size_t> first_;       // samples of shape i: points_[first_[i], first_[i + 1])
    std::vector<Block> blocks_;            // blocks of shape i: blocks_[block_first_[i], block_first_[i + 1])
    std::vector<std::size_t> block_first_; // size obstacleCount() + 1
};

} // namespace xgc2_math::scene
