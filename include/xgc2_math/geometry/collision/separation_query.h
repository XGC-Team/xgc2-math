#ifndef XGC2_MATH_GEOMETRY_COLLISION_SEPARATION_QUERY_H
#define XGC2_MATH_GEOMETRY_COLLISION_SEPARATION_QUERY_H

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "xgc2_math/geometry/math_helpers.h"

#ifndef XGC2_MATH_GEOMETRY_GJK_ALWAYS_TRUST_WARM_START
#define XGC2_MATH_GEOMETRY_GJK_ALWAYS_TRUST_WARM_START 1
#endif

namespace xgc2_math {
namespace gjk {

struct SeparationQuadruple {
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    double margin = std::numeric_limits<double>::infinity();
    Eigen::Vector3d point_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_b = Eigen::Vector3d::Zero();
};

struct Result {
    enum class Status {
        kSuccess,       // Produced a usable separation result from the current query.
        kOverlap,       // Current sets fell below the minimum distance; returned fallback/current separator.
        kMaxIterations, // Iteration budget exhausted; returned the latest usable separator with warning.
        kInvalid        // Invalid inputs prevented the query from running.
    };

    Status status = Status::kInvalid;
    SeparationQuadruple separator;
    int iterations = 0;
    double distance_gjk_time_us = 0.0;
    double guide_correction_time_us = 0.0;
    bool guide_attempted = false;
    bool guide_success = false;
};

struct WarmStart {
    SeparationQuadruple separator;
    bool valid = false; // True only after a previous query produced a usable separator.
};

namespace detail {

constexpr std::size_t kSimplexMaxVertices = 4;

struct Vertex {
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    Eigen::Vector3d w = Eigen::Vector3d::Zero();
    double weight = 0.0;
    std::size_t source = 0;
};

struct Simplex {
    std::array<Vertex, kSimplexMaxVertices> vertices{};
    std::size_t count = 0;
    Eigen::Vector3d closest = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_b = Eigen::Vector3d::Zero();

    void assignSingle(const Vertex& vertex) {
        vertices[0] = vertex;
        vertices[0].weight = 1.0;
        count = 1;
        updateClosest();
    }

    void append(const Vertex& vertex) { vertices[count++] = vertex; }

    void updateClosest() {
        point_a.setZero();
        point_b.setZero();
        closest.setZero();
        for (std::size_t i = 0; i < count; ++i) {
            point_a += vertices[i].weight * vertices[i].a;
            point_b += vertices[i].weight * vertices[i].b;
        }
        closest = point_b - point_a;
    }
};

inline void reduceToVertex(Simplex& simplex, std::size_t idx) {
    simplex.vertices[0] = simplex.vertices[idx];
    simplex.vertices[0].weight = 1.0;
    simplex.count = 1;
    simplex.updateClosest();
}

// NOTE (aliasing): the source indices may overlap the destination slots -- the
// tetrahedron face (0, 3, 1) reduces with idx1 = 3, idx2 = 1, so writing
// vertices[1] before reading vertices[idx2] would duplicate vertex 3 and drop
// vertex 1.  Snapshot the sources first; an in-place write order is not safe.
inline void reduceToSegment(Simplex& simplex, std::size_t idx0, std::size_t idx1, double w0, double w1) {
    const Vertex v0 = simplex.vertices[idx0];
    const Vertex v1 = simplex.vertices[idx1];
    simplex.vertices[0] = v0;
    simplex.vertices[1] = v1;
    simplex.vertices[0].weight = w0;
    simplex.vertices[1].weight = w1;
    simplex.count = 2;
    simplex.updateClosest();
}

inline void reduceToTriangle(Simplex& simplex, std::size_t idx0, std::size_t idx1, std::size_t idx2, double w0,
                             double w1, double w2) {
    const Vertex v0 = simplex.vertices[idx0];
    const Vertex v1 = simplex.vertices[idx1];
    const Vertex v2 = simplex.vertices[idx2];
    simplex.vertices[0] = v0;
    simplex.vertices[1] = v1;
    simplex.vertices[2] = v2;
    simplex.vertices[0].weight = w0;
    simplex.vertices[1].weight = w1;
    simplex.vertices[2].weight = w2;
    simplex.count = 3;
    simplex.updateClosest();
}

inline bool numericallySameVector(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) {
    const double scale = std::max(lhs.norm(), rhs.norm());
    if (scale == 0.0) {
        return (lhs.array() == rhs.array()).all();
    }
    return (lhs - rhs).norm() <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

inline long double dot3LongDouble(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) {
    return static_cast<long double>(lhs.x()) * rhs.x() + static_cast<long double>(lhs.y()) * rhs.y() +
           static_cast<long double>(lhs.z()) * rhs.z();
}

inline void deduplicateVertices(Simplex& simplex) {
    if (simplex.count <= 1) {
        return;
    }

    for (std::size_t i = 0; i < simplex.count; ++i) {
        for (std::size_t j = i + 1; j < simplex.count;) {
            if (numericallySameVector(simplex.vertices[i].w, simplex.vertices[j].w) &&
                numericallySameVector(simplex.vertices[i].a, simplex.vertices[j].a) &&
                numericallySameVector(simplex.vertices[i].b, simplex.vertices[j].b)) {
                simplex.vertices[i].weight += simplex.vertices[j].weight;
                for (std::size_t k = j; k + 1 < simplex.count; ++k) {
                    simplex.vertices[k] = simplex.vertices[k + 1];
                }
                --simplex.count;
                continue;
            }
            ++j;
        }
    }

    double weight_sum = 0.0;
    for (std::size_t i = 0; i < simplex.count; ++i) {
        weight_sum += simplex.vertices[i].weight;
    }
    if (weight_sum > 0.0) {
        for (std::size_t i = 0; i < simplex.count; ++i) {
            simplex.vertices[i].weight /= weight_sum;
        }
    }
    simplex.updateClosest();
}

inline bool hasEquivalentVertex(const Simplex& simplex, const Vertex& candidate) {
    for (std::size_t i = 0; i < simplex.count; ++i) {
        if (numericallySameVector(simplex.vertices[i].w, candidate.w) &&
            numericallySameVector(simplex.vertices[i].a, candidate.a) &&
            numericallySameVector(simplex.vertices[i].b, candidate.b)) {
            return true;
        }
    }
    return false;
}

inline void solveSegment(Simplex& simplex) {
    const Eigen::Vector3d& a = simplex.vertices[0].w;
    const Eigen::Vector3d& b = simplex.vertices[1].w;
    const Eigen::Vector3d ab = b - a;
    const long double denom = dot3LongDouble(ab, ab);
    if (denom == 0.0L) {
        reduceToVertex(simplex, 0);
        return;
    }

    const long double raw_t = -dot3LongDouble(a, ab) / denom;
    const double t = static_cast<double>(std::max(0.0L, std::min(1.0L, raw_t)));
    if (t <= 0.0) {
        reduceToVertex(simplex, 0);
        return;
    }
    if (t >= 1.0) {
        reduceToVertex(simplex, 1);
        return;
    }

    reduceToSegment(simplex, 0, 1, 1.0 - t, t);
}

inline void solveTriangle(Simplex& simplex) {
    const Eigen::Vector3d& a = simplex.vertices[0].w;
    const Eigen::Vector3d& b = simplex.vertices[1].w;
    const Eigen::Vector3d& c = simplex.vertices[2].w;
    const Eigen::Vector3d ab = b - a;
    const Eigen::Vector3d ac = c - a;
    const Eigen::Vector3d ap = -a;

    const long double d1 = dot3LongDouble(ab, ap);
    const long double d2 = dot3LongDouble(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        reduceToVertex(simplex, 0);
        return;
    }

    const Eigen::Vector3d bp = -b;
    const long double d3 = dot3LongDouble(ab, bp);
    const long double d4 = dot3LongDouble(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        reduceToVertex(simplex, 1);
        return;
    }

    const long double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = static_cast<double>(d1 / (d1 - d3));
        reduceToSegment(simplex, 0, 1, 1.0 - v, v);
        return;
    }

    const Eigen::Vector3d cp = -c;
    const long double d5 = dot3LongDouble(ab, cp);
    const long double d6 = dot3LongDouble(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        reduceToVertex(simplex, 2);
        return;
    }

    const long double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = static_cast<double>(d2 / (d2 - d6));
        reduceToSegment(simplex, 0, 2, 1.0 - w, w);
        return;
    }

    const long double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = static_cast<double>((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        reduceToSegment(simplex, 1, 2, 1.0 - w, w);
        return;
    }

    const long double denom = va + vb + vc;
    if (denom == 0.0L) {
        Simplex closest_edge;
        double closest_distance_sq = std::numeric_limits<double>::infinity();
        for (const std::array<std::size_t, 2> edge :
             {std::array<std::size_t, 2>{0, 1}, std::array<std::size_t, 2>{0, 2}, std::array<std::size_t, 2>{1, 2}}) {
            Simplex edge_simplex;
            edge_simplex.count = 2;
            edge_simplex.vertices[0] = simplex.vertices[edge[0]];
            edge_simplex.vertices[1] = simplex.vertices[edge[1]];
            solveSegment(edge_simplex);
            const double distance_sq = edge_simplex.closest.squaredNorm();
            if (distance_sq < closest_distance_sq) {
                closest_distance_sq = distance_sq;
                closest_edge = edge_simplex;
            }
        }
        simplex = closest_edge;
        return;
    }

    const double v = static_cast<double>(vb / denom);
    const double w = static_cast<double>(vc / denom);
    const double u = 1.0 - v - w;
    reduceToTriangle(simplex, 0, 1, 2, u, v, w);
}

inline long double orientation3d(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c,
                                 const Eigen::Vector3d& d) {
    const long double ab_x = static_cast<long double>(b.x()) - a.x();
    const long double ab_y = static_cast<long double>(b.y()) - a.y();
    const long double ab_z = static_cast<long double>(b.z()) - a.z();
    const long double ac_x = static_cast<long double>(c.x()) - a.x();
    const long double ac_y = static_cast<long double>(c.y()) - a.y();
    const long double ac_z = static_cast<long double>(c.z()) - a.z();
    const long double ad_x = static_cast<long double>(d.x()) - a.x();
    const long double ad_y = static_cast<long double>(d.y()) - a.y();
    const long double ad_z = static_cast<long double>(d.z()) - a.z();
    return ab_x * (ac_y * ad_z - ac_z * ad_y) - ab_y * (ac_x * ad_z - ac_z * ad_x) + ab_z * (ac_x * ad_y - ac_y * ad_x);
}

inline int orientationSign(long double value) {
    return value > 0.0L ? 1 : (value < 0.0L ? -1 : 0);
}

inline bool pointOutsideOfPlane(const Eigen::Vector3d& p, const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                                const Eigen::Vector3d& c, const Eigen::Vector3d& d) {
    const int sign_p = orientationSign(orientation3d(a, b, c, p));
    const int sign_d = orientationSign(orientation3d(a, b, c, d));
    return sign_p != 0 && sign_d != 0 && sign_p != sign_d;
}

inline bool solveTetrahedron(Simplex& simplex) {
    const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    const auto& a = simplex.vertices[0].w;
    const auto& b = simplex.vertices[1].w;
    const auto& c = simplex.vertices[2].w;
    const auto& d = simplex.vertices[3].w;

    struct FaceCandidate {
        bool valid = false;
        double dist_sq = std::numeric_limits<double>::infinity();
        std::size_t count = 0;
        std::array<std::size_t, 3> indices{0, 0, 0};
        std::array<double, 3> weights{0.0, 0.0, 0.0};
    };

    auto evaluate_face = [&](std::size_t idx0, std::size_t idx1, std::size_t idx2) -> FaceCandidate {
        FaceCandidate candidate;
        Simplex face_simplex;
        face_simplex.count = 3;
        face_simplex.vertices[0] = simplex.vertices[idx0];
        face_simplex.vertices[1] = simplex.vertices[idx1];
        face_simplex.vertices[2] = simplex.vertices[idx2];
        face_simplex.vertices[0].source = idx0;
        face_simplex.vertices[1].source = idx1;
        face_simplex.vertices[2].source = idx2;
        solveTriangle(face_simplex);
        candidate.valid = true;
        candidate.dist_sq = face_simplex.closest.squaredNorm();
        candidate.count = face_simplex.count;
        for (std::size_t i = 0; i < face_simplex.count; ++i) {
            candidate.indices[i] = face_simplex.vertices[i].source;
            candidate.weights[i] = face_simplex.vertices[i].weight;
        }
        return candidate;
    };

    std::array<FaceCandidate, 4> candidates{};
    std::size_t candidate_count = 0;
    bool reduced_degenerate_tetrahedron = false;
    double degenerate_edge_scale = 0.0;

    if (pointOutsideOfPlane(origin, a, b, c, d)) {
        candidates[candidate_count++] = evaluate_face(0, 1, 2);
    }
    if (pointOutsideOfPlane(origin, a, c, d, b)) {
        candidates[candidate_count++] = evaluate_face(0, 2, 3);
    }
    if (pointOutsideOfPlane(origin, a, d, b, c)) {
        candidates[candidate_count++] = evaluate_face(0, 3, 1);
    }
    if (pointOutsideOfPlane(origin, b, d, c, a)) {
        candidates[candidate_count++] = evaluate_face(1, 3, 2);
    }

    if (candidate_count == 0) {
        // No face classified the origin as outside.  That proves containment
        // only for a non-degenerate tetrahedron.  GJK support points can be
        // coplanar even for full-dimensional sets (for example a sphere
        // beside a box whose lower face shares z=0 with the sphere centre).
        // Treating such a zero-volume simplex as containing the origin
        // produces a false overlap.  Reduce a degenerate tetrahedron to its
        // closest face instead; a genuine lower-dimensional overlap will
        // still reduce to zero distance and be reported by the query.
        if (orientationSign(orientation3d(a, b, c, d)) != 0) {
            return true;
        }
        const double edge_scale =
            std::max({(b - a).norm(), (c - a).norm(), (d - a).norm(), (c - b).norm(), (d - b).norm(), (d - c).norm()});
        candidates[candidate_count++] = evaluate_face(0, 1, 2);
        candidates[candidate_count++] = evaluate_face(0, 2, 3);
        candidates[candidate_count++] = evaluate_face(0, 3, 1);
        candidates[candidate_count++] = evaluate_face(1, 3, 2);
        reduced_degenerate_tetrahedron = true;
        degenerate_edge_scale = edge_scale;
    }

    std::size_t best_idx = 0;
    for (std::size_t i = 1; i < candidate_count; ++i) {
        if (candidates[i].dist_sq < candidates[best_idx].dist_sq) {
            best_idx = i;
        }
    }

    const FaceCandidate& best = candidates[best_idx];
    if (!best.valid || best.count == 0) {
        return true;
    }
    if (reduced_degenerate_tetrahedron) {
        const double distance_tolerance = 64.0 * std::numeric_limits<double>::epsilon() * degenerate_edge_scale;
        if (best.dist_sq <= distance_tolerance * distance_tolerance) {
            return true;
        }
    }

    if (best.count == 1) {
        reduceToVertex(simplex, best.indices[0]);
    } else if (best.count == 2) {
        reduceToSegment(simplex, best.indices[0], best.indices[1], best.weights[0], best.weights[1]);
    } else {
        reduceToTriangle(simplex, best.indices[0], best.indices[1], best.indices[2], best.weights[0], best.weights[1],
                         best.weights[2]);
    }

    return false;
}

inline bool solveSimplex(Simplex& simplex) {
    switch (simplex.count) {
    case 1:
        simplex.vertices[0].weight = 1.0;
        simplex.updateClosest();
        deduplicateVertices(simplex);
        return false;
    case 2:
        solveSegment(simplex);
        deduplicateVertices(simplex);
        return false;
    case 3:
        solveTriangle(simplex);
        deduplicateVertices(simplex);
        return false;
    case 4:
        if (solveTetrahedron(simplex)) {
            return true;
        }
        deduplicateVertices(simplex);
        return false;
    default:
        return false;
    }
}

inline Eigen::Vector3d chooseInitialDirection(const Eigen::Vector3d& center_delta, const WarmStart* warm_start) {
    if (warm_start != nullptr && warm_start->valid && warm_start->separator.normal.squaredNorm() > 1e-12) {
        return warm_start->separator.normal;
    }
    if (center_delta.squaredNorm() > 1e-12) {
        return center_delta;
    }
    return Eigen::Vector3d::UnitX();
}

} // namespace detail
} // namespace gjk
} // namespace xgc2_math

#endif // XGC2_MATH_GEOMETRY_COLLISION_SEPARATION_QUERY_H
