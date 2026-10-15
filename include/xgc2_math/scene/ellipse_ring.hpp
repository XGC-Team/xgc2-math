#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xgc2_math::scene {

inline void validateEllipticRing(const Eigen::Vector2d& axes, double radius) {
    if (!axes.allFinite() || (axes.array() <= 0).any() || !std::isfinite(radius) || radius <= 0 ||
        radius >= axes.minCoeff()) {
        throw std::invalid_argument("ring requires positive finite radii and 0 < tubeRadius < min(radii)");
    }
}

// Euclidean distance to C(u)=(a cos u,b sin u,0), not distance in a scaled metric.
inline double ellipseCenterlineDistance(const Eigen::Vector3d& p, const Eigen::Vector2d& axes) {
    if (!p.allFinite() || !axes.allFinite() || (axes.array() <= 0).any()) {
        throw std::invalid_argument("invalid ellipse distance input");
    }
    long double a = axes.x(), b = axes.y(), x = std::abs(p.x()), y = std::abs(p.y());
    if (a < b) {
        std::swap(a, b);
        std::swap(x, y);
    }
    if (a == b) {
        return static_cast<double>(std::hypot(std::hypot(x, y) - a, static_cast<long double>(p.z())));
    }
    const long double delta = (a - b) * (a + b);
    long double qx = 0, qy = b;
    if (y == 0) {
        if (a * x < delta) {
            qx = a * a * x / delta;
            qy = b * std::sqrt(std::max(0.0L, 1 - (qx / a) * (qx / a)));
        } else {
            qx = a;
            qy = 0;
        }
    } else if (x != 0) {
        // h=lambda+b^2 keeps the nearest-point branch positive, avoiding
        // cancellation at the minor-axis singularity lambda=-b^2.
        long double lo = b * y, hi = a * x + b * y + a * a + b * b;
        for (unsigned i = 0; i < 160; ++i) {
            const long double h = hi > 4 * lo ? std::sqrt(lo) * std::sqrt(hi) : (lo + hi) / 2;
            if (h == lo || h == hi)
                break;
            const auto u = a * x / (h + delta), v = b * y / h;
            if (u * u + v * v > 1)
                lo = h;
            else
                hi = h;
        }
        const long double h = (lo + hi) / 2;
        qx = a * a * x / (h + delta);
        qy = b * b * y / h;
    }
    return static_cast<double>(std::hypot(std::hypot(x - qx, y - qy), static_cast<long double>(p.z())));
}

inline double ellipticRingResidual(const Eigen::Vector3d& p, const Eigen::Vector2d& axes, double radius) {
    return ellipseCenterlineDistance(p, axes) - radius;
}

// Distance along the inward planar normal to the ellipse's medial axis.
// For a>=b the meeting point is ((a^2-b^2)/a*cos(u),0).
inline double ellipticRingNormalCutDistance(const Eigen::Vector2d& axes, double u) {
    return axes.minCoeff() / axes.maxCoeff() * std::hypot(axes.x() * std::sin(u), axes.y() * std::cos(u));
}

namespace ring_detail {
// Fixed storage: ray queries allocate no memory. Derivative roots partition
// monotone intervals; testing those endpoints also finds even-multiplicity roots.
struct Polynomial {
    std::array<long double, 17> c{};
    int n = 0;
};
inline Polynomial add(Polynomial a, const Polynomial& b, long double factor = 1) {
    a.n = std::max(a.n, b.n);
    for (int i = 0; i <= b.n; ++i)
        a.c[static_cast<std::size_t>(i)] += factor * b.c[static_cast<std::size_t>(i)];
    return a;
}
inline Polynomial multiply(const Polynomial& a, const Polynomial& b) {
    Polynomial p;
    p.n = a.n + b.n;
    if (p.n > 16)
        throw std::logic_error("ring polynomial degree");
    for (int i = 0; i <= a.n; ++i)
        for (int j = 0; j <= b.n; ++j)
            p.c[static_cast<std::size_t>(i + j)] += a.c[static_cast<std::size_t>(i)] * b.c[static_cast<std::size_t>(j)];
    return p;
}
inline Polynomial scale(Polynomial a, long double x) {
    for (int i = 0; i <= a.n; ++i)
        a.c[static_cast<std::size_t>(i)] *= x;
    return a;
}
inline Polynomial derivative(const Polynomial& a) {
    Polynomial p;
    p.n = std::max(0, a.n - 1);
    for (int i = 1; i <= a.n; ++i)
        p.c[static_cast<std::size_t>(i - 1)] = i * a.c[static_cast<std::size_t>(i)];
    return p;
}
inline long double value(const Polynomial& p, long double x) {
    long double y = p.c[static_cast<std::size_t>(p.n)];
    for (int i = p.n - 1; i >= 0; --i)
        y = y * x + p.c[static_cast<std::size_t>(i)];
    return y;
}
struct Roots {
    std::array<long double, 32> x{};
    unsigned n = 0;
    void push(long double v) {
        if (n == x.size())
            throw std::runtime_error("ring root bound exceeded");
        x[n++] = v;
    }
};
inline Roots roots(Polynomial p, long double lo, long double hi) {
    long double norm = 0;
    for (int i = 0; i <= p.n; ++i)
        norm = std::max(norm, std::abs(p.c[static_cast<std::size_t>(i)]));
    Roots out;
    if (norm == 0)
        return out;
    for (int i = 0; i <= p.n; ++i)
        p.c[static_cast<std::size_t>(i)] /= norm;
    const long double eps = 128 * std::numeric_limits<long double>::epsilon();
    while (p.n > 0 && std::abs(p.c[static_cast<std::size_t>(p.n)]) <= eps)
        --p.n;
    if (p.n == 0)
        return out;
    const Roots critical = roots(derivative(p), lo, hi);
    Roots boundaries;
    boundaries.push(lo);
    for (unsigned i = 0; i < critical.n; ++i)
        boundaries.push(critical.x[i]);
    boundaries.push(hi);
    for (unsigned i = 0; i < boundaries.n; ++i) {
        if (std::abs(value(p, boundaries.x[i])) <= eps)
            out.push(boundaries.x[i]);
    }
    for (unsigned i = 1; i < boundaries.n; ++i) {
        auto a = boundaries.x[i - 1], b = boundaries.x[i];
        auto fa = value(p, a), fb = value(p, b);
        if ((fa < 0) == (fb < 0) || fa == 0 || fb == 0)
            continue;
        for (unsigned k = 0; k < 96; ++k) {
            const auto m = (a + b) / 2;
            if (m == a || m == b)
                break;
            const auto fm = value(p, m);
            if ((fm < 0) == (fa < 0)) {
                a = m;
                fa = fm;
            } else
                b = m;
        }
        out.push((a + b) / 2);
    }
    std::sort(out.x.begin(), out.x.begin() + out.n);
    unsigned n = 0;
    for (unsigned i = 0; i < out.n; ++i) {
        if (n == 0 || std::abs(out.x[i] - out.x[n - 1]) > 1024 * std::numeric_limits<long double>::epsilon())
            out.x[n++] = out.x[i];
    }
    out.n = n;
    return out;
}
} // namespace ring_detail

// The line intersects a union of balls centred on the ellipse. Each connected
// feasible angular arc gives one interval; never hull separated ring intervals.
// Returns the nearest forward entry, or the exit when starting inside, in the
// ray parameter t of origin+t*direction (metres for a unit direction).
inline double ellipticRingRayClosest(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction,
                                     const Eigen::Vector2d& axes, double radius, double maxRange) {
    using namespace ring_detail;
    validateEllipticRing(axes, radius);
    struct Interval {
        long double lo, hi;
    };
    std::array<Interval, 80> intervals{};
    unsigned count = 0;
    const long double d2 = static_cast<long double>(direction.x()) * direction.x() +
                           static_cast<long double>(direction.y()) * direction.y() +
                           static_cast<long double>(direction.z()) * direction.z();
    if (!origin.allFinite() || !direction.allFinite() || d2 == 0 || std::isnan(maxRange) || maxRange < 0)
        return std::numeric_limits<double>::infinity();
    Polynomial den;
    den.n = 2;
    den.c[0] = den.c[2] = 1;
    constexpr long double bound = 0.4142135623730950488016887242097L;
    const long double tolerance = 4096 * std::numeric_limits<long double>::epsilon();
    const auto append = [&](const Interval& interval) {
        if (count == intervals.size())
            throw std::runtime_error("ring interval bound");
        intervals[count++] = interval;
    };
    for (unsigned quarter = 0; quarter < 4; ++quarter) {
        const long double angle = quarter * 1.5707963267948966192313216916398L;
        const std::array<long double, 3> u{axes.x() * std::cos(angle), axes.y() * std::sin(angle), 0};
        const std::array<long double, 3> v{-axes.x() * std::sin(angle), axes.y() * std::cos(angle), 0};
        Polynomial t, h;
        std::array<Polynomial, 3> w;
        for (unsigned j = 0; j < 3; ++j) {
            w[j].n = 2;
            w[j].c[0] = u[j] - origin[j];
            w[j].c[1] = 2 * v[j];
            w[j].c[2] = -u[j] - origin[j];
            t = add(t, scale(w[j], direction[j]));
        }
        // |d cross W|^2 avoids cancellation when the origin lies far along d.
        for (unsigned j = 0; j < 3; ++j) {
            const auto cross =
                add(scale(w[(j + 1) % 3], direction[(j + 2) % 3]), scale(w[(j + 2) % 3], direction[(j + 1) % 3]), -1);
            h = add(h, multiply(cross, cross));
        }
        h = add(h, scale(multiply(den, den), d2 * radius * radius), -1);
        const auto ap = add(multiply(derivative(t), den), multiply(t, derivative(den)), -1);
        const auto bp = add(multiply(derivative(h), den), scale(multiply(h, derivative(den)), 2), -1);
        const auto stationary = add(scale(multiply(multiply(ap, ap), h), -4), multiply(bp, bp), -1);
        const auto edges = roots(h, -bound, bound), extrema = roots(stationary, -bound, bound);
        Roots cuts;
        cuts.push(-bound);
        for (unsigned j = 0; j < edges.n; ++j)
            cuts.push(edges.x[j]);
        cuts.push(bound);
        long double hscale = 0;
        for (int j = 0; j <= h.n; ++j)
            hscale += std::abs(h.c[static_cast<std::size_t>(j)]);
        const auto ball = [&](long double x) {
            const auto denominator = d2 * value(den, x), center = value(t, x);
            const auto half = std::sqrt(std::max(0.0L, -value(h, x)));
            return Interval{(center - half) / denominator, (center + half) / denominator};
        };
        for (unsigned j = 0; j < edges.n; ++j) {
            if (std::abs(value(h, edges.x[j])) <= tolerance * hscale)
                append(ball(edges.x[j]));
        }
        for (unsigned j = 1; j < cuts.n; ++j) {
            const auto lo = cuts.x[j - 1], hi = cuts.x[j];
            if (value(h, (lo + hi) / 2) > 0)
                continue;
            auto interval = ball(lo);
            const auto end = ball(hi);
            interval.lo = std::min(interval.lo, end.lo);
            interval.hi = std::max(interval.hi, end.hi);
            for (unsigned k = 0; k < extrema.n; ++k) {
                if (extrema.x[k] > lo && extrema.x[k] < hi) {
                    const auto x = ball(extrema.x[k]);
                    interval.lo = std::min(interval.lo, x.lo);
                    interval.hi = std::max(interval.hi, x.hi);
                }
            }
            append(interval);
        }
    }
    std::sort(intervals.begin(), intervals.begin() + count, [](auto a, auto b) {
        return a.lo < b.lo;
    });
    const long double eps = 64 * std::numeric_limits<long double>::epsilon() *
                            std::max({origin.norm(), axes.maxCoeff(), radius}) / std::sqrt(d2);
    // Input coordinates have double precision. Keep this boundary tolerance
    // separate from interval merging so it cannot fill a narrow ring hole.
    const long double boundaryEps =
        8 * std::numeric_limits<double>::epsilon() * std::max({origin.norm(), axes.maxCoeff(), radius}) / std::sqrt(d2);
    for (unsigned i = 0; i < count;) {
        auto x = intervals[i++];
        while (i < count && intervals[i].lo <= x.hi + eps)
            x.hi = std::max(x.hi, intervals[i++].hi);
        if (x.hi < -boundaryEps)
            continue;
        const auto hit = x.lo >= -boundaryEps ? std::max(0.0L, x.lo) : std::max(0.0L, x.hi);
        if (hit <= maxRange + boundaryEps)
            return static_cast<double>(std::min(hit, static_cast<long double>(maxRange)));
    }
    return std::numeric_limits<double>::infinity();
}

// Boundary samples only: inward planar normals stop at their medial-axis cut,
// so valid thick tubes (r>=min(a^2/b,b^2/a)) do not expose internal sheets.
// The XYZ GPU API observes this sampled map, not the analytic CPU surface.
inline std::vector<Eigen::Vector3d> sampleEllipticRingSurface(const Eigen::Vector2d& axes, double radius,
                                                              double spacing) {
    validateEllipticRing(axes, radius);
    if (!std::isfinite(spacing) || spacing <= 0)
        throw std::invalid_argument("invalid ring sample spacing");
    constexpr double pi = 3.14159265358979323846;
    const double nu = std::ceil(2 * pi * (axes.maxCoeff() + radius * axes.maxCoeff() / axes.minCoeff()) / spacing);
    const double nv = std::ceil(2 * pi * radius / spacing);
    if (nu * std::max(12.0, nv) > 16000000)
        throw std::length_error("ring surface sample work bound exceeded");
    const unsigned n = std::max(16u, static_cast<unsigned>(nu));
    std::vector<Eigen::Vector3d> out;
    for (unsigned i = 0; i < n; ++i) {
        const double u = 2 * pi * i / n;
        const Eigen::Vector3d c(axes.x() * std::cos(u), axes.y() * std::sin(u), 0);
        const double speed = std::hypot(axes.x() * std::sin(u), axes.y() * std::cos(u));
        const Eigen::Vector3d normal(axes.y() * std::cos(u) / speed, axes.x() * std::sin(u) / speed, 0);
        const double limit = std::acos(-std::min(1.0, ellipticRingNormalCutDistance(axes, u) / radius));
        const unsigned m = std::max(12u, static_cast<unsigned>(std::ceil(2 * limit * radius / spacing)));
        for (unsigned j = 0; j <= m; ++j) {
            const double v = -limit + 2 * limit * j / m;
            out.push_back(c + radius * (std::cos(v) * normal + Eigen::Vector3d(0, 0, std::sin(v))));
        }
    }
    return out;
}

// Conservative one-sided covering distance from the exact boundary to the
// samples above, excluding floating-point roundoff. In the regular domain it
// is <=spacing. Beyond the reach a square-root term bounds the clipped seam;
// spacing is not a claimed global Euclidean gap in that domain.
inline double ellipticRingSampleCoverBound(const Eigen::Vector2d& axes, double radius, double spacing) {
    validateEllipticRing(axes, radius);
    if (!std::isfinite(spacing) || spacing <= 0)
        throw std::invalid_argument("invalid ring sample spacing");
    constexpr double pi = 3.14159265358979323846;
    const double major = axes.maxCoeff(), minor = axes.minCoeff();
    const double lipschitz = major + radius * major / minor;
    const double n = std::max(16.0, std::ceil(2 * pi * lipschitz / spacing));
    const double halfStep = pi / n;
    const double seam = radius > minor * minor / major ? 2 * std::sqrt(radius * minor * halfStep) : 0;
    return lipschitz * halfStep + spacing / 2 + seam;
}

} // namespace xgc2_math::scene
