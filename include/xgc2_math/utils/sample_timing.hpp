#pragma once

#include <algorithm>
#include <cmath>

namespace xgc2_math {

constexpr double kMinimumSampleRateHz = 1.0e-3;

inline bool sampleStale(double now_sec, double stamp_sec, double timeout_sec) {
    if (!std::isfinite(now_sec) || !std::isfinite(stamp_sec) || !std::isfinite(timeout_sec) || timeout_sec < 0.0) {
        return true;
    }
    return now_sec - stamp_sec > timeout_sec;
}

inline bool sampleRateLow(bool sample_received, double period_sec, double min_rate_hz,
                          double min_rate_floor_hz = kMinimumSampleRateHz) {
    if (min_rate_hz <= 0.0 || !sample_received || !std::isfinite(period_sec) || period_sec <= 0.0) {
        return false;
    }
    return 1.0 / period_sec < std::max(min_rate_hz, min_rate_floor_hz);
}

inline bool sampleTimeJumped(double now_sec, double stamp_sec, double period_sec, double future_tolerance_sec = 0.05,
                             double backward_tolerance_sec = 0.05) {
    if (!std::isfinite(now_sec) || !std::isfinite(stamp_sec)) {
        return true;
    }
    if (stamp_sec > now_sec + future_tolerance_sec) {
        return true;
    }
    return std::isfinite(period_sec) && period_sec < -backward_tolerance_sec;
}

} // namespace xgc2_math
