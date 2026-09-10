#ifndef XGC2_MATH_RECURSIVE_LEAST_SQUARES_HPP
#define XGC2_MATH_RECURSIVE_LEAST_SQUARES_HPP

#include <algorithm>
#include <array>
#include <cmath>

#include "xgc2_math/observer/differentiator.hpp"
#include "xgc2_math/utils/status.hpp"

namespace xgc2_math {

enum class ScalarRecursiveLeastSquaresStatus {
    kInitialized,
    kAccepted,
    kHeldInvalidInput,
};

template <> struct StatusRegistry<ScalarRecursiveLeastSquaresStatus> {
    static constexpr std::array<StatusDescriptor<ScalarRecursiveLeastSquaresStatus>, 3> statuses{{
        {ScalarRecursiveLeastSquaresStatus::kInitialized, "initialized"},
        {ScalarRecursiveLeastSquaresStatus::kAccepted, "accepted"},
        {ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput, "held_invalid_input"},
    }};
};

struct ScalarRecursiveLeastSquaresOptions {
    double forgetting_factor{0.998};
    double initial_covariance{100.0};
    double min_abs_regressor{1.0e-6};
    double min_covariance{1.0e-9};
    double max_covariance{1.0e9};
};

inline bool isValid(const ScalarRecursiveLeastSquaresOptions& options) {
    return std::isfinite(options.forgetting_factor) && options.forgetting_factor > 0.0 &&
           options.forgetting_factor <= 1.0 && std::isfinite(options.initial_covariance) &&
           options.initial_covariance > 0.0 && std::isfinite(options.min_abs_regressor) &&
           options.min_abs_regressor >= 0.0 && std::isfinite(options.min_covariance) && options.min_covariance > 0.0 &&
           std::isfinite(options.max_covariance) && options.max_covariance >= options.min_covariance;
}

inline ScalarRecursiveLeastSquaresOptions normalized(ScalarRecursiveLeastSquaresOptions options) {
    const ScalarRecursiveLeastSquaresOptions defaults;
    if (!std::isfinite(options.forgetting_factor) || options.forgetting_factor <= 0.0 ||
        options.forgetting_factor > 1.0) {
        options.forgetting_factor = defaults.forgetting_factor;
    }
    if (!std::isfinite(options.initial_covariance) || options.initial_covariance <= 0.0) {
        options.initial_covariance = defaults.initial_covariance;
    }
    if (!std::isfinite(options.min_abs_regressor) || options.min_abs_regressor < 0.0) {
        options.min_abs_regressor = defaults.min_abs_regressor;
    }
    if (!std::isfinite(options.min_covariance) || options.min_covariance <= 0.0) {
        options.min_covariance = defaults.min_covariance;
    }
    if (!std::isfinite(options.max_covariance) || options.max_covariance < options.min_covariance) {
        options.max_covariance = std::max(defaults.max_covariance, options.min_covariance);
    }
    options.initial_covariance =
        std::clamp(options.initial_covariance, options.min_covariance, options.max_covariance);
    return options;
}

struct ScalarRecursiveLeastSquaresSample {
    double parameter{0.0};
    double covariance{100.0};
    double residual{0.0};
    double gain{0.0};
    ScalarRecursiveLeastSquaresStatus status{ScalarRecursiveLeastSquaresStatus::kInitialized};
    bool measurement_accepted{false};
};

class ScalarRecursiveLeastSquares {
  public:
    ScalarRecursiveLeastSquares() = default;

    explicit ScalarRecursiveLeastSquares(const ScalarRecursiveLeastSquaresOptions& options)
        : options_(normalized(options)) {
        covariance_ = options_.initial_covariance;
    }

    void setOptions(const ScalarRecursiveLeastSquaresOptions& options) {
        options_ = normalized(options);
        covariance_ = clampCovariance(covariance_);
    }

    const ScalarRecursiveLeastSquaresOptions& options() const { return options_; }

    void reset() {
        initialized_ = false;
        parameter_ = 0.0;
        covariance_ = clampCovariance(options_.initial_covariance);
    }

    void reset(double initial_parameter) {
        initialized_ = std::isfinite(initial_parameter);
        parameter_ = initialized_ ? initial_parameter : 0.0;
        covariance_ = clampCovariance(options_.initial_covariance);
    }

    void reset(double initial_parameter, double initial_covariance) {
        initialized_ = std::isfinite(initial_parameter);
        parameter_ = initialized_ ? initial_parameter : 0.0;
        covariance_ = clampCovariance(initial_covariance);
    }

    ScalarRecursiveLeastSquaresSample update(double measurement, double regressor) {
        if (!std::isfinite(measurement) || !std::isfinite(regressor)) {
            return sample(0.0, 0.0, ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput, false);
        }
        if (!initialized_) {
            return sample(0.0, 0.0, ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput, false);
        }
        if (std::fabs(regressor) < options_.min_abs_regressor) {
            return sample(0.0, 0.0, ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput, false);
        }

        const double denominator = options_.forgetting_factor + regressor * covariance_ * regressor;
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            return sample(0.0, 0.0, ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput, false);
        }

        const double gain = covariance_ * regressor / denominator;
        const double residual = measurement - regressor * parameter_;
        const double next_parameter = parameter_ + gain * residual;
        // Algebraically P / (lambda + phi * P * phi).  Avoid subtracting
        // gain * phi from one: it can round to one for informative samples,
        // spuriously resetting the covariance to its initial value.
        const double next_covariance = covariance_ / denominator;

        if (!std::isfinite(next_parameter) || !std::isfinite(next_covariance)) {
            return sample(residual, gain, ScalarRecursiveLeastSquaresStatus::kHeldInvalidInput, false);
        }

        parameter_ = next_parameter;
        // A positive update can underflow to zero; it belongs at the lower
        // bound, not at the initial covariance used for invalid reset inputs.
        covariance_ = std::clamp(next_covariance, options_.min_covariance, options_.max_covariance);
        return sample(residual, gain, ScalarRecursiveLeastSquaresStatus::kAccepted, true);
    }

    double parameter() const { return parameter_; }

    double covariance() const { return covariance_; }

    bool initialized() const { return initialized_; }

  private:
    double clampCovariance(double covariance) const {
        if (!std::isfinite(covariance) || covariance <= 0.0) {
            return options_.initial_covariance;
        }
        return std::clamp(covariance, options_.min_covariance, options_.max_covariance);
    }

    ScalarRecursiveLeastSquaresSample sample(double residual, double gain, ScalarRecursiveLeastSquaresStatus status,
                                             bool accepted) const {
        ScalarRecursiveLeastSquaresSample output;
        output.parameter = parameter_;
        output.covariance = covariance_;
        output.residual = residual;
        output.gain = gain;
        output.status = status;
        output.measurement_accepted = accepted;
        return output;
    }

    ScalarRecursiveLeastSquaresOptions options_{};
    double parameter_{0.0};
    double covariance_{100.0};
    bool initialized_{false};
};

} // namespace xgc2_math

#endif // XGC2_MATH_RECURSIVE_LEAST_SQUARES_HPP
