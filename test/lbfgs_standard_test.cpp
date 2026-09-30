#include "xgc2_math/optimization/lbfgs.hpp"

#include <iostream>

namespace {
namespace solver = xgc2_math::optimization::lbfgs;

struct Quadratic {
    int evaluations{0};

    static double evaluate(void* context, const Eigen::VectorXd& x, Eigen::VectorXd& gradient) {
        auto& self = *static_cast<Quadratic*>(context);
        ++self.evaluations;
        const Eigen::Vector3d target(1.0, -2.0, 3.0);
        const Eigen::Vector3d weights(1.0, 2.0, 5.0);
        const Eigen::Vector3d error = x - target;
        gradient = weights.cwiseProduct(error);
        return 0.5 * error.dot(gradient);
    }
};

bool defaultEigenApiConvergesOnKnownQuadratic() {
    Quadratic problem;
    Eigen::VectorXd x = Eigen::Vector3d(-4.0, 7.0, -2.0);
    double cost = 0.0;
    const solver::lbfgs_parameter_t parameters;
    const int result = solver::lbfgs_optimize(x, cost, Quadratic::evaluate, nullptr, nullptr, &problem, parameters);
    return result >= 0 && (x - Eigen::Vector3d(1.0, -2.0, 3.0)).norm() < 1e-4 && cost < 1e-8 && problem.evaluations > 1;
}

bool exactlyZeroInitialGradientStopsWithGradientToleranceDisabled() {
    Quadratic problem;
    Eigen::VectorXd x = Eigen::Vector3d(1.0, -2.0, 3.0);
    double cost = -1.0;
    solver::lbfgs_parameter_t parameters;
    parameters.g_epsilon = 0.0;
    const int result = solver::lbfgs_optimize(x, cost, Quadratic::evaluate, nullptr, nullptr, &problem, parameters);
    return result == solver::LBFGS_CONVERGENCE && problem.evaluations == 1 && cost == 0.0 && x.allFinite();
}
} // namespace

int main() {
    if (!defaultEigenApiConvergesOnKnownQuadratic()) {
        std::cerr << "LBFGS-Lite failed the known quadratic minimum check\n";
        return 1;
    }
    if (!exactlyZeroInitialGradientStopsWithGradientToleranceDisabled()) {
        std::cerr << "LBFGS-Lite failed the v2.3 zero initial gradient regression\n";
        return 1;
    }
    return 0;
}
