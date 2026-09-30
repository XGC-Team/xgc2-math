#pragma once

// Official LBFGS-Lite v2.3, unchanged. Source and MIT license are installed
// alongside the upstream header; see third_party/lbfgs_lite/source.json.
// Upstream narrows Eigen::Index to int. Keep its bytes intact while limiting
// the diagnostic exemption to this vendored include, not its callers.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
#include "xgc2_math/third_party/lbfgs_lite/lbfgs.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace xgc2_math::optimization {
namespace lbfgs = ::lbfgs;
} // namespace xgc2_math::optimization
