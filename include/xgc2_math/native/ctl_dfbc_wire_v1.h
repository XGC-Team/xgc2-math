#ifndef XGC2_MATH_NATIVE_CTL_DFBC_WIRE_V1_H
#define XGC2_MATH_NATIVE_CTL_DFBC_WIRE_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* "xgc.attitude_rate_cmd/1": geometric controller output (before any
 * vehicle-specific thrust mapping). */
typedef struct xgc_attitude_rate_cmd_v1 {
  double stamp;                 /* stamp of the state it was computed from */
  double specific_thrust;       /* m/s^2 along body z */
  double q_wxyz[4];             /* desired attitude */
  double body_rate[3];          /* rad/s */
  double position_error[3];
  uint32_t success;
  uint32_t flags;
} xgc_attitude_rate_cmd_v1;

#ifdef __cplusplus
#define XGC_SCHEMA_ASSERT static_assert
#else
#define XGC_SCHEMA_ASSERT _Static_assert
#endif
XGC_SCHEMA_ASSERT(sizeof(xgc_attitude_rate_cmd_v1) == 104, "xgc_attitude_rate_cmd_v1");
#undef XGC_SCHEMA_ASSERT

#ifdef __cplusplus
}
#endif
#endif
